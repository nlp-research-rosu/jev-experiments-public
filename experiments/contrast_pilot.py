"""Fixed-budget continuation controls for the small semantic contrast pilot."""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import random
import signal
import statistics
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.contrast_component_probe import evaluate
from experiments.judgment_pipeline import compare_probabilities, metrics_only, optimizer_for
from openjev.judgment_cli import load_judgment_engine
from openjev.judgment_model import frozen_digest
from openjev.judgment_training import bundle_loss, evaluate_bundles, load_bundles, prepare_bundle


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def evaluation_deadline(seconds):
    """Bound this standalone main-thread evaluation, including blocking calls."""
    if seconds <= 0 or signal.getitimer(signal.ITIMER_REAL) != (0, 0):
        raise ValueError("Need a positive evaluation bound and no active alarm")
    previous = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()

    def expired(signum, frame):
        raise TimeoutError("Pilot evaluation exceeded its cap")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
        if time.monotonic() - started > seconds:
            raise TimeoutError("Pilot evaluation exceeded its cap")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def validate_smoke(gate, fingerprint):
    if gate.get("mode") != "smoke" or gate.get("status") != "complete" or not gate.get("passed") or gate.get("fingerprint") != fingerprint:
        raise ValueError("A completed smoke gate for this exact source/data/policy is required")


def evaluate_broad(engine, prepared, output):
    """Stream raw logits immediately; aggregate with the unchanged evaluator."""
    identities = iter(b.id for b in prepared)

    class Recorder:
        def eval(self):
            engine.model.eval()

        def score_prompts(self, *args, **kwargs):
            identity = next(identities)
            logits = engine.model.score_prompts(*args, **kwargs)
            with output.open("a") as stream:
                stream.write(json.dumps({"id": identity, "logits": logits.detach().cpu().tolist()}, allow_nan=False) + "\n")
            return logits

    return evaluate_bundles(SimpleNamespace(model=Recorder(), unit_batch_size=engine.unit_batch_size), prepared)


def canonical_replay(record):
    if record.get("provenance", {}).get("assigned_split") != "train":
        raise ValueError("Replay must come exclusively from the original training partition")
    result = copy.deepcopy(record)
    result["examples"] = result["examples"][:1]
    result["relations"] = []
    return result


def build_schedule(contrast_ids, replay_pool, *, arm, steps, seed):
    replay_ids = [v for values in replay_pool.values() for v in values]
    if (
        arm not in ("control", "contrast") or steps < 1 or not contrast_ids or not replay_pool
        or any(not values for values in replay_pool.values())
        or len(set(replay_ids)) != len(replay_ids) or len(set(contrast_ids)) != len(contrast_ids)
        or set(contrast_ids) & set(replay_ids)
    ):
        raise ValueError("Invalid or overlapping training populations")
    sources = sorted(replay_pool)
    common_rng, intervention_rng, contrast_rng = random.Random(seed), random.Random(seed + 1), random.Random(seed + 2)
    order = []
    while len(order) < steps * 2:
        cycle = list(contrast_ids)
        contrast_rng.shuffle(cycle)
        order.extend(cycle)
    result = []
    for step in range(steps):
        batch = []
        for slot in range(4):
            source = sources[(4 * step + slot) % len(sources)]
            if slot < 2:
                kind, identity = "replay", common_rng.choice(replay_pool[source])
            elif arm == "control":
                kind, identity = "replay", intervention_rng.choice(replay_pool[source])
            else:
                kind, identity = "contrast", order[2 * step + slot - 2]
            batch.append({"kind": kind, "id": identity})
        result.append(batch)
    return result


def prepare_training(engine, data, output):
    contrast = {b["id"]: prepare_bundle(engine, b) for b in load_bundles(data / "train.jsonl")}
    pool, replay, excluded = {}, {}, []
    for record in load_bundles(Path("data/processed-v0.2/train.jsonl")):
        source = record["provenance"]["dataset"]
        if len(pool.get(source, [])) >= 100:
            continue
        item = canonical_replay(record)
        try:
            prepared = prepare_bundle(engine, item)
        except ValueError as exc:
            if "exceeds" not in str(exc):
                raise
            excluded.append(record["id"])
            continue
        replay[prepared.id] = prepared
        pool.setdefault(source, []).append(prepared.id)
    write(output / "training-populations.json", {
        "contrast_ids": list(contrast), "replay_by_source": pool, "excluded_oversize": excluded,
        "max_contrast_unit_tokens": max(len(p) for b in contrast.values() for p in b.prompts),
        "max_replay_unit_tokens": max(len(p) for b in replay.values() for p in b.prompts),
    })
    return contrast, replay, pool


def train_arm(engine, schedule, contrast, replay, output, *, max_seconds):
    model = engine.model
    inventory = model.parameter_inventory()
    frozen_before = frozen_digest(model)
    optimizer = optimizer_for(model, SimpleNamespace(lr=1e-4, head_lr=5e-5))
    started = time.monotonic()
    total_questions, total_rows = 0, 0
    history = []
    trainable = [p for p in model.parameters() if p.requires_grad]
    write(output / "schedule.json", schedule)
    for step, batch in enumerate(schedule, 1):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("Pilot arm exceeded its training cap; no complete result")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for item in batch:
            bundle = (contrast if item["kind"] == "contrast" else replay)[item["id"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.score_prompts(bundle.prompts, bundle.kinds, unit_batch_size=4)
                loss = bundle_loss(logits, bundle.groups, bundle.relations)["total"]
                (loss / len(batch)).backward()
            loss_total += loss.item() / len(batch)
            total_questions += len(bundle.groups)
            total_rows += len(bundle.prompts)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        if step == 1:
            gradient_check = {
                "adapters": any(p.grad is not None and p.grad.abs().max().item() > 0
                                for n, p in model.named_parameters() if ".lora_" in n),
                "heads": any(p.grad is not None and p.grad.abs().max().item() > 0
                             for n, p in model.named_parameters() if n.startswith(("binary.", "compatibility."))),
                "base_frozen": all(p.grad is None for p in model.parameters() if not p.requires_grad),
            }
            if not all(gradient_check.values()):
                raise RuntimeError("Pilot gradient policy failed")
        optimizer.step()
        entry = {"step": step, "loss": loss_total, "gradient_norm": norm.item(), "elapsed_seconds": time.monotonic() - started}
        history.append(entry)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        if step == 1 or step % 25 == 0:
            print(output.name, f"step {step}/{len(schedule)} loss {loss_total:.4f}", flush=True)
        if time.monotonic() - started > max_seconds:
            model.save_checkpoint(output / "partial-timeout", metadata={"completed": False, "steps": step})
            raise TimeoutError("Pilot arm crossed its training deadline during an update")
    optimizer.zero_grad(set_to_none=True)
    frozen_after = frozen_digest(model)
    if frozen_after != frozen_before:
        raise RuntimeError("Frozen backbone changed")
    model.eval()
    result = {
        "steps": len(history), "completed": len(history) == len(schedule), "inventory": inventory,
        "gradient_check": gradient_check, "frozen_before": frozen_before, "frozen_after": frozen_after,
        "questions_processed": total_questions, "candidate_rows_processed": total_rows,
        "elapsed_training_seconds": time.monotonic() - started,
    }
    result["checkpoint_id"] = model.save_checkpoint(output / "final", metadata=result)
    engine.model_id = result["checkpoint_id"]
    write(output / "training.json", result)
    return result


def evaluate_arm(engine, data, output):
    result = {}
    suites = {
        "validation": read(data / "validation/suite.json"),
        "test": read(data / "test/suite.json"),
        "regression": read("data/semantic-contrasts-v1/suite.json"),
    }
    for name, suite in suites.items():
        root = output / name
        root.mkdir()
        result[name], _ = evaluate(engine, suite, "full", root)
    broad = [prepare_bundle(engine, b) for b in load_bundles(Path("data/processed-v0.2/test.jsonl"))]
    broad_results = evaluate_broad(engine, broad, output / "broad-raw.jsonl")
    write(output / "broad-test.json", broad_results)
    result["broad"] = metrics_only(broad_results)
    write(output / "evaluation.json", result)
    return result


def scaling_gate(evaluations):
    def metrics(arm):
        test = evaluations[arm]["test"]["summary"]
        return {
            "accuracy": test["overall"]["accuracy"],
            "macro_binary_balanced_accuracy": statistics.mean(
                v["balanced_binary_accuracy"] for v in test["by_question"].values() if "balanced_binary_accuracy" in v
            ),
            "macro_pair_correctness": statistics.mean(test["relations"][k]["both_correct_rate"]
                                                       for k in ("flip", "question_contrast", "invariant")),
            "broad_accuracy": evaluations[arm]["broad"]["canonical"]["accuracy"],
        }
    measured = {arm: metrics(arm) for arm in ("starting", "control", "contrast")}
    a, b, c = (measured[k] for k in ("starting", "control", "contrast"))
    checks = {
        "test_gain_vs_control_at_least_10pp": c["accuracy"] >= b["accuracy"] + 0.10 - 1e-12,
        "balanced_binary_improves_over_both": c["macro_binary_balanced_accuracy"] > max(a["macro_binary_balanced_accuracy"], b["macro_binary_balanced_accuracy"]),
        "mean_pair_correctness_improves_over_both": c["macro_pair_correctness"] > max(a["macro_pair_correctness"], b["macro_pair_correctness"]),
        "broad_retention_within_2pp": c["broad_accuracy"] >= a["broad_accuracy"] - 0.02 - 1e-12,
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": measured}


def run(args, report):
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    for split in ("train", "validation", "test"):
        if sha(args.data / split / "suite.json") != read(args.data / split / "manifest.json")["suite_sha256"]:
            raise ValueError("Frozen pilot suite changed")
    report["fingerprint"] = {
        "sources": {str(p): sha(p) for p in [Path(__file__), Path("experiments/contrast_pilot_data.py"),
                    *[Path("experiments") / name for name in ("contrast_component_probe.py", "semantic_contrasts.py", "judgment_pipeline.py")],
                    *[Path("src/openjev") / name for name in ("judgment_training.py", "judgment_model.py", "judgments.py", "judgment_cli.py", "runtime.py")]]},
        "data": {str(p): sha(p) for p in [*(args.data / f"{s}.jsonl" for s in ("train", "validation", "test")),
                   *(args.data / s / "suite.json" for s in ("train", "validation", "test")),
                   Path("data/semantic-contrasts-v1/suite.json"),
                   Path("data/processed-v0.2/train.jsonl"), Path("data/processed-v0.2/test.jsonl")]},
        "starting_checkpoint": read(args.checkpoint / "checkpoint.json")["checkpoint_id"],
        "seed": 42, "steps_per_arm": args.steps, "lr": 1e-4, "head_lr": 5e-5,
        "max_input_tokens": 1536, "unit_batch_size": 4, "replay_pool_per_source": 100,
        "max_training_seconds": args.max_seconds, "max_evaluation_seconds": args.evaluation_max_seconds,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "fla-core")},
    }
    engine = load_judgment_engine(checkpoint=args.checkpoint, backend="fla", trainable=True, max_input_tokens=1536, unit_batch_size=4)
    contrast, replay, pool = prepare_training(engine, args.data, args.output)
    report["evaluations"], report["training"] = {}, {}
    if args.mode == "smoke":
        root = args.output / "smoke"
        root.mkdir()
        # Include the longest case explicitly before one ordinary matched step.
        largest = max(contrast.values(), key=lambda b: max(map(len, b.prompts))).id
        schedule = build_schedule(list(contrast), pool, arm="contrast", steps=2, seed=42)
        schedule[0][2] = {"kind": "contrast", "id": largest}
        report["training"]["smoke"] = train_arm(engine, schedule, contrast, replay, root, max_seconds=args.max_seconds)
        request = read(args.data / "train/suite.json")["cases"][0]["request"]
        before = engine.evaluate(request, details=True, cached=False)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = load_judgment_engine(checkpoint=root / "final", backend="fla", unit_batch_size=4)
        after = reloaded.evaluate(request, details=True, cached=False)
        report["reload_max_difference"] = compare_probabilities(before["answers"], after["answers"], tolerance=1e-5)
        report.update(status="complete", passed=True)
    else:
        gate = read(args.gate)
        validate_smoke(gate, report["fingerprint"])
        root = args.output / "starting"
        root.mkdir()
        with evaluation_deadline(args.evaluation_max_seconds):
            report["evaluations"]["starting"] = evaluate_arm(engine, args.data, root)
        write(args.output / "report.json", report)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        for arm in ("control", "contrast"):
            torch.manual_seed(42)
            engine = load_judgment_engine(checkpoint=args.checkpoint, backend="fla", trainable=True, max_input_tokens=1536, unit_batch_size=4)
            root = args.output / arm
            root.mkdir()
            schedule = build_schedule(list(contrast), pool, arm=arm, steps=args.steps, seed=42)
            report["training"][arm] = train_arm(engine, schedule, contrast, replay, root, max_seconds=args.max_seconds)
            with evaluation_deadline(args.evaluation_max_seconds):
                report["evaluations"][arm] = evaluate_arm(engine, args.data, root)
            write(args.output / "report.json", report)
            del engine
            gc.collect()
            torch.cuda.empty_cache()
        report["scaling_gate"] = scaling_gate(report["evaluations"])
        report.update(status="complete", passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "run"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", default=Path("data/contrast-pilot-v1"), type=Path)
    parser.add_argument("--checkpoint", default=Path("checkpoints/judgment-full-v0.2/final"), type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--evaluation-max-seconds", type=float, default=1200)
    args = parser.parse_args()
    if args.steps < 1 or args.max_seconds <= 0 or args.evaluation_max_seconds <= 0 or (args.mode == "run" and not args.gate):
        parser.error("Positive bounds and a smoke gate for the run are required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "mode": args.mode, "passed": False, "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output / "report.json", report)


if __name__ == "__main__":
    main()
