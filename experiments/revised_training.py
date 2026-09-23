"""Primitive-balanced fresh-versus-continued training with validation-only selection."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.contrast_pilot import evaluation_deadline
from experiments.judgment_pipeline import compare_probabilities, optimizer_for
from experiments.revised_evaluation import (
    broad_suite,
    evaluate_cases,
    readiness,
    select_checkpoint,
    validation_objective,
    write,
)
from openjev.judgment_cli import load_judgment_engine
from openjev.judgment_model import frozen_digest
from openjev.judgment_training import bundle_loss, load_bundles, prepare_bundle

KINDS = ("noul", "choice", "score")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_group(record):
    group = record.get("group_id", record.get("provenance", {}).get("family"))
    if not isinstance(group, str) or not group:
        raise ValueError("Every training record must name its source group")
    return group


def make_schedule(pools, *, steps, seed):
    if steps < 1 or set(pools) != {"broad", "contrast"}:
        raise ValueError("Invalid training schedule")
    ids = []
    for origin in pools:
        if set(pools[origin]) != set(KINDS):
            raise ValueError("Every origin needs every primitive")
        for sources in pools[origin].values():
            if not sources or any(not group for group in sources.values()):
                raise ValueError("Training groups must be nonempty")
            ids.extend(value for group in sources.values() for value in group)
    if len(ids) != len(set(ids)):
        raise ValueError("Training IDs must be globally unique")
    streams = {}
    for origin in sorted(pools):
        for kind in KINDS:
            for source, values in sorted(pools[origin][kind].items()):
                key = f"{seed}/{origin}/{kind}/{source}"
                rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))
                order = []
                while len(order) < steps:
                    cycle = list(values)
                    rng.shuffle(cycle)
                    order.extend(cycle)
                streams[origin, kind, source] = iter(order)
    result = []
    for step in range(steps):
        batch = []
        for kind in KINDS:
            for origin in ("broad", "contrast"):
                sources = sorted(pools[origin][kind])
                source = sources[step % len(sources)]
                batch.append({"origin": origin, "primitive": kind, "source": source,
                              "id": next(streams[origin, kind, source])})
        result.append(batch)
    return result


def new_engine(arm, checkpoint):
    engine = load_judgment_engine(checkpoint=checkpoint if arm == "continued" else None,
                                  trainable=True, backend="fla", max_input_tokens=1536, unit_batch_size=4)
    if arm == "fresh":
        engine.model.add_lora(rank=8, alpha=16)
    engine.model.parameter_inventory()
    return engine


def prepare_populations(engine, data, output):
    pools = {origin: {kind: {} for kind in KINDS} for origin in ("broad", "contrast")}
    prepared, population, excluded = {}, [], []
    broad = load_bundles(Path("data/processed-v0.2/train.jsonl"))
    for row in sorted(broad, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()):
        if row["provenance"]["assigned_split"] != "train":
            raise ValueError("Replay contains nontraining source")
        source = row["provenance"]["dataset"]
        example = row["examples"][0]
        kind = example["question"]["type"]
        group = pools["broad"][kind].setdefault(source, [])
        if len(group) >= 800:
            continue
        identity = "broad/" + row["id"]
        item = {"id": identity, "examples": [example], "relations": [], "provenance": row["provenance"]}
        try:
            bundle = prepare_bundle(engine, item)
        except ValueError as error:
            if "exceeds" not in str(error):
                raise
            excluded.append(identity)
            continue
        prepared[identity] = bundle
        group.append(identity)
        population.append({"id": identity, "group_id": source_group(row), "origin": "broad", "source": source, "primitive": kind})
    for row in load_bundles(data / "train.jsonl"):
        for index, example in enumerate(row["examples"]):
            kind = example["question"]["type"]
            identity = "contrast/" + row["id"] + "/" + str(index)
            source = "revised"
            item = {"id": identity, "examples": [example], "relations": [], "provenance": {"dataset": source}}
            prepared[identity] = prepare_bundle(engine, item)
            pools["contrast"][kind].setdefault(source, []).append(identity)
            population.append({"id": identity, "group_id": source_group(row), "origin": "contrast", "primitive": kind})
    write(output / "populations.json", {"pools": pools, "records": population, "excluded_oversize": excluded,
          "max_unit_tokens": max(len(p) for b in prepared.values() for p in b.prompts)})
    write(output / "prepared-prompt-hashes.json", {key: hashlib.sha256(json.dumps(value.prompts).encode()).hexdigest() for key, value in prepared.items()})
    return prepared, pools


def validation_records():
    counts, records = {}, []
    for row in sorted(load_bundles(Path("data/processed-v0.2/validation.jsonl")), key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()):
        if row["provenance"]["assigned_split"] != "validation":
            raise ValueError("Wrong validation partition")
        source = row["provenance"]["dataset"]
        if counts.get(source, 0) < 50:
            counts[source] = counts.get(source, 0) + 1
            records.append(row)
    if len(records) != 300:
        raise ValueError("Need all six validation sources")
    return records


def evaluate_pair(engine, suite, broad, output, deadline):
    output.mkdir(parents=True, exist_ok=False)
    with evaluation_deadline(deadline):
        result = {"revised": evaluate_cases(engine, suite, output / "revised"),
                  "broad": evaluate_cases(engine, broad_suite(broad), output / "broad")}
    write(output / "evaluation.json", result)
    return result


def save(engine, path, arm, step, optimizer=None):
    identity = engine.model.save_checkpoint(path, metadata={"study": "revised-initialization-v1", "arm": arm, "step": step},
                                            optimizer=optimizer, training_state={"arm": arm, "step": step})
    engine.model_id = identity
    return identity


def fit_arm(engine, arm, schedule, prepared, data, broad_validation, args, root, *, smoke=False):
    model = engine.model
    frozen_before = frozen_digest(model)
    optimizer = optimizer_for(model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
    trainable = [p for p in model.parameters() if p.requires_grad]
    curve, elapsed, history, step = [], 0., [], 0
    update_started = None
    suite = read(data / "validation/suite.json")
    checkpoints = (0, 250, 500, 1000) if not smoke else ()

    def validate(at_step):
        checkpoint = root / f"step-{at_step:04d}"
        identity = save(engine, checkpoint, arm, at_step, optimizer)
        values = evaluate_pair(engine, suite, broad_validation, root / f"validation-{at_step:04d}", args.evaluation_max_seconds)
        curve.append({"status": "complete", "step": at_step, "checkpoint": str(checkpoint), "checkpoint_id": identity,
                      "objective": validation_objective(values), "evaluation": values})
        write(root / "validation-curve.json", curve)

    try:
        if 0 in checkpoints:
            validate(0)
        for step, batch in enumerate(schedule, 1):
            if elapsed > args.max_seconds:
                raise TimeoutError("Training time budget exceeded")
            update_started = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total, per_kind = 0., {kind: 0. for kind in KINDS}
            for item in batch:
                bundle = prepared[item["id"]]
                if len(bundle.groups) != 1 or bundle.groups[0].primitive != item["primitive"]:
                    raise ValueError("Schedule must contain complete single-question groups")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model.score_prompts(bundle.prompts, bundle.kinds, unit_batch_size=4)
                    loss = bundle_loss(logits, bundle.groups, [])["supervised"]
                    (loss / 6).backward()
                total += loss.item() / 6
                per_kind[item["primitive"]] += loss.item() / 2
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            if step == 1:
                gradient_check = {"adapters": any(p.grad is not None and p.grad.abs().max().item() > 0 for n, p in model.named_parameters() if ".lora_" in n),
                                  "heads": any(p.grad is not None and p.grad.abs().max().item() > 0 for n, p in model.named_parameters() if n.startswith(("binary.", "compatibility."))),
                                  "frozen": all(p.grad is None for p in model.parameters() if not p.requires_grad)}
                if not all(gradient_check.values()):
                    raise RuntimeError("Gradient policy check failed")
            optimizer.step()
            engine.synchronize()
            elapsed += time.monotonic() - update_started
            update_started = None
            entry = {"step": step, "loss": total, "by_primitive": per_kind, "gradient_norm": norm.item(), "training_seconds": elapsed}
            history.append(entry)
            with (root / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
            if step == 1 or step % 50 == 0:
                print(arm, f"step {step}/{len(schedule)} loss {total:.4f}; training {elapsed:.1f}s", flush=True)
            if elapsed > args.max_seconds:
                raise TimeoutError("Training time budget exceeded in update")
            if step in checkpoints:
                validate(step)
        optimizer.zero_grad(set_to_none=True)
        frozen_after = frozen_digest(model)
        if frozen_after != frozen_before:
            raise RuntimeError("Frozen original weights changed")
        result = {"status": "complete", "steps": step, "training_seconds": elapsed, "gradient_check": gradient_check,
                  "frozen_before": frozen_before, "frozen_after": frozen_after, "inventory": model.parameter_inventory()}
        if smoke:
            result["checkpoint_id"] = save(engine, root / "final", arm, step, optimizer)
        else:
            result["selected"] = select_checkpoint(curve)
            result["still_improving_last_interval"] = curve[-1]["objective"] < curve[-2]["objective"] - .01
        write(root / "training.json", result)
        return result
    except BaseException:
        if update_started is not None:
            elapsed += time.monotonic() - update_started
        optimizer.zero_grad(set_to_none=True)
        try:
            model.save_checkpoint(root / f"partial-completed-{len(history):04d}",
                                  metadata={"study": "revised-initialization-v1", "arm": arm, "status": "failed",
                                            "completed_updates": len(history), "attempted_step": step, "resumable": False},
                                  optimizer=optimizer, training_state={"step": len(history), "attempted_step": step, "resumable": False})
        except Exception:
            pass
        write(root / "training.json", {"status": "failed", "steps": len(history), "training_seconds": elapsed})
        raise


def fingerprint(args):
    files = [Path(__file__), *[Path("experiments") / name for name in (
        "contrast_revision_data.py", "revised_evaluation.py", "revised_metrics.py", "semantic_contrasts.py", "contrast_pilot.py", "judgment_pipeline.py")],
        *[Path("src/openjev") / name for name in ("judgment_model.py", "judgment_training.py", "judgments.py", "judgment_cli.py", "runtime.py")]]
    data = [*(args.data / split / "suite.json" for split in ("train", "validation", "test")),
            args.data / "train.jsonl", *[Path("data/processed-v0.2") / (split + ".jsonl") for split in ("train", "validation", "test")],
            Path("data/semantic-contrasts-v1/suite.json")]
    return {"sources": {str(p): sha(p) for p in files}, "data": {str(p): sha(p) for p in data},
            "starting_checkpoint_id": read(args.checkpoint / "checkpoint.json")["checkpoint_id"],
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "fla-core")},
            "steps": 1000, "seed": 42, "lr": 5e-5, "head_lr": 2.5e-5,
            "max_tokens": 1536, "unit_batch_size": 4, "max_training_seconds": args.max_seconds,
            "max_evaluation_seconds": args.evaluation_max_seconds}


def run(args, report):
    report["fingerprint"] = fingerprint(args)
    for split in ("train", "validation", "test"):
        if sha(args.data / split / "suite.json") != read(args.data / split / "manifest.json")["suite_sha256"]:
            raise ValueError("Frozen data differs from manifest")
    if args.mode == "run":
        gate = read(args.gate)
        if gate.get("mode") != "smoke" or gate.get("status") != "complete" or gate.get("fingerprint") != report["fingerprint"]:
            raise ValueError("Matching completed source/data smoke gate required")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    engine = new_engine("continued", args.checkpoint)
    prepared, pools = prepare_populations(engine, args.data, args.output)
    broad_validation = validation_records()
    write(args.output / "validation-ids.json", [r["id"] for r in broad_validation])
    schedule = make_schedule(pools, steps=1000, seed=42)
    write(args.output / "schedule.json", schedule)
    report["schedule_sha256"] = sha(args.output / "schedule.json")
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    report["arms"] = {}
    for arm in ("fresh", "continued"):
        torch.manual_seed(42)
        engine = new_engine(arm, args.checkpoint)
        root = args.output / arm
        root.mkdir()
        actual_schedule = schedule[:2] if args.mode == "smoke" else schedule
        report["arms"][arm] = fit_arm(engine, arm, actual_schedule, prepared, args.data, broad_validation, args, root, smoke=args.mode == "smoke")
        if args.mode == "smoke":
            request = read(args.data / "train/suite.json")["cases"][0]["request"]
            a = engine.evaluate(request, cached=False, details=True)
            del engine
            gc.collect()
            torch.cuda.empty_cache()
            engine = load_judgment_engine(checkpoint=root / "final", backend="fla", unit_batch_size=4)
            b = engine.evaluate(request, cached=False, details=True)
            report["arms"][arm]["reload_max_difference"] = compare_probabilities(a["answers"], b["answers"], tolerance=1e-5)
        write(args.output / "report.json", report)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    if args.mode == "run":
        selected = {arm: report["arms"][arm]["selected"] for arm in ("fresh", "continued")}
        write(args.output / "selection-locked.json", selected)
        report["tests"] = {}
        test_suite = read(args.data / "test/suite.json")
        broad_test = load_bundles(Path("data/processed-v0.2/test.jsonl"))
        for arm in ("reference", "fresh", "continued"):
            checkpoint = args.checkpoint if arm == "reference" else Path(selected[arm]["checkpoint"])
            engine = load_judgment_engine(checkpoint=checkpoint, backend="fla", unit_batch_size=4, max_input_tokens=1536)
            root = args.output / (arm + "-test")
            result = evaluate_pair(engine, test_suite, broad_test, root, args.evaluation_max_seconds)
            with evaluation_deadline(args.evaluation_max_seconds):
                result["legacy"] = evaluate_cases(engine, read("data/semantic-contrasts-v1/suite.json"), root / "legacy")
            report["tests"][arm] = result
            if arm != "reference":
                rows = read(root / "revised/predictions.json")
                report["arms"][arm]["readiness"] = readiness(result, report["tests"]["reference"], rows)
            write(args.output / "report.json", report)
            del engine
            gc.collect()
            torch.cuda.empty_cache()
    report["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/contrast-revision-v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/judgment-full-v0.2/final"))
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--evaluation-max-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.max_seconds <= 0 or args.evaluation_max_seconds <= 0 or (args.mode == "run" and not args.gate):
        parser.error("Positive time budgets and an exact smoke gate are required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "mode": args.mode, "started_utc": datetime.now(timezone.utc).isoformat()}
    write(args.output / "report.json", report)
    try:
        run(args, report)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output / "report.json", report)


if __name__ == "__main__":
    main()
