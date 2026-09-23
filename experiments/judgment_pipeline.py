"""Gated baseline, smoke training, checkpoint checks and bounded full training."""

import argparse
import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import torch

from openjev.judgment_cli import load_judgment_engine, read_json
from openjev.judgment_model import JudgmentEngine, JudgmentModel, frozen_digest
from openjev.judgment_training import (
    PreparedBundle,
    TrainingGroup,
    bundle_loss,
    evaluate_bundles,
    load_bundles,
    prepare_bundle,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(args):
    files = [
        Path(__file__),
        *[
            Path("src/openjev") / f
            for f in ("judgments.py", "judgment_model.py", "judgment_training.py", "judgment_cli.py")
        ],
    ]
    return {
        "source_sha256": {str(p): sha(p) for p in files},
        "data_sha256": {
            name: sha(args.data / (name + ".jsonl")) for name in ("train", "validation", "calibration", "test")
        },
        "packages": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "fla-core", "pyarrow")
        },
        "backend": args.backend,
        "unit_batch_size": args.unit_batch_size,
        "accumulation": args.accumulation,
        "training_token_limit": args.max_tokens,
        "rank": 8,
        "alpha": 16,
        "adapter_lr": args.lr,
        "head_lr": args.head_lr,
        "consistency_weight": args.consistency_weight,
        "seed": args.seed,
    }


def write_report(path, result):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_gate(gate, fp):
    if (
        gate.get("stage") != "smoke"
        or not gate.get("passed")
        or not gate.get("completed_requested_pass")
        or gate.get("fingerprint") != fp
    ):
        raise ValueError(
            "full training requires a completed passing smoke gate for this exact source/data/runtime policy"
        )


def compare_probabilities(a, b, *, tolerance=0.03):
    """Check matching tree topology and bounded BF16 cache/full differences."""

    def collect(node, path=()):
        if isinstance(node, dict):
            if isinstance(node.get("type"), str) and isinstance(node.get("probabilities"), dict):
                return {path + (key,): value for key, value in node["probabilities"].items()}
            return {p: v for key, value in node.items() for p, v in collect(value, path + (key,)).items()}
        if isinstance(node, list):
            return {p: v for key, value in enumerate(node) for p, v in collect(value, path + (key,)).items()}
        return {}

    x, y = collect(a), collect(b)
    if not x or x.keys() != y.keys():
        raise ValueError("cached/full answer paths differ")
    maximum = max(abs(x[key] - y[key]) for key in x)
    if not math.isfinite(maximum) or maximum > tolerance:
        raise ValueError(f"cached/full probabilities differ by {maximum}, exceeding {tolerance}")
    return maximum


def prepare_split(engine, path, max_tokens, cache_root):
    key = hashlib.sha256(
        (
            sha(path)
            + sha("src/openjev/judgments.py")
            + sha("src/openjev/judgment_training.py")
            + str(max_tokens)
            + engine.model.revision
        ).encode()
    ).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / (key + ".jsonl.gz")
    counts_path = cache_root / (key + ".counts.json")
    if cache.exists() and counts_path.exists():
        result = []
        with gzip.open(cache, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                row["groups"] = [TrainingGroup(**{**g, "indices": tuple(g["indices"])}) for g in row["groups"]]
                result.append(PreparedBundle(**row))
        return result, json.loads(counts_path.read_text())
    records = load_bundles(path)
    result, rejected = [], []
    for index, record in enumerate(records):
        prepared = prepare_bundle(engine, record)
        length = max(map(len, prepared.prompts))
        if length > max_tokens:
            rejected.append({"id": record["id"], "tokens": length})
            continue
        # Validate target/relation alignment without evaluating a neural model.
        bundle_loss(torch.zeros(len(prepared.prompts)), prepared.groups, prepared.relations)
        result.append(prepared)
        if (index + 1) % 2000 == 0:
            print(f"Prepared {path.stem} {index + 1}/{len(records)}", flush=True)
    if not result:
        raise ValueError("no eligible records after token bound")
    counts = {
        "source_bundles": len(records),
        "eligible_bundles": len(result),
        "max_tokens": max_tokens,
        "rejected_oversized": rejected,
        "units": sum(len(r.prompts) for r in result),
    }
    temporary = cache.with_suffix(".tmp.gz")
    with gzip.open(temporary, "wt") as stream:
        for row in result:
            stream.write(json.dumps(asdict(row), separators=(",", ":")) + "\n")
    temporary.replace(cache)
    counts_path.write_text(json.dumps(counts, indent=2) + "\n")
    return result, counts


def balanced_sample(prepared, per_source):
    counts, sample = {}, []
    for row in sorted(prepared, key=lambda r: hashlib.sha256(r.id.encode()).hexdigest()):
        count = counts.get(row.source, 0)
        if count < per_source:
            sample.append(row)
            counts[row.source] = count + 1
    return sample


def metrics_only(result):
    return {k: v for k, v in result.items() if k != "predictions"}


def optimizer_for(model, args):
    adapters = [p for n, p in model.named_parameters() if p.requires_grad and ".lora_" in n]
    heads = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(("compatibility.", "binary."))]
    return torch.optim.AdamW(
        [{"params": adapters, "lr": args.lr}, {"params": heads, "lr": args.head_lr}], weight_decay=0.01
    )


def train(engine, prepared, validation, args, report):
    model = engine.model
    optimizer = optimizer_for(model, args)
    trainable = [p for p in model.parameters() if p.requires_grad]
    progress = {"step": 0, "epoch": 0, "position": 0, "bundles_seen": 0}
    if args.resume:
        saved = torch.load(args.resume / "training.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(saved["optimizer"])
        progress = saved["state"]
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if args.stage == "smoke":
        training = balanced_sample(prepared, 2)
        target_steps = args.smoke_steps
        epochs = math.ceil(target_steps * args.accumulation / len(training)) + 1
    else:
        training = prepared
        target_steps = math.ceil(len(training) / args.accumulation) * args.epochs
        epochs = args.epochs
    report["training_bundles"] = len(training)
    report["target_steps"] = target_steps
    initial = evaluate_bundles(engine, training if args.stage == "smoke" else validation)
    report["before_training"] = metrics_only(initial)
    write_report(args.output / "before-predictions.json", initial)
    frozen_before = frozen_digest(model)
    report["frozen_before"] = frozen_before
    history = []
    started = time.monotonic()
    completed = False
    batch_rng = None
    batch_cuda_rng = None
    try:
        for epoch in range(progress["epoch"], epochs):
            order = list(range(len(training)))
            random.Random(args.seed + epoch).shuffle(order)
            begin = progress["position"] if epoch == progress["epoch"] else 0
            for position in range(begin, len(order), args.accumulation):
                if progress["step"] >= target_steps:
                    completed = True
                    break
                if time.monotonic() - started >= args.max_seconds:
                    break
                selected = order[position : position + args.accumulation]
                batch_rng = torch.get_rng_state()
                batch_cuda_rng = torch.cuda.get_rng_state_all()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                sup = cons = total = 0.0
                step_started = time.monotonic()
                for index in selected:
                    bundle = training[index]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model.score_prompts(bundle.prompts, bundle.kinds, unit_batch_size=args.unit_batch_size)
                        losses = bundle_loss(
                            logits, bundle.groups, bundle.relations, consistency_weight=args.consistency_weight
                        )
                        (losses["total"] / len(selected)).backward()
                    sup += losses["supervised"].item() / len(selected)
                    cons += losses["consistency"].item() / len(selected)
                    total += losses["total"].item() / len(selected)
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                if progress["step"] == 0:
                    report["first_gradient_check"] = {
                        "adapter_nonzero": any(
                            p.grad is not None and p.grad.abs().max().item() > 0
                            for n, p in model.named_parameters()
                            if ".lora_B" in n
                        ),
                        "readout_nonzero": any(
                            p.grad is not None and p.grad.abs().max().item() > 0
                            for n, p in model.named_parameters()
                            if n.startswith(("compatibility.", "binary."))
                        ),
                        "no_frozen_gradients": all(p.grad is None for p in model.parameters() if not p.requires_grad),
                    }
                    if not all(report["first_gradient_check"].values()):
                        raise RuntimeError("gradient policy check failed")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                progress.update(
                    step=progress["step"] + 1,
                    epoch=epoch,
                    position=position + len(selected),
                    bundles_seen=progress["bundles_seen"] + len(selected),
                )
                engine.model_id = f"openjev-judgment-v0.2/run-{report['run_id']}/step-{progress['step']}"
                batch_rng = batch_cuda_rng = None
                entry = {
                    **progress,
                    "loss": total,
                    "supervised": sup,
                    "consistency": cons,
                    "gradient_norm": norm.item(),
                    "seconds": time.monotonic() - step_started,
                }
                history.append(entry)
                with (args.output / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
                if progress["step"] % 10 == 0 or progress["step"] == 1:
                    print(
                        f"Step {progress['step']}/{target_steps}; loss {total:.4f}; {entry['seconds']:.2f}s; peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                        flush=True,
                    )
                    report.update(
                        progress=progress.copy(),
                        elapsed_training_seconds=time.monotonic() - started,
                        last_training=entry,
                    )
                    write_report(args.output / "report.json", report)
                if args.stage == "full" and progress["step"] % args.checkpoint_every == 0:
                    model.save_checkpoint(
                        args.output / f"step-{progress['step']:06d}",
                        metadata={
                            "stage": args.stage,
                            "progress": progress.copy(),
                            "fingerprint": report["fingerprint"],
                        },
                        optimizer=optimizer,
                        training_state=progress.copy(),
                    )
                    evaluation = evaluate_bundles(engine, validation)
                    write_report(args.output / f"validation-{progress['step']:06d}.json", evaluation)
                if progress["step"] >= target_steps:
                    completed = True
                    break
            if completed or time.monotonic() - started >= args.max_seconds:
                break
            progress.update(epoch=epoch + 1, position=0)
        if args.stage == "full" and progress["epoch"] >= epochs:
            completed = True
    except BaseException as exc:
        optimizer.zero_grad(set_to_none=True)
        if batch_rng is not None:
            torch.set_rng_state(batch_rng)
            torch.cuda.set_rng_state_all(batch_cuda_rng)
        model.save_checkpoint(
            args.output / f"interrupted-{progress['step']:06d}",
            metadata={
                "stage": args.stage,
                "progress": progress.copy(),
                "fingerprint": report["fingerprint"],
                "error": repr(exc),
            },
            optimizer=optimizer,
            training_state=progress.copy(),
        )
        report.update(status="failed", error=repr(exc), progress=progress)
        write_report(args.output / "report.json", report)
        raise
    report.update(
        progress=progress.copy(),
        completed_requested_pass=completed,
        elapsed_training_seconds=time.monotonic() - started,
    )
    frozen_after = frozen_digest(model)
    report["frozen_after"] = frozen_after
    if frozen_after != frozen_before:
        raise RuntimeError("frozen base weights changed")
    model.eval()
    report["after_training"] = metrics_only(evaluate_bundles(engine, training if args.stage == "smoke" else validation))
    model.save_checkpoint(
        args.output / "final",
        metadata={"stage": args.stage, "progress": progress.copy(), "fingerprint": report["fingerprint"]},
        optimizer=optimizer,
        training_state=progress.copy(),
    )
    return training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed-v0.2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=Path("reports/judgment-smoke/report.json"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--backend", choices=("reference", "fla"), default="fla")
    parser.add_argument("--unit-batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--smoke-steps", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=10800)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        args.output.exists()
        or min(
            args.unit_batch_size,
            args.max_tokens,
            args.accumulation,
            args.smoke_steps,
            args.epochs,
            args.checkpoint_every,
        )
        < 1
        or args.max_seconds <= 0
    ):
        parser.error("choose a new output and positive bounds")
    fp = fingerprint(args)
    if args.stage == "full":
        gate = read_json(args.gate)
        try:
            validate_gate(gate, fp)
        except ValueError as exc:
            parser.error(str(exc))
    if args.resume:
        saved_metadata = read_json(args.resume / "checkpoint.json")["metadata"]
        if saved_metadata.get("fingerprint") != fp or saved_metadata.get("stage") != args.stage:
            parser.error("resume checkpoint source/data/runtime policy differs")
    args.output.mkdir(parents=True)
    report = {
        "stage": args.stage,
        "status": "preparing",
        "passed": False,
        "fingerprint": fp,
        "run_id": uuid.uuid4().hex,
    }
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status="failed", passed=False, error=repr(exc))
        write_report(args.output / "report.json", report)
        raise


def run(args, report):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    write_report(args.output / "report.json", report)
    engine = load_judgment_engine(
        checkpoint=args.resume, backend=args.backend, unit_batch_size=args.unit_batch_size, trainable=bool(args.resume)
    )
    prepared, counts = prepare_split(engine, args.data / "train.jsonl", args.max_tokens, Path(".cache/judgment-tokens"))
    validation_all, validation_counts = prepare_split(
        engine, args.data / "validation.jsonl", args.max_tokens, Path(".cache/judgment-tokens")
    )
    validation = balanced_sample(validation_all, 4)
    report.update(data_filter=counts, validation_filter=validation_counts)
    if args.stage == "smoke":
        request = read_json("data/integration/nested-request.json")
        a = engine.evaluate(request, details=True)
        b = engine.evaluate(request, details=True, cached=False)
        write_report(args.output / "nested-cached.json", a)
        write_report(args.output / "nested-independent.json", b)
        # Reassembly/serialization must preserve the complete caller-provided tree.
        if read_json(args.output / "nested-cached.json") != a:
            raise RuntimeError("JSON round trip changed nested response")
        report["nested_values_equal"] = a["values"] == b["values"]
        report["nested_max_probability_difference"] = compare_probabilities(a["answers"], b["answers"])
        baseline = evaluate_bundles(engine, validation)
        write_report(args.output / "untrained-validation.json", baseline)
        report["untrained_validation"] = metrics_only(baseline)
    elif not args.resume:
        test, test_counts = prepare_split(
            engine, args.data / "test.jsonl", args.max_tokens, Path(".cache/judgment-tokens")
        )
        baseline = evaluate_bundles(engine, test)
        write_report(args.output / "untrained-test.json", baseline)
        report["untrained_test"] = metrics_only(baseline)
        report["test_filter"] = test_counts
        del test
    if not args.resume:
        report["parameter_policy"] = engine.model.add_lora()
    else:
        report["parameter_policy"] = engine.model.parameter_inventory()
    torch.cuda.reset_peak_memory_stats()
    if args.stage == "smoke":
        # Exercise real longest/most-candidate bundles, not just easy short samples.
        probes = {
            b.id: b
            for b in (
                max(prepared, key=lambda b: max(map(len, b.prompts))),
                max(prepared, key=lambda b: len(b.prompts)),
            )
        }
        report["gradient_stress"] = []
        engine.model.train()
        for probe in probes.values():
            engine.model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = bundle_loss(
                    engine.model.score_prompts(probe.prompts, probe.kinds, unit_batch_size=args.unit_batch_size),
                    probe.groups,
                    probe.relations,
                    consistency_weight=args.consistency_weight,
                )
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in engine.model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
            )
            report["gradient_stress"].append(
                {
                    "bundle": probe.id,
                    "max_tokens": max(map(len, probe.prompts)),
                    "units": len(probe.prompts),
                    "gradient_norm": norm.item(),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
        engine.model.zero_grad(set_to_none=True)
        print("Longest-input / largest-group backward checks passed", flush=True)
    report["status"] = "training"
    write_report(args.output / "report.json", report)
    training = train(engine, prepared, validation, args, report)
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    final_eval = evaluate_bundles(engine, validation)
    write_report(args.output / "validation-final.json", final_eval)
    request = read_json("data/integration/nested-request.json")
    before_reload = engine.evaluate(request, details=True, cached=False)
    _, tokens, kinds = engine.prepare(request)
    with torch.inference_mode():
        expected_logits = engine.model.score_prompts(tokens, kinds).cpu()
    tokenizer = engine.tokenizer
    del engine, training
    gc.collect()
    torch.cuda.empty_cache()
    model, metadata = JudgmentModel.load_checkpoint(
        args.output / "final", kernel_backend=args.backend, trainable=args.stage == "smoke"
    )
    engine = JudgmentEngine(model, tokenizer, model_id=model.checkpoint_id, unit_batch_size=args.unit_batch_size)
    with torch.inference_mode():
        actual_logits = model.score_prompts(tokens, kinds).cpu()
    torch.testing.assert_close(actual_logits, expected_logits, atol=1e-4, rtol=1e-4)
    after_reload = engine.evaluate(request, details=True, cached=False)
    report["reload_max_logit_difference"] = (actual_logits - expected_logits).abs().max().item()
    report["reload_values_equal"] = before_reload["values"] == after_reload["values"]
    if not report["reload_values_equal"]:
        raise RuntimeError("checkpoint reload changed decisions")
    write_report(args.output / "nested-reloaded.json", after_reload)
    if args.stage == "smoke":
        if not report["completed_requested_pass"]:
            raise RuntimeError("smoke hit its time cap before completing the requested steps")
        if report["after_training"]["nll"] >= 0.9 * report["before_training"]["nll"]:
            raise RuntimeError("tiny-sample loss did not decrease by at least 10 percent")
        optimizer = optimizer_for(model, args)
        saved = torch.load(args.output / "final/training.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(saved["optimizer"])
        model.train()
        probe = prepared[0]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = bundle_loss(
                model.score_prompts(probe.prompts, probe.kinds, unit_batch_size=args.unit_batch_size),
                probe.groups,
                probe.relations,
                consistency_weight=args.consistency_weight,
            )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
        report["resume_update_finite"] = math.isfinite(losses["total"].item())
    else:
        validation_final = evaluate_bundles(engine, validation_all)
        write_report(args.output / "validation-all-final.json", validation_final)
        report["validation_all"] = metrics_only(validation_final)
        test, test_counts = prepare_split(
            engine, args.data / "test.jsonl", args.max_tokens, Path(".cache/judgment-tokens")
        )
        report["test_filter"] = test_counts
        test_eval = evaluate_bundles(engine, test)
        write_report(args.output / "test-final.json", test_eval)
        report["test"] = metrics_only(test_eval)
    report.update(
        passed=True, status="complete" if report["completed_requested_pass"] else "time_capped_checkpoint_saved"
    )
    write_report(args.output / "report.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("fingerprint", "parameter_policy", "data_filter", "validation_filter", "test_filter")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
