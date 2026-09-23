"""Bounded, paired-language v1 training from one frozen v0.2 checkpoint.

The runner deliberately owns the experimental control plane: it checks that the
two rendered training corpora differ only in text fields, fixes one six-slot
replay schedule, and refuses to train unless a matching completed smoke record
exists.  It never uses test results to select a checkpoint.
"""

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
from experiments.revised_evaluation import broad_suite, evaluate_cases, select_checkpoint, validation_objective, write
from experiments.revised_metrics import validate_suite
from openjev.judgment_cli import load_judgment_engine
from openjev.judgment_model import frozen_digest
from openjev.judgment_training import bundle_loss, load_bundles, prepare_bundle

KINDS = ("noul", "choice", "score")
ARMS = ("template", "natural")
STUDY = "paired-language-v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def source_group(record):
    group = record.get("group_id")
    if not isinstance(group, str) or not group:
        raise ValueError("Every paired training record needs a nonempty group_id")
    return group


def _without_rendering(record):
    """Return the declared semantic payload, removing the two allowed text fields."""
    result = json.loads(json.dumps(record))
    for example in result.get("examples", []):
        example.pop("state", None)
        question = example.get("question")
        if isinstance(question, dict):
            question.pop("instructions", None)
    return result


def _pair_map(records, arm):
    result = {}
    for record in records:
        identity = record.get("id")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"{arm} has a record without an id")
        if identity in result:
            raise ValueError(f"{arm} has a duplicate training id: {identity}")
        if record.get("provenance", {}).get("assigned_split") != "train":
            raise ValueError(f"{arm}/{identity} is not assigned to train")
        source_group(record)
        result[identity] = record
    if not result:
        raise ValueError(f"{arm} training population is empty")
    return result


def compare_paired_records(template, natural):
    """Validate all non-rendered paired semantics, including question order/criteria."""
    templates, naturals = _pair_map(template, "template"), _pair_map(natural, "natural")
    if templates.keys() != naturals.keys():
        raise ValueError("template and natural training IDs differ")
    differing_renderings = []
    for identity in sorted(templates):
        left, right = templates[identity], naturals[identity]
        if _without_rendering(left) != _without_rendering(right):
            raise ValueError(f"paired semantic mismatch for {identity}; only state/instructions may differ")
        a, b = left["examples"], right["examples"]
        if len(a) != len(b) or not a:
            raise ValueError(f"paired example count mismatch for {identity}")
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            if x.get("target") != y.get("target"):
                raise ValueError(f"paired target mismatch for {identity}/{index}")
            qx, qy = x.get("question"), y.get("question")
            if not isinstance(qx, dict) or not isinstance(qy, dict):
                raise ValueError(f"paired record needs questions: {identity}/{index}")
            if qx.get("type") != qy.get("type") or qx.get("criteria") != qy.get("criteria"):
                raise ValueError(f"paired question type/criteria mismatch for {identity}/{index}")
            if isinstance(qx.get("criteria"), dict) and list(qx["criteria"]) != list(qy["criteria"]):
                raise ValueError(f"paired candidate order mismatch for {identity}/{index}")
            if list(qx) != list(qy):
                raise ValueError(f"paired question field order mismatch for {identity}/{index}")
            if x.get("state") != y.get("state") or qx.get("instructions") != qy.get("instructions"):
                differing_renderings.append(f"{identity}/{index}")
    return {"records": len(templates), "rendering_differences": differing_renderings}


def _suite_names(suite):
    ids, families = set(), set()
    for case in suite["cases"]:
        ids.add(case["id"])
        families.add(case["family_id"])
    return ids, families


def validate_population_boundaries(template, natural, validation, test):
    """Reject accidental train/evaluation leakage by either record or family name."""
    validate_suite(validation)
    validate_suite(test)
    eval_ids, eval_families = _suite_names(validation)
    test_ids, test_families = _suite_names(test)
    if eval_ids & test_ids or eval_families & test_families:
        raise ValueError("validation/test overlap in record or family IDs")

    def signatures(suite):
        return {json.dumps(c["request"], sort_keys=True, ensure_ascii=False) for c in suite["cases"] if "request" in c}

    if signatures(validation) & signatures(test):
        raise ValueError("validation/test overlap in exact model requests")
    eval_ids |= test_ids
    eval_families |= test_families
    for arm, records in (("template", template), ("natural", natural)):
        for record in records:
            if record["id"] in eval_ids or source_group(record) in eval_families:
                raise ValueError(f"{arm}/{record['id']} leaks into validation or test")


def _all_hashes(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _all_hashes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_hashes(item)
    elif isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower()):
        yield value.lower()


def validate_manifest(root):
    """Require the frozen manifest to bind every paired input, without assuming its key layout."""
    root = Path(root)
    manifest_path = root / "paired-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("missing paired-manifest.json")
    manifest = read(manifest_path)
    required = {
        sha(root / "template/train.jsonl"),
        sha(root / "natural/train.jsonl"),
        sha(root / "validation/suite.json"),
        sha(root / "test/suite.json"),
    }
    present = set(_all_hashes(manifest))
    if not required <= present:
        raise ValueError("paired manifest does not bind all training and evaluation file hashes")
    return {"path": str(manifest_path), "sha256": sha(manifest_path), "bound_hashes": sorted(required)}


def validate_data(root):
    root = Path(root)
    paths = {arm: root / arm / "train.jsonl" for arm in ARMS}
    validation_path, test_path = root / "validation/suite.json", root / "test/suite.json"
    template, natural = load_bundles(paths["template"]), load_bundles(paths["natural"])
    validation, test = read(validation_path), read(test_path)
    pairing = compare_paired_records(template, natural)
    validate_population_boundaries(template, natural, validation, test)
    return {
        "template": template,
        "natural": natural,
        "validation": validation,
        "test": test,
        "pairing": pairing,
        "manifest": validate_manifest(root),
    }


def make_schedule(pools, *, steps, seed):
    """One original/new slot per primitive, in the contract's fixed order."""
    if steps < 1 or set(pools) != {"original", "new"}:
        raise ValueError("invalid paired schedule")
    all_ids = []
    for origin in ("original", "new"):
        if set(pools[origin]) != set(KINDS):
            raise ValueError("every origin needs Noul, Choice, and Score")
        for kind in KINDS:
            sources = pools[origin][kind]
            if not sources or any(not values for values in sources.values()):
                raise ValueError("every scheduled source must be nonempty")
            all_ids.extend(value for values in sources.values() for value in values)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("training schedule IDs must be globally unique")
    streams = {}
    for origin in ("original", "new"):
        for kind in KINDS:
            for source, values in sorted(pools[origin][kind].items()):
                rng = random.Random(int(hashlib.sha256(f"{seed}/{origin}/{kind}/{source}".encode()).hexdigest(), 16))
                order = []
                while len(order) < steps:
                    cycle = list(values)
                    rng.shuffle(cycle)
                    order.extend(cycle)
                streams[origin, kind, source] = iter(order)
    schedule = []
    for step in range(steps):
        row = []
        for kind in KINDS:
            for origin in ("original", "new"):
                sources = sorted(pools[origin][kind])
                source = sources[step % len(sources)]
                row.append(
                    {"origin": origin, "primitive": kind, "source": source, "id": next(streams[origin, kind, source])}
                )
        schedule.append(row)
    return schedule


def _single_question_bundle(identity, example, source):
    return {"id": identity, "examples": [example], "relations": [], "provenance": {"dataset": source}}


def prepare_paired_example(engine, arm, identity, index, example):
    """Prepare one new unit; an oversize member invalidates its entire paired run."""
    item_id = f"new/{identity}/{index}"
    try:
        return item_id, prepare_bundle(engine, _single_question_bundle(item_id, example, STUDY))
    except ValueError as error:
        if "exceeds" in str(error):
            raise ValueError(f"oversized paired example {arm}/{identity}/{index}; no dropping allowed") from error
        raise


def prepare_populations(engine, data, output):
    """Prepare both text arms and the shared replay; paired oversize inputs are fatal."""
    pools = {origin: {kind: {} for kind in KINDS} for origin in ("original", "new")}
    prepared = {arm: {} for arm in ARMS}
    broad_exclusions = []
    # The source ordering is canonical and shared even when an old source has an oversize unit.
    selected = {}
    for row in sorted(
        load_bundles("data/processed-v0.2/train.jsonl"), key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()
    ):
        provenance = row.get("provenance", {})
        if provenance.get("assigned_split") != "train":
            raise ValueError("original replay contains a nontraining record")
        source = provenance.get("dataset")
        if not isinstance(source, str) or not source:
            raise ValueError("original replay record has no dataset")
        if len(selected.get(source, [])) >= 800:
            continue
        example = row["examples"][0]
        kind = example["question"]["type"]
        identity = "original/" + row["id"]
        try:
            bundle = prepare_bundle(engine, _single_question_bundle(identity, example, source))
        except ValueError as error:
            if "exceeds" not in str(error):
                raise
            broad_exclusions.append({"id": row["id"], "source": source, "reason": str(error)})
            continue
        selected.setdefault(source, []).append((identity, bundle, kind))
    for source, rows in selected.items():
        if len(rows) != 800:
            raise ValueError(f"original replay source {source} has {len(rows)} eligible records, need 800")
        for identity, bundle, kind in rows:
            for arm in ARMS:
                prepared[arm][identity] = bundle
            pools["original"][kind].setdefault(source, []).append(identity)

    records = {arm: load_bundles(data[arm]) for arm in ARMS}
    compare_paired_records(records["template"], records["natural"])
    for identity in sorted(record["id"] for record in records["template"]):
        for arm in ARMS:
            record = next(r for r in records[arm] if r["id"] == identity)
            for index, example in enumerate(record["examples"]):
                kind = example["question"]["type"]
                if kind not in KINDS:
                    raise ValueError(f"unknown paired primitive {kind}")
                item_id, prepared[arm][item_id] = prepare_paired_example(engine, arm, identity, index, example)
                if arm == "template":
                    category = record.get("provenance", {}).get("category", "paired-language-v1")
                    pools["new"][kind].setdefault(category, []).append(item_id)
    write(
        output / "populations.json",
        {
            "pools": pools,
            "original_exclusions": broad_exclusions,
            "paired_exclusions": [],
            "rule": "paired examples are rejected, never dropped",
        },
    )
    prompt_hashes = {
        arm: {
            identity: hashlib.sha256(json.dumps(bundle.prompts).encode()).hexdigest()
            for identity, bundle in values.items()
        }
        for arm, values in prepared.items()
    }
    if any(
        prompt_hashes["template"][key] != prompt_hashes["natural"][key]
        for key in prompt_hashes["template"]
        if key.startswith("original/")
    ):
        raise RuntimeError("Original replay prompts differ between arms")
    write(output / "prepared-prompt-hashes.json", prompt_hashes)
    return prepared, pools, broad_exclusions


def new_engine(checkpoint):
    engine = load_judgment_engine(
        checkpoint=checkpoint, trainable=True, backend="fla", max_input_tokens=1536, unit_batch_size=4
    )
    engine.model.parameter_inventory()
    return engine


def broad_validation_records():
    """The fixed 300 canonical broad validation records: 50 from each source."""
    counts, rows = {}, []
    for row in sorted(
        load_bundles("data/processed-v0.2/validation.jsonl"), key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()
    ):
        provenance = row.get("provenance", {})
        if provenance.get("assigned_split") != "validation":
            raise ValueError("original broad validation contains a nonvalidation record")
        source = provenance.get("dataset")
        if counts.get(source, 0) < 50:
            counts[source] = counts.get(source, 0) + 1
            rows.append(row)
    if len(rows) != 300 or any(count != 50 for count in counts.values()):
        raise ValueError("need exactly 50 canonical validation records per original source")
    return rows


def checkpoint_id(path):
    info = read(Path(path) / "checkpoint.json")
    return info["checkpoint_id"]


def save(engine, path, arm, step, initial_checkpoint_id, optimizer=None, **extra):
    metadata = {"study": STUDY, "arm": arm, "step": step, "initial_checkpoint_id": initial_checkpoint_id, **extra}
    identity = engine.model.save_checkpoint(
        path, metadata=metadata, optimizer=optimizer, training_state={"arm": arm, "step": step}
    )
    engine.model_id = identity
    return identity


def evaluate_pair(engine, suite, broad, output, deadline):
    output.mkdir(parents=True, exist_ok=False)
    with evaluation_deadline(deadline):
        result = {
            "paired": evaluate_cases(engine, suite, output / "paired"),
            "broad": evaluate_cases(engine, broad_suite(broad), output / "broad"),
        }
    write(output / "evaluation.json", result)
    # Preserve the old objective's expected key while retaining precise run labels on disk.
    return {"revised": result["paired"], "broad": result["broad"]}


def fit_arm(
    engine, arm, schedule, prepared, validation, broad_validation, args, root, initial_checkpoint_id, *, smoke=False
):
    model, optimizer = engine.model, optimizer_for(engine.model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
    frozen_before, trainable = frozen_digest(model), [p for p in model.parameters() if p.requires_grad]
    checkpoints = () if smoke else (0, 250, 500, 1000)
    history, curve, elapsed, gradient_check = [], [], 0.0, None

    def validate(step):
        checkpoint = root / f"step-{step:04d}"
        identity = save(engine, checkpoint, arm, step, initial_checkpoint_id, optimizer)
        value = evaluate_pair(
            engine, validation, broad_validation, root / f"validation-{step:04d}", args.evaluation_max_seconds
        )
        curve.append(
            {
                "status": "complete",
                "step": step,
                "checkpoint": str(checkpoint),
                "checkpoint_id": identity,
                "objective": validation_objective(value),
                "evaluation": value,
            }
        )
        write(root / "validation-curve.json", curve)

    try:
        if 0 in checkpoints:
            validate(0)
        for step, batch in enumerate(schedule, 1):
            started = time.monotonic()
            if elapsed >= args.max_seconds:
                raise TimeoutError("training time cap reached")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total, primitive_loss = 0.0, {kind: 0.0 for kind in KINDS}
            for item in batch:
                bundle = prepared[item["id"]]
                if len(bundle.groups) != 1 or bundle.groups[0].primitive != item["primitive"]:
                    raise ValueError("six-slot schedule does not match prepared primitive")
                with torch.autocast(
                    device_type=model.device.type, dtype=torch.bfloat16, enabled=model.device.type == "cuda"
                ):
                    loss = bundle_loss(
                        model.score_prompts(bundle.prompts, bundle.kinds, unit_batch_size=4), bundle.groups, []
                    )["supervised"]
                    (loss / 6).backward()
                total += loss.item() / 6
                primitive_loss[item["primitive"]] += loss.item() / 2
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            if step == 1:
                gradient_check = {
                    "adapters": any(
                        p.grad is not None and p.grad.abs().max().item() > 0
                        for n, p in model.named_parameters()
                        if ".lora_" in n
                    ),
                    "heads": any(
                        p.grad is not None and p.grad.abs().max().item() > 0
                        for n, p in model.named_parameters()
                        if n.startswith(("binary.", "compatibility."))
                    ),
                    "frozen": all(p.grad is None for p in model.parameters() if not p.requires_grad),
                    "finite_norm": bool(torch.isfinite(norm).item()),
                }
                if not all(gradient_check.values()):
                    raise RuntimeError("finite gradient/frozen-weight smoke check failed")
            optimizer.step()
            engine.synchronize()
            elapsed += time.monotonic() - started
            entry = {
                "step": step,
                "loss": total,
                "by_primitive": primitive_loss,
                "gradient_norm": norm.item(),
                "training_seconds": elapsed,
            }
            history.append(entry)
            write_jsonl(root / "history.jsonl", entry)
            if step == 1 or step % 50 == 0:
                print(arm, f"step {step}/{len(schedule)} loss {total:.4f}; training {elapsed:.1f}s", flush=True)
            if elapsed > args.max_seconds:
                raise TimeoutError("training time cap exceeded in completed update")
            if step in checkpoints:
                validate(step)
        optimizer.zero_grad(set_to_none=True)
        frozen_after = frozen_digest(model)
        if frozen_before != frozen_after:
            raise RuntimeError("frozen original weights changed")
        result = {
            "status": "complete",
            "completed_updates": len(history),
            "training_seconds": elapsed,
            "gradient_check": gradient_check,
            "frozen_before": frozen_before,
            "frozen_after": frozen_after,
            "initial_checkpoint_id": initial_checkpoint_id,
            "inventory": model.parameter_inventory(),
        }
        if smoke:
            result["checkpoint_id"] = save(
                engine, root / "final", arm, len(history), initial_checkpoint_id, optimizer, mode="smoke"
            )
        else:
            result["selected"] = select_checkpoint(curve)
            result["still_improving_last_interval"] = curve[-1]["objective"] < curve[-2]["objective"] - 0.01
        write(root / "training.json", result)
        return result
    except BaseException as error:
        optimizer.zero_grad(set_to_none=True)
        try:
            save(
                engine,
                root / f"partial-{len(history):04d}",
                arm,
                len(history),
                initial_checkpoint_id,
                optimizer,
                status="failed",
                completed_updates=len(history),
                attempted_step=step if "step" in locals() else 0,
                resumable=False,
                error=repr(error),
            )
        except Exception:
            pass
        write(
            root / "training.json",
            {
                "status": "failed",
                "completed_updates": len(history),
                "training_seconds": elapsed,
                "initial_checkpoint_id": initial_checkpoint_id,
                "error": repr(error),
            },
        )
        raise


def fingerprint(args):
    source = [
        Path(__file__),
        Path("experiments/revised_evaluation.py"),
        Path("experiments/revised_metrics.py"),
        Path("experiments/judgment_pipeline.py"),
        Path("src/openjev/judgment_model.py"),
        Path("src/openjev/judgment_training.py"),
        Path("src/openjev/judgment_cli.py"),
        Path("src/openjev/judgments.py"),
        Path("src/openjev/runtime.py"),
        Path("experiments/semantic_contrasts.py"),
        Path("experiments/contrast_pilot.py"),
    ]
    data = (
        [args.data / arm / "train.jsonl" for arm in ARMS]
        + [args.data / split / "suite.json" for split in ("validation", "test")]
        + [args.data / "paired-manifest.json"]
        + [Path("data/processed-v0.2") / f"{split}.jsonl" for split in ("train", "validation", "test")]
        + [Path("data/semantic-contrasts-v1/suite.json")]
    )
    packages = {}
    for name in ("torch", "transformers", "peft", "fla-core"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "sources": {str(p): sha(p) for p in source},
        "data": {str(p): sha(p) for p in data},
        "starting_checkpoint_id": checkpoint_id(args.checkpoint),
        "packages": packages,
        "steps": 1000,
        "seed": 42,
        "rank": 8,
        "alpha": 16,
        "adapter_lr": 5e-5,
        "head_lr": 2.5e-5,
        "max_tokens": 1536,
        "unit_batch_size": 4,
        "max_training_seconds": args.max_seconds,
        "max_evaluation_seconds": args.evaluation_max_seconds,
    }


def validate_smoke_gate(gate, expected_fingerprint):
    if (
        gate.get("mode") != "smoke"
        or gate.get("status") != "complete"
        or not gate.get("passed")
        or not gate.get("completed_requested_pass")
        or gate.get("fingerprint") != expected_fingerprint
    ):
        raise ValueError("a matching completed passing paired smoke gate is required")


def run(args, report):
    validated = validate_data(args.data)
    report.update(fingerprint=fingerprint(args), pairing=validated["pairing"], manifest=validated["manifest"])
    if args.mode == "run":
        validate_smoke_gate(read(args.gate), report["fingerprint"])
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    bootstrap = new_engine(args.checkpoint)
    prepared, pools, broad_exclusions = prepare_populations(
        bootstrap, {arm: args.data / arm / "train.jsonl" for arm in ARMS}, args.output
    )
    schedule = make_schedule(pools, steps=1000, seed=42)
    write(args.output / "schedule.json", schedule)
    report.update(
        schedule_sha256=sha(args.output / "schedule.json"),
        original_exclusions=broad_exclusions,
        initial_checkpoint_id=checkpoint_id(args.checkpoint),
        arms={},
        arm_controls={
            "shared_schedule_sha256": sha(args.output / "schedule.json"),
            "initial_checkpoint_ids": {},
            "paired_semantic_records": validated["pairing"]["records"],
        },
    )
    del bootstrap
    gc.collect()
    torch.cuda.empty_cache()
    broad_validation = broad_validation_records()
    for arm in ARMS:
        torch.manual_seed(42)
        engine = new_engine(args.checkpoint)
        if engine.model.checkpoint_id != report["initial_checkpoint_id"]:
            raise RuntimeError(f"{arm} loaded a different initial checkpoint")
        root = args.output / arm
        root.mkdir()
        actual = schedule[:2] if args.mode == "smoke" else schedule
        result = fit_arm(
            engine,
            arm,
            actual,
            prepared[arm],
            validated["validation"],
            broad_validation,
            args,
            root,
            report["initial_checkpoint_id"],
            smoke=args.mode == "smoke",
        )
        if result["initial_checkpoint_id"] != report["initial_checkpoint_id"]:
            raise RuntimeError(f"{arm} did not start from the declared v0.2 checkpoint")
        report["arm_controls"]["initial_checkpoint_ids"][arm] = result["initial_checkpoint_id"]
        if args.mode == "smoke":
            request = validated["validation"]["cases"][0]["request"]
            before = engine.evaluate(request, cached=False, details=True)
            del engine
            gc.collect()
            torch.cuda.empty_cache()
            engine = load_judgment_engine(
                checkpoint=root / "final", backend="fla", max_input_tokens=1536, unit_batch_size=4
            )
            after = engine.evaluate(request, cached=False, details=True)
            result["reload_max_difference"] = compare_probabilities(before["answers"], after["answers"], tolerance=1e-5)
        report["arms"][arm] = result
        write(args.output / "report.json", report)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    if args.mode == "run":
        selected = {arm: report["arms"][arm]["selected"] for arm in ARMS}
        write(args.output / "selection-locked.json", selected)
        report["tests"] = {}
        broad_test = load_bundles("data/processed-v0.2/test.jsonl")
        for arm in ("reference",) + ARMS:
            checkpoint = args.checkpoint if arm == "reference" else Path(selected[arm]["checkpoint"])
            engine = load_judgment_engine(
                checkpoint=checkpoint, backend="fla", max_input_tokens=1536, unit_batch_size=4
            )
            root = args.output / f"{arm}-test"
            result = evaluate_pair(engine, validated["test"], broad_test, root, args.evaluation_max_seconds)
            with evaluation_deadline(args.evaluation_max_seconds):
                result["semantic114"] = evaluate_cases(
                    engine, read("data/semantic-contrasts-v1/suite.json"), root / "semantic114"
                )
            report["tests"][arm] = result
            write(args.output / "report.json", report)
            del engine
            gc.collect()
            torch.cuda.empty_cache()
        # A separately declared fixed-update comparison prevents different
        # validation-selected stopping points from obscuring the data effect.
        report["fixed_budget_tests"] = {}
        for arm in ARMS:
            if selected[arm]["step"] == 1000:
                report["fixed_budget_tests"][arm] = {
                    "checkpoint": selected[arm]["checkpoint"],
                    "reused_selected": True,
                    "predictions_root": str(args.output / f"{arm}-test"),
                    "results": report["tests"][arm],
                }
                continue
            checkpoint = args.output / arm / "step-1000"
            engine = load_judgment_engine(
                checkpoint=checkpoint, backend="fla", max_input_tokens=1536, unit_batch_size=4
            )
            root = args.output / f"{arm}-fixed-1000-test"
            result = evaluate_pair(engine, validated["test"], broad_test, root, args.evaluation_max_seconds)
            with evaluation_deadline(args.evaluation_max_seconds):
                result["semantic114"] = evaluate_cases(
                    engine, read("data/semantic-contrasts-v1/suite.json"), root / "semantic114"
                )
            report["fixed_budget_tests"][arm] = {
                "checkpoint": str(checkpoint),
                "reused_selected": False,
                "predictions_root": str(root),
                "results": result,
            }
            write(args.output / "report.json", report)
            del engine
            gc.collect()
            torch.cuda.empty_cache()
    report["completed_requested_pass"] = all(
        v["completed_updates"] == (2 if args.mode == "smoke" else 1000) for v in report["arms"].values()
    )
    report["passed"] = args.mode == "smoke" and report["completed_requested_pass"]
    report["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/paired-language-v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/judgment-full-v0.2/final"))
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--evaluation-max-seconds", type=float, default=1800)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.max_seconds <= 0
        or args.evaluation_max_seconds <= 0
        or (args.mode == "run" and not args.gate)
    ):
        parser.error("choose a new output, positive caps, and a smoke gate for a full run")
    args.output.mkdir(parents=True)
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
