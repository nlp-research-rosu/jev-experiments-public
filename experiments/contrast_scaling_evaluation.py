"""Post-training validation selection and frozen tests for contrast scaling."""

import argparse
import gc
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.contrast_pilot import evaluation_deadline
from experiments.contrast_scaling_training import ENDPOINTS
from experiments.paired_language_training import broad_validation_records
from experiments.revised_evaluation import broad_suite, evaluate_cases, validation_objective, write
from openjev.judgment_cli import load_judgment_engine
from openjev.judgment_training import load_bundles


def _checked_curve(points, required):
    if len(points) != len(required) or {p.get("step") for p in points} != set(required):
        raise ValueError("validation must cover every declared checkpoint exactly once")
    if any(
        p.get("status") != "complete"
        or type(p.get("objective")) not in (int, float)
        or not math.isfinite(p["objective"])
        or not p.get("checkpoint_id")
        or not p.get("checkpoint")
        for p in points
    ):
        raise ValueError("only complete finite validation points can select checkpoints")
    return {p["step"]: p for p in points}


def choose_checkpoints(primary_curve, repeat_curve):
    primary = _checked_curve(primary_curve, ENDPOINTS)
    repeated = _checked_curve(repeat_curve, (0, 200, 400, 600, 800, 1000))
    for step in (0, 200):
        if primary[step]["checkpoint_id"] != repeated[step]["checkpoint_id"]:
            raise ValueError("repeat branch must share the initial 200-family prefix")
    selected = {}
    endpoints = {}
    for size in (200, 1000, 5000):
        selected[f"data_{size}"] = min(
            (p for s, p in primary.items() if s <= size), key=lambda p: (p["objective"], p["step"])
        )
        endpoints[f"data_{size}"] = primary[size]
    selected["repeat_200"] = min(repeated.values(), key=lambda p: (p["objective"], p["step"]))
    endpoints["repeat_200"] = repeated[1000]
    return {
        "baseline": primary[0],
        "endpoints": endpoints,
        "selected": selected,
        "primary_interpretation": "Complete-pass endpoints use 200/1000/5000 unique families and respectively 200/1000/5000 updates.",
        "compute_control": "Repeat200 at1000updates versus1000unique families at1000updates. The5000endpoint also uses more compute.",
    }


def unique_test_checkpoints(selection):
    by_id = {}
    named = [("baseline", selection["baseline"])]
    named += [(f"endpoint/{name}", p) for name, p in selection["endpoints"].items()]
    named += [(f"selected/{name}", p) for name, p in selection["selected"].items()]
    for role, point in named:
        identity = point["checkpoint_id"]
        if identity not in by_id:
            by_id[identity] = {"checkpoint_id": identity, "checkpoint": point["checkpoint"], "roles": []}
        by_id[identity]["roles"].append(role)
    return list(by_id.values())


def materialize_curve(points, evaluations):
    """Reuse identical predictions while retaining each stream/update location."""
    return [{**evaluations[point["checkpoint_id"]], **point} for point in points]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_evaluation_freeze(data):
    freeze = read(data / "evaluation-freeze.json")
    for name in ("test", "validation"):
        if sha(data / name / "suite.json") != freeze["suites"][name]["suite_sha256"]:
            raise ValueError(f"{name} differs from frozen evaluation")
    return freeze


def _engine(checkpoint):
    return load_judgment_engine(checkpoint=checkpoint, backend="fla", max_input_tokens=1536, unit_batch_size=4)


def _identity(path):
    return read(path / "checkpoint.json")["checkpoint_id"]


def run(args, report):
    train = read(args.training / "report.json")
    if (
        train.get("status") != "complete"
        or train["primary"]["completed_updates"] != 5000
        or train["repeat_200"]["completed_updates"] != 1000
    ):
        raise ValueError("complete5000primary and1000repeat training is required before selection")
    for stream in ("primary", "repeat_200"):
        if train[stream]["frozen_before"] != train[stream]["frozen_after"]:
            raise ValueError("frozen original weights changed")
    freeze = verify_evaluation_freeze(args.data)
    report.update(
        training_report_sha256=sha(args.training / "report.json"),
        freeze=freeze,
        source_sha256=sha(__file__),
        validation={},
        curves={},
        tests={},
    )
    validation = read(args.data / "validation/suite.json")
    broad = broad_suite(broad_validation_records())
    points = []
    for stream, steps in [("primary", ENDPOINTS), ("repeat-200", (400, 600, 800, 1000))]:
        for step in steps:
            path = args.training / stream / f"step-{step:04d}"
            points.append({"stream": stream, "step": step, "checkpoint": str(path), "checkpoint_id": _identity(path)})
    for point in points:
        identity = point["checkpoint_id"]
        if identity in report["validation"]:
            continue
        key = f"{point['stream']}-{point['step']:04d}"
        engine = _engine(point["checkpoint"])
        if engine.model.checkpoint_id != identity:
            raise ValueError("checkpoint identity changed while loading")
        root = args.output / "validation" / key
        with evaluation_deadline(args.max_evaluation_seconds):
            value = {
                "revised": evaluate_cases(engine, validation, root / "new"),
                "broad": evaluate_cases(engine, broad, root / "broad"),
            }
        result = {
            **point,
            "status": "complete",
            "objective": validation_objective(value),
            "evaluation": value,
            "predictions_root": str(root),
        }
        report["validation"][identity] = result
        write(args.output / "report.json", report)
        print("validation", key, "objective", result["objective"], flush=True)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    primary = materialize_curve([p for p in points if p["stream"] == "primary"], report["validation"])
    repeat = [p for p in primary if p["step"] in (0, 200)] + [
        *materialize_curve([p for p in points if p["stream"] == "repeat-200"], report["validation"])
    ]
    report["curves"] = {"primary": primary, "repeat_200": repeat}
    selection = choose_checkpoints(primary, repeat)
    selection["locked_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "selection-locked.json", selection)
    report["selection"] = selection
    write(args.output / "report.json", report)
    # Test text/outcomes are only consumed after every selection is durably locked.
    verify_evaluation_freeze(args.data)
    test = read(args.data / "test/suite.json")
    broad_test = broad_suite(load_bundles("data/processed-v0.2/test.jsonl"))
    legacy = read("data/semantic-contrasts-v1/suite.json")
    prior = read("data/paired-language-clarification-v1/test/suite.json")
    for index, job in enumerate(unique_test_checkpoints(selection)):
        engine = _engine(job["checkpoint"])
        if engine.model.checkpoint_id != job["checkpoint_id"]:
            raise ValueError("test checkpoint identity changed")
        root = args.output / "tests" / f"model-{index:02d}"
        results = {}
        for name, suite in [
            ("new", test),
            ("broad", broad_test),
            ("semantic114", legacy),
            ("prior_paired_clarified", prior),
        ]:
            with evaluation_deadline(args.max_evaluation_seconds):
                results[name] = evaluate_cases(engine, suite, root / name)
            print("test", index, name, "complete", flush=True)
        report["tests"][job["checkpoint_id"]] = {**job, "predictions_root": str(root), "results": results}
        write(args.output / "report.json", report)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    report["status"] = "complete"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("checkpoints/contrast-scaling-v1"))
    parser.add_argument("--data", type=Path, default=Path("data/contrast-scaling-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evaluation-seconds", type=float, default=3600)
    args = parser.parse_args()
    if args.output.exists() or args.max_evaluation_seconds <= 0:
        parser.error("new output and positive evaluation cap required")
    args.output.mkdir(parents=True)
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    write(args.output / "report.json", report)
    try:
        run(args, report)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        write(args.output / "report.json", report)


if __name__ == "__main__":
    main()
