"""Retrospective inference-only wording/layout probe; no training or selection."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from experiments.contrast_component_probe import compare_predictions, evaluate
from experiments.contrast_pilot_data import DOMAINS, make_state, questions
from openjev.judgment_cli import load_judgment_engine


def probe_suite(*, nested, heldout_wording):
    suite = json.loads(Path("data/contrast-pilot-v1/test/suite.json").read_text())
    suite["cases"] = [c for c in suite["cases"] if c["variant"] in ("success", "pending", "wrong_target", "reversed")]
    domain_map = {d[0]: d for d in DOMAINS["test"]}
    for case in suite["cases"]:
        case["request"] = {
            "state": make_state(domain_map[case["domain"]], case["variant"], text_style=nested),
            "questions": questions(heldout=heldout_wording),
        }
    ids = {c["id"] for c in suite["cases"]}
    suite["relations"] = [r for r in suite["relations"] if all(r[e]["case_id"] in ids for e in ("left", "right"))]
    return copy.deepcopy(suite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    engine = load_judgment_engine(checkpoint=Path("checkpoints/contrast-pilot-v1/contrast/final"), backend="fla", unit_batch_size=4)
    report = {
        "status": "running", "checkpoint_id": engine.model_id,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Retrospective 16-case subset from inspected test. Layout also changes task fields into a prose brief. No weights/defaults/gold labels changed; not a fresh holdout.",
        "conditions": {},
    }
    for nested in (False, True):
        for wording in (False, True):
            name = ("nested" if nested else "structured") + "-" + ("heldout-wording" if wording else "training-wording")
            root = args.output / name
            root.mkdir()
            suite = probe_suite(nested=nested, heldout_wording=wording)
            (root / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
            result, rows = evaluate(engine, suite, "full", root)
            if nested and wording:
                ids = {c["id"] for c in suite["cases"]}
                previous = json.loads(Path("checkpoints/contrast-pilot-v1/contrast/test/predictions.json").read_text())
                reproduction = compare_predictions([r for r in previous if r["case_id"] in ids], rows)
                if reproduction["max_probability_difference"] > 1e-5 or reproduction["label_changes"]:
                    raise RuntimeError("Original condition failed to reproduce")
                result["reproduction"] = reproduction
            report["conditions"][name] = result
            (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["status"] = "complete"
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
