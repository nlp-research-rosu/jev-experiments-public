"""Diagnose changed decisions and largest padding-related probability differences."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.aligned_fewshot import compare_execution
from experiments.answer_ablation import REVISION
from experiments.fewshot_stability import read_cases
from experiments.padded_batching import CONDITIONS, archived_scorer, measure, serial_scorer, sha
from openjev.engine import score_token_prompts
from openjev.runtime import load_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/batching-comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/batching-precision.json"))
    args = parser.parse_args()
    partial = args.output.with_suffix(".partial.json")
    if args.output.exists() or partial.exists():
        parser.error("Refusing to overwrite existing results")
    source = json.loads(args.source.read_text())
    selected = []
    for condition in CONDITIONS:
        old = {r["id"]: r for r in source["results"][f"{condition}/old"] if r["repeat"] == 0}
        padded = {r["id"]: r for r in source["results"][f"{condition}/padded"] if r["repeat"] == 0}
        delta = {
            i: max(
                abs(x - y)
                for f in row["probabilities"]
                for x, y in zip(row["probabilities"][f], old[i]["probabilities"][f], strict=True)
            )
            for i, row in padded.items()
        }
        ids = {i for i, row in padded.items() if row["values"] != old[i]["values"]}
        ids.add(max(delta, key=delta.get))
        selected.extend((condition, i, delta[i]) for i in sorted(ids))
    engine = load_engine("Qwen/Qwen3.5-2B", revision=REVISION)
    engine.model.float()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = {c["id"]: c for c in read_cases(Path("data/holdout-v4.json"))}
    examples = read_cases(Path("data/fewshot-examples-v1.json"))
    scorers = {"old": archived_scorer(), "padded": score_token_prompts, "independent": serial_scorer}
    report = {
        "metadata": {
            "source_sha256": sha(args.source),
            "script_sha256": sha(__file__),
            "model": engine.model_id,
            "revision": engine.revision,
            "dtype": "BF16 checkpoint values promoted to FP32; TF32 disabled",
            "attention_backend": "EFFICIENT_ATTENTION with explicitly repeated KV heads",
            "selection": "All v4 case/condition decision disagreements plus the largest probability delta per condition",
            "scope": "Numerical execution diagnostic, not an accuracy or latency selection experiment; upstream excluded",
        },
        "checks": [],
    }
    torch.cuda.reset_peak_memory_stats()
    with ExitStack() as stack:
        stack.enter_context(patch("transformers.integrations.sdpa_attention.use_gqa_in_sdpa", return_value=False))
        stack.enter_context(sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION))
        for condition, case_id, bf16_delta in selected:
            case = cases[case_id]
            rows = {name: measure(engine, case, examples, condition, scorer) for name, scorer in scorers.items()}
            for row in rows.values():
                row["repeat"] = 0
            check = {
                "condition": condition,
                "id": case_id,
                "bf16_max_abs_probability_difference": bf16_delta,
                "rows": rows,
                "comparisons": {
                    name: compare_execution([case], [rows["padded"]], [rows[name]]) for name in ("old", "independent")
                },
            }
            report["checks"].append(check)
            report["metadata"]["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(condition, case_id, json.dumps(check["comparisons"]), flush=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    partial.unlink()


if __name__ == "__main__":
    main()
