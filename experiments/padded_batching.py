"""Compare the archived length-grouped scorer with padded batches on fixed prompts."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import statistics
import subprocess
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.aligned_fewshot import compare_execution
from experiments.answer_ablation import REVISION
from experiments.compact_fewshot import decide
from experiments.fewshot_stability import ExampleJSONEngine, read_cases
from openjev.benchmark import load_cases, make_summary
from openjev.engine import score_token_prompts
from openjev.runtime import load_engine

ARCHIVE = Path("reports/sources/engine_length_grouped_v1.py")
CONDITIONS = ("original_0", "field_questions_4", "aligned_chat_4")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archived_scorer():
    spec = importlib.util.spec_from_file_location("openjev._length_grouped_reference", ARCHIVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_token_prompts


def measure(engine, case, examples, condition, scorer):
    """Instrument real backbone calls; the temporary scorer swap is sequential."""
    trace, scoring_ms = [], []

    def capture(module, args, kwargs):
        inputs = args[0] if args else kwargs["input_ids"]
        trace.append(list(inputs.shape))

    def timed_score(*args, **kwargs):
        engine.synchronize()
        started = time.perf_counter()
        result = scorer(*args, **kwargs)
        engine.synchronize()
        scoring_ms.append((time.perf_counter() - started) * 1000)
        return result

    with ExitStack() as stack:
        hook = engine.model.model.register_forward_pre_hook(capture, with_kwargs=True)
        stack.callback(hook.remove)
        stack.enter_context(patch("experiments.aligned_fewshot.score_token_prompts", timed_score))
        stack.enter_context(patch("openjev.engine.score_token_prompts", timed_score))
        if condition == "upstream":
            row = engine.decide(case)
            _, prompts = engine._prepare(case)
            row.update(prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(), layout="s0_forward")
        elif condition == "json_4":
            row = ExampleJSONEngine(engine, examples[:4]).generate_json(case)
        else:
            row = decide(engine, case, examples, condition)
    if row["forward_calls"] != len(trace):
        raise AssertionError("Reported forward count does not match observed backbone calls")
    row.update(backbone_shapes=trace, scoring_ms=sum(scoring_ms), processed_tokens=sum(b * s for b, s in trace))
    return row


def serial_scorer(model, prompts, **kwargs):
    kwargs["parallel"] = False
    return score_token_prompts(model, prompts, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--include-json", action="store_true")
    parser.add_argument("--include-upstream", action="store_true")
    parser.add_argument("--independent", action="store_true")
    parser.add_argument("--float32", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".partial.json").exists() or args.repeats < 1:
        parser.error("Choose an unused output and positive repeat count")
    data = Path("data/holdout-v4.json")
    examples_path = Path("data/fewshot-examples-v1.json")
    cases, examples = read_cases(data), read_cases(examples_path)
    if args.case_ids:
        if set(args.case_ids) - {c["id"] for c in cases}:
            parser.error("Unknown case IDs")
        cases = [c for c in cases if c["id"] in args.case_ids]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("Limit must be positive")
        cases = cases[: args.limit]
    upstream, upstream_hashes = load_cases(Path("data"), "upstream") if args.include_upstream else ([], {})
    scorers = {"old": archived_scorer(), "padded": score_token_prompts}
    if args.independent:
        scorers["independent"] = serial_scorer
    torch.manual_seed(42)
    gpu_start = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.free", "--format=csv,noheader"], text=True
    ).strip()
    engine = load_engine("Qwen/Qwen3.5-2B", revision=REVISION, max_input_tokens=16384)
    if args.float32:
        # Promote the same BF16 checkpoint values to isolate execution arithmetic.
        engine.model.float()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    jobs = [
        (f"{condition}/{name}", condition, scorer) for condition in args.conditions for name, scorer in scorers.items()
    ]
    if args.include_json:
        jobs.append(("json_4", "json_4", score_token_prompts))
    upstream_jobs = [(f"upstream/{name}", "upstream", scorer) for name, scorer in scorers.items()]
    report = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": engine.model_id,
            "revision": engine.revision,
            "dtype": "BF16 checkpoint promoted to FP32" if args.float32 else "BF16 backbone; FP32 selected head",
            "trained": False,
            "prompts_changed": False,
            "dataset_role": "Previously inspected v4; execution/speed comparison, not a new accuracy holdout",
            "unique_cases": len(cases),
            "repeats": args.repeats,
            "branch_batch_size": engine.branch_batch_size,
            "seed": 42,
            "warmups_per_job": 1,
            "gpu_at_start": gpu_start,
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers")},
            "file_sha256": {str(p): sha(p) for p in (data, examples_path, ARCHIVE)},
            "upstream_sha256": upstream_hashes,
            "source_sha256": {
                str(p): sha(p) for root in ("src/openjev", "experiments") for p in sorted(Path(root).glob("*.py"))
            },
            "timing": "Synchronized end-to-end including prompt preparation, scoring and decoding; additional synchronized scoring time excludes prompt/answer handling. Jobs rotate across cases/repetitions.",
        },
        "results": {name: [] for name, _, _ in jobs + (upstream_jobs if upstream else [])},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.json")
    with ExitStack() as stack:
        if args.float32:
            stack.enter_context(patch("transformers.integrations.sdpa_attention.use_gqa_in_sdpa", return_value=False))
            stack.enter_context(sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION))
            report["metadata"]["attention_backend"] = "EFFICIENT_ATTENTION with explicitly repeated KV heads"
        for _, condition, scorer in jobs:
            measure(engine, cases[0], examples, condition, scorer)
        if upstream:
            for _, condition, scorer in upstream_jobs:
                measure(engine, upstream[0], examples, condition, scorer)
        torch.cuda.reset_peak_memory_stats()
        for dataset, pending in ((cases, jobs), (upstream, upstream_jobs)):
            for index, case in enumerate(dataset):
                for repeat in range(args.repeats):
                    offset = (index + repeat) % len(pending)
                    for name, condition, scorer in pending[offset:] + pending[:offset]:
                        row = measure(engine, case, examples, condition, scorer)
                        row["repeat"] = repeat
                        report["results"][name].append(row)
                report["metadata"]["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                report["metadata"]["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
                partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(f"Completed {index + 1}/{len(dataset)}: {case['id']}", flush=True)
    report["summaries"] = {}
    for name, rows in report["results"].items():
        summary = make_summary(upstream if name.startswith("upstream/") else cases, rows)
        summary["median_scoring_ms"] = statistics.median(r["scoring_ms"] for r in rows)
        summary["forward_calls"] = sorted({r["forward_calls"] for r in rows})
        report["summaries"][name] = summary
    report["comparisons"] = {}
    for condition in args.conditions + (["upstream"] if upstream else []):
        selected_cases = upstream if condition == "upstream" else cases
        padded = report["results"][f"{condition}/padded"]
        for name in scorers:
            if name == "padded":
                continue
            reference = report["results"][f"{condition}/{name}"]
            first = [r for r in reference if r["repeat"] == 0]
            comparison = compare_execution(selected_cases, padded, first)
            by_key = {(r["id"], r["repeat"]): r for r in reference}
            comparison["median_paired_speedup"] = statistics.median(
                by_key[r["id"], r["repeat"]]["elapsed_ms"] / r["elapsed_ms"] for r in padded
            )
            report["comparisons"][f"{condition}/padded_vs_{name}"] = comparison
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    partial.unlink()
    print(json.dumps(report["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
