"""Run a local benchmark, retaining raw outputs and reproducibility metadata."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .benchmark import load_cases, make_summary, write_markdown
from .runtime import load_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--suite", choices=["diagnostic", "upstream", "all"], default="diagnostic")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/experiment.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--check-reference", type=int, default=3)
    parser.add_argument(
        "--reference-tolerance",
        type=float,
        default=0.02,
        help="Maximum cached/independent absolute probability difference (default: 0.02)",
    )
    parser.add_argument("--branch-batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--modes", nargs="+", choices=["parallel", "independent", "json"], default=["parallel", "json"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.warmups < 0
        or args.check_reference < 0
        or args.branch_batch_size < 1
        or args.max_input_tokens < 1
        or args.max_new_tokens < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("counts and limits must be positive (warmups/reference checks may be zero)")
    if not 0 <= args.reference_tolerance <= 1:
        parser.error("reference tolerance must be finite and between zero and one")
    if len(set(args.modes)) != len(args.modes):
        parser.error("modes must be unique")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}; choose another path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases, hashes = load_cases(args.data_dir, args.suite)
    if args.limit:
        cases = cases[: args.limit]
    torch.manual_seed(42)
    started = time.perf_counter()
    engine = load_engine(
        args.model,
        revision=args.revision,
        device=args.device,
        branch_batch_size=args.branch_batch_size,
        max_input_tokens=args.max_input_tokens,
    )
    source_root = Path(__file__).parent
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "revision": engine.revision,
        "dtype": str(next(engine.model.parameters()).dtype),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate", "tokenizers")},
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
        "cuda_runtime": torch.version.cuda,
        "load_seconds": time.perf_counter() - started,
        "suite": args.suite,
        "unique_cases": len(cases),
        "repeats": args.repeats,
        "warmups": args.warmups,
        "reference_check_count": args.check_reference,
        "reference_tolerance": args.reference_tolerance,
        "dataset_sha256": hashes,
        "seed": 42,
        "branch_batch_size": args.branch_batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_root.glob("*.py"))},
        "calibrated": False,
        "trained": False,
    }
    if args.device == "cuda":
        meta["gpu_at_start"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        torch.cuda.reset_peak_memory_stats()
    report = {"metadata": meta, "reference_checks": [], "results": {m: [] for m in args.modes}}
    for case in cases[: args.check_reference]:
        a, b = engine.decide(case), engine.decide(case, parallel=False)
        difference = max(
            abs(x - y)
            for field in a["probabilities"]
            for x, y in zip(a["probabilities"][field], b["probabilities"][field], strict=True)
        )
        check = {
            "id": case["id"],
            "max_abs_probability_difference": difference,
            "choice_disagreements": sum(a["values"][f] != b["values"][f] for f in a["values"]),
            "parallel": a,
            "independent": b,
        }
        report["reference_checks"].append(check)
        print(f"reference {case['id']}: max probability difference {difference:.6f}", flush=True)
        if difference > args.reference_tolerance:
            args.output.with_suffix(".failed-reference.json").write_text(json.dumps(report, indent=2))
            raise RuntimeError(
                f"cached/reference probabilities differ by more than {args.reference_tolerance}; "
                "investigate before benchmarking"
            )

    def run(mode, case):
        if mode == "json":
            return engine.generate_json(case, max_new_tokens=args.max_new_tokens)
        return engine.decide(case, parallel=(mode == "parallel"))

    for _ in range(args.warmups):
        for mode in args.modes:
            run(mode, cases[0])
    for repeat in range(args.repeats):
        for i, case in enumerate(cases):
            rotation = (repeat + i) % len(args.modes)
            order = args.modes[rotation:] + args.modes[:rotation]
            for mode in order:
                row = run(mode, case)
                row["repeat"] = repeat
                report["results"][mode].append(row)
                print(
                    f"repeat {repeat + 1}/{args.repeats} case {i + 1}/{len(cases)} {case['id']} {mode}: {row['elapsed_ms']:.1f} ms",
                    flush=True,
                )
            args.output.with_suffix(".partial.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    report["summaries"] = {m: make_summary(cases, rows) for m, rows in report["results"].items()}
    if "parallel" in args.modes and "json" in args.modes:
        lookup = {(r["id"], r["repeat"]): r["elapsed_ms"] for r in report["results"]["parallel"]}
        report["paired_speedup_median"] = statistics.median(
            r["elapsed_ms"] / lookup[r["id"], r["repeat"]] for r in report["results"]["json"]
        )
    if args.device == "cuda":
        meta["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        meta["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    meta["total_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    args.output.with_suffix(".partial.json").unlink(missing_ok=True)
    print(f"Saved {args.output} and {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
