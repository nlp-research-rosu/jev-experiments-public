"""Datasets, accuracy/latency summaries, and portable experiment reports."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from .metrics import summarize
from .schema import parse_schema, valid_answer


def load_cases(data_dir: Path, suite: str):
    paths = []
    if suite in ("diagnostic", "all"):
        paths.append(data_dir / "diagnostic.json")
    if suite in ("upstream", "all"):
        paths.extend(sorted(p for p in (data_dir / "upstream").glob("*.json") if p.name != "PROVENANCE.json"))
    cases, provenance = [], {}
    for path in paths:
        content = path.read_bytes()
        provenance[str(path.relative_to(data_dir))] = hashlib.sha256(content).hexdigest()
        value = json.loads(content)
        if "cases" in value:
            cases.extend({"schema": value["schema"], **case} for case in value["cases"])
        else:
            cases.append(value)
    if not cases:
        raise ValueError("no cases found")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case IDs")
    for case in cases:
        fields = parse_schema(case["schema"])
        if not isinstance(case.get("context"), str) or not case["context"].strip():
            raise ValueError(f"{case['id']}: missing text context")
        if "expected" in case and not valid_answer(case["expected"], fields):
            raise ValueError(f"{case['id']}: invalid or incomplete gold labels")
    return cases, provenance


def make_summary(cases, rows):
    first = {r["id"]: r for r in rows if r["repeat"] == 0}
    result = summarize(cases, [first[c["id"]] for c in cases])
    latencies = [r["elapsed_ms"] for r in rows]
    result.update(
        {
            "timed_evaluations": len(rows),
            "median_ms": statistics.median(latencies),
            "mean_ms": statistics.mean(latencies),
            "p95_ms": sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)],
            "repeat_value_disagreements": sum(r["values"] != first[r["id"]]["values"] for r in rows if r["repeat"] > 0),
        }
    )
    return result


def write_markdown(report, output: Path):
    meta = report["metadata"]

    def percent(value):
        return "unlabeled" if value is None else f"{100 * value:.1f}%"

    lines = [
        "# Parallel decision experiment",
        "",
        f"Model: `{meta['model']}` at `{meta['revision']}`. GPU: {meta['device_name']}.",
        "",
        f"Suite: **{meta['suite']}**, {meta['unique_cases']} unique cases; {meta['repeats']} timing repetitions per case.",
        "Quality metrics use the first repetition only. Timing excludes download/loading and includes prompt preparation, GPU execution, and answer assembly.",
        "",
        "| Method | Field accuracy | Exact cases | Valid schema | Median ms | p95 ms | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, s in report["summaries"].items():
        brier = "—" if s["brier"] is None else f"{s['brier']:.4f}"
        lines.append(
            f"| {mode} | {percent(s['field_accuracy'])} | {percent(s['exact_case_accuracy'])} | {percent(s['schema_validity'])} | {s['median_ms']:.1f} | {s['p95_ms']:.1f} | {brier} |"
        )
    if "paired_speedup_median" in report:
        lines += ["", f"Median paired JSON/parallel latency ratio: **{report['paired_speedup_median']:.2f}×**."]
    lines += ["", "## Cached versus independent reference", ""]
    if "reference_tolerance" in meta:
        lines += [f"Configured maximum absolute probability difference: {meta['reference_tolerance']:.3f}.", ""]
    for check in report["reference_checks"]:
        lines.append(
            f"- {check['id']}: max probability difference {check['max_abs_probability_difference']:.6f}; choice disagreements {check['choice_disagreements']}."
        )
    lines += ["", "## Errors", ""]
    for mode, s in report["summaries"].items():
        lines.append(f"### {mode}")
        lines.append("")
        if not s["errors"]:
            lines.append("No labeled errors." if s["labeled_fields"] else "No gold labels; accuracy is not measured.")
        for e in s["errors"]:
            lines.append(f"- `{e['id']}` / `{e['field']}`: expected `{e['expected']}`, got `{e['actual']}`.")
        lines.append("")
    lines += [
        "## Interpretation limits",
        "",
        "The diagnostic set is small, synthetic, and hand-authored. Upstream scenarios have no gold labels. Neither establishes production quality or parity with Jev.",
        "Candidate probabilities are normalized scores, not calibrated confidence. The Brier score is the sum over all classes, averaged over labeled fields; ECE uses ten equal-width confidence bins.",
        "The parallel path uses verified single-token option codes and independently answers each field. JSON generation uses original answer values and can condition later fields on earlier ones, so these are different inference objectives and prompts.",
        "The JSON baseline is greedy ordinary generation, without grammar constraints. This is not a comparison against an optimized structured-output serving engine.",
        "Reported speed is for this implementation, model, hardware, input sizes, and current GPU load. All forward calls, including prefill, are counted in the raw report.",
        "",
    ]
    output.write_text("\n".join(lines))
