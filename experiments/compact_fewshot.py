"""Fixed compact/shared single-code prompts; development selection then holdout."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

from experiments.aligned_fewshot import compare_execution
from experiments.aligned_fewshot import decide as aligned_decide
from experiments.answer_ablation import REVISION
from experiments.fewshot_stability import (
    LAYOUTS,
    ExampleJSONEngine,
    layout_id,
    prepare_messages,
    read_cases,
    stability_summary,
    validate_split,
)
from openjev.benchmark import make_summary
from openjev.runtime import load_engine
from openjev.schema import same_value, valid_answer

CANDIDATES = ("field_4", "shared_4", "field_questions_4")


def prepare_compact_messages(context, fields, examples, *, style="field", shift=0, reverse=False):
    if style not in ("field", "shared", "field_questions"):
        raise ValueError("Unknown compact style")
    targets, ordered, codes = prepare_messages(context, fields, [], shift=shift, reverse=reverse)
    if not examples:
        return targets, ordered, codes
    if any(not valid_answer(e["expected"], fields) for e in examples):
        raise ValueError("Demonstration labels must cover the complete schema")
    schema = [
        {"field": f.name, "question": f.description, "options": dict(zip(cs, f.choices, strict=True))}
        for f, cs in zip(ordered, codes, strict=True)
    ]
    system = {
        "role": "system",
        "content": targets[0][0]["content"]
        + "\n\nSCHEMA FOR EVERY REQUEST:\n"
        + json.dumps(schema, ensure_ascii=False),
    }

    def history_for(indices):
        history = [system]
        for example in examples:
            for i in indices:
                f, cs = ordered[i], codes[i]
                code = cs[next(j for j, v in enumerate(f.choices) if same_value(v, example["expected"][f.name]))]
                question = (
                    f"\nQuestion for {json.dumps(f.name)}: choose using the schema above.\nAnswer only the option code."
                )
                if style == "field_questions":
                    question = (
                        f"\nQuestion for {json.dumps(f.name)}: {f.description}\nOptions:\n"
                        + "\n".join(
                            f"{c}: {json.dumps(v, ensure_ascii=False)}" for c, v in zip(cs, f.choices, strict=True)
                        )
                        + "\nAnswer only the option code."
                    )
                history.extend(
                    [
                        {
                            "role": "user",
                            "content": "CONTEXT (data):\n" + example["context"] + question,
                        },
                        {"role": "assistant", "content": code},
                    ]
                )
        return history

    if style == "shared":
        shared = history_for(range(len(fields)))
        histories = [shared + [target[-1]] for target in targets]
    else:
        histories = [history_for([i]) + [target[-1]] for i, target in enumerate(targets)]
    return histories, ordered, codes


def select_candidate(summaries, baseline):
    checks = {}
    for name, s in summaries.items():
        checks[name] = {
            "mean_accuracy": s["mean_field_accuracy"] >= baseline["mean_field_accuracy"] - 0.01 - 1e-12,
            "stable_fraction": s["stable_field_fraction"] >= baseline["stable_field_fraction"] - 0.02 - 1e-12,
            "worst_layout": s["worst_layout_accuracy"] >= baseline["worst_layout_accuracy"] - 0.02 - 1e-12,
            "speed": s["per_layout"]["s0_forward"]["median_ms"]
            <= 0.75 * baseline["per_layout"]["s0_forward"]["median_ms"],
        }
    eligible = sorted(name for name, results in checks.items() if all(results.values()))
    if eligible:
        selected = min(
            eligible,
            key=lambda n: (
                summaries[n]["per_layout"]["s0_forward"]["median_ms"],
                -summaries[n]["mean_field_accuracy"],
                n,
            ),
        )
    else:
        selected = min(
            summaries,
            key=lambda n: (
                -summaries[n]["mean_field_accuracy"],
                -summaries[n]["stable_field_fraction"],
                summaries[n]["per_layout"]["s0_forward"]["median_ms"],
                n,
            ),
        )
    return {
        "selected": selected,
        "eligible": eligible,
        "checks": checks,
        "status": "eligible" if eligible else "exploratory_fallback",
    }


def decide(engine, case, examples, condition, shift=0, reverse=False, *, parallel=True):
    if condition in CANDIDATES:
        return aligned_decide(
            engine,
            case,
            examples,
            "aligned_chat_4",
            shift,
            reverse,
            parallel=parallel,
            prepare_fn=partial(prepare_compact_messages, style=condition.removesuffix("_4")),
        )
    return aligned_decide(engine, case, examples, condition, shift, reverse, parallel=parallel)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["development", "holdout"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("reports/compact-selection.json"))
    parser.add_argument("--protocol", type=Path, default=Path("reports/compact-protocol.json"))
    args = parser.parse_args()
    development = args.stage == "development"
    if args.output.exists() or (development and args.selection.exists()):
        parser.error("Use new output and development selection paths")
    protocol_path = args.protocol
    protocol = json.loads(protocol_path.read_text())
    for path, digest in protocol["files"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    data = Path("data/holdout-v3.json" if development else "data/holdout-v4.json")
    cases = read_cases(data)
    examples = read_cases(Path("data/fewshot-examples-v1.json"))
    validate_split(cases, examples)
    if development:
        conditions = protocol["candidates"]
        if not conditions or any(c not in CANDIDATES for c in conditions):
            raise ValueError("Protocol contains unknown candidates")
    else:
        selection = json.loads(args.selection.read_text())
        if selection["protocol_sha256"] != sha(protocol_path) or selection["holdout_sha256"] != sha(data):
            raise ValueError("Selection does not match frozen protocol/holdout")
        if selection["development_sha256"] != sha(selection["development_report"]):
            raise ValueError("Development report changed after selection")
        conditions = ["original_0", "aligned_chat_4", selection["selected"]]
        for path in ("data/diagnostic.json", "data/holdout-v1.json", "data/holdout-v2.json", "data/holdout-v3.json"):
            validate_split(cases, read_cases(Path(path)))
    torch.manual_seed(42)
    engine = load_engine("Qwen/Qwen3.5-2B", revision=REVISION)
    json_engine = None if development else ExampleJSONEngine(engine, examples[:4])
    modes = conditions + ([] if development else ["json_4", "serial_selected"])
    report = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": engine.model_id,
            "revision": engine.revision,
            "stage": args.stage,
            "data": str(data),
            "trained": False,
            "calibrated": False,
            "dtype": "BF16 backbone; FP32 selected output head",
            "seed": 42,
            "unique_cases": len(cases),
            "identity_repeats": 3,
            "warmups_per_condition": 1,
            "layout_ids": [layout_id(*layout) for layout in LAYOUTS],
            "file_sha256": {str(p): sha(p) for p in (data, protocol_path, Path("data/fewshot-examples-v1.json"))},
            "source_sha256": {
                str(p): sha(p) for root in ("src/openjev", "experiments") for p in sorted(Path(root).glob("*.py"))
            },
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers")},
            "gpu_at_start": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.free", "--format=csv,noheader"], text=True
            ).strip(),
        },
        "results": {mode: [] for mode in modes},
    }
    if not development:
        report["metadata"]["selection_sha256"] = sha(args.selection)
        report["metadata"]["selected"] = selection["selected"]
    warmup = read_cases(Path("data/holdout-v3.json"))[0]
    for condition in conditions:
        decide(engine, warmup, examples, condition)
    if json_engine:
        json_engine.generate_json(warmup)
        decide(engine, warmup, examples, selection["selected"], parallel=False)
    if development:
        previous = json.loads(Path("reports/aligned-holdout.json").read_text())
        report["reproduction_checks"] = []
        for condition in ("original_0", "aligned_chat_4"):
            old = {
                r["id"]: r
                for r in previous["results"]["parallel_" + condition]
                if r["repeat"] == 0 and r["layout"] == "s0_forward"
            }
            for case in cases[:4]:
                row = decide(engine, case, examples, condition)
                if (
                    row["prompt_sha256"] != old[case["id"]]["prompt_sha256"]
                    or row["values"] != old[case["id"]]["values"]
                ):
                    raise ValueError("Previous baseline does not reproduce")
                report["reproduction_checks"].append(
                    {"condition": condition, "id": case["id"], "same_prompt_and_answer": True}
                )
        print("Eight old-control prompt/answer checks passed", flush=True)
    torch.cuda.reset_peak_memory_stats()
    for case_index, case in enumerate(cases):
        jobs = [(condition, shift, reverse, 0) for condition in conditions for shift, reverse in LAYOUTS]
        jobs += [(condition, 0, False, rep) for condition in conditions for rep in (1, 2)]
        if not development:
            jobs += [("json_4", 0, False, rep) for rep in range(3)] + [("serial_selected", 0, False, 0)]
        offset = case_index % len(jobs)
        for mode, shift, reverse, repeat in jobs[offset:] + jobs[:offset]:
            if mode == "json_4":
                row = json_engine.generate_json(case)
            elif mode == "serial_selected":
                row = decide(engine, case, examples, selection["selected"], parallel=False)
            else:
                row = decide(engine, case, examples, mode, shift, reverse)
            row.update(repeat=repeat, layout=layout_id(shift, reverse))
            report["results"][mode].append(row)
        args.output.with_suffix(".partial.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"{case_index + 1}/{len(cases)} {case['id']}", flush=True)
    for rows in report["results"].values():
        rows.sort(key=lambda r: (r["repeat"], r["id"], r["layout"]))
    report["summaries"] = {
        mode: stability_summary(cases, rows, report["metadata"]["layout_ids"])
        if mode in conditions
        else make_summary(cases, rows)
        for mode, rows in report["results"].items()
    }
    if not development:
        report["execution_comparison"] = compare_execution(
            cases, report["results"][selection["selected"]], report["results"]["serial_selected"]
        )
    report["metadata"]["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["metadata"]["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".partial.json").unlink()
    if development:
        pool = dict(report["summaries"])
        if "previous_development" in protocol:
            pool.update(json.loads(Path(protocol["previous_development"]).read_text())["summaries"])
        choice = select_candidate(pool, previous["summaries"]["parallel_aligned_chat_4"])
        choice.update(
            created_utc=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=sha(protocol_path),
            development_report=str(args.output),
            development_sha256=sha(args.output),
            holdout_sha256=sha("data/holdout-v4.json"),
            holdout_scored_at_selection=False,
        )
        args.selection.write_text(json.dumps(choice, indent=2) + "\n")
        print("Selection:", choice, flush=True)
    for mode, summary in report["summaries"].items():
        print(
            mode,
            {
                k: summary[k]
                for k in ("mean_field_accuracy", "stable_field_fraction", "field_accuracy", "median_ms")
                if k in summary
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
