"""Test field-specific single-code demonstrations without changing model weights."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.answer_ablation import REVISION, decode_scores
from experiments.fewshot_stability import (
    LAYOUTS,
    ExampleJSONEngine,
    layout_id,
    prepare_messages,
    read_cases,
    stability_summary,
    tokenize_messages,
    validate_split,
)
from openjev.benchmark import make_summary
from openjev.engine import score_token_prompts
from openjev.runtime import load_engine
from openjev.schema import parse_schema, same_value, valid_answer

CONDITIONS = ("original_0", "object_4", "aligned_chat_4")


def prepare_aligned_messages(context, fields, examples, *, shift=0, reverse=False):
    targets, ordered, codes = prepare_messages(context, fields, [], shift=shift, reverse=reverse)
    histories = [[target[0]] for target in targets]
    for example in examples:
        if not valid_answer(example["expected"], fields):
            raise ValueError("Demonstration labels must cover the complete schema")
        questions, _, _ = prepare_messages(example["context"], fields, [], shift=shift, reverse=reverse)
        for history, question, field, cs in zip(histories, questions, ordered, codes, strict=True):
            answer = cs[next(i for i, v in enumerate(field.choices) if same_value(v, example["expected"][field.name]))]
            history.extend([question[-1], {"role": "assistant", "content": answer}])
    for history, target in zip(histories, targets, strict=True):
        history.append(target[-1])
    return histories, ordered, codes


@torch.inference_mode()
def decide(engine, case, examples, condition, shift=0, reverse=False, *, parallel=True, prepare_fn=None):
    engine.synchronize()
    started = time.perf_counter()
    fields = parse_schema(case["schema"])
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    prepare = prepare_fn or (prepare_aligned_messages if condition == "aligned_chat_4" else prepare_messages)
    messages, ordered, codes = prepare(
        case["context"], fields, [] if condition == "original_0" else examples[:4], shift=shift, reverse=reverse
    )
    prompts = tokenize_messages(engine.tokenizer, messages)
    if max(map(len, prompts)) > engine.max_input_tokens:
        raise ValueError("Prompt exceeds input limit")
    candidates = []
    for message, prompt, cs in zip(messages, prompts, codes, strict=True):
        rendered = engine.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = []
        for code in cs:
            encoded = engine.tokenizer.encode(code, add_special_tokens=False)
            if len(encoded) != 1 or engine.tokenizer.decode(encoded) != code:
                raise ValueError("Answer code must be one exact token")
            if engine.tokenizer.encode(rendered + code, add_special_tokens=False) != prompt + encoded:
                raise ValueError("Answer changes prompt token boundary")
            ids.append(encoded[0])
        if len(set(ids)) != len(ids):
            raise ValueError("Answer token collision")
        candidates.append(ids)
    union = list(dict.fromkeys(t for ids in candidates for t in ids))
    indices = {token: i for i, token in enumerate(union)}
    scores, stats = score_token_prompts(
        engine.model,
        prompts,
        parallel=parallel,
        candidate_ids=union,
        branch_batch_size=engine.branch_batch_size,
    )
    probabilities = [
        scores[i, [indices[t] for t in ids]].float().softmax(-1).cpu().tolist() for i, ids in enumerate(candidates)
    ]
    values, probabilities = decode_scores(fields, ordered, probabilities)
    engine.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "id": case["id"],
        "values": values,
        "probabilities": probabilities,
        "elapsed_ms": elapsed_ms,
        "input_tokens": max(map(len, prompts)),
        "total_prompt_tokens": sum(map(len, prompts)),
        "prompt_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        "layout": layout_id(shift, reverse),
        "repeat": 0,
        "parallel": parallel,
        **stats,
    }


def compare_execution(cases, parallel_rows, serial_rows):
    cached = {r["id"]: r for r in parallel_rows if r["repeat"] == 0 and r["layout"] == "s0_forward"}
    serial = {r["id"]: r for r in serial_rows}
    differences, disagreements = [], []
    for case in cases:
        a, b = cached[case["id"]], serial[case["id"]]
        if a["prompt_sha256"] != b["prompt_sha256"]:
            raise ValueError("Serial and parallel prompts differ")
        for field in parse_schema(case["schema"]):
            differences.extend(
                abs(x - y) for x, y in zip(a["probabilities"][field.name], b["probabilities"][field.name], strict=True)
            )
            if not same_value(a["values"][field.name], b["values"][field.name]):
                disagreements.append(
                    {
                        "id": case["id"],
                        "field": field.name,
                        "parallel": a["values"][field.name],
                        "serial": b["values"][field.name],
                    }
                )
    return {
        "compared_fields": sum(len(parse_schema(c["schema"])) for c in cases),
        "identical_prompt_hashes": True,
        "max_abs_probability_difference": max(differences),
        "choice_disagreements": len(disagreements),
        "disagreements": disagreements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    data = Path("data/fewshot-development-v1.json" if args.smoke else "data/holdout-v3.json")
    examples_path = Path("data/fewshot-examples-v1.json")
    protocol_path = Path("reports/aligned-protocol.json")
    protocol = json.loads(protocol_path.read_text())
    for path, digest in protocol["files"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen input changed: {path}")
    cases, examples = read_cases(data), read_cases(examples_path)
    if args.smoke:
        cases = cases[:8]
    validate_split(cases, examples)
    if not args.smoke:
        for path in ("data/diagnostic.json", "data/holdout-v1.json", "data/holdout-v2.json"):
            validate_split(cases, read_cases(Path(path)))
    layouts = [(0, False)] if args.smoke else LAYOUTS
    repeats = 1 if args.smoke else 3
    torch.manual_seed(42)
    engine = load_engine("Qwen/Qwen3.5-2B", revision=REVISION)
    json_engines = {} if args.smoke else {s: ExampleJSONEngine(engine, examples[:s]) for s in (0, 4)}
    modes = [f"parallel_{c}" for c in CONDITIONS]
    if not args.smoke:
        modes += [f"serial_{c}" for c in CONDITIONS] + [f"json_{s}" for s in json_engines]
    report = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": engine.model_id,
            "revision": engine.revision,
            "dtype": "BF16 backbone; FP32 selected output head",
            "trained": False,
            "calibrated": False,
            "smoke": args.smoke,
            "data": str(data),
            "file_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (data, examples_path, protocol_path)
            },
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for root in ("src/openjev", "experiments")
                for p in sorted(Path(root).glob("*.py"))
            },
            "identity_repeats": repeats,
            "unique_cases": len(cases),
            "layout_ids": [layout_id(*layout) for layout in layouts],
            "seed": 42,
            "warmups_per_condition": 1,
            "gpu_at_start": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.free", "--format=csv,noheader"], text=True
            ).strip(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        },
        "results": {mode: [] for mode in modes},
    }
    # Warm up on already-inspected development data, never using holdout labels.
    warmup = read_cases(Path("data/fewshot-development-v1.json"))[0]
    for condition in CONDITIONS:
        decide(engine, warmup, examples, condition)
        if not args.smoke:
            decide(engine, warmup, examples, condition, parallel=False)
    for wrapper in json_engines.values():
        wrapper.generate_json(warmup)
    torch.cuda.reset_peak_memory_stats()
    for case_index, case in enumerate(cases):
        jobs = [(f"parallel_{c}", c, shift, reverse, 0) for c in CONDITIONS for shift, reverse in layouts]
        jobs += [(f"parallel_{c}", c, 0, False, rep) for c in CONDITIONS for rep in range(1, repeats)]
        if not args.smoke:
            jobs += [(f"serial_{c}", c, 0, False, 0) for c in CONDITIONS]
            jobs += [(f"json_{s}", s, 0, False, rep) for s in json_engines for rep in range(repeats)]
        offset = case_index % len(jobs)
        for mode, condition, shift, reverse, repeat in jobs[offset:] + jobs[:offset]:
            row = (
                json_engines[condition].generate_json(case)
                if mode.startswith("json")
                else decide(engine, case, examples, condition, shift, reverse, parallel=mode.startswith("parallel"))
            )
            row.update(repeat=repeat, layout=layout_id(shift, reverse))
            report["results"][mode].append(row)
        args.output.with_suffix(".partial.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"{case_index + 1}/{len(cases)} {case['id']}", flush=True)
    for rows in report["results"].values():
        rows.sort(key=lambda r: (r["repeat"], r["id"], r["layout"]))
    report["summaries"] = {
        mode: stability_summary(cases, rows, report["metadata"]["layout_ids"])
        if mode.startswith("parallel")
        else make_summary(cases, rows)
        for mode, rows in report["results"].items()
    }
    if not args.smoke:
        report["execution_comparisons"] = {
            c: compare_execution(cases, report["results"][f"parallel_{c}"], report["results"][f"serial_{c}"])
            for c in CONDITIONS
        }
    report["metadata"]["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["metadata"]["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".partial.json").unlink()
    for mode, summary in report["summaries"].items():
        print(
            mode,
            {
                k: summary[k]
                for k in ("mean_field_accuracy", "field_accuracy", "median_ms", "stable_field_fraction")
                if k in summary
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
