"""Warm, local end-to-end latency and numerical checks for trained judgments."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from openjev.judgment_cli import load_judgment_engine, read_json
from openjev.judgment_training import evaluate_bundles, load_bundles, prepare_bundle
from openjev.judgments import assemble_response


def timing_summary(values):
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("nonempty finite nonnegative timings required")
    ordered = sorted(values)
    return {
        "samples": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[math.ceil(0.95 * len(values)) - 1],
        "p99_ms": ordered[math.ceil(0.99 * len(values)) - 1],
        "mean_ms": statistics.mean(values),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def leaves(tree, path=()):
    if isinstance(tree, dict):
        if isinstance(tree.get("type"), str) and "probabilities" in tree:
            return {path: tree}
        return {p: v for k, value in tree.items() for p, v in leaves(value, path + (k,)).items()}
    if isinstance(tree, list):
        return {p: v for k, value in enumerate(tree) for p, v in leaves(value, path + (k,)).items()}
    raise ValueError("invalid answer tree")


def decision(leaf):
    if leaf["type"] == "choice":
        return leaf["choice"]
    if leaf["type"] == "noul":
        return leaf["noul"] >= 0.5
    return max(leaf["probabilities"], key=leaf["probabilities"].get)


def compare_answers(a, b):
    a, b = leaves(a), leaves(b)
    if a.keys() != b.keys() or not a:
        raise ValueError("answer paths differ")
    changes, maximum, mean, scores = 0, 0.0, [], []
    for path, left in a.items():
        right = b[path]
        if left["type"] != right["type"] or left["probabilities"].keys() != right["probabilities"].keys():
            raise ValueError("answer spaces differ")
        delta = max(abs(p - right["probabilities"][key]) for key, p in left["probabilities"].items())
        if not math.isfinite(delta):
            raise ValueError("nonfinite answer")
        maximum = max(maximum, delta)
        mean.append(delta)
        changes += decision(left) != decision(right)
        if left["type"] == "score":
            scores.append(abs(left["score"] - right["score"]))
    return {
        "questions": len(a),
        "decision_changes": changes,
        "max_probability_difference": maximum,
        "mean_question_max_difference": statistics.mean(mean),
        "max_score_value_difference": max(scores, default=0),
    }


def make_request(history, questions, *, choices=None):
    if questions < 1 or (choices is not None and not 1 <= choices <= 255):
        raise ValueError("invalid request size")
    state = {
        "customer": {"message": "I was charged twice. Please refund the duplicate payment."},
        "order": {"payments": [24.5, 24.5], "currency": "USD"},
        "policy": {"duplicate_payments": "Refund the duplicate charge."},
        "history": history,
    }
    templates = [
        {
            "type": "choice",
            "instructions": "Which department should handle this customer request?",
            "criteria": {
                "billing": "Payments and refunds",
                "technical": "Software faults",
                "sales": "Buying a product",
                "shipping": "Delivery and tracking",
            },
        },
        {
            "type": "noul",
            "instructions": "Does the customer explicitly request money back?",
            "criteria": {"true": "A refund is explicitly requested.", "false": "A refund is not explicitly requested."},
        },
        {
            "type": "score",
            "instructions": "How strongly does the policy support refunding the duplicate payment?",
            "criteria": ["The policy contradicts a refund.", "The policy is unclear.", "The policy supports a refund."],
        },
    ]
    if choices is not None:
        templates = [
            {
                "type": "choice",
                "instructions": "Select the department matching the customer's request.",
                "criteria": {
                    "billing": "Payments and refunds",
                    **{
                        f"department_{i}": f"Unrelated archived department number {i}, not handling refunds."
                        for i in range(1, choices)
                    },
                },
            }
        ]
    rows = []
    for i in range(questions):
        question = copy.deepcopy(templates[i % len(templates)])
        # Distinct instructions prevent every repeated question sharing its entire suffix.
        question["instructions"] += f" This is independent policy check {i + 1}."
        rows.append({"answer": question})
    return {"state": state, "questions": {"decisions": rows}}


def cases_for(engine):
    seed = "Archived note: the customer previously asked about delivery, and the previous order was delivered successfully. "
    tokens = engine.tokenizer.encode(seed * 500, add_special_tokens=False)
    cases = [
        {
            "id": "nested-realistic",
            "request": read_json("data/integration/nested-request.json"),
            "role": "hand-authored integration case; no quality claim",
        }
    ]
    for history_tokens, counts in ((128, (1, 5, 10, 25, 50)), (512, (5, 10)), (2048, (5, 10)), (4096, (10,))):
        text = engine.tokenizer.decode(tokens[:history_tokens])
        for count in counts:
            cases.append(
                {
                    "id": f"context-{history_tokens}-questions-{count}",
                    "request": make_request(text, count),
                    "role": "synthetic scaling only",
                }
            )
    for choices in (32, 128):
        cases.append(
            {
                "id": f"choices-{choices}",
                "request": make_request("", 1, choices=choices),
                "role": "synthetic candidate-count scaling only",
            }
        )
    return cases


@torch.inference_mode()
def measure(engine, serialized, *, cached, batch_size):
    engine.synchronize()
    started = time.perf_counter()
    request = json.loads(serialized)
    compiled, prompts, kinds = engine.prepare(request)
    prepared_at = time.perf_counter()
    event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    event_start.record()
    if cached:
        logits, stats = engine.model.score_cached(prompts, kinds, unit_batch_size=batch_size)
    else:
        logits = engine.model.score_prompts(prompts, kinds, unit_batch_size=batch_size)
        stats = {"forward_calls": math.ceil(len(prompts) / batch_size), "prefix_tokens": 0}
    event_end.record()
    engine.synchronize()
    scored_at = time.perf_counter()
    raw = logits.cpu().tolist()
    response = assemble_response(compiled, raw)
    response["usage"] = {
        "input_tokens": sum(map(len, prompts)),
        "max_unit_tokens": max(map(len, prompts)),
        "model_units": len(prompts),
        "output_tokens": 0,
        **stats,
    }
    rendered = json.dumps(response, allow_nan=False, separators=(",", ":"))
    finished = time.perf_counter()
    # Metadata/diagnostics are outside the timed request. CUDA events include stream
    # idle gaps caused by host launches, and must not be called pure kernel time.
    return {
        "end_to_end_ms": (finished - started) * 1000,
        "prepare_ms": (prepared_at - started) * 1000,
        "scoring_wall_ms": (scored_at - prepared_at) * 1000,
        "output_ms": (finished - scored_at) * 1000,
        "cuda_stream_ms": event_start.elapsed_time(event_end),
        "output_bytes": len(rendered.encode()),
        "questions": len(compiled.questions),
        "usage": response["usage"],
        "answers": response["answers"],
        "logits": raw,
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_status():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def write(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def quality_probe(engine, prepared, saved_reference, *, cached=False, batch_size=4):
    """All original test cases once, preserving canonical and pair metrics."""
    original = engine.model.score_prompts
    if cached:
        engine.model.score_prompts = lambda prompts, kinds, **kwargs: engine.model.score_cached(
            prompts, kinds, unit_batch_size=batch_size
        )[0]
    engine.unit_batch_size = batch_size
    try:
        result = evaluate_bundles(engine, prepared)
    finally:
        engine.model.score_prompts = original
    baseline = {r["id"]: r for r in saved_reference["predictions"]}
    difference, flips, questions, max_logit_delta = [], 0, 0, 0.0
    for row in result["predictions"]:
        before = baseline[row["id"]]
        max_logit_delta = max(
            max_logit_delta, max(abs(a - b) for a, b in zip(row["logits"], before["logits"], strict=True))
        )
        # First view only: do not count augmented views as independent quality cases.
        p, q = row["probabilities"][0], before["probabilities"][0]
        difference.append(max(abs(a - b) for a, b in zip(p, q, strict=True)))
        flips += max(range(len(p)), key=p.__getitem__) != max(range(len(q)), key=q.__getitem__)
        questions += 1
    result["vs_saved_unmerged_independent"] = {
        "original_cases": questions,
        "decision_changes": flips,
        "max_probability_difference": max(difference),
        "mean_probability_difference": statistics.mean(difference),
        "max_logit_difference_all_views": max_logit_delta,
    }
    return result


def run(args, report):
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    load_started = time.perf_counter()
    engine = load_judgment_engine(checkpoint=args.checkpoint, backend="fla", max_input_tokens=8192)
    report["load_seconds"] = time.perf_counter() - load_started
    report["checkpoint_id"] = engine.model_id
    prepared, saved = None, None
    if not args.skip_quality:
        reference_dir = Path("checkpoints/judgment-full-v0.2")
        reference_info = read_json(reference_dir / "final/checkpoint.json")
        data = Path("data/processed-v0.2/test.jsonl")
        if reference_info["checkpoint_id"] != engine.model_id or reference_info["metadata"]["fingerprint"][
            "data_sha256"
        ]["test"] != sha(data):
            raise ValueError(
                "quality reference does not match checkpoint/test data; use --skip-quality for other models"
            )
        saved = read_json(reference_dir / "test-final.json")
        prepared = [prepare_bundle(engine, record) for record in load_bundles(data)]
        report["quality_reference"] = {
            "checkpoint_id": reference_info["checkpoint_id"],
            "test_sha256": sha(data),
            "predictions_sha256": sha(reference_dir / "test-final.json"),
        }
        report["quality"] = {}
    cases = cases_for(engine)
    if args.case_ids:
        cases = [case for case in cases if case["id"] in args.case_ids]
        if set(args.case_ids) != {case["id"] for case in cases}:
            raise ValueError("unknown case IDs")
    write(args.output / "requests.json", cases)
    report["requests_sha256"] = sha(args.output / "requests.json")
    modes = {"cached-4": (True, 4), "independent-4": (False, 4), "cached-32": (True, 32), "cached-all": (True, 255)}
    modes = {name: spec for name, spec in modes.items() if name in args.modes}
    references = {}
    for representation in ("adapters", "merged"):
        if representation == "merged":
            print("Merging adapters into an in-memory inference copy", flush=True)
            engine.model.backbone = engine.model.backbone.merge_and_unload(safe_merge=True)
            engine.model_id = report["checkpoint_id"] + "/merged-bf16"
            engine.model.lora_settings = None
            engine.model.eval()
            report["merged_note"] = (
                "In-memory BF16 merge only; original checkpoint unchanged; may change outputs through rounding."
            )
        report["results"][representation] = {}
        for case in cases:
            identifier = case["id"]
            payload = json.dumps(case["request"], allow_nan=False)
            initial = measure(engine, payload, cached=False, batch_size=4)
            if representation == "adapters":
                references[identifier] = initial
            case_result = {
                "questions": initial["questions"],
                "state_tokens": len(
                    engine.tokenizer.encode(
                        json.dumps(case["request"]["state"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                        add_special_tokens=False,
                    )
                ),
                "usage": initial["usage"],
                "role": case["role"],
                "first_independent_call_ms": initial["end_to_end_ms"],
                "modes": {},
            }
            report["results"][representation][identifier] = case_result
            for mode, (cached, batch_size) in modes.items():
                for _ in range(args.warmups):
                    measure(engine, payload, cached=cached, batch_size=batch_size)
                case_result["modes"][mode] = {"samples": [], "repeat_max_probability_difference": 0.0}
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(args.repeats):
                names = list(modes)
                offset = repeat % len(names)
                for mode in names[offset:] + names[:offset]:
                    cached, batch_size = modes[mode]
                    measured = measure(engine, payload, cached=cached, batch_size=batch_size)
                    result = case_result["modes"][mode]
                    if repeat == 0:
                        result["answers"] = measured["answers"]
                        result["usage"] = measured["usage"]
                        result["vs_same_representation_independent"] = compare_answers(
                            initial["answers"], measured["answers"]
                        )
                        result["vs_original_checkpoint_independent"] = compare_answers(
                            references[identifier]["answers"], measured["answers"]
                        )
                    else:
                        drift = compare_answers(result["answers"], measured["answers"])
                        result["repeat_max_probability_difference"] = max(
                            result["repeat_max_probability_difference"], drift["max_probability_difference"]
                        )
                    result["samples"].append(
                        {
                            "repeat": repeat,
                            **{
                                key: value
                                for key, value in measured.items()
                                if key not in ("answers", "logits", "usage", "questions")
                            },
                        }
                    )
            for result in case_result["modes"].values():
                result["timing"] = {
                    metric: timing_summary([row[metric] for row in result["samples"]])
                    for metric in ("end_to_end_ms", "prepare_ms", "scoring_wall_ms", "output_ms", "cuda_stream_ms")
                }
            case_result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            case_result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            case_result["gpu_status"] = gpu_status()
            write(args.output / "report.json", report)
            print(
                representation,
                identifier,
                {
                    mode: round(value["timing"]["end_to_end_ms"]["p50_ms"], 2)
                    for mode, value in case_result["modes"].items()
                },
                flush=True,
            )
        if not args.skip_quality and representation == "adapters":
            print("Quality check adapters-cached-4", flush=True)
            result = quality_probe(engine, prepared, saved, cached=True, batch_size=4)
            write(args.output / "adapters-cached-4-quality.json", result)
            report["quality"]["adapters-cached-4"] = {k: v for k, v in result.items() if k != "predictions"}
            write(args.output / "report.json", report)
    if not args.skip_quality:
        for name, cached in (("merged-independent", False), ("merged-cached-all", True)):
            print("Quality check", name, flush=True)
            result = quality_probe(engine, prepared, saved, cached=cached, batch_size=255 if cached else 4)
            write(args.output / (name + "-quality.json"), result)
            report["quality"][name] = {k: v for k, v in result.items() if k != "predictions"}
            write(args.output / "report.json", report)
    report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(), gpu_end=gpu_status())
    write(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/judgment-full-v0.2/final"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("cached-4", "independent-4", "cached-32", "cached-all"),
        default=["cached-4", "independent-4", "cached-32", "cached-all"],
    )
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or args.warmups < 1:
        parser.error("choose unused output and positive counts")
    args.output.mkdir(parents=True)
    sources = [Path(__file__), *Path("src/openjev").glob("judgment*.py")]
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "repeats": args.repeats,
        "warmups_per_case_mode": args.warmups,
        "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "fla-core")},
        "source_sha256": {str(p): sha(p) for p in sources},
        "gpu_start": gpu_status(),
        "precision": "BF16 base, FP32 LoRA/readouts, TF32 disabled; no quantization",
        "timing_scope": "JSON parsing, compile/render/tokenize, synchronized scoring, answer assembly, JSON serialization; excludes model loading, file/network I/O; no previous request KV reuse",
        "cuda_stream_timing_note": "CUDA elapsed events include host-launch idle gaps; not pure GPU kernel time",
        "quality_scope": "Saved test set inspected previously; execution regression checks, not new generalization estimate",
        "results": {},
    }
    write(args.output / "report.json", report)
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        write(args.output / "report.json", report)
        raise


if __name__ == "__main__":
    main()
