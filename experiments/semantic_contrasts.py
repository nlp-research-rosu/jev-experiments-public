"""Freeze and evaluate semantic contrast diagnostics without changing model weights."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from openjev.judgment_cli import load_judgment_engine, read_json
from openjev.judgments import compile_request


def target_label(value):
    return ("true" if value else "false") if type(value) is bool else value


def answer_space(question):
    return {"false", "true"} if question["type"] == "noul" else set(question["criteria"])


def validate_suite(suite):
    cases = suite["cases"]
    by_id = {c["id"]: c for c in cases}
    if not cases or len(by_id) != len(cases):
        raise ValueError("nonempty unique case IDs required")
    for case in cases:
        request = case["request"]
        if set(request) != {"state", "questions"}:
            raise ValueError("only state and questions may enter model")
        compile_request(request)
        if set(request["questions"]) != set(case["expected"]) or set(case["rationale"]) != set(case["expected"]):
            raise ValueError("incomplete targets or rationales")
        for qid, q in request["questions"].items():
            expected = case["expected"][qid]
            if q["type"] == "noul":
                if type(expected) is not bool:
                    raise ValueError("binary targets must be hard booleans")
            elif q["type"] != "choice" or not isinstance(expected, str) or expected not in q["criteria"]:
                raise ValueError("only valid binary/categorical targets supported")
    seen = set()
    for relation in suite["relations"]:
        if relation["id"] in seen or relation["kind"] not in ("flip", "question_contrast", "invariant"):
            raise ValueError("duplicate or unsupported relation")
        seen.add(relation["id"])
        try:
            left, right = relation["left"], relation["right"]
            a, b = by_id[left["case_id"]], by_id[right["case_id"]]
            qa, qb = a["request"]["questions"][left["question_id"]], b["request"]["questions"][right["question_id"]]
            ya, yb = a["expected"][left["question_id"]], b["expected"][right["question_id"]]
        except KeyError as exc:
            raise ValueError("missing relation endpoint") from exc
        if a["family_id"] != b["family_id"] or qa["type"] != qb["type"] or answer_space(qa) != answer_space(qb):
            raise ValueError("relation family or answer spaces differ")
        if (ya == yb) != (relation["kind"] == "invariant"):
            raise ValueError("relation conflicts with declared targets")
        if relation["kind"] == "question_contrast" and a["id"] != b["id"]:
            raise ValueError("question contrast must hold state fixed")


def prediction_view(question, expected, answer):
    label = target_label(expected)
    if question["type"] == "noul":
        p = {"false": 1 - answer["noul"], "true": answer["noul"]}
        z = answer["details"].get("calibrated_logit", answer["details"]["raw_logit"])
        nll = max(z, 0) - z * float(expected) + math.log1p(math.exp(-abs(z)))
    else:
        p = answer["probabilities"]
        z = answer["details"].get("calibrated_logits", answer["details"]["raw_logits"])
        peak = max(z.values())
        nll = peak + math.log(sum(math.exp(v - peak) for v in z.values())) - z[label]
    if set(p) != answer_space(question) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
        raise ValueError("invalid model probabilities")
    if not math.isclose(sum(p.values()), 1, abs_tol=1e-6) or not math.isfinite(nll):
        raise ValueError("invalid probability sum or loss")
    maximum = max(p.values())
    top = [key for key, value in p.items() if abs(value - maximum) <= 1e-12]
    predicted = top[0] if len(top) == 1 else None
    return {
        "expected": label,
        "predicted": predicted,
        "probabilities": p,
        "correct": predicted == label,
        "ambiguous_top": len(top) != 1,
        "nll": nll,
        "brier": sum((value - float(key == label)) ** 2 for key, value in p.items()),
        "max_probability": maximum,
    }


def relation_metrics(relation, lookup, *, tolerance=0.05):
    def endpoint(e):
        return lookup[e["case_id"], e["question_id"]]

    a, b = endpoint(relation["left"]), endpoint(relation["right"])
    pa, pb = a["probabilities"], b["probabilities"]
    if pa.keys() != pb.keys():
        raise ValueError("relation distributions do not align")
    result = {
        "both_correct": a["predicted"] == a["expected"] and b["predicted"] == b["expected"],
        "total_variation": 0.5 * sum(abs(pa[key] - pb[key]) for key in pa),
        "same_prediction": a["predicted"] == b["predicted"],
    }
    if relation["kind"] == "invariant":
        result["within_tolerance"] = result["total_variation"] <= tolerance
    else:
        ma = pa[a["expected"]] - pb[a["expected"]]
        mb = pb[b["expected"]] - pa[b["expected"]]
        result.update(direction_correct=ma > 0 and mb > 0, directional_margin=(ma + mb) / 2)
    return result


def summarize(rows, relations):
    def group(items):
        correct = sum(r["prediction"]["correct"] for r in items)
        classes = Counter(r["prediction"]["expected"] for r in items)
        result = {
            "questions": len(items),
            "correct": correct,
            "accuracy": correct / len(items),
            "expected_class_counts": dict(classes),
            "nll": statistics.mean(r["prediction"]["nll"] for r in items),
            "brier": statistics.mean(r["prediction"]["brier"] for r in items),
            "high_probability_errors": sum(
                not r["prediction"]["correct"] and r["prediction"]["max_probability"] >= 0.95 for r in items
            ),
        }
        if set(classes) == {"false", "true"}:
            result["balanced_binary_accuracy"] = statistics.mean(
                sum(r["prediction"]["correct"] for r in items if r["prediction"]["expected"] == label) / classes[label]
                for label in ("false", "true")
            )
        return result

    result = {"overall": group(rows), "by_question": {}, "by_domain": {}, "by_family": {}, "relations": {}}
    for key, output in (("question_id", "by_question"), ("domain", "by_domain"), ("family_id", "by_family")):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[key]].append(row)
        result[output] = {k: group(v) for k, v in grouped.items()}
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    result["complete_cases"] = {
        "cases": len(by_case),
        "all_fields_correct": sum(all(r["prediction"]["correct"] for r in rs) for rs in by_case.values()),
    }
    for kind in sorted({r["kind"] for r in relations}):
        chosen = [r["metrics"] for r in relations if r["kind"] == kind]
        stats = {
            "pairs": len(chosen),
            "both_correct": sum(r["both_correct"] for r in chosen),
            "both_correct_rate": statistics.mean(r["both_correct"] for r in chosen),
            "mean_total_variation": statistics.mean(r["total_variation"] for r in chosen),
        }
        if kind == "invariant":
            stats.update(
                within_tolerance=sum(r["within_tolerance"] for r in chosen),
                max_total_variation=max(r["total_variation"] for r in chosen),
            )
        else:
            stats.update(
                direction_correct=sum(r["direction_correct"] for r in chosen),
                mean_directional_margin=statistics.mean(r["directional_margin"] for r in chosen),
            )
        result["relations"][kind] = stats
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze(path):
    from experiments.contrast_action_cases import build_action_suite
    from experiments.contrast_bank_cases import build_bank_suite

    if path.exists():
        raise FileExistsError("choose a new diagnostic version; do not overwrite frozen cases")
    components = {"execution": build_action_suite(), "destination": build_bank_suite()}
    suite = {
        "contract": {name: data["contract"] for name, data in components.items()},
        "cases": [case for data in components.values() for case in data["cases"]],
        "relations": [relation for data in components.values() for relation in data["relations"]],
    }
    validate_suite(suite)
    path.mkdir(parents=True)
    write(path / "suite.json", suite)
    manifest = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "role": "diagnostic evaluation only; not training data",
        "suite_sha256": sha(path / "suite.json"),
        "cases": len(suite["cases"]),
        "questions": sum(len(c["expected"]) for c in suite["cases"]),
        "families": sorted({c["family_id"] for c in suite["cases"]}),
        "relations": dict(Counter(r["kind"] for r in suite["relations"])),
        "invariance_tv_tolerance": 0.05,
        "source_sha256": {
            str(p): sha(p)
            for p in (
                Path(__file__),
                Path("experiments/contrast_action_cases.py"),
                Path("experiments/contrast_bank_cases.py"),
            )
        },
        "label_policy": "Explicit evidence-relative synthetic contracts, hand-assigned labels reviewed before model evaluation. No soft probability gold. Claim and fact are distinct, not complements.",
        "reference_methodology": [
            "https://aclanthology.org/2020.acl-main.442/",
            "https://aclanthology.org/2020.findings-emnlp.117/",
        ],
        "limits": "Authored templates, six families, correlated views; not population accuracy or untouched generalization after inspection. Some checks are exactly computable in software.",
    }
    write(path / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def run(args, report):
    manifest = read_json(args.suite / "manifest.json")
    if sha(args.suite / "suite.json") != manifest["suite_sha256"]:
        raise ValueError("frozen suite hash mismatch")
    suite = read_json(args.suite / "suite.json")
    validate_suite(suite)
    report["manifest"] = manifest
    report["results"] = {}
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_rows, output_relations = {}, {}
    for variant in ("untrained", "trained"):
        engine = load_judgment_engine(
            checkpoint=args.checkpoint if variant == "trained" else None,
            backend="fla",
            unit_batch_size=4,
            max_input_tokens=4096,
        )
        report["results"][variant] = {"model_id": engine.model_id}
        engine.evaluate(suite["cases"][0]["request"], cached=False, details=True)
        torch.cuda.reset_peak_memory_stats()
        rows, lookup = [], {}
        for index, case in enumerate(suite["cases"]):
            answer = engine.evaluate(case["request"], cached=False, details=True)
            for qid, q in case["request"]["questions"].items():
                pred = prediction_view(q, case["expected"][qid], answer["answers"][qid])
                row = {
                    "case_id": case["id"],
                    "family_id": case["family_id"],
                    "domain": case["domain"],
                    "variant": case["variant"],
                    "question_id": qid,
                    "prediction": pred,
                }
                rows.append(row)
                lookup[case["id"], qid] = pred
            with (args.output / (variant + "-responses.jsonl")).open("a") as stream:
                stream.write(json.dumps({"case_id": case["id"], "response": answer}, allow_nan=False) + "\n")
            if (index + 1) % 10 == 0:
                print(f"{variant} {index + 1}/{len(suite['cases'])}: {case['id']}", flush=True)
                report["progress"] = {"model": variant, "cases": index + 1}
                write(args.output / "report.json", report)
        relations = [
            {**r, "metrics": relation_metrics(r, lookup, tolerance=manifest["invariance_tv_tolerance"])}
            for r in suite["relations"]
        ]
        output_rows[variant], output_relations[variant] = rows, relations
        write(args.output / (variant + "-predictions.json"), rows)
        write(args.output / (variant + "-relations.json"), relations)
        report["results"][variant].update(
            summary=summarize(rows, relations), peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30
        )
        write(args.output / "report.json", report)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    before = {(r["case_id"], r["question_id"]): r for r in output_rows["untrained"]}
    improved, regressed, retained = [], [], 0
    for row in output_rows["trained"]:
        old = before[row["case_id"], row["question_id"]]
        a, b = old["prediction"]["correct"], row["prediction"]["correct"]
        record = {"before": old, "after": row}
        if b and not a:
            improved.append(record)
        if a and not b:
            regressed.append(record)
        retained += a and b
    before_pairs = {r["id"]: r for r in output_relations["untrained"]}
    pair_changes = Counter()
    for row in output_relations["trained"]:
        a, b = before_pairs[row["id"]]["metrics"]["both_correct"], row["metrics"]["both_correct"]
        pair_changes["retained" if a and b else "regressed" if a else "improved" if b else "both_failed"] += 1
    report["changes"] = {
        "improved_questions": len(improved),
        "regressed_questions": len(regressed),
        "retained_correct_questions": retained,
        "contrast_pairs": dict(pair_changes),
    }
    write(args.output / "changed-questions.json", {"improved": improved, "regressed": regressed})
    report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    parser.add_argument("--suite", type=Path, default=Path("data/semantic-contrasts-v1"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/judgment-full-v0.2/final"))
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.suite)
        return
    if args.output is None or args.output.exists():
        parser.error("choose a new output directory")
    args.output.mkdir(parents=True)
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(p): sha(p) for p in [Path(__file__), *Path("src/openjev").glob("judgment*.py")]},
        "execution": "Unmerged BF16 + FP32 readouts/LoRA; independent full-input scoring in groups of four; FLA; no truncation, calibration or training.",
        "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "fla-core")},
    }
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        write(args.output / "report.json", report)
        raise


if __name__ == "__main__":
    main()
