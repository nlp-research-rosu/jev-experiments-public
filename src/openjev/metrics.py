"""Accuracy and probability diagnostics; normalization is not calibration."""

import math
import statistics

from .schema import parse_schema, same_value, valid_answer


def summarize(cases: list[dict], results: list[dict]) -> dict:
    if not cases or len(cases) != len(results):
        raise ValueError("need one result per nonempty case list")
    total_fields = correct_fields = exact_cases = labeled_cases = valid = 0
    briers, losses, confidences, latencies, errors = [], [], [], [], []
    for case, result in zip(cases, results, strict=True):
        if case["id"] != result["id"]:
            raise ValueError("result IDs must match cases in order")
        fields = parse_schema(case["schema"])
        answer = result.get("values", {})
        valid += valid_answer(answer, fields)
        elapsed = result["elapsed_ms"]
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("latencies must be finite and nonnegative")
        latencies.append(elapsed)
        if "expected" not in case:
            continue
        gold = case["expected"]
        if not valid_answer(gold, fields):
            raise ValueError(f"{case['id']}: expected must cover the entire schema with valid values")
        labeled_cases += 1
        exact_cases += valid_answer(answer, fields) and all(same_value(answer[f.name], gold[f.name]) for f in fields)
        for field in fields:
            total_fields += 1
            prediction = answer.get(field.name) if isinstance(answer, dict) else None
            correct = same_value(prediction, gold[field.name])
            correct_fields += correct
            if not correct:
                errors.append(
                    {"id": case["id"], "field": field.name, "expected": gold[field.name], "actual": prediction}
                )
            probs = result.get("probabilities", {}).get(field.name)
            if probs is None:
                continue
            if (
                len(probs) != len(field.choices)
                or any(not math.isfinite(p) or p < 0 or p > 1 for p in probs)
                or not math.isclose(sum(probs), 1.0, abs_tol=1e-5)
            ):
                raise ValueError("probabilities must be a finite normalized vector matching choices")
            idx = next(i for i, c in enumerate(field.choices) if same_value(c, gold[field.name]))
            briers.append(sum((p - (i == idx)) ** 2 for i, p in enumerate(probs)))
            losses.append(-math.log(max(probs[idx], 1e-12)))
            confidences.append((max(probs), int(correct)))
    bins = []
    for i in range(10):
        members = [(p, c) for p, c in confidences if min(int(p * 10), 9) == i]
        bins.append(
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "count": len(members),
                "mean_confidence": statistics.mean(p for p, _ in members) if members else None,
                "accuracy": statistics.mean(c for _, c in members) if members else None,
            }
        )
    ece = (
        (sum(b["count"] * abs(b["mean_confidence"] - b["accuracy"]) for b in bins if b["count"]) / len(confidences))
        if confidences
        else None
    )
    return {
        "evaluations": len(cases),
        "labeled_evaluations": labeled_cases,
        "labeled_fields": total_fields,
        "schema_validity": valid / len(cases),
        "field_accuracy": correct_fields / total_fields if total_fields else None,
        "exact_case_accuracy": exact_cases / labeled_cases if labeled_cases else None,
        "brier": statistics.mean(briers) if briers else None,
        "log_loss": statistics.mean(losses) if losses else None,
        "probability_coverage": len(briers) / total_fields if total_fields else None,
        "ece_10_bins": ece,
        "calibration_bins": bins,
        "median_ms": statistics.median(latencies),
        "mean_ms": statistics.mean(latencies),
        "p95_ms": sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)],
        "errors": errors,
    }
