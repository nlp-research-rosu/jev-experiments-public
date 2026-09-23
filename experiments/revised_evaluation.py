"""Streaming evaluation for the revised initialization experiment."""

import json

from experiments.revised_metrics import prediction_view, relation_metrics, summarize, validate_suite


def write(path, value):
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def evaluate_cases(engine, suite, output):
    validate_suite(suite)
    output.mkdir(parents=True, exist_ok=False)
    rows, lookup = [], {}
    max_tokens = 0
    for index, case in enumerate(suite["cases"], 1):
        response = engine.evaluate(case["request"], cached=False, details=True)
        with (output / "responses.jsonl").open("a") as stream:
            stream.write(json.dumps({"case_id": case["id"], "response": response}, allow_nan=False) + "\n")
        max_tokens = max(max_tokens, response["usage"]["max_unit_tokens"])
        for qid, question in case["request"]["questions"].items():
            pred = prediction_view(question, case["expected"][qid], response["answers"][qid])
            row = {"case_id": case["id"], "question_id": qid, "family_id": case["family_id"],
                   "domain": case["domain"], "variant": case["variant"], "layout": case.get("layout", "unspecified"),
                   "prediction": pred}
            rows.append(row)
            lookup[case["id"], qid] = pred
            with (output / "predictions.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        if index % 50 == 0:
            print(output.name, f"{index}/{len(suite['cases'])} cases", flush=True)
    relations = [{**r, "metrics": relation_metrics(r, lookup)} for r in suite["relations"]]
    result = {"summary": summarize(rows, relations), "max_unit_tokens": max_tokens}
    write(output / "predictions.json", rows)
    write(output / "relations.json", relations)
    write(output / "summary.json", result)
    return result


def broad_suite(records):
    cases = []
    for record in records:
        example = record["examples"][0]
        kind = example["question"]["type"]
        target = example["target"]
        y = target[{"noul": "truth", "choice": "choice", "score": "level_index"}[kind]]
        source = record["provenance"]["dataset"]
        cases.append({"id": record["id"], "family_id": record["group_id"], "domain": source,
                      "variant": "canonical", "layout": "original",
                      "request": {"state": example["state"], "questions": {source: example["question"]}},
                      "expected": {source: y}, "rationale": {source: "Preserved original canonical source label"}})
    return {"cases": cases, "relations": [], "contract": "Original labeled task collection, canonical first views only"}


def macro_nll(result):
    parts = result["summary"]["by_primitive"]
    if set(parts) != {"noul", "choice", "score"}:
        raise ValueError("Validation must cover every primitive")
    return sum(parts[k]["nll"] for k in sorted(parts)) / 3


def validation_objective(evaluation):
    return .5 * macro_nll(evaluation["revised"]) + .5 * macro_nll(evaluation["broad"])


def select_checkpoint(curve):
    if not curve or any(point.get("status") != "complete" for point in curve):
        raise ValueError("Only complete validation observations may select a checkpoint")
    return min(curve, key=lambda point: (point["objective"], point["step"]))


def readiness(evaluation, reference, rows):
    summary = evaluation["revised"]["summary"]
    broad, original = evaluation["broad"]["summary"], reference["broad"]["summary"]
    binary = [v["balanced_binary_accuracy"] for v in summary["by_question"].values() if "balanced_binary_accuracy" in v]
    checks = {"revised_accuracy_90pct": summary["overall"]["accuracy"] >= .90,
              "macro_binary_balanced_85pct": bool(binary) and sum(binary) / len(binary) >= .85,
              "layout_gap_at_most_5pp": max(v["accuracy"] for v in summary["by_layout"].values()) - min(v["accuracy"] for v in summary["by_layout"].values()) <= .05 + 1e-12,
              "broad_accuracy_within_2pp": broad["overall"]["accuracy"] >= original["overall"]["accuracy"] - .02 - 1e-12,
              "broad_nll_within_005": broad["overall"]["nll"] <= original["overall"]["nll"] + .05,
              "revised_nll_not_worse": summary["overall"]["nll"] <= reference["revised"]["summary"]["overall"]["nll"]}
    completion = summary["by_question_by_layout"].get("completed", {})
    for layout in ("flat", "nested", "prose"):
        part = completion.get(layout, {})
        checks[f"completion_recall_{layout}"] = part.get("positive_recall") is not None and part["positive_recall"] >= .90
        checks[f"completion_specificity_{layout}"] = part.get("negative_specificity") is not None and part["negative_specificity"] >= .90
    for kind, minimum in (("flip", .80), ("question_contrast", .80), ("invariant", .85), ("layout_invariant", .85)):
        checks[f"pair_{kind}"] = summary["relations"].get(kind, {}).get("both_correct_rate", -1) >= minimum
    unknown = {}
    for family, qid in (("action", "status"), ("registry", "record_status")):
        population = [r for r in rows if r["question_id"] == qid and r["prediction"]["expected"].lower() == "unknown"]
        recall = sum(r["prediction"]["correct"] for r in population) / len(population) if population else None
        unknown[family] = {"count": len(population), "recall": recall}
        checks[f"unknown_recall_{family}"] = recall is not None and recall >= .90
    for source, baseline in original["by_domain"].items():
        if "score_mean_absolute_error" not in baseline:
            continue
        actual = broad["by_domain"][source]
        checks[f"score_accuracy_{source}"] = actual["accuracy"] >= baseline["accuracy"] - .02 - 1e-12
        checks[f"score_mae_{source}"] = actual["score_mean_absolute_error"] <= baseline["score_mean_absolute_error"] + .05
    return {"passed": all(checks.values()), "checks": checks, "unknown": unknown,
            "macro_binary_balanced_accuracy": sum(binary) / len(binary) if binary else None}
