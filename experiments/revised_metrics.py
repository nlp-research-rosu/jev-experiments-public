"""Versioned Noul/Choice/Score behavioral metrics; prior frozen evaluators stay intact."""

import math
import statistics
from collections import Counter, defaultdict

from experiments.semantic_contrasts import relation_metrics as original_relation_metrics
from openjev.judgments import compile_request

INVARIANTS = {"invariant", "layout_invariant"}


def target_label(value):
    if type(value) is bool:
        return "true" if value else "false"
    return str(value)


def answer_space(question):
    kind = question["type"]
    if kind == "noul":
        return {"false", "true"}
    if kind == "score":
        return {str(i) for i in range(len(question["criteria"]))}
    return set(question["criteria"])


def validate_suite(suite):
    cases = suite["cases"]
    by_id = {c["id"]: c for c in cases}
    if not cases or len(by_id) != len(cases):
        raise ValueError("Need nonempty unique cases")
    for case in cases:
        request = case["request"]
        if set(request) != {"state", "questions"}:
            raise ValueError("Only state and questions may enter model")
        compile_request(request)
        if set(case["expected"]) != set(request["questions"]) or set(case["rationale"]) != set(case["expected"]):
            raise ValueError("Missing targets or rationales")
        for qid, q in request["questions"].items():
            y = case["expected"][qid]
            if q["type"] == "noul":
                valid = type(y) is bool
            elif q["type"] == "score":
                valid = type(y) is int and 0 <= y < len(q["criteria"])
            else:
                valid = isinstance(y, str) and y in q["criteria"]
            if not valid:
                raise ValueError("Invalid primitive target")
    seen = set()
    for relation in suite["relations"]:
        kind = relation["kind"]
        if relation["id"] in seen or kind not in INVARIANTS | {"flip", "question_contrast"}:
            raise ValueError("Invalid or duplicate relation")
        seen.add(relation["id"])
        try:
            left, right = relation["left"], relation["right"]
            a, b = by_id[left["case_id"]], by_id[right["case_id"]]
            qa, qb = a["request"]["questions"][left["question_id"]], b["request"]["questions"][right["question_id"]]
            ya, yb = a["expected"][left["question_id"]], b["expected"][right["question_id"]]
        except KeyError as error:
            raise ValueError("Missing relation endpoint") from error
        if a["family_id"] != b["family_id"] or qa["type"] != qb["type"] or answer_space(qa) != answer_space(qb):
            raise ValueError("Relation family or answer spaces differ")
        if (ya == yb) != (kind in INVARIANTS):
            raise ValueError("Relation contradicts targets")
        if kind == "question_contrast" and a["id"] != b["id"]:
            raise ValueError("Question contrast must preserve the record")


def prediction_view(question, expected, answer):
    kind, label = question["type"], target_label(expected)
    details = answer["details"]
    if kind == "noul":
        p = {"false": 1 - answer["noul"], "true": answer["noul"]}
        z = details["calibrated_logit"] if "calibrated_logit" in details else details["raw_logit"]
        nll = max(z, 0) - z * float(expected) + math.log1p(math.exp(-abs(z)))
    else:
        p = answer["probabilities"]
        z = details["calibrated_logits"] if "calibrated_logits" in details else details["raw_logits"]
        peak = max(z.values())
        nll = peak + math.log(sum(math.exp(v - peak) for v in z.values())) - z[label]
    if set(p) != answer_space(question) or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
        raise ValueError("Invalid model probability vector")
    if not math.isfinite(nll) or not math.isclose(sum(p.values()), 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Invalid model probability sum or loss")
    maximum = max(p.values())
    top = [key for key, value in p.items() if abs(value - maximum) <= 1e-12]
    predicted = top[0] if len(top) == 1 else None
    result = {
        "primitive": kind, "expected": label, "predicted": predicted, "probabilities": p,
        "correct": predicted == label, "ambiguous_top": len(top) != 1, "nll": nll,
        "brier": sum((value - float(key == label)) ** 2 for key, value in p.items()), "max_probability": maximum,
    }
    if kind == "score":
        mean = sum(int(key) * value for key, value in p.items())
        error = abs(mean - expected)
        result.update(score_mean=mean, score_target=expected, score_absolute_error=error,
                      score_normalized_absolute_error=error / max(1, len(p) - 1), returned_score=answer["score"])
    return result


def relation_metrics(relation, lookup, *, tolerance=.05):
    actual = {**relation, "kind": "invariant"} if relation["kind"] in INVARIANTS else relation
    return original_relation_metrics(actual, lookup, tolerance=tolerance)


def summarize(rows, relations):
    def primitive(row):
        p = row["prediction"]
        return p.get("primitive", "noul" if set(p["probabilities"]) == {"false", "true"} else "choice")

    def group(items):
        classes = Counter(r["prediction"]["expected"] for r in items)
        result = {
            "questions": len(items), "correct": sum(r["prediction"]["correct"] for r in items),
            "expected_class_counts": dict(classes),
            "nll": statistics.mean(r["prediction"]["nll"] for r in items),
            "brier": statistics.mean(r["prediction"]["brier"] for r in items),
            "high_probability_errors": sum(not r["prediction"]["correct"] and r["prediction"]["max_probability"] >= .95 for r in items),
        }
        result["accuracy"] = result["correct"] / len(items)
        if all(primitive(r) == "noul" for r in items):
            positives = [r["prediction"] for r in items if r["prediction"]["expected"] == "true"]
            negatives = [r["prediction"] for r in items if r["prediction"]["expected"] == "false"]
            tp = sum(p["predicted"] == "true" for p in positives)
            tn = sum(p["predicted"] == "false" for p in negatives)
            result.update(positive_count=len(positives), negative_count=len(negatives), true_positives=tp, true_negatives=tn,
                          false_positives=sum(p["predicted"] == "true" for p in negatives),
                          false_negatives=sum(p["predicted"] == "false" for p in positives),
                          positive_recall=tp / len(positives) if positives else None,
                          negative_specificity=tn / len(negatives) if negatives else None)
            if positives and negatives:
                result["balanced_binary_accuracy"] = (result["positive_recall"] + result["negative_specificity"]) / 2
        scores = [r["prediction"] for r in items if primitive(r) == "score"]
        if scores:
            result.update(score_questions=len(scores), score_mean_absolute_error=statistics.mean(p["score_absolute_error"] for p in scores),
                          score_mean_normalized_absolute_error=statistics.mean(p["score_normalized_absolute_error"] for p in scores))
        return result

    result = {"overall": group(rows), "relations": {}}
    selectors = {"by_question": lambda r: r["question_id"], "by_domain": lambda r: r["domain"],
                 "by_family": lambda r: r["family_id"], "by_layout": lambda r: r.get("layout", "unspecified"),
                 "by_primitive": primitive}
    for name, selector in selectors.items():
        groups = defaultdict(list)
        for row in rows:
            groups[selector(row)].append(row)
        result[name] = {key: group(value) for key, value in groups.items()}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["question_id"]][row.get("layout", "unspecified")].append(row)
    result["by_question_by_layout"] = {qid: {layout: group(value) for layout, value in layouts.items()} for qid, layouts in grouped.items()}
    cases = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)
    result["complete_cases"] = {"cases": len(cases), "all_fields_correct": sum(all(r["prediction"]["correct"] for r in rs) for rs in cases.values())}
    for kind in sorted({r["kind"] for r in relations}):
        chosen = [r["metrics"] for r in relations if r["kind"] == kind]
        stats = {"pairs": len(chosen), "both_correct": sum(r["both_correct"] for r in chosen),
                 "both_correct_rate": statistics.mean(r["both_correct"] for r in chosen),
                 "mean_total_variation": statistics.mean(r["total_variation"] for r in chosen)}
        if kind in INVARIANTS:
            stats.update(within_tolerance=sum(r["within_tolerance"] for r in chosen), max_total_variation=max(r["total_variation"] for r in chosen))
        else:
            stats["direction_correct"] = sum(r["direction_correct"] for r in chosen)
        result["relations"][kind] = stats
    return result
