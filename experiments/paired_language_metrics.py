"""Category/predicate matrices and paired family uncertainty for language experiments."""

import math
import random
import statistics
from collections import Counter, defaultdict

from experiments.revised_metrics import relation_metrics, target_label


def _validate(suite, rows):
    cases = {c["id"]: c for c in suite["cases"]}
    expected = {(c["id"], q) for c in suite["cases"] for q in c["expected"]}
    actual = [(r["case_id"], r["question_id"]) for r in rows]
    if len(cases) != len(suite["cases"]) or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Prediction coverage differs from the complete suite")
    for r in rows:
        if r["prediction"]["expected"] != target_label(cases[r["case_id"]]["expected"][r["question_id"]]):
            raise ValueError("Prediction target differs from frozen label")
    return cases


def _group(rows):
    if not rows:
        return {"questions": 0, "correct": 0, "accuracy": None}
    p = [r["prediction"] for r in rows]
    result = {
        "questions": len(p),
        "correct": sum(x["correct"] for x in p),
        "accuracy": statistics.mean(x["correct"] for x in p),
        "nll": statistics.mean(x["nll"] for x in p),
        "probability_nll_floor_1e12": statistics.mean(
            -math.log(max(x["probabilities"][x["expected"]], 1e-12)) for x in p
        ),
        "zero_target_probabilities": sum(x["probabilities"][x["expected"]] == 0 for x in p),
        "brier": statistics.mean(x["brier"] for x in p),
        "confident_wrong_90": sum(not x["correct"] and x["max_probability"] >= 0.9 for x in p),
        "confident_wrong_95": sum(not x["correct"] and x["max_probability"] >= 0.95 for x in p),
        "class_counts": dict(Counter(x["expected"] for x in p)),
    }
    confusion = defaultdict(Counter)
    for x in p:
        confusion[x["expected"]][x["predicted"] if x["predicted"] is not None else "__tie__"] += 1
    result["confusion"] = {k: dict(v) for k, v in confusion.items()}
    if all(x["primitive"] == "noul" for x in p):
        pos = [x for x in p if x["expected"] == "true"]
        neg = [x for x in p if x["expected"] == "false"]
        tp = sum(x["predicted"] == "true" for x in pos)
        tn = sum(x["predicted"] == "false" for x in neg)
        recall, specificity = tp / len(pos) if pos else None, tn / len(neg) if neg else None
        result.update(
            positive_count=len(pos),
            negative_count=len(neg),
            true_positives=tp,
            true_negatives=tn,
            sensitivity=recall,
            specificity=specificity,
            balanced_accuracy=(recall + specificity) / 2 if pos and neg else None,
        )
    scores = [x for x in p if x["primitive"] == "score"]
    if scores:
        result.update(
            score_count=len(scores),
            score_mean_absolute_error=statistics.mean(x["score_absolute_error"] for x in scores),
            score_normalized_mean_absolute_error=statistics.mean(x["score_normalized_absolute_error"] for x in scores),
        )
    bins = []
    for i in range(10):
        part = [x for x in p if min(9, int(x["max_probability"] * 10)) == i]
        if part:
            bins.append(
                {
                    "lower": i / 10,
                    "upper": (i + 1) / 10,
                    "count": len(part),
                    "mean_selected_probability": statistics.mean(x["max_probability"] for x in part),
                    "accuracy": statistics.mean(x["correct"] for x in part),
                }
            )
    result["reliability_bins"] = bins
    result["ece_10_bin"] = sum(x["count"] * abs(x["accuracy"] - x["mean_selected_probability"]) for x in bins) / len(p)
    return result


def _unknown(rows):
    choices = [r["prediction"] for r in rows if r["prediction"]["primitive"] == "choice"]
    expected = [p for p in choices if p["expected"].lower() == "unknown"]
    predicted = [p for p in choices if str(p["predicted"]).lower() == "unknown"]
    tp = sum(p["correct"] for p in expected)
    return {
        "count": len(expected),
        "predicted": len(predicted),
        "true_positive": tp,
        "recall": tp / len(expected) if expected else None,
        "precision": tp / len(predicted) if predicted else None,
    }


def _relations(relations, lookup):
    groups = defaultdict(list)
    for r in relations:
        groups[r["kind"]].append(relation_metrics(r, lookup))
    result = {}
    for k, xs in groups.items():
        result[k] = {
            "pairs": len(xs),
            "both_correct": sum(x["both_correct"] for x in xs),
            "both_correct_rate": statistics.mean(x["both_correct"] for x in xs),
            "mean_probability_distance": statistics.mean(x["total_variation"] for x in xs),
        }
        if k in ("invariant", "layout_invariant"):
            result[k]["within_005"] = sum(x["within_tolerance"] for x in xs)
    return result


def matrix_summary(suite, rows):
    cases = _validate(suite, rows)
    lookup = {(r["case_id"], r["question_id"]): r["prediction"] for r in rows}

    def block(part):
        ids = {r["case_id"] for r in part}
        rels = [r for r in suite["relations"] if r["left"]["case_id"] in ids and r["right"]["case_id"] in ids]
        return {
            "overall": _group(part),
            "unknown": _unknown(part),
            "relations": _relations(rels, lookup),
            "by_primitive": {
                p: _group([r for r in part if r["prediction"]["primitive"] == p]) for p in ("noul", "choice", "score")
            },
        }

    result = block(rows)
    for name, field in [
        ("by_category", "category"),
        ("by_general_category", "domain"),
        ("by_family", "family_id"),
        ("by_layout", "layout"),
    ]:
        groups = defaultdict(list)
        for r in rows:
            groups[cases[r["case_id"]].get(field, "unspecified")].append(r)
        result[name] = {key: block(part) for key, part in groups.items()}
    predicates = defaultdict(list)
    for r in rows:
        case = cases[r["case_id"]]
        tags = case.get("predicate_tags", {})
        if isinstance(tags, dict):
            tag = tags.get(r["question_id"], r["question_id"])
        elif isinstance(tags, list) and all(isinstance(value, str) for value in tags):
            # Case-level facets do not identify the predicate of an individual
            # question. Keep its category and slot explicit rather than guessing.
            tag = f"{case.get('category', 'unspecified')}/{r['question_id']}"
        else:
            raise ValueError("predicate_tags must be a question mapping or a list of case tags")
        predicates[tag].append(r)
    result["by_predicate"] = {key: _group(part) for key, part in predicates.items()}
    per_case = defaultdict(list)
    for row in rows:
        per_case[row["case_id"]].append(row["prediction"]["correct"])
    result["whole_cases"] = {"cases": len(per_case), "all_correct": sum(all(v) for v in per_case.values())}
    result["macro_category_accuracy"] = statistics.mean(
        x["overall"]["accuracy"] for x in result["by_category"].values()
    )
    result["metric_policy"] = {
        "confident_wrong": "Selected answer probability, not Jev confidence field.",
        "brier": "Sum over classes, including both binary classes.",
        "unknown": "Choice label named unknown; missing support is not automatically a false world state.",
        "ece": "10 equal-width selected-probability bins; exploratory, not a proof of calibration.",
        "predicate_grouping": "Explicit question tags when mapped; case-tag lists use category/question ID without inferring predicate labels.",
    }
    result["metric_policy"]["nll"] = (
        "nll retains source evaluator loss (stable raw logits locally). probability_nll_floor_1e12 uses returned probabilities consistently for cross-system comparisons; API rounding can lose information."
    )
    return result


def paired_family_comparison(suite, control, natural, *, draws=2000, seed=42):
    cases = _validate(suite, control)
    _validate(suite, natural)
    if draws < 1:
        raise ValueError("Need positive bootstrap draws")
    families = defaultdict(list)
    for c in cases.values():
        if c["family_id"] not in families[c["category"]]:
            families[c["category"]].append(c["family_id"])
    summaries = []
    for rows in (control, natural):
        acc = defaultdict(
            lambda: {"questions": 0, "correct": 0, "nll": 0.0, "flip": [0, 0], "question_contrast": [0, 0]}
        )
        lookup = {(r["case_id"], r["question_id"]): r["prediction"] for r in rows}
        for r in rows:
            a = acc[cases[r["case_id"]]["family_id"]]
            a["questions"] += 1
            a["correct"] += r["prediction"]["correct"]
            a["nll"] += r["prediction"]["nll"]
        for rel in suite["relations"]:
            if rel["kind"] in ("flip", "question_contrast"):
                a = acc[cases[rel["left"]["case_id"]]["family_id"]][rel["kind"]]
                a[1] += 1
                a[0] += all(
                    lookup[end["case_id"], end["question_id"]]["correct"] for end in (rel["left"], rel["right"])
                )
        summaries.append(acc)

    def calculate(ids, metric):
        values = []
        for summary in summaries:
            if metric in ("accuracy", "nll"):
                num = sum(summary[f]["correct" if metric == "accuracy" else "nll"] for f in ids)
                den = sum(summary[f]["questions"] for f in ids)
            else:
                num = sum(summary[f][metric][0] for f in ids)
                den = sum(summary[f][metric][1] for f in ids)
            values.append(num / den if den else None)
        return values[1] - values[0] if None not in values else None

    rng = random.Random(seed)
    all_ids = [f for fs in families.values() for f in fs]
    metrics = ("accuracy", "nll", "flip", "question_contrast")
    samples = {m: [] for m in metrics}
    for _ in range(draws):
        ids = [rng.choice(fs) for fs in families.values() for _ in fs]
        for m in metrics:
            value = calculate(ids, m)
            if value is not None:
                samples[m].append(value)
    result = {
        "families": len(all_ids),
        "resampling_unit": "family_within_category",
        "draws": draws,
        "seed": seed,
        "direction": "natural minus template; positive accuracy/pair and negative NLL differences favor natural.",
    }
    for m in metrics:
        xs = sorted(samples[m])
        result[m] = {
            "difference": calculate(all_ids, m),
            "interval_95": [xs[math.floor(0.025 * (len(xs) - 1))], xs[math.ceil(0.975 * (len(xs) - 1))]]
            if xs
            else None,
        }
    return result
