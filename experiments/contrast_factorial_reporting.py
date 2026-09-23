"""Paired, category-stratified effects for the approved four-arm pilot."""

import math
import random
import statistics
from collections import defaultdict

from experiments.revised_metrics import INVARIANTS, relation_metrics, summarize, target_label, validate_suite


def _effects(means):
    a, b, c, d = (means[key] for key in "ABCD")
    return {
        "rubric_at_low": b - a,
        "language_at_narrow": c - a,
        "rubric_at_high": d - c,
        "language_at_broad": d - b,
        "interaction": (d - c) - (b - a),
        "rubric_average": ((b - a) + (d - c)) / 2,
        "language_average": ((c - a) + (d - b)) / 2,
    }


def _percentile(values, quantile):
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def estimate_factorial_effects(arm_values, categories, *, bootstrap_samples=2000, seed=42):
    """Estimate effects from one scalar per assessment family per arm.

    For primary pair accuracy that scalar is the family's fraction of valid
    contrast pairs with both endpoints correct. Average families within category
    and categories equally. Other scalar metrics may use the same estimator,
    but ratios such as pooled Unknown precision require count-based statistics.
    The same sampled families are used for all four arms in every bootstrap.
    """
    if set(arm_values) != set("ABCD") or not categories:
        raise ValueError("four arms and nonempty family categories are required")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1 or type(seed) is not int:
        raise ValueError("invalid bootstrap settings")
    ids = set(categories)
    for rows in arm_values.values():
        if set(rows) != ids or any(type(v) not in (float, int) or not math.isfinite(v) for v in rows.values()):
            raise ValueError("all arms need the same finite family population")
    strata = defaultdict(list)
    for family in sorted(ids):
        category = categories[family]
        if not isinstance(category, str) or not category:
            raise ValueError("categories must be nonempty strings")
        strata[category].append(family)

    def means_for(groups):
        return {
            arm: statistics.mean(statistics.mean(arm_values[arm][family] for family in group) for group in groups)
            for arm in "ABCD"
        }

    groups = [strata[category] for category in sorted(strata)]
    means = means_for(groups)
    point = _effects(means)
    draws = {name: [] for name in point}
    rng = random.Random(seed)
    for _ in range(bootstrap_samples):
        sampled = [[rng.choice(group) for _ in group] for group in groups]
        for name, value in _effects(means_for(sampled)).items():
            draws[name].append(value)
    return {
        "means": means,
        "effects": {
            name: {"estimate": value, "ci95": [_percentile(draws[name], 0.025), _percentile(draws[name], 0.975)]}
            for name, value in point.items()
        },
        "families": len(ids),
        "categories": len(strata),
        "family_counts": {category: len(families) for category, families in strata.items()},
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "aggregation": "Equal family weight within category; equal category weight. Paired family resampling.",
        "limitation": "Corpus uncertainty for one training seed; not a training-seed variance estimate.",
    }


def _extra_metrics(rows):
    result = summarize(rows, [])["overall"]
    result["confidence"] = {}
    for threshold in (0.90, 0.95):
        covered = [r for r in rows if r["prediction"]["max_probability"] >= threshold]
        wrong = sum(not r["prediction"]["correct"] for r in covered)
        result["confidence"][f"{threshold:.2f}"] = {
            "covered": len(covered),
            "coverage": len(covered) / len(rows),
            "wrong": wrong,
            "error_rate_among_covered": wrong / len(covered) if covered else None,
        }
    choice = [r["prediction"] for r in rows if r["prediction"]["primitive"] == "choice"]
    gold_unknown = sum(p["expected"].lower() == "unknown" for p in choice)
    predicted_unknown = sum((p["predicted"] or "").lower() == "unknown" for p in choice)
    correct_unknown = sum(p["expected"].lower() == "unknown" and p["correct"] for p in choice)
    result["unknown"] = {
        "choice_questions": len(choice),
        "gold": gold_unknown,
        "predicted": predicted_unknown,
        "correct": correct_unknown,
        "recall": correct_unknown / gold_unknown if gold_unknown else None,
        "precision": correct_unknown / predicted_unknown if predicted_unknown else None,
    }
    return result


def summarize_factorial(suite, rows):
    """Join by exact identities; report the predeclared family-macro primary metric."""
    validate_suite(suite)
    cases = {case["id"]: case for case in suite["cases"]}
    expected_keys = {(case["id"], qid) for case in suite["cases"] for qid in case["expected"]}
    keys = [(row["case_id"], row["question_id"]) for row in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected_keys:
        raise ValueError("prediction population differs from the complete suite")
    family_categories = {}
    lookup = {}
    grouped = {name: defaultdict(list) for name in ("by_category", "by_primitive", "by_score_k")}
    for row in rows:
        case, qid = cases[row["case_id"]], row["question_id"]
        category = case.get("category", case["domain"])
        family = case["family_id"]
        if family in family_categories and family_categories[family] != category:
            raise ValueError("family spans categories")
        family_categories[family] = category
        question, pred = case["request"]["questions"][qid], row["prediction"]
        if (
            row["family_id"] != family
            or pred["expected"] != target_label(case["expected"][qid])
            or pred["primitive"] != question["type"]
        ):
            raise ValueError("prediction metadata or target is misbound")
        if pred["correct"] != (pred["predicted"] == pred["expected"]):
            raise ValueError("prediction correctness is inconsistent")
        lookup[row["case_id"], qid] = pred
        grouped["by_category"][category].append(row)
        grouped["by_primitive"][question["type"]].append(row)
        if question["type"] == "score":
            grouped["by_score_k"][str(len(question["criteria"]))].append(row)
    relations = [{**r, "metrics": relation_metrics(r, lookup)} for r in suite["relations"]]
    result = summarize(rows, relations)
    result["overall"] = _extra_metrics(rows)
    for name, groups in grouped.items():
        result[name] = {key: _extra_metrics(part) for key, part in sorted(groups.items())}
    contrast_families, axes = defaultdict(list), defaultdict(list)
    invariants = []
    for relation in relations:
        metric = relation["metrics"]
        if relation["kind"] in INVARIANTS:
            invariants.append(metric)
            continue
        axis = relation.get("axis", "question" if relation["kind"] == "question_contrast" else None)
        if axis not in {"evidence", "rubric", "question"}:
            raise ValueError("contrast relation has no declared axis")
        family = cases[relation["left"]["case_id"]]["family_id"]
        contrast_families[family].append(metric["both_correct"])
        axes[axis].append(metric["both_correct"])
    if relations and set(contrast_families) != set(family_categories):
        raise ValueError("every assessment family needs a meaningful contrast")
    family_values = {family: statistics.mean(values) for family, values in contrast_families.items()}
    category_values = defaultdict(list)
    for family, value in family_values.items():
        category_values[family_categories[family]].append(value)
    category_means = {key: statistics.mean(values) for key, values in category_values.items()}
    contrasts = [value for values in contrast_families.values() for value in values]
    result["primary"] = {
        "category_macro": statistics.mean(category_means.values()) if category_means else None,
        "family_values": family_values,
        "family_categories": family_categories,
        "category_values": category_means,
        "pairs": len(contrasts),
        "both_correct": sum(contrasts),
        "pooled_pair_accuracy": statistics.mean(contrasts) if contrasts else None,
    }
    result["contrast_axes"] = {
        axis: {"pairs": len(values), "both_correct": sum(values), "both_correct_rate": statistics.mean(values)}
        for axis, values in sorted(axes.items())
    }
    result["invariants"] = {
        "pairs": len(invariants),
        "both_correct": sum(v["both_correct"] for v in invariants),
        "both_correct_rate": statistics.mean(v["both_correct"] for v in invariants) if invariants else None,
        "mean_total_variation": statistics.mean(v["total_variation"] for v in invariants) if invariants else None,
    }
    return result
