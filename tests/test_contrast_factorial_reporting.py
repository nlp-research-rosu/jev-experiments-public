"""Independent arithmetic and pairing checks for the four-arm comparison."""

import pytest


def test_factorial_interaction_and_paired_intervals_have_known_answers():
    from experiments.contrast_factorial_reporting import estimate_factorial_effects

    a = {"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4}
    values = {
        arm: {k: v + shift for k, v in a.items()} for arm, shift in [("A", 0), ("B", 0.1), ("C", 0.2), ("D", 0.5)]
    }
    out = estimate_factorial_effects(
        values, {"f1": "x", "f2": "x", "f3": "y", "f4": "y"}, bootstrap_samples=100, seed=42
    )
    expected = {
        "rubric_at_low": 0.1,
        "language_at_narrow": 0.2,
        "rubric_at_high": 0.3,
        "language_at_broad": 0.4,
        "interaction": 0.2,
    }
    for name, value in expected.items():
        assert out["effects"][name]["estimate"] == pytest.approx(value)
        assert out["effects"][name]["ci95"] == pytest.approx([value, value])


def test_category_macro_does_not_let_a_larger_category_dominate():
    from experiments.contrast_factorial_reporting import estimate_factorial_effects

    values = {
        "A": {"a": 0.0, "b": 0.0, "c": 0.0},
        "B": {"a": 1.0, "b": 1.0, "c": 0.0},
        "C": {"a": 0.0, "b": 0.0, "c": 0.0},
        "D": {"a": 1.0, "b": 1.0, "c": 0.0},
    }
    out = estimate_factorial_effects(values, {"a": "many", "b": "many", "c": "one"}, bootstrap_samples=20)
    assert out["means"]["B"] == 0.5
    assert out["effects"]["rubric_at_low"]["estimate"] == 0.5


def test_missing_arm_family_and_nonfinite_values_fail_closed():
    from experiments.contrast_factorial_reporting import estimate_factorial_effects

    vals = {a: {"f": 0.0} for a in "ABCD"}
    with pytest.raises(ValueError):
        estimate_factorial_effects({a: v for a, v in vals.items() if a != "D"}, {"f": "c"})
    vals["D"] = {}
    with pytest.raises(ValueError):
        estimate_factorial_effects(vals, {"f": "c"})
    vals["D"] = {"f": float("nan")}
    with pytest.raises(ValueError):
        estimate_factorial_effects(vals, {"f": "c"})


def _binary_suite_and_rows():
    from experiments.revised_metrics import prediction_view

    suite = {"cases": [], "relations": []}
    rows = []
    for identity, gold in [("left", True), ("right", False)]:
        question = {"type": "noul", "instructions": "Was the requested operation completed?"}
        suite["cases"].append(
            {
                "id": identity,
                "family_id": "f",
                "domain": "d",
                "category": "completion",
                "variant": identity,
                "request": {"state": {"record": identity}, "questions": {"q": question}},
                "expected": {"q": gold},
                "rationale": {"q": "Independent fixture."},
            }
        )
        rows.append(
            {
                "case_id": identity,
                "question_id": "q",
                "family_id": "f",
                "domain": "d",
                "variant": identity,
                "prediction": prediction_view(
                    question, gold, {"noul": 0.96, "details": {"raw_logit": 3.1780538303479458}}
                ),
            }
        )
    suite["relations"] = [
        {
            "id": "contrast",
            "kind": "flip",
            "axis": "evidence",
            "left": {"case_id": "left", "question_id": "q"},
            "right": {"case_id": "right", "question_id": "q"},
        }
    ]
    return suite, rows


def test_constant_answer_fails_pair_metric_and_reports_confidence_coverage():
    from experiments.contrast_factorial_reporting import summarize_factorial

    suite, rows = _binary_suite_and_rows()
    result = summarize_factorial(suite, rows)
    assert result["overall"]["accuracy"] == 0.5
    assert result["primary"]["category_macro"] == 0
    assert result["primary"]["family_values"] == {"f": 0}
    assert result["contrast_axes"]["evidence"]["both_correct"] == 0
    assert result["overall"]["confidence"]["0.90"] == {
        "covered": 2,
        "coverage": 1.0,
        "wrong": 1,
        "error_rate_among_covered": 0.5,
    }
    assert result["by_category"]["completion"]["questions"] == 2


def test_summary_refuses_missing_duplicate_or_misbound_prediction_rows():
    import copy

    from experiments.contrast_factorial_reporting import summarize_factorial

    suite, rows = _binary_suite_and_rows()
    for invalid in [rows[:1], rows + rows[:1]]:
        with pytest.raises(ValueError):
            summarize_factorial(suite, invalid)
    bad = copy.deepcopy(rows)
    bad[0]["family_id"] = "another"
    with pytest.raises(ValueError):
        summarize_factorial(suite, bad)
    bad = copy.deepcopy(rows)
    bad[1]["prediction"]["expected"] = "true"
    with pytest.raises(ValueError):
        summarize_factorial(suite, bad)


def test_invariants_are_not_counted_as_successful_semantic_contrasts():
    import copy

    from experiments.contrast_factorial_reporting import summarize_factorial

    suite, rows = _binary_suite_and_rows()
    duplicate = copy.deepcopy(suite["cases"][0])
    duplicate["id"] = "same-truth"
    suite["cases"].append(duplicate)
    row = copy.deepcopy(rows[0])
    row["case_id"] = "same-truth"
    rows.append(row)
    suite["relations"].append(
        {
            "id": "stable",
            "kind": "invariant",
            "axis": "evidence",
            "left": {"case_id": "left", "question_id": "q"},
            "right": {"case_id": "same-truth", "question_id": "q"},
        }
    )
    result = summarize_factorial(suite, rows)
    assert result["primary"]["pairs"] == 1
    assert result["primary"]["category_macro"] == 0
    assert result["invariants"]["both_correct_rate"] == 1


def test_unknown_precision_and_recall_use_choice_predictions_not_low_confidence():
    import math

    from experiments.contrast_factorial_reporting import summarize_factorial
    from experiments.revised_metrics import prediction_view

    suite = {"cases": [], "relations": []}
    rows = []
    question = {
        "type": "choice",
        "instructions": "What is established?",
        "criteria": {"yes": "Established true.", "no": "Established false.", "unknown": "Unresolved."},
    }
    for i, (gold, values) in enumerate(
        [
            ("unknown", {"yes": 0.1, "no": 0.2, "unknown": 0.7}),
            ("unknown", {"yes": 0.5, "no": 0.3, "unknown": 0.2}),
            ("yes", {"yes": 0.3, "no": 0.1, "unknown": 0.6}),
        ]
    ):
        identity = str(i)
        suite["cases"].append(
            {
                "id": identity,
                "family_id": identity,
                "domain": "d",
                "variant": "v",
                "request": {"state": identity, "questions": {"q": question}},
                "expected": {"q": gold},
                "rationale": {"q": "Fixture"},
            }
        )
        rows.append(
            {
                "case_id": identity,
                "family_id": identity,
                "domain": "d",
                "question_id": "q",
                "prediction": prediction_view(
                    question,
                    gold,
                    {"probabilities": values, "details": {"raw_logits": {k: math.log(v) for k, v in values.items()}}},
                ),
            }
        )
    metric = summarize_factorial(suite, rows)["overall"]["unknown"]
    assert metric == {"choice_questions": 3, "gold": 2, "predicted": 2, "correct": 1, "recall": 0.5, "precision": 0.5}
