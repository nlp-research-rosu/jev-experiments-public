import importlib.util
import math

import pytest


def module():
    assert importlib.util.find_spec("experiments.paired_language_metrics") is not None, (
        "paired matrix metrics not implemented"
    )
    from experiments import paired_language_metrics

    return paired_language_metrics


def row(case, qid, expected, probabilities, primitive="choice"):
    maximum = max(probabilities.values())
    top = [k for k, v in probabilities.items() if v == maximum]
    predicted = top[0] if len(top) == 1 else None
    return {
        "case_id": case,
        "question_id": qid,
        "prediction": {
            "primitive": primitive,
            "expected": expected,
            "predicted": predicted,
            "probabilities": probabilities,
            "correct": predicted == expected,
            "max_probability": maximum,
            "nll": -math.log(probabilities[expected]),
            "brier": sum((v - float(k == expected)) ** 2 for k, v in probabilities.items()),
        },
    }


def fixtures():
    suite = {
        "cases": [
            {
                "id": "a",
                "family_id": "fa",
                "domain": "evidence",
                "category": "claim_vs_completion",
                "predicate_tags": {"n1": "claim", "n2": "confirmed", "c1": "outcome"},
                "expected": {"n1": True, "n2": False, "c1": "unknown"},
            },
            {
                "id": "b",
                "family_id": "fb",
                "domain": "binding",
                "category": "entity_binding",
                "predicate_tags": {"c1": "outcome"},
                "expected": {"c1": "done"},
            },
        ],
        "relations": [
            {
                "id": "q",
                "kind": "question_contrast",
                "left": {"case_id": "a", "question_id": "n1"},
                "right": {"case_id": "a", "question_id": "n2"},
            }
        ],
    }
    rows = [
        row("a", "n1", "true", {"true": 0.96, "false": 0.04}, "noul"),
        row("a", "n2", "false", {"true": 0.96, "false": 0.04}, "noul"),
        row("a", "c1", "unknown", {"unknown": 0.8, "done": 0.2}),
        row("b", "c1", "done", {"unknown": 0.91, "done": 0.09}),
    ]
    return suite, rows


def test_unknown_precision_recall_and_confident_errors_are_not_accuracy_aliases():
    m = module()
    s, rows = fixtures()
    r = m.matrix_summary(s, rows)
    assert r["overall"]["accuracy"] == 0.5
    assert r["overall"]["confident_wrong_90"] == 2
    assert r["overall"]["confident_wrong_95"] == 1
    assert r["unknown"]["count"] == 1 and r["unknown"]["predicted"] == 2
    assert r["unknown"]["recall"] == 1 and r["unknown"]["precision"] == 0.5
    assert r["by_category"]["claim_vs_completion"]["relations"]["question_contrast"]["both_correct"] == 0
    assert r["by_category"]["claim_vs_completion"]["overall"]["correct"] == 2
    assert r["whole_cases"]["all_correct"] == 0


def test_matrix_rejects_missing_predictions_instead_of_reporting_partial_success():
    m = module()
    s, rows = fixtures()
    with pytest.raises(ValueError, match="coverage"):
        m.matrix_summary(s, rows[:-1])


def test_case_level_tag_lists_do_not_become_question_predicate_labels():
    """Independent scaling authors supplied case tags, not a per-question map."""
    m = module()
    suite, rows = fixtures()
    original = m.matrix_summary(suite, rows)
    suite["cases"][0]["predicate_tags"] = ["explicit_authority", "missing_evidence_semantics"]

    result = m.matrix_summary(suite, rows)

    assert result["overall"] == original["overall"]
    assert result["relations"] == original["relations"]
    assert set(result["by_predicate"]) == {
        "claim_vs_completion/n1", "claim_vs_completion/n2", "claim_vs_completion/c1", "outcome",
    }
    assert result["by_predicate"]["claim_vs_completion/n2"]["correct"] == 0
    assert result["by_predicate"]["outcome"]["questions"] == 1
    assert suite["cases"][0]["predicate_tags"] == ["explicit_authority", "missing_evidence_semantics"]


def test_pair_bootstrap_resamples_families_and_reports_the_direction():
    m = module()
    s, control = fixtures()
    natural = [{**r, "prediction": dict(r["prediction"])} for r in control]
    for r in natural:
        r["prediction"]["correct"] = True
        r["prediction"]["predicted"] = r["prediction"]["expected"]
        r["prediction"]["nll"] = 0.01
    d = m.paired_family_comparison(s, control, natural, draws=200, seed=7)
    assert d["families"] == 2 and d["resampling_unit"] == "family_within_category"
    assert d["accuracy"]["difference"] == 0.5
    assert d["accuracy"]["interval_95"] == [0.5, 0.5]
    assert d["question_contrast"]["difference"] == 1
    assert d["nll"]["difference"] < 0


def test_common_probability_nll_keeps_raw_logit_loss_separate():
    m = module()
    s, rows = fixtures()
    rows[1]["prediction"].update(probabilities={"true": 1.0, "false": 0.0}, max_probability=1.0, nll=50.0)
    r = m.matrix_summary(s, rows)
    assert r["overall"]["zero_target_probabilities"] == 1
    expected = (
        sum(-math.log(max(x["prediction"]["probabilities"][x["prediction"]["expected"]], 1e-12)) for x in rows) / 4
    )
    assert r["overall"]["probability_nll_floor_1e12"] == pytest.approx(expected)
    assert r["overall"]["nll"] > r["overall"]["probability_nll_floor_1e12"]
