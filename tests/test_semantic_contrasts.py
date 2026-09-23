import math

import pytest


def test_correct_probability_order_does_not_hide_a_failed_positive_example():
    from experiments.semantic_contrasts import relation_metrics

    relation = {
        "kind": "flip",
        "left": {"case_id": "a", "question_id": "q"},
        "right": {"case_id": "b", "question_id": "q"},
    }
    lookup = {
        ("a", "q"): {"expected": "false", "predicted": "false", "probabilities": {"false": 0.9, "true": 0.1}},
        ("b", "q"): {"expected": "true", "predicted": "false", "probabilities": {"false": 0.6, "true": 0.4}},
    }
    result = relation_metrics(relation, lookup)
    assert result["direction_correct"] is True
    assert result["both_correct"] is False
    assert result["directional_margin"] == pytest.approx(0.3)


def test_stable_but_wrong_answers_do_not_count_as_correct_invariance_pairs():
    from experiments.semantic_contrasts import relation_metrics

    relation = {
        "kind": "invariant",
        "left": {"case_id": "a", "question_id": "q"},
        "right": {"case_id": "b", "question_id": "q"},
    }
    wrong = {"expected": "false", "predicted": "true", "probabilities": {"false": 0.01, "true": 0.99}}
    result = relation_metrics(relation, {("a", "q"): wrong, ("b", "q"): wrong})
    assert result["within_tolerance"] is True
    assert result["both_correct"] is False
    assert result["total_variation"] == 0


def test_probability_loss_uses_logits_when_sigmoid_saturates():
    from experiments.semantic_contrasts import prediction_view

    q = {"type": "noul"}
    answer = {"type": "noul", "noul": 1.0, "details": {"raw_logit": 40.0}}
    row = prediction_view(q, False, answer)
    assert row["nll"] == pytest.approx(40.0)
    assert not row["correct"]
    assert row["brier"] == 2.0


def test_categorical_loss_is_stable_and_ambiguous_top_is_explicit():
    from experiments.semantic_contrasts import prediction_view

    q = {"type": "choice", "criteria": {"yes": "yes", "unknown": "unknown"}}
    answer = {
        "type": "choice",
        "choice": "yes",
        "probabilities": {"yes": 0.5, "unknown": 0.5},
        "details": {"raw_logits": {"yes": 100.0, "unknown": 100.0}},
    }
    row = prediction_view(q, "unknown", answer)
    assert row["nll"] == pytest.approx(math.log(2))
    assert row["predicted"] is None
    assert row["ambiguous_top"] is True


def test_suite_validator_checks_relations_and_metadata_boundaries():
    from experiments.semantic_contrasts import validate_suite

    case = {
        "id": "a",
        "family_id": "one",
        "domain": "test",
        "variant": "a",
        "expected": {"q": True},
        "rationale": {"q": "test"},
        "request": {"state": {}, "questions": {"q": {"type": "noul"}}},
    }
    suite = {"cases": [case], "relations": [], "contract": "test"}
    validate_suite(suite)
    suite["relations"] = [
        {
            "id": "bad",
            "kind": "flip",
            "left": {"case_id": "a", "question_id": "q"},
            "right": {"case_id": "missing", "question_id": "q"},
            "reason": "bad",
        }
    ]
    with pytest.raises(ValueError):
        validate_suite(suite)
