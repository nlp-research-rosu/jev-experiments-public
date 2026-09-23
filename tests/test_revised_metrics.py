import math

import pytest


def test_score_distribution_quality_and_mean_error_are_separate():
    from experiments.revised_metrics import prediction_view

    q = {"type": "score", "criteria": ["low", "middle", "high"]}
    p = {"0": .49, "1": .02, "2": .49}
    a = {"type": "score", "score": 1, "probabilities": p,
         "details": {"raw_logits": {k: math.log(v) for k, v in p.items()}}}
    view = prediction_view(q, 1, a)
    assert view["score_absolute_error"] == 0
    assert view["nll"] == pytest.approx(-math.log(.02))
    assert not view["correct"] and view["ambiguous_top"]


def test_layout_invariance_is_not_scored_as_a_label_flip():
    from experiments.revised_metrics import relation_metrics, summarize

    p = {"expected": "true", "predicted": "true", "probabilities": {"true": .9, "false": .1},
         "primitive": "noul", "correct": True, "nll": -math.log(.9), "brier": .02, "max_probability": .9}
    rows = [{"case_id": cid, "question_id": "q", "family_id": "f", "domain": "d", "variant": "v",
             "layout": layout, "prediction": p} for cid, layout in [("a", "flat"), ("b", "nested")]]
    relation = {"kind": "layout_invariant", "left": {"case_id": "a", "question_id": "q"},
                "right": {"case_id": "b", "question_id": "q"}}
    relation["metrics"] = relation_metrics(relation, {(r["case_id"], "q"): p for r in rows})
    summary = summarize(rows, [relation])
    assert summary["relations"]["layout_invariant"]["within_tolerance"] == 1
    assert summary["by_question_by_layout"]["q"]["nested"]["positive_recall"] == 1
    assert summary["by_question_by_layout"]["q"]["flat"]["negative_count"] == 0


def test_validator_checks_score_targets_and_truth_preserving_layout_relations():
    from experiments.revised_metrics import validate_suite

    def case(cid, target):
        return {"id": cid, "family_id": "f", "domain": "d", "variant": "v",
                "request": {"state": "record", "questions": {"q": {"type": "score", "criteria": ["low", "high"]}}},
                "expected": {"q": target}, "rationale": {"q": "Authored"}}
    suite = {"cases": [case("a", 0), case("b", 0)], "relations": [{"id": "r", "kind": "layout_invariant",
             "left": {"case_id": "a", "question_id": "q"}, "right": {"case_id": "b", "question_id": "q"}}]}
    validate_suite(suite)
    suite["cases"][1]["expected"]["q"] = 1
    with pytest.raises(ValueError):
        validate_suite(suite)
    suite["relations"] = []
    suite["cases"][1]["expected"]["q"] = True
    with pytest.raises(ValueError):
        validate_suite(suite)


def test_local_binary_nll_uses_unsaturated_logit():
    from experiments.revised_metrics import prediction_view

    view = prediction_view({"type": "noul"}, False, {"noul": 1., "details": {"raw_logit": 40.}})
    assert view["nll"] == pytest.approx(40)
    assert not view["correct"]
