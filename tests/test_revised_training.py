import pytest


def test_schedule_equalizes_primitive_gradient_weight_and_reuses_identical_stream():
    from experiments.revised_training import make_schedule

    pools = {origin: {kind: {"source": [origin + kind + str(i) for i in range(4)]}
                      for kind in ("noul", "choice", "score")} for origin in ("broad", "contrast")}
    schedule = make_schedule(pools, steps=8, seed=42)
    assert schedule == make_schedule(pools, steps=8, seed=42)
    assert len(schedule) == 8 and all(len(step) == 6 for step in schedule)
    for step in schedule:
        assert [(item["primitive"], item["origin"]) for item in step] == [
            ("noul", "broad"), ("noul", "contrast"), ("choice", "broad"),
            ("choice", "contrast"), ("score", "broad"), ("score", "contrast")]
    assert len({step[0]["id"] for step in schedule[:4]}) == 4


def test_schedule_rejects_missing_primitive_or_ambiguous_ids():
    from experiments.revised_training import make_schedule

    pools = {origin: {kind: {"source": ["same"]} for kind in ("noul", "choice", "score")}
             for origin in ("broad", "contrast")}
    with pytest.raises(ValueError):
        make_schedule(pools, steps=2, seed=42)


def test_selection_uses_only_completed_validation_objectives_and_breaks_ties_early():
    from experiments.revised_evaluation import select_checkpoint

    curve = [{"status": "complete", "objective": x, "step": s} for s, x in [(0, .5), (250, .2), (500, .2)]]
    assert select_checkpoint(curve)["step"] == 250
    curve.append({"status": "failed", "objective": .01, "step": 1000})
    with pytest.raises(ValueError):
        select_checkpoint(curve)


def test_revised_group_identity_uses_declared_family_without_assuming_broad_schema():
    from experiments.revised_training import source_group

    assert source_group({"group_id": "public-passage", "provenance": {"dataset": "boolq"}}) == "public-passage"
    assert source_group({"provenance": {"family": "action/ledger"}}) == "action/ledger"
    with pytest.raises(ValueError):
        source_group({"provenance": {}})


def test_readiness_rejects_score_regression_hidden_by_unchanged_overall_accuracy():
    import copy

    from experiments.revised_evaluation import readiness

    revised = {"summary": {"overall": {"accuracy": .95, "nll": .1},
               "by_question": {"completed": {"balanced_binary_accuracy": .95}},
               "by_layout": {k: {"accuracy": .95} for k in ("flat", "nested", "prose")},
               "by_question_by_layout": {"completed": {k: {"positive_recall": .95, "negative_specificity": .95}
                                                        for k in ("flat", "nested", "prose")}},
               "relations": {k: {"both_correct_rate": .95} for k in ("flip", "question_contrast", "invariant", "layout_invariant")}}}
    broad = {"summary": {"overall": {"accuracy": .9, "nll": .2},
             "by_domain": {"wine_quality": {"accuracy": .5, "score_mean_absolute_error": .6},
                           "oracle_policy": {"accuracy": 1., "score_mean_absolute_error": .01}}}}
    evaluation = {"revised": revised, "broad": broad}
    reference = copy.deepcopy(evaluation)
    rows = [{"question_id": qid, "prediction": {"expected": "unknown", "correct": True}}
            for qid in ("status", "record_status")]
    assert readiness(evaluation, reference, rows)["passed"] is True
    evaluation["broad"]["summary"]["by_domain"]["wine_quality"]["accuracy"] = .46
    result = readiness(evaluation, reference, rows)
    assert result["passed"] is False and result["checks"]["score_accuracy_wine_quality"] is False
    evaluation = copy.deepcopy(reference)
    evaluation["broad"]["summary"]["by_domain"]["oracle_policy"]["score_mean_absolute_error"] = .08
    assert readiness(evaluation, reference, rows)["checks"]["score_mae_oracle_policy"] is False


def test_readiness_does_not_hide_one_layouts_positive_completion_failure():
    from experiments.revised_evaluation import summarize

    # Literal predictions demonstrate that perfect negative accuracy cannot mask zero positive recall.
    rows = []
    for identity, expected in [("positive", "true"), ("negative", "false")]:
        rows.append({"case_id": identity, "question_id": "completed", "family_id": "family", "domain": "domain",
                     "variant": identity, "layout": "nested", "prediction": {
                         "primitive": "noul", "expected": expected, "predicted": "false", "correct": expected == "false",
                         "probabilities": {"false": .9, "true": .1}, "nll": .1, "brier": .1, "max_probability": .9}})
    metric = summarize(rows, [])["by_question_by_layout"]["completed"]["nested"]
    assert metric["positive_recall"] == 0 and metric["negative_specificity"] == 1
    assert metric["balanced_binary_accuracy"] == .5
