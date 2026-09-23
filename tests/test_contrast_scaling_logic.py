import copy

import pytest


def record():
    return {
        "query": {"actor": "Ari", "operation": "dispatch", "target": "parcel-7", "run_id": "r7"},
        "window": [10, 20],
        "trusted_sources": ["ledger", "inspector"],
        "statements": [],
        "approvals": [],
        "executions": [],
        "reversals": [],
        "corroborations": [],
        "items": [],
        "inventory_complete": True,
    }


def execution(**edits):
    return {
        "actor": "Ari",
        "operation": "dispatch",
        "target": "parcel-7",
        "run_id": "r7",
        "time": 15,
        "source": "ledger",
        "verified": True,
        "mode": "live",
        "outcome": "success",
        **edits,
    }


def statement(**edits):
    return {
        "id": "m1",
        "speaker": "Ari",
        "actor": "Ari",
        "operation": "dispatch",
        "target": "parcel-7",
        "tense": "completed",
        "polarity": "positive",
        "endorsed": False,
        **edits,
    }


def test_claim_intent_and_operational_evidence_are_distinct():
    from experiments.contrast_scaling_logic import derive_features

    r = record()
    r["statements"] = [statement(tense="future")]
    f = derive_features(r)
    assert f["intent"] and not f["claim"] and not f["invocation"] and f["unknown"]
    r["statements"] = [statement()]
    r["executions"] = [execution(outcome="no_effect")]
    f = derive_features(r)
    assert f["claim"] and f["invocation"] and f["failed"] and not f["completed"] and not f["unknown"]


@pytest.mark.parametrize(
    "change",
    [{"target": "parcel-8"}, {"actor": "Bo"}, {"operation": "inspect"}, {"run_id": "r8"}, {"time": 9}, {"time": 21}],
)
def test_exact_subject_and_inclusive_window_bind_the_execution(change):
    from experiments.contrast_scaling_logic import derive_features

    r = record()
    r["executions"] = [execution(**change)]
    f = derive_features(r)
    assert not f["completed"] and f["unknown"]


@pytest.mark.parametrize("time", [10, 20])
def test_both_window_endpoints_count(time):
    from experiments.contrast_scaling_logic import derive_features

    r = record()
    r["executions"] = [execution(time=time)]
    assert derive_features(r)["completed"]


@pytest.mark.parametrize(
    "change",
    [{"mode": "dry_run"}, {"verified": False}, {"source": "rumor"}, {"outcome": "timeout"}, {"outcome": "missing"}],
)
def test_nonconfirming_outcomes_remain_unknown(change):
    from experiments.contrast_scaling_logic import derive_features

    r = record()
    r["executions"] = [execution(**change)]
    f = derive_features(r)
    assert f["invocation"] and f["unknown"] and not f["completed"] and not f["failed"]


def test_conflicting_final_reports_do_not_establish_either_outcome():
    from experiments.contrast_scaling_logic import derive_features

    r = record()
    r["executions"] = [execution(), execution(source="inspector", outcome="no_effect")]
    f = derive_features(r)
    assert f["unknown"] and not f["completed"] and not f["failed"] and f["success_report"] and f["failure_report"]


def test_reversal_requires_matching_verified_link_after_success():
    from experiments.contrast_scaling_logic import derive_features

    r = record()
    r["executions"] = [execution()]
    undo = execution(time=17, outcome="success")
    undo["reverses"] = "r7"
    r["reversals"] = [undo]
    f = derive_features(r)
    assert f["completed"] and f["reversed"] and not f["effective"]
    r["reversals"][0]["reverses"] = "other"
    assert derive_features(r)["effective"]
    r["reversals"][0].update(reverses="r7", time=14)
    assert derive_features(r)["effective"]


def test_approval_and_endorsement_do_not_create_completion():
    from experiments.contrast_scaling_logic import derive_features

    r = record()
    r["approvals"] = [execution(decision="allow")]
    r["statements"] = [statement(endorsed=True)]
    f = derive_features(r)
    assert f["approved"] and f["endorsement"] and f["claim"] and not f["completed"]


def test_corroboration_must_be_independent_matching_authenticated_source():
    from experiments.contrast_scaling_logic import derive_features

    r = record()
    r["executions"] = [execution()]
    r["corroborations"] = [execution(source="ledger")]
    assert not derive_features(r)["corroborated"]
    r["corroborations"] = [execution(source="inspector")]
    assert derive_features(r)["corroborated"]
    r["corroborations"][0]["run_id"] = "other"
    assert not derive_features(r)["corroborated"]


def test_quantifiers_use_distinct_items_and_unknown_is_not_failure():
    from experiments.contrast_scaling_logic import derive_features

    r = record()
    r["items"] = [
        {"id": "x", "status": "success", "verified": True},
        {"id": "y", "status": "unknown", "verified": True},
        {"id": "z", "status": "no_effect", "verified": True},
    ]
    f = derive_features(r)
    assert f["success_count"] == 1 and f["failure_count"] == 1 and f["unknown_count"] == 1
    assert f["any_success"] and f["any_failure"] and not f["all_success"] and f["item_unknown"]
    r["items"].append(copy.deepcopy(r["items"][0]))
    assert derive_features(r)["success_count"] == 1
    r["items"].append({"id": "x", "status": "no_effect", "verified": True})
    assert derive_features(r)["success_count"] == 0 and derive_features(r)["unknown_count"] == 2


def test_supplied_rule_changes_score_on_the_same_evidence():
    from experiments.contrast_scaling_logic import derive_features, eval_rule, select_level

    r = record()
    r["approvals"] = [execution(decision="allow")]
    f = derive_features(r)
    execution_scale = [True, {"feature": "invocation"}, {"feature": "completed"}]
    approval_scale = [True, {"feature": "approved"}, {"feature": "completed"}]
    assert select_level(execution_scale, f) == 0
    assert select_level(approval_scale, f) == 1
    assert eval_rule({"all": [{"feature": "approved"}, {"not": {"feature": "completed"}}]}, f)
    assert not eval_rule({"at_least": {"count": 2, "of": [{"feature": "approved"}, {"feature": "completed"}]}}, f)


def test_unknown_program_or_feature_is_rejected_not_silently_false():
    from experiments.contrast_scaling_logic import eval_rule, select_choice, select_level

    with pytest.raises(ValueError):
        eval_rule({"feature": "typo"}, {})
    with pytest.raises(ValueError):
        eval_rule({"unexpected": True}, {})
    with pytest.raises(ValueError):
        select_level([False, False], {})
    with pytest.raises(ValueError):
        select_choice({"a": True, "b": True}, {})


def test_numeric_active_flags_cannot_create_inconsistent_known_states():
    from experiments.contrast_scaling_logic import derive_features

    r = record()
    for invalid in (0, 1, "unknown", None):
        r["state_events"] = [
            {"target": "parcel-7", "time": 15, "source": "ledger", "verified": True, "active": invalid}
        ]
        f = derive_features(r)
        assert not f["current_known"]
        assert not f["active_now"] and not f["inactive_now"]
    r["state_events"] = [
        {"target": "parcel-7", "time": 12, "source": "ledger", "verified": True, "active": True},
        {"target": "parcel-7", "time": 18, "source": "ledger", "verified": True, "active": False},
    ]
    r["query_time"] = 15
    f = derive_features(r)
    assert f["active_earlier"] and f["inactive_now"] and f["current_known"]


def test_unrelated_success_reports_do_not_establish_requested_completion():
    from experiments.contrast_scaling_logic import derive_features

    r = record()
    r["executions"] = [execution(target="parcel-8")]
    f = derive_features(r)
    assert f["any_live_success"] and f["other_live_success"] and not f["completed"]
    r["executions"] = [execution()]
    f = derive_features(r)
    assert f["any_live_success"] and not f["other_live_success"] and f["completed"]
    r["executions"] = [execution(target="parcel-8", verified=False), execution(run_id="r9", mode="dry_run")]
    f = derive_features(r)
    assert not f["any_live_success"] and not f["other_live_success"]
