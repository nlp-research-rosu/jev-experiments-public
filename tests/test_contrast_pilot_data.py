from collections import Counter


def test_splits_keep_domains_and_all_variants_together():
    from experiments.contrast_pilot_data import build_splits
    from experiments.semantic_contrasts import validate_suite

    splits = build_splits()
    assert {k: len(v["cases"]) for k, v in splits.items()} == {"train": 300, "validation": 50, "test": 100}
    seen_domains = set()
    seen_states = set()
    for suite in splits.values():
        validate_suite(suite)
        domains = {c["domain"] for c in suite["cases"]}
        assert not domains & seen_domains
        seen_domains |= domains
        assert set(Counter(c["domain"] for c in suite["cases"]).values()) == {25}
        for case in suite["cases"]:
            import json

            state = json.dumps(case["request"]["state"], sort_keys=True)
            assert state not in seen_states
            seen_states.add(state)
            assert len(case["expected"]) == 7


def test_claim_evidence_truth_table_and_explicit_unknown_controls():
    from experiments.contrast_pilot_data import build_splits

    for suite in build_splits().values():
        cases = suite["cases"]
        assert {(c["expected"]["claimed"], c["expected"]["completed"]) for c in cases} == {
            (False, False), (False, True), (True, False), (True, True),
        }
        first = cases[0]["domain"]
        by_variant = {c["variant"]: c for c in cases if c["domain"] == first}
        for variant in ["pending", "timeout", "unverified", "conflicting", "wrong_target", "wrong_operation", "future", "stale"]:
            y = by_variant[variant]["expected"]
            assert y["completed"] is False and y["current_effect"] is False and y["status"] == "unknown"
        assert by_variant["claim_only"]["expected"] == {
            "planned": False, "claimed": True, "invoked": False, "completed": False,
            "current_effect": False, "authorized": True, "status": "unknown",
        }
        assert by_variant["reversed"]["expected"] == {
            "planned": True, "claimed": False, "invoked": True, "completed": True,
            "current_effect": False, "authorized": True, "status": "reversed",
        }
        assert by_variant["unauthorized_success"]["expected"]["completed"] is True
        assert by_variant["unauthorized_success"]["expected"]["authorized"] is False


def test_unknown_after_uncertain_reversal_does_not_erase_prior_completion():
    from experiments.contrast_pilot_data import build_splits

    case = next(c for c in build_splits()["train"]["cases"] if c["variant"] == "uncertain_reversal")
    assert case["expected"]["completed"] is True
    assert case["expected"]["current_effect"] is False
    assert case["expected"]["status"] == "unknown"
    events = case["request"]["state"]["events"]
    assert events[-1]["reverses"] == events[0]["id"]
    assert events[-1]["result"]["outcome"] == "timeout"


def test_training_bundles_exclude_evaluator_metadata_from_model_inputs():
    from experiments.contrast_pilot_data import build_splits, training_bundle

    case = build_splits()["train"]["cases"][0]
    bundle = training_bundle(case)
    assert len(bundle["examples"]) == 7
    assert bundle["provenance"]["dataset"] == "contrast_pilot"
    for example in bundle["examples"]:
        assert set(example) == {"state", "question", "target"}
        assert example["state"] == case["request"]["state"]
    assert bundle["relations"] == []


def test_rendered_evidence_preserves_authority_time_and_reversal_distinctions():
    from experiments.contrast_pilot_data import build_splits

    splits = build_splits()
    cases = {c["variant"]: c["request"]["state"] for c in splits["train"]["cases"] if c["domain"] == "feature"}
    assert cases["simulation"]["events"][0]["mode"] == "simulation"
    assert "No real effect" in cases["simulation"]["events"][0]["result"]["detail"]
    assert cases["unverified"]["events"][0]["result"]["authenticated"] is False
    results = cases["conflicting"]["events"][0]["result"]
    assert [r["outcome"] for r in results] == ["applied", "rejected"]
    assert all(r["authenticated"] and r["source"] == "executor" for r in results)
    assert cases["stale"]["events"][0]["at"] < cases["stale"]["task"]["window_start"]
    assert cases["future"]["events"][0]["at"] > cases["future"]["task"]["window_end"]
    event, reversal = cases["unrelated_reversal"]["events"]
    assert reversal["target"] != event["target"] and reversal["reverses"] != event["id"]
    assert cases["wrong_target"]["events"][0]["target"] != cases["wrong_target"]["task"]["target"]
    assert cases["wrong_operation"]["events"][0]["operation"] != cases["wrong_operation"]["task"]["operation"]
    heldout = next(c["request"]["state"] for c in splits["test"]["cases"] if c["domain"] == "package" and c["variant"] == "success")
    assert "release_package" in heldout["request_brief"] and "package-comet" in heldout["request_brief"]
    assert heldout["record"]["events"][0]["operation"] == "release_package"
    assert heldout["record"]["events"][0]["result"]["outcome"] == "applied"
