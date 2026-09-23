"""Behavioral checks for the bounded revision corpus."""

from collections import Counter


def _family_cases(suite, family, variant):
    return [
        case
        for case in suite["cases"]
        if case["family_id"] == family and case["variant"] == variant
    ]


def test_builds_disjoint_layout_complete_splits_with_primitive_targets():
    """Catches a generator that drops a layout, leaks a domain, or changes target shapes."""
    from experiments.contrast_revision_data import build_splits, training_bundle

    splits = build_splits()
    assert {name: len(suite["cases"]) for name, suite in splits.items()} == {
        "train": 660,
        "validation": 144,
        "test": 237,
    }
    seen_domains = set()
    historical_domains = {
        "feature", "article", "projector", "reviewer", "newsletter", "webinar", "backup", "profile",
        "calendar", "folder", "alert", "artifact", "playlist", "workspace", "package", "course", "parcel",
        "catalog", "support", "refund", "appointment", "access", "payment", "shipping",
    }
    for split, suite in splits.items():
        domains = {case["domain"] for case in suite["cases"]}
        assert not domains & seen_domains
        assert not domains & historical_domains
        seen_domains |= domains
        assert {case["layout"] for case in suite["cases"]} == {"flat", "nested", "prose"}
        assert {relation["kind"] for relation in suite["relations"]} >= {
            "flip", "question_contrast", "invariant", "layout_invariant"
        }
        for case in suite["cases"]:
            questions = case["request"]["questions"]
            assert Counter(question["type"] for question in questions.values()) == {
                "noul": 3,
                "choice": 1,
                "score": 1,
            }
            assert set(case["expected"]) == set(questions) == set(case["rationale"])
            bundle = training_bundle(case)
            assert bundle["relations"] == []
            assert {frozenset(example) for example in bundle["examples"]} == {frozenset({"state", "question", "target"})}
            target_keys = {"noul": "truth", "choice": "choice", "score": "level_index"}
            for (qid, question), example in zip(questions.items(), bundle["examples"], strict=True):
                assert example["target"] == {target_keys[question["type"]]: case["expected"][qid]}
            assert all("expected" not in example["state"] for example in bundle["examples"] if isinstance(example["state"], dict))


def test_action_gold_treats_old_success_as_out_of_scope_but_keeps_matching_reversal_historical():
    """Catches the old ambiguity where an earlier success establishes this operation's score."""
    from experiments.contrast_revision_data import build_splits

    suite = build_splits()["train"]
    family = "action/ledger"
    old_success = _family_cases(suite, family, "old_success")[0]["expected"]
    reversed_effect = _family_cases(suite, family, "reversed")[0]["expected"]
    uncertain_reversal = _family_cases(suite, family, "uncertain_reversal")[0]["expected"]
    unauthorized = _family_cases(suite, family, "unauthorized_success")[0]["expected"]

    assert old_success == {
        "claimed_completion": False,
        "completed": False,
        "current_effect": False,
        "status": "unknown",
        "execution_stage": 0,
    }
    assert reversed_effect["completed"] is True
    assert reversed_effect["current_effect"] is False
    assert reversed_effect["status"] == "reversed"
    assert reversed_effect["execution_stage"] == 2
    assert uncertain_reversal["completed"] is True
    assert uncertain_reversal["status"] == "unknown"
    assert unauthorized["completed"] is True
    unauthorized_state = _family_cases(suite, family, "unauthorized_success")[0]["request"]["state"]
    assert unauthorized_state["policy"]["approval"] == "denied"


def test_registry_gold_counts_only_independently_verified_checks_and_keeps_claims_separate():
    """Catches treating a claim, missing baseline, or unverified snapshot as a registry fact."""
    from experiments.contrast_revision_data import build_splits

    suite = build_splits()["train"]
    family = "registry/roster"
    by_variant = {case["variant"]: case["expected"] for case in suite["cases"] if case["family_id"] == family and case["layout"] == "flat"}

    assert by_variant["baseline_absent"] == {
        "explicit_change_claim": False,
        "record_confirms_change": False,
        "policy_allows": True,
        "record_status": "unknown",
        "verification_stage": 0,
    }
    assert by_variant["claimed_unverified"] == {
        "explicit_change_claim": True,
        "record_confirms_change": False,
        "policy_allows": True,
        "record_status": "unknown",
        "verification_stage": 1,
    }
    assert by_variant["verified_change"]["record_confirms_change"] is True
    assert by_variant["verified_change"]["verification_stage"] == 2
    assert by_variant["verified_unchanged"]["record_status"] == "unchanged"
    assert {(labels["explicit_change_claim"], labels["record_confirms_change"]) for labels in by_variant.values()} == {
        (False, False), (False, True), (True, False), (True, True)
    }


def test_layout_renderings_preserve_facts_and_test_has_new_question_wording():
    """Catches representation-specific gold or a held-out wording copied from training."""
    from experiments.contrast_revision_data import build_splits

    splits = build_splits()
    train = splits["train"]
    family = "action/ledger"
    rendered = _family_cases(train, family, "success")
    assert {case["layout"] for case in rendered} == {"flat", "nested", "prose"}
    assert {tuple(sorted(case["expected"].items())) for case in rendered} == {
        tuple(sorted(rendered[0]["expected"].items()))
    }
    assert isinstance(next(case for case in rendered if case["layout"] == "prose")["request"]["state"], str)
    assert isinstance(next(case for case in rendered if case["layout"] == "flat")["request"]["state"], dict)
    assert "task" in next(case for case in rendered if case["layout"] == "nested")["request"]["state"]
    prose = next(case for case in rendered if case["layout"] == "prose")["request"]["state"]
    flat = next(case for case in rendered if case["layout"] == "flat")["request"]["state"]
    assert flat["scope"]["description"] in prose
    assert flat["policy"]["rule"] in prose
    assert flat["evidence_policy"] in prose
    assert "contradictory results for the same invocation" in prose

    train_wordings = {
        question["instructions"]
        for case in train["cases"]
        for question in case["request"]["questions"].values()
    }
    test_wordings = {
        question["instructions"]
        for case in splits["test"]["cases"]
        for question in case["request"]["questions"].values()
    }
    assert test_wordings - train_wordings


def test_rendered_evidence_exposes_missing_unverified_and_conflicting_unknown_controls():
    """Catches evidence renderers that hide the facts supporting the distinct unknown labels."""
    from experiments.contrast_revision_data import build_splits

    suite = build_splits()["train"]
    action = {
        case["variant"]: case["request"]["state"]
        for case in suite["cases"]
        if case["family_id"] == "action/ledger" and case["layout"] == "flat"
    }
    assert action["pending"]["events"][0]["result"] is None
    assert action["failed"]["events"][0]["result"]["outcome"] == "rejected"
    assert action["unverified"]["events"][0]["result"]["authenticated"] is False
    assert [row["outcome"] for row in action["conflicting"]["events"][0]["result"]] == ["applied", "rejected"]
    assert action["old_success"]["events"][0]["at"] < action["old_success"]["scope"]["window_start"]
    assert action["future_success"]["events"][0]["at"] > action["future_success"]["scope"]["window_end"]
    wrong_target = action["wrong_target"]["events"][0]
    assert wrong_target["target"] == action["wrong_target"]["scope"]["target"] + "-OTHER"
    assert wrong_target["target"] in wrong_target["result"]["detail"]

    registry = {
        case["variant"]: case["request"]["state"]
        for case in suite["cases"]
        if case["family_id"] == "registry/roster" and case["layout"] == "flat"
    }
    assert registry["baseline_absent"]["trusted_baseline"] is None
    assert registry["unverified_snapshot"]["snapshots"][0]["authenticated"] is False
    assert len(registry["conflicting_snapshots"]["snapshots"]) == 2
    assert {row["value"] for row in registry["conflicting_snapshots"]["snapshots"]} == {"Blair", "Casey"}
    intent = registry["intent_only"]
    assert intent["trusted_baseline"]["trusted"] is True and intent["snapshots"] == []
    intent_gold = next(
        case["expected"]
        for case in suite["cases"]
        if case["family_id"] == "registry/roster" and case["variant"] == "intent_only" and case["layout"] == "flat"
    )
    assert intent_gold["verification_stage"] == 1
    denial = registry["claimed_unchanged"]
    assert denial["assistant_statement"] == "I have not changed the on-call lead in staff roster."
