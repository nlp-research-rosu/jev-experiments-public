def case(suite, suffix):
    return next(c for c in suite["cases"] if c["domain"] == "support" and c["variant"] == suffix)


def test_claim_and_confirmation_cover_all_four_truth_combinations():
    from experiments.contrast_action_cases import build_action_suite

    suite = build_action_suite()
    combinations = {
        (c["expected"]["claimed_done"], c["expected"]["completion_confirmed"])
        for c in suite["cases"]
        if c["domain"] == "support"
    }
    assert combinations == {(False, False), (False, True), (True, False), (True, True)}
    assert case(suite, "claim_only")["expected"]["outcome_status"] == "not_established"


def test_timeout_is_unknown_but_explicit_failure_confirms_no_effect():
    from experiments.contrast_action_cases import build_action_suite

    suite = build_action_suite()
    failed, timed_out = case(suite, "failed_call"), case(suite, "timeout_call")
    assert failed["expected"]["invocation_shown"] is True
    assert failed["expected"]["completion_confirmed"] is False
    assert failed["expected"]["outcome_status"] == "confirmed_no_effect"
    assert timed_out["expected"]["outcome_status"] == "not_established"


def test_permission_is_independent_of_execution_and_dry_run_is_not_completion():
    from experiments.contrast_action_cases import build_action_suite

    suite = build_action_suite()
    unauthorized = case(suite, "unauthorized_success")["expected"]
    assert unauthorized["completion_confirmed"] is True
    assert unauthorized["permitted"] is False
    simulated = case(suite, "simulation_success")["expected"]
    assert simulated["invocation_shown"] is True
    assert simulated["completion_confirmed"] is False
    assert simulated["outcome_status"] == "confirmed_no_effect"


def test_reversal_and_binding_change_only_the_relevant_outcome():
    from experiments.contrast_action_cases import build_action_suite

    suite = build_action_suite()
    reversed_case = case(suite, "reversed")["expected"]
    assert reversed_case["completion_confirmed"] is True
    assert reversed_case["effective_now"] is False
    assert reversed_case["outcome_status"] == "confirmed_reversed"
    assert case(suite, "unrelated_reversal")["expected"]["effective_now"] is True
    assert case(suite, "wrong_target_success")["expected"]["completion_confirmed"] is False
    assert case(suite, "wrong_operation_success")["expected"]["completion_confirmed"] is False


def test_relation_endpoints_have_valid_semantic_targets_without_label_leakage():
    from experiments.contrast_action_cases import build_action_suite
    from openjev.judgments import compile_request

    suite = build_action_suite()
    by_id = {c["id"]: c for c in suite["cases"]}
    assert len(by_id) == len(suite["cases"])
    for c in suite["cases"]:
        assert set(c["request"]) == {"state", "questions"}
        assert set(c["expected"]) == set(c["request"]["questions"])
        assert len(compile_request(c["request"]).questions) == len(c["expected"])
    for relation in suite["relations"]:
        left, right = relation["left"], relation["right"]
        a = by_id[left["case_id"]]["expected"][left["question_id"]]
        b = by_id[right["case_id"]]["expected"][right["question_id"]]
        assert (a == b) == (relation["kind"] == "invariant")
