import copy
from collections import Counter, defaultdict

QUESTION_IDS = ("n1", "n2", "n3", "c1", "s1")


def seeds():
    from experiments.paired_language_data import build_training_seeds

    return build_training_seeds()


def cases_for(data, category, family="01"):
    prefix = f"train/{category}/{family}/"
    return {case["variant"]: case for case in data["cases"] if case["id"].startswith(prefix)}


def _changed_paths(left, right, prefix=""):
    if type(left) is not type(right):
        return {prefix}
    if isinstance(left, dict):
        paths = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths |= _changed_paths(left[key], right[key], path)
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return {prefix}
        return set().union(*(_changed_paths(a, b, f"{prefix}[{i}]") for i, (a, b) in enumerate(zip(left, right))))
    return set() if left == right else {prefix}


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def _matching_invocation(case, **overrides):
    requested = case["facts"]["context"]["requested_action"]
    invocation = {
        "id": "test-added-invocation",
        "actor": requested["actor"],
        "operation": requested["operation"],
        "target": requested["target"],
        "time": "2026-01-10T12:00:00Z",
        "mode": "real",
        "result": "success",
        "authenticated": True,
    }
    invocation.update(overrides)
    return invocation


def test_seed_population_preserves_interface_without_emitting_data(tmp_path, monkeypatch):
    from openjev.judgments import compile_request

    monkeypatch.chdir(tmp_path)
    data = seeds()
    assert list(tmp_path.iterdir()) == []
    assert data["metadata"]["families"] == 200
    assert data["metadata"]["cases"] == 800
    assert data["metadata"]["judgments"] == 4000
    assert len(data["relations"]) == 1000
    assert len(data["families"]) == 200
    assert len({family["workflow"]["scope_id"] for family in data["families"]}) == 200
    for case in data["cases"]:
        assert list(case["request"]["questions"]) == list(QUESTION_IDS)
        assert set(case["expected"]) == set(QUESTION_IDS)
        assert set(case["request"]["state"]) == {"context", "policy", "record"}
        assert case["request"]["state"] == case["facts"]
        compile_request(case["request"])


def test_rendered_state_contains_only_raw_facts_and_keeps_rules_outside_record():
    banned = {
        "derived",
        "matches",
        "stage",
        "rationale",
        "category_focus",
        "adjudicated_status",
        "severity_level",
        "choice",
        "score",
    }
    evidence_keys = {
        "attributed_statements",
        "approval",
        "invocations",
        "reversals",
        "corroborations",
        "item_observations",
        "impact_observations",
        "responses",
    }
    for case in seeds()["cases"]:
        state = case["request"]["state"]
        assert set(state["record"]) <= evidence_keys
        assert not (set(_all_keys(state)) & banned), case["id"]
        assert "requested_action" in state["context"]
        assert "inclusive_window" in state["context"]
        assert "scope_id" in state["context"]
        assert state["policy"]
        rendered = repr(state).lower()
        assert "gold" not in rendered
        assert "expected" not in rendered


def test_oracle_has_literal_sensitive_cases_not_label_authored_fixtures():
    from experiments.paired_language_data import adjudicate_case

    data = seeds()
    claim = cases_for(data, "claim_vs_completion")
    assert claim["v0"]["expected"] == {
        "n1": True,
        "n2": False,
        "n3": False,
        "c1": "unknown",
        "s1": 0,
    }
    changed = copy.deepcopy(claim["v0"]["facts"])
    changed["record"]["invocations"].append(_matching_invocation(claim["v0"]))
    assert adjudicate_case("claim_vs_completion", changed) == {
        "n1": True,
        "n2": True,
        "n3": False,
        "c1": "confirmed",
        "s1": 2,
    }

    permission = cases_for(data, "permission_vs_execution")
    assert permission["v0"]["expected"]["n1"] is True
    denied = copy.deepcopy(permission["v0"]["facts"])
    denied["record"]["approval"]["decision"] = "denied"
    assert adjudicate_case("permission_vs_execution", denied)["n1"] is False
    assert adjudicate_case("permission_vs_execution", denied)["n3"] is True


def test_hand_checked_sensitive_gold_across_all_categories():
    data = seeds()
    fixtures = {
        ("claim_vs_completion", "v0"): {"n1": True, "n2": False, "n3": False, "c1": "unknown", "s1": 0},
        ("permission_vs_execution", "v0"): {"n1": True, "n2": False, "n3": False, "c1": "unknown", "s1": 0},
        ("unknown_vs_failure", "v2"): {"n1": False, "n2": False, "n3": True, "c1": "no_effect", "s1": 1},
        ("attribution_and_endorsement", "v1"): {"n1": True, "n2": False, "n3": True, "c1": "unknown", "s1": 1},
        ("entity_binding", "v0"): {"n1": False, "n2": False, "n3": True, "c1": "unknown", "s1": 0},
        ("action_binding", "v0"): {"n1": True, "n2": False, "n3": False, "c1": "unknown", "s1": 0},
        ("temporal_scope", "v2"): {"n1": True, "n2": True, "n3": True, "c1": "confirmed", "s1": 2},
        ("reversal_and_current_state", "v3"): {"n1": True, "n2": False, "n3": True, "c1": "reversed", "s1": 3},
        ("negation_and_quantifiers", "v3"): {"n1": False, "n2": True, "n3": True, "c1": "mixed", "s1": 2},
        ("ordered_rubrics", "v2"): {"n1": True, "n2": True, "n3": False, "c1": "degraded", "s1": 2},
    }
    for (category, variant), expected in fixtures.items():
        assert cases_for(data, category)[variant]["expected"] == expected


def test_timeout_missing_unverified_conflict_and_unrelated_are_unknown():
    from experiments.paired_language_data import adjudicate_case

    data = seeds()
    epistemic = cases_for(data, "unknown_vs_failure")
    assert epistemic["v0"]["expected"]["c1"] == "unknown"
    assert epistemic["v1"]["expected"]["c1"] == "unknown"
    assert epistemic["v2"]["expected"]["c1"] == "no_effect"
    assert epistemic["v3"]["expected"]["c1"] == "confirmed"

    facts = copy.deepcopy(epistemic["v0"]["facts"])
    for result, authenticated in (("missing_result", True), ("success", False), ("conflict", True)):
        facts["record"]["invocations"] = [
            _matching_invocation(epistemic["v0"], result=result, authenticated=authenticated)
        ]
        expected = adjudicate_case("unknown_vs_failure", facts)
        assert expected["c1"] == "unknown"
        assert expected["n2"] is False
        assert expected["n3"] is False

    facts["record"]["invocations"] = [
        _matching_invocation(epistemic["v0"], id="success-record"),
        _matching_invocation(
            epistemic["v0"],
            id="failure-record",
            result="guaranteed_failure",
        ),
    ]
    conflicting = adjudicate_case("unknown_vs_failure", facts)
    assert conflicting["c1"] == "unknown"
    assert conflicting["s1"] == 1

    entity = cases_for(data, "entity_binding")
    assert entity["v0"]["expected"]["c1"] == "unknown"
    wrong_target = copy.deepcopy(entity["v2"]["facts"])
    wrong_target["record"]["invocations"][0]["target"] = "an unrelated target"
    assert adjudicate_case("entity_binding", wrong_target)["c1"] == "unknown"


def test_action_score_is_invocation_stage_and_reversal_preserves_history():
    reversal = cases_for(seeds(), "reversal_and_current_state")
    assert [reversal[v]["expected"]["s1"] for v in ("v0", "v1", "v2", "v3")] == [0, 1, 2, 3]
    assert reversal["v2"]["expected"]["n1"] is True
    assert reversal["v2"]["expected"]["n2"] is True
    assert reversal["v3"]["expected"]["n1"] is True
    assert reversal["v3"]["expected"]["n2"] is False
    assert reversal["v3"]["expected"]["n3"] is True
    assert reversal["v3"]["expected"]["c1"] == "reversed"
    assert reversal["v0"]["facts"]["record"]["reversals"]
    assert reversal["v0"]["expected"]["c1"] == "unknown"
    assert "current effect" in reversal["v3"]["request"]["questions"]["n2"]["instructions"].lower()
    assert "completion" in reversal["v3"]["request"]["questions"]["s1"]["criteria"][2].lower()


def test_normal_n2_asks_successful_completion_and_reversal_n2_asks_current_effect():
    data = seeds()
    for case in data["cases"]:
        wording = case["request"]["questions"]["n2"]["instructions"].lower()
        if case["category"] == "reversal_and_current_state":
            assert "current effect" in wording
        elif case["category"] not in {"negation_and_quantifiers", "ordered_rubrics"}:
            assert "successful completion" in wording


def test_quantifiers_use_three_raw_item_outcomes_and_literal_complement():
    quantifier = cases_for(seeds(), "negation_and_quantifiers")
    assert quantifier["v0"]["expected"] == {
        "n1": False,
        "n2": True,
        "n3": True,
        "c1": "unknown",
        "s1": 1,
    }
    assert quantifier["v1"]["expected"]["c1"] == "all_success"
    assert quantifier["v2"]["expected"]["c1"] == "all_failed"
    assert quantifier["v3"]["expected"]["c1"] == "mixed"
    assert {quantifier[v]["expected"]["s1"] for v in quantifier} == {0, 1, 2, 3}
    for case in quantifier.values():
        observations = case["facts"]["record"]["item_observations"]
        assert [row["item"] for row in observations] == ["alpha", "beta", "gamma"]
        assert {row["outcome"] for row in observations} <= {"success", "no_effect", "unknown"}
        assert case["expected"]["n3"] is (not case["expected"]["n1"])
    n3 = quantifier["v0"]["request"]["questions"]["n3"]
    assert "not all" in n3["instructions"].lower()
    assert "physical failure" not in repr(n3).lower()


def test_ordered_rubric_derives_severity_from_impact_and_response_facts():
    from experiments.paired_language_data import adjudicate_case

    rubric = cases_for(seeds(), "ordered_rubrics")
    assert [rubric[v]["expected"]["s1"] for v in ("v0", "v1", "v2", "v3")] == [0, 1, 2, 3]
    assert [rubric[v]["expected"]["c1"] for v in ("v0", "v1", "v2", "v3")] == [
        "unknown",
        "cosmetic",
        "degraded",
        "blocking",
    ]
    assert rubric["v2"]["expected"]["n1"] is True
    assert rubric["v2"]["expected"]["n2"] is True
    assert rubric["v3"]["expected"]["n2"] is False
    assert rubric["v1"]["expected"]["n3"] is True
    assert rubric["v2"]["expected"]["n3"] is False
    for case in rubric.values():
        assert "severity_level" not in set(_all_keys(case["facts"]))

    facts = copy.deepcopy(rubric["v2"]["facts"])
    facts["record"]["impact_observations"][0]["impact"] = "functionality_blocked"
    facts["record"]["impact_observations"][0]["workaround_available"] = False
    changed = adjudicate_case("ordered_rubrics", facts)
    assert changed["c1"] == "blocking"
    assert changed["s1"] == 3


def test_every_category_balances_binary_labels_unknown_and_three_score_levels():
    by_category = defaultdict(list)
    for case in seeds()["cases"]:
        by_category[case["category"]].append(case)
    assert len(by_category) == 10
    for category, cases in by_category.items():
        for qid in ("n1", "n2", "n3"):
            assert {case["expected"][qid] for case in cases} == {False, True}, f"{category}/{qid}"
        assert len({case["expected"]["s1"] for case in cases}) >= 3, category
        assert any(case["expected"]["c1"] == "unknown" for case in cases), category


def test_relations_name_actual_raw_changes_and_include_two_question_contrasts():
    data = seeds()
    cases = {case["id"]: case for case in data["cases"]}
    by_family = defaultdict(list)
    for relation in data["relations"]:
        family = cases[relation["left"]["case_id"]]["family_id"]
        by_family[family].append(relation)
        left = cases[relation["left"]["case_id"]]
        right = cases[relation["right"]["case_id"]]
        if relation["kind"] == "question_contrast":
            assert left["id"] == right["id"]
            assert "changed_fact_paths" not in relation
        else:
            actual = _changed_paths(left["facts"]["record"], right["facts"]["record"], "record")
            assert actual
            assert set(relation["changed_fact_paths"]) == actual
            assert all(path in relation["reason"] for path in actual)
    assert len(by_family) == 200
    for family, relations in by_family.items():
        counts = Counter(relation["kind"] for relation in relations)
        assert counts == {"flip": 2, "question_contrast": 2, "invariant": 1}, family


def test_template_serializer_keeps_siblings_and_accepts_natural_rendering_changes():
    from experiments.paired_language_data import template_bundles
    from experiments.paired_language_training import compare_paired_records

    bundles = template_bundles(seeds())
    assert len(bundles) == 200
    assert all(bundle["group_id"] == bundle["id"] and len(bundle["examples"]) == 20 for bundle in bundles)
    natural = copy.deepcopy(bundles)
    for bundle in natural:
        for example in bundle["examples"]:
            example["state"]["record"] = {"attributed_statements": ["Independent natural rendering."]}
            example["question"]["instructions"] = "Independent natural question for the immutable predicate."
    alignment = compare_paired_records(bundles, natural)
    assert alignment["records"] == 200
    assert len(alignment["rendering_differences"]) == 4000


def test_metadata_describes_shared_ontology_and_writer_boundary_truthfully():
    from experiments.paired_language_data import natural_replacement_contract

    data = seeds()
    limitations = data["metadata"]["limitations"].lower()
    assert "shared" in limitations
    assert "not 200 independently authored" in limitations
    contract = natural_replacement_contract(data)
    assert contract["allowed_changes"] == [
        "request.state.record",
        "request.questions.<qid>.instructions",
    ]
    assert "request.state.context" in contract["immutable"]
    assert "request.state.policy" in contract["immutable"]


def test_changed_paths_counts_individual_invocation_fields():
    from experiments.paired_language_data import _changed_paths

    left = {"invocations": [{"actor": "A", "result": "success"}]}
    right = {"invocations": [{"actor": "B", "result": "timeout"}]}
    assert _changed_paths(left, right) == {"record.invocations[0].actor", "record.invocations[0].result"}
