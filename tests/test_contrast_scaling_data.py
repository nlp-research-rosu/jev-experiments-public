import copy
import json

import pytest


CATEGORIES = (
    "claim_vs_completion", "permission_vs_execution", "unknown_vs_failure",
    "attribution_and_endorsement", "entity_binding", "action_binding",
    "temporal_scope", "reversal_and_current_state", "negation_and_quantifiers",
    "ordered_rubrics",
)


def _blueprint(category, number=0):
    return {
        "id": f"{category}/writer-{number}", "category": category,
        "scenario_domain": f"{category.replace('_', ' ')} desk",
        "operation": {"verb": "dispatch", "past": "dispatched", "noun": "dispatch"},
        "other_operation": {"verb": "archive", "past": "archived", "noun": "archive"},
        "target_template": "case-{identifier}",
        "sources": {"primary": "north desk", "secondary": "central desk", "untrusted": "south desk"},
        "evidence_heading": "Recorded observations", "question_prefix": "Using the supplied record,",
        "score_title": "Supplied review scale",
        "claim_template": "{speaker} says {actor} {past} {target}.",
        "intent_template": "{speaker} says {actor} will {verb} {target}.",
        "denial_template": "{speaker} denies that {actor} did {verb} {target}.",
        "endorsement_template": "{authority} endorses {speaker}'s statement that {actor} {past} {target}.",
    }


def _bank():
    return [_blueprint(category, number) for category in CATEGORIES for number in range(20)]


def test_build_family_derives_complete_hard_targets_and_keeps_private_labels_out_of_state():
    """Replacing oracle-derived labels with a hidden state field would make this fail."""
    from experiments.contrast_scaling_data import build_family

    family = build_family(_blueprint("claim_vs_completion"), 0)
    assert [case["variant"] for case in family["cases"]] == ["v0", "v1", "v2", "v3"]
    assert all(tuple(case["request"]["questions"]) == ("n1", "n2", "n3", "c1", "s1") for case in family["cases"])
    assert all(set(case["expected"]) == {"n1", "n2", "n3", "c1", "s1"} for case in family["cases"])
    assert all(isinstance(case["expected"][qid], bool) for case in family["cases"] for qid in ("n1", "n2", "n3"))
    encoded = json.dumps([case["request"]["state"] for case in family["cases"]], sort_keys=True)
    assert "raw_facts" not in encoded and "expected" not in encoded and "private_programs" not in encoded
    assert "raw_facts" in family["private_provenance"] and "programs" in family["private_provenance"]


def test_seed_changes_the_reproducible_latent_configuration_without_changing_family_identity():
    """Ignoring the seed would make nominally independent generation runs identical."""
    from experiments.contrast_scaling_data import build_family

    blueprint = _blueprint("claim_vs_completion")
    first, second = build_family(blueprint, 7, seed=42), build_family(blueprint, 7, seed=43)
    assert first["id"] == second["id"]
    assert first["private_provenance"]["semantic_configuration"] != second["private_provenance"]["semantic_configuration"]
    assert first["cases"][0]["request"]["state"] != second["cases"][0]["request"]["state"]


def test_same_facts_with_different_rubrics_changes_only_the_score_definition():
    """Changing facts to obtain a rubric flip would defeat the counterfactual."""
    from experiments.contrast_scaling_data import build_family

    family = build_family(_blueprint("ordered_rubrics"), 3)
    first, second = family["cases"][:2]
    assert first["request"]["state"]["evidence"] == second["request"]["state"]["evidence"]
    assert first["request"]["state"]["rules"] == second["request"]["state"]["rules"]
    assert first["request"]["state"]["choice_definitions"] == second["request"]["state"]["choice_definitions"]
    assert first["request"]["state"]["score_levels"] != second["request"]["state"]["score_levels"]
    assert first["expected"]["s1"] != second["expected"]["s1"]
    assert all("{\"feature\"" not in level for level in first["request"]["state"]["score_levels"] + second["request"]["state"]["score_levels"])
    assert family["private_provenance"]["programs"][0]["score"][0] is True


def test_training_bundles_preserve_cases_and_expose_only_model_inputs():
    """Putting provenance or relations into training units would leak evaluator metadata."""
    from experiments.contrast_scaling_data import build_family, training_bundles

    family = build_family(_blueprint("permission_vs_execution"), 1)
    bundle = training_bundles({"families": [family]})[0]
    assert len(bundle["examples"]) == 20
    assert bundle["relations"] == []
    assert [row["question_id"] for row in bundle["examples"]] == ["n1", "n2", "n3", "c1", "s1"] * 4
    assert all(set(row) == {"state", "question", "target", "case_id", "question_id"} for row in bundle["examples"])
    assert all(set(row["target"]) in ({"truth"}, {"choice"}, {"level_index"}) for row in bundle["examples"])
    assert "raw_facts" not in json.dumps(bundle, sort_keys=True)


def test_training_suite_is_deterministic_balanced_and_prefix_nested():
    """Grouping categories or changing the prefix order would invalidate scale comparisons."""
    from experiments.contrast_scaling_data import build_training_suite

    suite = build_training_suite(_bank(), families_per_category=20, seed=42)
    ids = suite["ordered_family_ids"]
    assert suite == build_training_suite(_bank(), families_per_category=20, seed=42)
    assert len(ids) == 200
    assert [family["category"] for family in suite["families"][:10]] == list(CATEGORIES)
    assert suite["nested_prefixes"]["200"] == ids
    assert len({family["private_provenance"]["semantic_signature"] for family in suite["families"]}) > 10


def test_semantic_configuration_cycle_has_five_hundred_non_name_variations_per_category():
    """A 25-program cycle cannot stand in for the requested 5,000 semantic families."""
    from experiments.contrast_scaling_data import build_training_suite

    suite = build_training_suite(_bank(), families_per_category=500)
    by_category = {category: [] for category in CATEGORIES}
    for family in suite["families"]:
        by_category[family["category"]].append(family["private_provenance"]["semantic_configuration"])
    assert all(set(configurations) == set(range(500)) for configurations in by_category.values())
    assert {family["private_provenance"]["evidence_shape"] for family in suite["families"]} == set(range(20))
    assert suite["diversity"]["semantic_configurations_per_category"] == 500
    assert suite["diversity"]["normalized_joint_fact_program_signatures"] == 5000
    assert suite["diversity"]["normalized_semantic_fact_signatures"] < 5000
    assert suite["diversity"]["semantic_program_signatures"] <= 5000
    assert set(suite["diversity"]["normalized_signature_counts"].values()) == {1}


def test_choice_rules_are_exclusive_and_declared_relations_match_oracle_targets():
    """An ambiguous Choice rule or manufactured relation must reject the family."""
    from experiments.contrast_scaling_data import build_family, validate_family
    from experiments.contrast_scaling_logic import derive_features, select_choice

    family = build_family(_blueprint("unknown_vs_failure"), 4)
    validate_family(family)
    for case, raw, programs in zip(family["cases"], family["private_provenance"]["raw_facts"], family["private_provenance"]["programs"]):
        assert select_choice(programs["choice"], derive_features(raw)) == case["expected"]["c1"]
    broken = copy.deepcopy(family)
    broken["relations"][0]["expected_equal"] = not broken["relations"][0]["expected_equal"]
    with pytest.raises(ValueError, match="relation"):
        validate_family(broken)


@pytest.mark.parametrize("category", ["entity_binding", "action_binding"])
def test_binding_choice_partitions_exact_results_from_other_success_reports(category):
    """Collapsing a trusted out-of-scope success report into unknown loses the binding distinction."""
    from experiments.contrast_scaling_data import build_family

    family = build_family(_blueprint(category), 0)
    first, third = family["cases"][0], family["cases"][2]
    assert set(first["request"]["questions"]["c1"]["criteria"]) == {
        "exact_success", "exact_failure", "unrelated_success", "unknown",
    }
    assert first["expected"]["c1"] == "unrelated_success"
    assert third["expected"]["c1"] == "exact_success"
    assert "different actor, operation, target, or run" in first["request"]["questions"]["c1"]["criteria"]["unrelated_success"]


def test_claims_denials_and_endorsements_are_rendered_messages_not_state_flags():
    """Replacing textual assertions with latent tense or endorsement flags would leak annotation machinery."""
    from experiments.contrast_scaling_data import build_family

    claim = build_family(_blueprint("claim_vs_completion"), 0)
    attribution = build_family(_blueprint("attribution_and_endorsement"), 0)
    denial_messages = claim["cases"][2]["request"]["state"]["evidence"]["messages"]
    endorsed_messages = attribution["cases"][2]["request"]["state"]["evidence"]["messages"]
    assert any("denies" in text for text in denial_messages)
    assert any("endorses" in text for text in endorsed_messages)
    assert attribution["cases"][0]["expected"]["n1"] is False
    assert attribution["cases"][2]["expected"]["n1"] is True


def test_placeholder_validation_rejects_unknown_format_fields_and_missing_bank_coverage():
    """Permissive template fields or incomplete banks can silently leak unsupported semantics."""
    from experiments.contrast_scaling_data import build_training_suite, validate_blueprint

    broken = _blueprint("claim_vs_completion")
    broken["claim_template"] = "{actor} {unknown}"
    with pytest.raises(ValueError, match="placeholder"):
        validate_blueprint(broken)
    with pytest.raises(ValueError, match="20 blueprints"):
        build_training_suite([_blueprint(category) for category in CATEGORIES], families_per_category=1)


def test_reversal_and_ordered_rubric_questions_match_hand_adjudicated_predicates():
    """Keeping question prose separate from its private program swaps the first predicates."""
    from experiments.contrast_scaling_data import build_family

    reversal = build_family(_blueprint("reversal_and_current_state"), 0)["cases"][0]
    assert [reversal["expected"][qid] for qid in ("n1", "n2", "n3")] == [True, False, True]
    assert "historical completion" in reversal["request"]["questions"]["n1"]["instructions"]
    assert "effect remains established" in reversal["request"]["questions"]["n2"]["instructions"]
    assert "successful reversal" in reversal["request"]["questions"]["n3"]["instructions"]

    ordered = build_family(_blueprint("ordered_rubrics"), 0)["cases"][0]
    assert [ordered["expected"][qid] for qid in ("n1", "n2", "n3")] == [True, False, True]
    assert "finite measurement is recorded" in ordered["request"]["questions"]["n1"]["instructions"]
    assert "recorded measurement is >= 50" in ordered["request"]["questions"]["n2"]["instructions"]


def test_public_state_exposes_every_raw_premise_needed_to_reproduce_gold():
    """Dropping query time, inventory scope, or reversal bindings makes public adjudication impossible."""
    from experiments.contrast_scaling_data import build_family

    temporal_family = build_family(_blueprint("temporal_scope"), 0)
    temporal = temporal_family["cases"][0]["request"]["state"]
    assert temporal["evidence"]["query_time"] == temporal_family["private_provenance"]["raw_facts"][0]["query_time"]
    assert "latest admissible state event" in temporal["rules"]["interpretation"]["state"]

    inventory_family = build_family(_blueprint("negation_and_quantifiers"), 0)
    inventory = inventory_family["cases"][0]["request"]["state"]
    assert inventory["evidence"]["inventory_scope"] == {
        "item_ids": inventory_family["private_provenance"]["raw_facts"][0]["item_ids"],
        "complete": True,
    }
    assert "declared item IDs" in inventory["rules"]["interpretation"]["inventory"]

    reversal = build_family(_blueprint("reversal_and_current_state"), 0)["cases"][0]["request"]["state"]
    record = reversal["evidence"]["reversal_records"][0]
    assert record["actor"] == reversal["rules"]["scope"]["actor"]
    assert record["operation"] == reversal["rules"]["scope"]["operation"]
    assert "reverses" in reversal["rules"]["interpretation"]["reversal"]


def test_boolean_rule_renderer_preserves_nested_scope_with_explicit_grouping():
    """Flattening mixed all/any/not expressions changes their ordinary-language truth conditions."""
    from experiments.contrast_scaling_data import _rule_text

    rule = {
        "all": [
            {"any": [{"feature": "claim"}, {"feature": "completed"}]},
            {"not": {"feature": "completed"}},
        ]
    }
    assert _rule_text(rule) == (
        "((the record contains an affirmative completed claim for the scoped action OR the record establishes historical "
        "completion for the exact scoped action from admissible evidence) AND (NOT (the record establishes historical "
        "completion for the exact scoped action from admissible evidence)))"
    )


def test_negated_completion_gloss_keeps_unknown_world_state_epistemically_open():
    """Rendering not(completed) as non-occurrence would turn absent proof into a false world fact."""
    from experiments.contrast_scaling_data import _rule_text, build_family

    case = build_family(_blueprint("unknown_vs_failure"), 0)["cases"][0]
    assert case["expected"]["c1"] == "unknown"
    rendered = _rule_text({"not": {"feature": "completed"}})
    assert rendered == (
        "(NOT (the record establishes historical completion for the exact scoped action from admissible evidence))"
    )
    assert "did not happen" not in rendered
    incomplete_scope = _rule_text({"not": {"feature": "all_success"}})
    assert incomplete_scope == (
        "(NOT (the explicitly complete inventory record establishes verified success for every declared item))"
    )
    assert "failed" not in incomplete_scope


def test_ordered_rubric_score_flip_is_caused_by_visible_numeric_cutoffs():
    """Boolean side conditions cannot stand in for the promised variable numeric rubric scale."""
    from experiments.contrast_scaling_data import build_family

    family = build_family(_blueprint("ordered_rubrics"), 0)
    raw = family["private_provenance"]["raw_facts"][0]
    score_a = family["private_provenance"]["programs"][0]["score"]
    score_b = family["private_provenance"]["programs"][1]["score"]
    assert set(score_a[1]) == {"compare"}
    assert set(score_b[1]) == {"compare"}
    assert score_a[1]["compare"]["right"] == raw["measurement"]
    assert score_b[1]["compare"]["right"] > raw["measurement"]
    assert family["cases"][0]["expected"]["s1"] != family["cases"][1]["expected"]["s1"]


def test_generated_records_cover_conflict_corroboration_and_inventory_unknown_branches():
    """Nominal shapes made only of irrelevant execution rows leave promised oracle branches dead."""
    from experiments.contrast_scaling_data import build_family
    from experiments.contrast_scaling_logic import derive_features

    outcome_features = []
    quantifier_features = []
    quantifier_raw = []
    for instance in range(20):
        outcome = build_family(_blueprint("claim_vs_completion"), instance)
        outcome_features.extend(derive_features(raw) for raw in outcome["private_provenance"]["raw_facts"])
        quantifier = build_family(_blueprint("negation_and_quantifiers"), instance)
        quantifier_features.extend(derive_features(raw) for raw in quantifier["private_provenance"]["raw_facts"])
        quantifier_raw.extend(quantifier["private_provenance"]["raw_facts"])

    assert {features["conflict"] for features in outcome_features} == {False, True}
    assert {features["corroborated"] for features in outcome_features} == {False, True}
    assert {features["item_unknown"] for features in quantifier_features} == {False, True}
    assert {features["inventory_complete"] for features in quantifier_features} == {False, True}
    assert any(len({row["id"] for row in raw["items"]}) < len(raw["item_ids"]) for raw in quantifier_raw)
    assert any(any(row["verified"] is False for row in raw["items"]) for raw in quantifier_raw)
    assert any(
        len({row["status"] for row in raw["items"] if row["id"] == item_id and row["verified"] is True}) > 1
        for raw in quantifier_raw for item_id in raw["item_ids"]
    )


def test_surface_roles_rotate_without_outcome_markers_or_counterfactual_drift():
    """Fixed Mira/other-* bindings let a model memorize surface tokens instead of exact scope."""
    from experiments.contrast_scaling_data import build_family

    blueprint = _blueprint("entity_binding")
    families = [build_family(blueprint, instance) for instance in range(20)]
    requested_actors = set()
    requested_operations = set()
    trusted_source_sets = set()
    for family in families:
        raws = family["private_provenance"]["raw_facts"]
        requested_actors.add(raws[0]["query"]["actor"])
        requested_operations.add(raws[0]["query"]["operation"])
        trusted_source_sets.add(tuple(sorted(raws[0]["trusted_sources"])))
        assert all(raw["query"] == raws[0]["query"] for raw in raws)
        assert all(raw["trusted_sources"] == raws[0]["trusted_sources"] for raw in raws)
        encoded = json.dumps(raws, sort_keys=True)
        assert "other-target" not in encoded
        assert "other-run" not in encoded
        assert "item-a" not in encoded and "item-b" not in encoded
        for raw in raws:
            assert all(row["target"].startswith("case-") for row in raw["executions"])

    assert len(requested_actors) > 1
    assert requested_operations == {"dispatch", "archive"}
    assert len(trusted_source_sets) > 1


def test_surface_permutations_preserve_oracle_labels_and_normalized_signature():
    """Neutral renaming is an invariance and must not inflate semantic diversity."""
    from experiments.contrast_scaling_data import build_family

    first = build_family(_blueprint("claim_vs_completion", 0), 7)
    second = build_family(_blueprint("claim_vs_completion", 1), 7)
    assert first["private_provenance"]["raw_facts"][0] != second["private_provenance"]["raw_facts"][0]
    assert [case["expected"] for case in first["cases"]] == [case["expected"] for case in second["cases"]]
    assert first["private_provenance"]["semantic_signature"] == second["private_provenance"]["semantic_signature"]


@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.parametrize("instance", [0, 3, 19, 20, 99, 499])
def test_every_category_and_representative_configuration_passes_canonical_validation(category, instance):
    """Local shape validation cannot substitute for the unchanged evaluator contract."""
    from experiments.contrast_scaling_data import build_family
    from experiments.revised_metrics import validate_suite

    family = build_family(_blueprint(category), instance)
    validate_suite({"cases": family["cases"], "relations": family["relations"]})
    for case in family["cases"]:
        assert set(case["rationale"]) == set(case["expected"])
        assert len(case["request"]["questions"]["s1"]["criteria"]) == len(
            family["cases"][0]["request"]["questions"]["s1"]["criteria"]
        )
    for relation in family["relations"]:
        if relation["kind"] == "question_contrast":
            assert relation["left"]["case_id"] == relation["right"]["case_id"]
            left = next(case for case in family["cases"] if case["id"] == relation["left"]["case_id"])
            assert left["expected"][relation["left"]["question_id"]] != left["expected"][relation["right"]["question_id"]]


def test_nested_prefixes_are_stratified_over_evidence_shapes_and_rule_forms():
    """Mapping ordered instances directly to configurations confounds data scale with complexity."""
    from experiments.contrast_scaling_data import build_training_suite

    suite = build_training_suite(_bank(), families_per_category=100, seed=42)
    small = build_training_suite(_bank(), families_per_category=20, seed=42)
    assert suite["ordered_family_ids"][:200] == small["ordered_family_ids"]
    for prefix_size, expected_rule_forms in ((200, 20), (1000, 25)):
        prefix = suite["families"][:prefix_size]
        for category in CATEGORIES:
            rows = [family["private_provenance"] for family in prefix if family["category"] == category]
            assert len({row["evidence_shape"] for row in rows}) == 20
            assert len({row["rule_form"] for row in rows}) == expected_rule_forms
        coverage = suite["diversity"]["prefix_configuration_coverage"][str(prefix_size)]
        assert set(coverage["evidence_shapes_by_category"].values()) == {20}
        assert set(coverage["rule_forms_by_category"].values()) == {expected_rule_forms}


def test_semantic_signature_ignores_pure_source_renames():
    """Source aliases must encode trusted membership and equality, not spelling."""
    from experiments.contrast_scaling_data import _signature, build_family

    family = build_family(_blueprint("claim_vs_completion"), 0)
    raw = family["private_provenance"]["raw_facts"][0]
    programs = family["private_provenance"]["programs"][0]
    renamed = copy.deepcopy(raw)
    aliases = {"north desk": "primary ledger", "central desk": "secondary ledger", "south desk": "street wire"}
    renamed["trusted_sources"] = [aliases[value] for value in renamed["trusted_sources"]]
    for collection in ("statements", "approvals", "executions", "reversals", "corroborations", "state_events"):
        for row in renamed[collection]:
            row["source"] = aliases[row["source"]]
    assert _signature(raw, programs) == _signature(renamed, programs)


@pytest.mark.parametrize("binding", ["actor", "operation", "target", "run_id"])
def test_semantic_signature_preserves_exact_requested_bindings(binding):
    """Collapsing role values hides exact-request versus other-binding semantics."""
    from experiments.contrast_scaling_data import _signature, build_family

    family = build_family(_blueprint("claim_vs_completion"), 0)
    raw = copy.deepcopy(family["private_provenance"]["raw_facts"][2])
    programs = family["private_provenance"]["programs"][2]
    changed = copy.deepcopy(raw)
    changed["executions"][0][binding] = f"other-{binding}"
    assert _signature(raw, programs) != _signature(changed, programs)


def test_semantic_signature_preserves_source_trust_membership():
    """Trust membership is semantic even when source spelling is not."""
    from experiments.contrast_scaling_data import _signature, build_family

    family = build_family(_blueprint("claim_vs_completion"), 0)
    raw = family["private_provenance"]["raw_facts"][0]
    programs = family["private_provenance"]["programs"][0]
    changed = copy.deepcopy(raw)
    changed["trusted_sources"] = ["central desk", "south desk"]
    assert _signature(raw, programs) != _signature(changed, programs)


def test_semantic_signature_ignores_query_identity_renames_and_nonce_values():
    """Family identity strings and renamed domain entities are not semantic diversity."""
    from experiments.contrast_scaling_data import _signature, build_family

    family = build_family(_blueprint("reversal_and_current_state"), 0)
    raw = family["private_provenance"]["raw_facts"][0]
    programs = family["private_provenance"]["programs"][0]
    renamed = copy.deepcopy(raw)
    replacements = {
        "actor": (raw["query"]["actor"], "requested-person"),
        "operation": (raw["query"]["operation"], "requested-operation"),
        "target": (raw["query"]["target"], "requested-target"),
        "run_id": (raw["query"]["run_id"], "requested-run"),
        "reverses": (raw["query"]["run_id"], "requested-run"),
    }
    for key, (old, new) in replacements.items():
        if renamed["query"].get(key) == old:
            renamed["query"][key] = new
        for collection in ("statements", "approvals", "executions", "reversals", "corroborations"):
            for row in renamed[collection]:
                if row.get(key) == old:
                    row[key] = new
    assert _signature(raw, programs) == _signature(renamed, programs)


def test_full_population_is_200_language_blueprints_times_25_distinct_semantic_instances():
    """A nonce or template count cannot stand in for normalized fact/program diversity."""
    from experiments.contrast_scaling_data import _signature, build_training_suite
    from experiments.revised_metrics import validate_suite

    suite = build_training_suite(_bank(), families_per_category=500)
    assert suite["diversity"]["authored_language_blueprints"] == 200
    assert suite["diversity"]["semantic_instances_per_blueprint"] == 25
    assert suite["diversity"]["normalized_semantic_fact_program_families"] == 5000
    assert suite["diversity"]["normalized_joint_fact_program_signatures"] == 5000
    assert suite["diversity"]["surface_invariance"]["counted_as_semantic_diversity"] is False
    assert suite["diversity"]["normalized_semantic_fact_signatures"] == len({
        _signature(family["private_provenance"]["raw_facts"][0], {})
        for family in suite["families"]
    })
    assert suite["diversity"]["semantic_program_signatures"] == len({
        json.dumps(family["private_provenance"]["programs"][0], sort_keys=True)
        for family in suite["families"]
    })
    assert "not 5,000 independent language templates" in suite["metadata"]["limitations"]
    for size in ("200", "1000", "5000"):
        coverage = suite["diversity"]["semantic_coverage_by_prefix"][size]
        assert coverage["numeric_threshold_truths"]["true"] > 0
        assert coverage["numeric_threshold_truths"]["false"] > 0
        assert coverage["distinct_numeric_cutoffs"] > 1
        assert coverage["distinct_measurements"] > 1
        assert all(coverage["numeric_threshold_positions"][position] > 0 for position in ("below", "boundary", "above"))
        assert coverage["same_evidence_numeric_score_flips"] > 0
        assert coverage["conflicting_outcome_unknown"] > 0
        assert coverage["conflicting_choice_unknown"] > 0
        assert coverage["corroborated_score_operand_true"] > 0
        for feature in ("completed", "failed", "unknown", "conflict", "corroborated", "inventory_complete", "item_unknown"):
            assert coverage["feature_truths"][feature]["true"] > 0
            assert coverage["feature_truths"][feature]["false"] > 0
        for branch in ("missing_item", "unverified_item", "conflicting_item", "incomplete_inventory"):
            assert coverage["inventory_branches"][branch] > 0
    validate_suite({
        "cases": [case for family in suite["families"] for case in family["cases"]],
        "relations": [relation for family in suite["families"] for relation in family["relations"]],
    })
    for family in suite["families"]:
        v0, v1, v2, v3 = family["cases"]
        assert v0["request"]["state"]["evidence"] == v1["request"]["state"]["evidence"]
        assert v2["request"]["state"]["evidence"] == v3["request"]["state"]["evidence"]
        assert v0["request"]["state"]["rules"] == v2["request"]["state"]["rules"]
        assert v1["request"]["state"]["rules"] == v3["request"]["state"]["rules"]
        assert v0["request"]["questions"] == v2["request"]["questions"]
        assert v1["request"]["questions"] == v3["request"]["questions"]
