"""Independent raw-observation counterexamples and factorial contract checks."""

import copy
import importlib.util
import json
import re
import string
from collections import defaultdict
from pathlib import Path

import pytest


def factory():
    assert importlib.util.find_spec("experiments.contrast_factorial_data") is not None, "semantic factory missing"
    from experiments import contrast_factorial_data
    return contrast_factorial_data


def raw():
    return {"query": {"actor": "Ada", "operation": "seal", "target": "box", "run_id": "r1"},
            "window": [0, 10], "query_time": 3, "trusted_sources": ["ledger", "inspector"],
            "statements": [], "executions": [], "approvals": [], "corroborations": [], "reversals": [],
            "state_events": [], "item_ids": ["a", "b", "c", "d"], "items": [], "inventory_complete": True,
            "measurement": None}


def execution(**changes):
    return {"actor": "Ada", "operation": "seal", "target": "box", "run_id": "r1", "source": "ledger",
            "time": 5, "verified": True, "mode": "live", "outcome": "success", **changes}


def test_claim_and_completion_are_independent_and_untrusted_success_is_not_completion():
    f = factory()
    for claimed in (False, True):
        for completed in (False, True):
            r = raw()
            if claimed:
                r["statements"] = [{**r["query"], "source": "speaker", "time": 4, "tense": "completed",
                                    "polarity": "positive", "endorsed": False}]
            if completed:
                r["executions"] = [execution()]
            got = f.derive_features(r)
            assert (got["claim"], got["completed"]) == (claimed, completed)
    r["executions"] = [execution(verified=False)]
    assert not f.derive_features(r)["completed"]
    assert f.derive_features(r)["unknown"]


@pytest.mark.parametrize("changes", [{"actor": "Bea"}, {"operation": "open"}, {"target": "crate"},
                                    {"run_id": "r2"}, {"time": 11}, {"mode": "simulation"},
                                    {"source": "untrusted"}, {"outcome": "timeout"}])
def test_binding_provenance_and_timeout_counterexamples(changes):
    f = factory()
    r = raw()
    r["executions"] = [execution(**changes)]
    assert f.derive_features(r)["completed"] is False
    assert f.derive_features(r)["failed"] is False


def test_conflict_is_unknown_and_wrong_target_reversal_does_not_erase_success():
    f = factory()
    r = raw()
    r["executions"] = [execution(), execution(outcome="no_effect")]
    got = f.derive_features(r)
    assert got["unknown"] and got["conflict"] and not got["failed"]
    r["executions"] = [execution()]
    r["reversals"] = [execution(target="crate", reverses="r1", time=8)]
    assert f.derive_features(r)["effective"]
    r["reversals"][0]["target"] = "box"
    assert f.derive_features(r)["completed"] and not f.derive_features(r)["effective"]


@pytest.mark.parametrize("changes", [{"reverses": "r2"}, {"outcome": "no_effect"}, {"time": 4},
                                    {"verified": False}, {"source": "untrusted"}])
def test_failed_or_unbound_reversal_does_not_remove_effect(changes):
    f = factory()
    r = raw()
    r["executions"] = [execution()]
    r["reversals"] = [{**execution(reverses="r1", time=8), **changes}]
    assert f.derive_features(r)["effective"]
    assert not f.derive_features(r)["reversed"]


def test_permission_time_inventory_and_current_conflict_are_not_claim_shortcuts():
    f = factory()
    r = raw()
    r["approvals"] = [execution(decision="allow", time=2), execution(decision="deny", time=7)]
    r["executions"] = [execution()]
    got = f.derive_features(r)
    assert got["forbidden"] and got["completed"] and not got["approved"]
    r["items"] = [{"id": x, "verified": True, "status": "success"} for x in r["item_ids"]]
    r["inventory_complete"] = False
    assert not f.derive_features(r)["all_success"]
    r["state_events"] = [execution(time=1, active=True), execution(time=9, active=True),
                         execution(time=9, active=False)]
    got = f.derive_features(r)
    assert got["ever_active"] and got["active_earlier"] and not got["current_known"]


def toy_card(category, identity="toy", author="synthetic"):
    templates = {
        "approval": "{source} at {time}: {actor} / {operation} / {target}; decision={decision}; verified={verified}.",
        "corroboration": "At {time}, {source} corroboration for {actor} / {operation} / {target}, run {run_id}: verified={verified}; mode={mode}; outcome={outcome}.",
        "reversal": "At {time}, {source} reversal for {actor} / {operation} / {target}, run {run_id}, reverses={reverses}: verified={verified}; mode={mode}; outcome={outcome}.",
        "state_event": "{source} at {time} reports target={target}, active={active}, verified={verified}.",
        "item": "Item {id} has reported status={status}, verified={verified}.",
        "measurement": "The recorded measurement is {value}.",
    }
    required = {"entity_binding": [], "action_binding": [], "temporal_scope": ["state_event"],
                "reversal_and_current_state": ["approval", "corroboration", "reversal"],
                "negation_and_quantifiers": ["item"], "ordered_rubrics": ["measurement"]}.get(
                    category, ["approval", "corroboration"])
    return {"schema_version": 1, "id": identity, "author": author, "category": category,
            "heading": "Synthetic recorded observations",
            "claim_template": "At {time}, {source} states that {actor} carried out {operation} on {target}.",
            "intent_template": "At {time}, {source} says {actor} intends to carry out {operation} on {target}.",
            "denial_template": "At {time}, {source} denies that {actor} carried out {operation} on {target}.",
            "endorsement_template": "At {time}, {source} endorses the assertion that {actor} carried out {operation} on {target}.",
            "execution_template": "Run {run_id}: {actor} / {operation} / {target}; {source}, {time}; verified={verified}; mode={mode}; outcome={outcome}.",
            "observation_templates": {key: templates[key] for key in required},
            "construction_note": "Clearly synthetic test fixture, never an approved authored card."}


def toy_bank():
    return [toy_card(c, f"toy/{a}/{c}/{i}", a) for c in factory().CATEGORIES
            for a in ("synthetic-a", "synthetic-b") for i in range(20)]


def test_templates_keep_annotation_flags_private_and_reject_unsafe_placeholders():
    f = factory()
    card = toy_card("claim_vs_completion")
    f.validate_blueprint(card)
    for bad in ("{actor.name}", "{actor!r}", "{actor:20}", "{actor}{actor}", "{gold}"):
        broken = copy.deepcopy(card)
        broken["claim_template"] = broken["claim_template"].replace("{actor}", bad)
        with pytest.raises(ValueError):
            f.validate_blueprint(broken)
    r = raw()
    r["statements"] = [{**r["query"], "time": 1, "source": "speaker", "tense": "completed",
                         "polarity": "positive", "endorsed": True}]
    evidence = f.render_evidence(r, card)
    text = json.dumps(evidence)
    assert "endorses the assertion" in text
    assert all(secret not in text for secret in ("tense", "polarity", "endorsed", "construction_note", "synthetic-a"))


def test_operational_rows_have_exactly_the_fields_preserved_by_their_template():
    f = factory()
    fields = {"executions": f.EXECUTION_SLOTS, "approvals": f.OBSERVATION_SLOTS["approval"],
              "reversals": f.OBSERVATION_SLOTS["reversal"], "corroborations": f.OBSERVATION_SLOTS["corroboration"],
              "items": f.OBSERVATION_SLOTS["item"], "state_events": f.OBSERVATION_SLOTS["state_event"]}
    for category in f.CATEGORIES:
        for record in f._worlds(category)[0]:
            for kind, expected in fields.items():
                for row in record[kind]:
                    assert set(row) == expected


def test_twenty_family_prototype_has_real_contrasts_and_legal_witnesses():
    f = factory()
    for category in f.CATEGORIES:
        for i in (0, 1):
            plan = f.make_plan(category, i, 2 + i, 4 + i)
            assert len(plan["records"]) == 2
            assert plan["records"][0] != plan["records"][1]
            for arm in ("narrow", "broad"):
                for program in plan["programs"][arm]:
                    witnesses = program["witnesses"]
                    assert set(map(int, witnesses)) == set(range(len(program["rules"])))
                    for level, witness in witnesses.items():
                        assert f.select_level(program["rules"], f.derive_features(witness)) == int(level)
            family = f.render_family(plan, toy_card(category), "broad")
            from experiments.revised_metrics import validate_suite
            validate_suite(family)
            assert {r["contrast_axis"] for r in family["relations"]} >= {"evidence", "rubric", "question"}


@pytest.fixture(scope="module")
def matched():
    return factory().build_matched_suites(toy_bank())


def test_all_factor_matches_levels_and_compiler_inputs(matched):
    f = factory()
    audit = f.validate_matched_suites(matched)
    assert audit["families"] == 400 and audit["cases_per_arm"] == 1600
    assert audit["shared_language_families"] == 200
    assert all(count > 0 for levels in audit["level_coverage"].values() for count in levels.values())
    assert all(count > 0 for levels in audit["variant_level_coverage"].values() for count in levels.values())
    assert all(len(cells) == 4 for cells in audit["independent_fact_crosses"].values())
    for arm, suite in matched.items():
        bundles = f.training_bundles(suite, arm)
        assert len(bundles) == 400 and all(len(b["examples"]) == 20 for b in bundles)
        assert all(b["provenance"]["assigned_split"] == "train" for b in bundles)
    changed = copy.deepcopy(matched)
    changed["C"]["cases"][0]["expected"]["n1"] = not changed["C"]["cases"][0]["expected"]["n1"]
    with pytest.raises(ValueError):
        f.validate_matched_suites(changed)


def test_emission_refuses_missing_freeze_and_unreviewed_bank(tmp_path):
    f = factory()
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"schema_version": 1, "blueprints": toy_bank()}))
    with pytest.raises(ValueError, match="freeze|repository root"):
        f.emit_study(tmp_path, "bank.json", "missing.json", "base", "output")
    (tmp_path / "freeze.json").write_text("{}")
    with pytest.raises(ValueError, match="freeze|approval|repository root"):
        f.emit_study(tmp_path, "bank.json", "freeze.json", "base", "output")
    assert not (tmp_path / "output").exists()


def test_hand_adjudicated_five_level_rubrics_are_distinct_semantic_rules():
    f = factory()

    def levels(category, record):
        return tuple(f.select_level(f._program(category, 5, 0, v)["rules"], f.derive_features(record)) for v in (0, 1))

    r = raw()
    r["statements"] = [{**r["query"], "source": "speaker", "time": 4, "tense": "completed",
                         "polarity": "positive", "endorsed": True}]
    r["approvals"] = [execution(decision="allow")]
    assert levels("claim_vs_completion", r) == (2, 1)  # assertion+adoption versus permission alone
    r = raw()
    r["executions"] = [execution(verified=False)]
    assert levels("entity_binding", r) == (3, 2)  # success report, but not established execution
    r["executions"] = [execution(outcome="timeout")]
    assert levels("unknown_vs_failure", r) == (1, 1)
    r["executions"] = [execution(outcome="no_effect")]
    assert levels("unknown_vs_failure", r) == (3, 2)  # resolved outcome versus successful completion
    r = raw()
    r["items"] = [{"id": "a", "verified": True, "status": "success"},
                  {"id": "b", "verified": True, "status": "success"},
                  {"id": "c", "verified": True, "status": "no_effect"}]
    assert levels("negation_and_quantifiers", r) == (2, 3)  # two successful, three resolved
    r = raw()
    r["state_events"] = [execution(time=2, active=False), execution(time=8, active=True)]
    assert levels("temporal_scope", r) == (3, 4)
    r = raw()
    r["measurement"] = 40
    assert levels("ordered_rubrics", r) == (2, 1)  # 40 threshold versus 47 threshold
    r["measurement"] = None
    assert levels("ordered_rubrics", r) == (0, 0)


def test_reversal_conditions_are_visible_in_every_candidate_branch():
    f = factory()
    plan = f.make_plan("reversal_and_current_state", 3, 3, 5)
    family = f.render_family(plan, toy_card("reversal_and_current_state"), "broad")
    from openjev.judgments import compile_request
    for case in family["cases"]:
        for unit in compile_request(case["request"]).units:
            rules = unit.payload["state"]["rules"]
            assert "reverses" in rules and "no earlier" in rules and "actor/action/target" in rules


def test_validator_recomputes_oracle_and_evidence(matched):
    f = factory()
    changed = copy.deepcopy(matched)
    for arm in "ABCD":
        family = changed[arm]["families"][0]
        family["private_provenance"]["plan"]["records"][0]["measurement"] = 999
    with pytest.raises(ValueError, match="fact"):
        f.validate_matched_suites(changed)
    changed = copy.deepcopy(matched)
    for arm in "ABCD":
        changed[arm]["cases"][0]["request"]["state"]["evidence"]["heading"] = "Manufactured observation"
    with pytest.raises(ValueError, match="render|evidence"):
        f.validate_matched_suites(changed)
    changed = copy.deepcopy(matched)
    for arm in "ABCD":
        changed[arm]["cases"] = copy.deepcopy(changed[arm]["cases"])
        changed[arm]["cases"][0]["request"]["state"]["evidence"]["heading"] = "Detached suite view"
    with pytest.raises(ValueError, match="flattened|family"):
        f.validate_matched_suites(changed)


def test_every_family_has_nontrivial_language_content(matched):
    f = factory()
    for family in matched["A"]["families"]:
        card = toy_card(family["category"])
        alternate = copy.deepcopy(card)
        alternate["execution_template"] = "Source {source} logged outcome={outcome} for {actor} undertaking {operation} on {target}; time={time}, run={run_id}, mode={mode}, verified={verified}."
        for key in alternate["observation_templates"]:
            alternate["observation_templates"][key] = "Observation received: " + alternate["observation_templates"][key]
        changed = False
        for record in family["private_provenance"]["plan"]["records"]:
            original, other = f.render_evidence(record, card), f.render_evidence(record, alternate)
            primary = {k: v for k, v in original.items() if k not in {"heading", "statements"}}
            other_primary = {k: v for k, v in other.items() if k not in {"heading", "statements"}}
            changed |= primary != other_primary
        assert changed, family["id"]


def test_language_quality_ignores_ids_headings_and_punctuation(matched):
    f = factory()
    cards = toy_bank()
    cards[0]["heading"] = "A different heading"
    cards[1]["execution_template"] += "!"
    report = f.language_quality_report(cards)
    assert report["blueprint_ids"] == 400
    assert all(row["distinct_primary_constructions"] == 1 for row in report["categories"].values())
    with pytest.raises(ValueError, match="language"):
        f.validate_matched_suites(matched, require_language_variation=True)


def test_training_bundles_match_actual_compiler_and_hard_target_interface(matched):
    from openjev.judgment_training import prepare_bundle, target_vector
    from openjev.judgments import compile_request, render_unit_messages

    class CPUCompiler:
        def prepare(self, request):
            compiled = compile_request(request)
            # UTF-8 bytes are a reversible serializer probe, not a tokenizer preflight.
            encoded = [list(json.dumps(render_unit_messages(unit), sort_keys=True).encode()) for unit in compiled.units]
            return compiled, encoded, [unit.readout_kind for unit in compiled.units]

    f = factory()
    for arm in "ABCD":
        bundles = f.training_bundles(matched[arm], arm)
        for index in range(40):
            bundle = prepare_bundle(CPUCompiler(), bundles[index])
            assert len(bundle.groups) == 20
            assert [g.primitive for g in bundle.groups] == ["noul", "noul", "noul", "choice", "score"] * 4
            assert all(set(target_vector(g)) <= {0.0, 1.0} and sum(target_vector(g)) == 1 for g in bundle.groups)
            family = matched[arm]["families"][index]
            expected = [list(json.dumps(render_unit_messages(unit), sort_keys=True).encode())
                        for case in family["cases"] for unit in compile_request(case["request"]).units]
            assert bundle.prompts == expected
            assert bundle.relations == []


def test_symmetric_aliases_and_affine_clock_preserve_all_raw_semantics(matched):
    f = factory()
    roles = defaultdict(lambda: defaultdict(set))
    clocks = set()
    shortcut_correct = positives = total = 0
    for family in matched["A"]["families"]:
        plan = family["private_provenance"]["plan"]
        assert "surface_context" in plan, "reversible role-exchangeable context missing"
        context = plan["surface_context"]
        assert {v: k for k, v in context["aliases"].items()} == context["inverse_aliases"]
        clocks.add((context["time_scale"], context["time_offset"]))
        for role, pair in {"actor": ("operator", "other operator"), "operation": ("seal", "open"),
                           "target": ("parcel", "crate"), "run": ("run", "other run")}.items():
            for index, original in enumerate(pair):
                roles[role][context["aliases"][original]].add(index)
        for original in ("ledger", "inspector", "untrusted", "speaker"):
            roles["source"][context["aliases"][original]].add(original in {"ledger", "inspector"})
        for world_index, record in zip(plan["world_indices"], plan["records"]):
            normalized = f.normalize_record(record, context)
            assert normalized == f._worlds(plan["category"])[0][world_index]
            assert f.derive_features(record) == f.derive_features(normalized)
            assert f.transform_record(normalized, context) == record
            if plan["category"] in {"entity_binding", "action_binding"}:
                accepted = [r for r in record["executions"] if r["actor"].startswith("operator-")
                            and r["operation"] == "seal" and r["target"].startswith("parcel-")
                            and r["run_id"].startswith("run-") and r["source"] in {"ledger", "inspector"}
                            and r["verified"] and r["mode"] == "live"]
                predicted = any(r["outcome"] == "success" for r in accepted) and not any(r["outcome"] == "no_effect" for r in accepted)
                truth = f.derive_features(record)["completed"]
                shortcut_correct += predicted == truth
                positives += truth
                total += 1
    assert total == 160 and positives > 0 and shortcut_correct < total
    assert len(clocks) > 20
    assert all(len(values) == 2 for group in roles.values() for values in group.values())


def test_alias_and_clock_values_survive_primary_template_rendering(matched):
    f = factory()
    mapping = {"executions": ("execution_template", f.EXECUTION_SLOTS),
               "approvals": ("approval", f.OBSERVATION_SLOTS["approval"]),
               "corroborations": ("corroboration", f.OBSERVATION_SLOTS["corroboration"]),
               "reversals": ("reversal", f.OBSERVATION_SLOTS["reversal"]),
               "state_events": ("state_event", f.OBSERVATION_SLOTS["state_event"]),
               "items": ("item", f.OBSERVATION_SLOTS["item"])}
    for family in matched["A"]["families"]:
        plan, card = family["private_provenance"]["plan"], family["private_provenance"]["blueprint"]
        for record in plan["records"]:
            evidence = f.render_evidence(record, card)
            for kind, (template_name, slots) in mapping.items():
                if not record[kind]:
                    continue
                template = card[template_name] if template_name == "execution_template" else card["observation_templates"][template_name]
                pattern = "".join(re.escape(literal) + (f"(?P<{name}>.*?)" if name else "")
                                  for literal, name, _, _ in string.Formatter().parse(template))
                for row, rendered in zip(record[kind], evidence[kind], strict=True):
                    parsed = re.fullmatch(pattern, rendered)
                    assert parsed is not None
                    assert parsed.groupdict() == {key: str(row[key]).lower() if type(row[key]) is bool else str(row[key]) for key in slots}


@pytest.mark.parametrize("field", ["scope", "rules", "score_rule", "score_levels", "score_selection", "score_ordinality",
                                    "choice_definitions", "noul_question", "choice_question", "score_question"])
def test_public_meaning_corruption_shared_across_arms_is_rejected(matched, field):
    f = factory()
    modified = copy.deepcopy(matched)
    for arm in "ABCD":
        for case in modified[arm]["families"][0]["cases"]:
            state, questions = case["request"]["state"], case["request"]["questions"]
            if field == "scope":
                state["scope"]["query"]["operation"] = "NONMATCHING"
            elif field.endswith("question"):
                qid = {"noul_question": "n1", "choice_question": "c1", "score_question": "s1"}[field]
                questions[qid]["instructions"] = "Ignore the supplied meaning and select a fixed answer."
            elif field == "score_levels":
                state[field] = ["Every case is level zero."] * len(state[field])
            elif field == "choice_definitions":
                state[field] = {key: "This always applies." for key in state[field]}
            else:
                state[field] = "Only level zero may be selected, regardless of observations."
    with pytest.raises(ValueError, match="canonical|public|request|oracle"):
        f.validate_matched_suites(modified)


def test_canonical_request_reconstruction_compares_actual_serialized_types(matched):
    f = factory()
    modified = copy.deepcopy(matched)
    for arm in "ABCD":
        for case in modified[arm]["families"][0]["cases"]:
            scope = case["request"]["state"]["scope"]
            scope["query_time"] = float(scope["query_time"])
    with pytest.raises(ValueError, match="canonical|public|request"):
        f.validate_matched_suites(modified)


def test_legal_raw_world_membership_does_not_equate_boolean_verification_with_integer():
    f = factory()
    plan = f.make_plan("entity_binding", 3, 3, 5)
    row = next(row for record in plan["records"] for row in record["executions"] if row["verified"] is True)
    row["verified"] = 1
    f.bind_plan_fingerprints(plan)
    with pytest.raises(ValueError, match="raw|world|fact"):
        f.validate_plan(plan)


@pytest.mark.parametrize("damage", ["index", "witness", "witness_level", "program_rule", "program_text", "private_query"])
def test_semantic_audit_validates_actual_facts_programs_and_witnesses(damage):
    f = factory()
    plan = f.make_plan("claim_vs_completion", 3, 3, 5)
    if damage == "index":
        plan["world_indices"] = [999999, 999998]
    elif damage == "witness":
        plan["programs"]["broad"][0]["witnesses"]["0"] = {}
    elif damage == "witness_level":
        witnesses = plan["programs"]["broad"][0]["witnesses"]
        witnesses["0"] = copy.deepcopy(witnesses["4"])
    elif damage == "program_rule":
        plan["programs"]["broad"][0]["rules"][1] = False
    elif damage == "program_text":
        plan["programs"]["broad"][0]["instruction"] = "Select zero."
    else:
        plan["records"][0]["query"]["operation"] = "NONMATCHING"
    # Deliberately repair untrusted fingerprint declarations; semantic binding must still fail.
    if hasattr(f, "bind_plan_fingerprints"):
        f.bind_plan_fingerprints(plan)
    assert hasattr(f, "validate_plan"), "validated legal-world/program/witness binding missing"
    with pytest.raises(ValueError):
        f.validate_plan(plan)


def authorized_gate_fixture(tmp_path, monkeypatch):
    f = factory()
    monkeypatch.setattr(f, "REPO_ROOT", tmp_path.resolve(), raising=False)
    from experiments.contrast_factorial_training import tree_sha256
    from openjev.judgment_model import checkpoint_identity
    def save(path, value):
        path = tmp_path / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    base = tmp_path / "checkpoints/judgment-full-v0.2/final"
    for part in ("training.pt", "readouts.safetensors", "adapter/adapter_model.safetensors", "adapter/adapter_config.json"):
        target = base / part
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"small CPU-only identity fixture")
    info = {"format": "openjev-judgment-v0.2", "model_id": "Qwen/Qwen3.5-2B", "revision": "fixture-revision",
            "lora": {"rank": 8, "alpha": 16}}
    info["checkpoint_id"] = checkpoint_identity(base, info)
    save("checkpoints/judgment-full-v0.2/final/checkpoint.json", info)
    artifact = tmp_path / "opaque-evaluation-artifact.bin"
    artifact.write_bytes(b"\xffopaque bytes: never parse evaluation narrative\x00")
    fingerprint = f.file_sha256(artifact)
    contract = save("docs/contrast-factorial-contract-v1.md", {"fixture": True})
    save("data/contrast-factorial-v1/legacy/manifest.json", {})
    freeze = {"study": "contrast-factorial-v1", "status": "frozen", "reviewed_requests": 480,
              "reviewed_judgments": 2400, "unresolved_review_flags": 0, "no_model_predictions_used": True,
              "training_language_authored_before_freeze": False, "suites": {},
              "legacy_manifest_sha256": f.file_sha256(tmp_path / "data/contrast-factorial-v1/legacy/manifest.json")}
    for split, families, per_category in (("assessment", 80, 8), ("calibration", 40, 4)):
        freeze["suites"][split] = {"path": artifact.name, "suite_sha256": fingerprint, "families": families,
                                   "cases": families * 4, "judgments": families * 20,
                                   "category_counts": {c: per_category for c in f.CATEGORIES}}
    for key in ("author_sources", "review_sources", "source_sha256"):
        freeze[key] = {artifact.name: fingerprint}
    freeze_path = save("data/contrast-factorial-v1/evaluation-freeze.json", freeze)
    approval = {"study": "contrast-factorial-v1", "status": "approved", "base_checkpoint": str(base.relative_to(tmp_path)),
                "base_checkpoint_sha256": tree_sha256(base), "base_checkpoint_id": info["checkpoint_id"],
                "model_id": info["model_id"], "revision": info["revision"],
                "evaluation_freeze_path": str(freeze_path.relative_to(tmp_path)),
                "evaluation_freeze_sha256": f.file_sha256(freeze_path),
                "contract_path": str(contract.relative_to(tmp_path)), "contract_sha256": f.file_sha256(contract),
                "arms": list("ABCD"), "updates_per_arm": 400}
    save("reports/contrast-factorial-v1/STUDY_APPROVAL.json", approval)
    cards = toy_bank()
    bank_hash = f.digest(cards)
    review = {"kind": "independent_language_review", "approved": True, "bank_sha256": bank_hash,
              "semantic_preservation": "pass", "construction_diversity": "pass", "reviewer_identity": "independent-reviewer",
              "reviewed_blueprint_ids": [b["id"] for b in cards], "unresolved_flags": []}
    save("review.json", review)
    bank = {"schema_version": 1, "blueprints": cards,
            "review": {"approved": True, "bank_sha256": bank_hash, "review_path": "review.json"}}
    save("bank.json", bank)
    return approval, freeze, bank, review, save


def test_metadata_only_pinned_emission_preflight_does_not_decode_suites(tmp_path, monkeypatch):
    f = factory()
    approval, _, _, _, _ = authorized_gate_fixture(tmp_path, monkeypatch)
    assert hasattr(f, "validate_emission_inputs"), "root-pinned metadata preflight missing"
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path.name != "opaque-evaluation-artifact.bin", "evaluation artifact decoded"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    result = f.validate_emission_inputs(tmp_path, "bank.json", approval["evaluation_freeze_path"], approval["base_checkpoint"])
    assert result["approval_sha256"] == f.file_sha256(tmp_path / "reports/contrast-factorial-v1/STUDY_APPROVAL.json")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("damage", ["missing_approval", "unpinned_base", "base_component", "base_identity", "invalid_freeze",
                                    "freeze_count", "freeze_flag", "freeze_hash", "artifact_hash", "self_review", "hardlink_review",
                                    "pilot_review", "review_flags", "review_kind", "review_semantics", "review_diversity", "reviewer"])
def test_emission_preflight_rejects_false_authority_and_partial_review(tmp_path, monkeypatch, damage):
    f = factory()
    approval, freeze, bank, review, save = authorized_gate_fixture(tmp_path, monkeypatch)
    assert hasattr(f, "validate_emission_inputs"), "root-pinned metadata preflight missing"
    base = approval["base_checkpoint"]
    if damage == "missing_approval":
        (tmp_path / "reports/contrast-factorial-v1/STUDY_APPROVAL.json").unlink()
    elif damage == "unpinned_base":
        base = "arbitrary-base"
    elif damage == "base_component":
        (tmp_path / base / "training.pt").unlink()
    elif damage == "base_identity":
        info = json.loads((tmp_path / base / "checkpoint.json").read_text())
        info["checkpoint_id"] = "invalid-identity"
        save(base + "/checkpoint.json", info)
    elif damage == "invalid_freeze":
        (tmp_path / approval["evaluation_freeze_path"]).write_text("not a freeze")
    elif damage in {"freeze_count", "freeze_flag", "freeze_hash"}:
        if damage == "freeze_count":
            freeze["reviewed_requests"] = 20
        elif damage == "freeze_flag":
            freeze["unresolved_review_flags"] = 1
        else:
            freeze["status"] = "draft"
        save(approval["evaluation_freeze_path"], freeze)
        if damage != "freeze_hash":
            approval["evaluation_freeze_sha256"] = f.file_sha256(tmp_path / approval["evaluation_freeze_path"])
            save("reports/contrast-factorial-v1/STUDY_APPROVAL.json", approval)
    elif damage == "artifact_hash":
        (tmp_path / "opaque-evaluation-artifact.bin").write_bytes(b"changed")
    elif damage in {"self_review", "hardlink_review"}:
        bank.update(review)
        bank["review"]["review_path"] = "bank.json" if damage == "self_review" else "bank-link.json"
        save("bank.json", bank)
        if damage == "hardlink_review":
            (tmp_path / "bank-link.json").hardlink_to(tmp_path / "bank.json")
    else:
        if damage == "pilot_review":
            review["reviewed_blueprint_ids"] = review["reviewed_blueprint_ids"][:20]
        elif damage == "review_flags":
            review["unresolved_flags"] = ["ambiguous"]
        elif damage == "review_kind":
            review["kind"] = "self_review"
        elif damage == "review_semantics":
            review["semantic_preservation"] = "pending"
        elif damage == "review_diversity":
            review["construction_diversity"] = "pending"
        else:
            review["reviewer_identity"] = ""
        save("review.json", review)
    with pytest.raises(ValueError):
        f.validate_emission_inputs(tmp_path, "bank.json", approval["evaluation_freeze_path"], base)


def test_caller_repository_cannot_supply_its_own_root_approval(tmp_path, monkeypatch):
    f = factory()
    actual_root = Path(__file__).resolve().parents[1]
    approval, _, _, _, _ = authorized_gate_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(f, "REPO_ROOT", actual_root)
    with pytest.raises(ValueError, match="repository root"):
        f.validate_emission_inputs(tmp_path, "bank.json", approval["evaluation_freeze_path"], approval["base_checkpoint"])
