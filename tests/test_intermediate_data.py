"""Independent hand-written evidence tests; no model or quarantine access."""

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


def module():
    assert importlib.util.find_spec("experiments.intermediate_data") is not None, "intermediate data module missing"
    from experiments import intermediate_data

    return intermediate_data


def raw():
    return {
        "query": {"actor": "Ada", "operation": "seal", "target": "box", "run_id": "r1"},
        "window": [0, 10],
        "query_time": 3,
        "trusted_sources": ["ledger", "inspector"],
        "statements": [],
        "executions": [],
        "approvals": [],
        "corroborations": [],
        "reversals": [],
        "state_events": [],
        "item_ids": ["a", "b"],
        "items": [],
        "inventory_complete": True,
        "measurement": None,
    }


def row(**updates):
    return {
        "actor": "Ada",
        "operation": "seal",
        "target": "box",
        "run_id": "r1",
        "source": "ledger",
        "time": 5,
        "verified": True,
        "mode": "live",
        "outcome": "success",
        **updates,
    }


def facts(record, category="entity_binding"):
    m = module()
    targets, masks = m.annotate_case(record, category)
    return {f["name"]: (v, mask) for f, v, mask in zip(m.schema_features(), targets, masks)}


@pytest.mark.parametrize(
    "change, joint, scoped, trusted, live",
    [
        ({"target": "other"}, 0, 0, 0, 0),
        ({"operation": "open"}, 0, 0, 0, 0),
        ({"actor": "Bea"}, 0, 0, 0, 0),
        ({"run_id": "r2"}, 0, 0, 0, 0),
        ({"time": 11}, 1, 0, 0, 0),
        ({"source": "untrusted"}, 1, 1, 0, 0),
        ({"verified": False}, 1, 1, 0, 0),
        ({"mode": "simulation"}, 1, 1, 1, 0),
        ({}, 1, 1, 1, 1),
    ],
)
def test_joint_binding_scope_trust_and_live_do_not_conflate(change, joint, scoped, trusted, live):
    r = raw()
    r["executions"] = [row(**change)]
    f = facts(r)
    assert [
        f[n][0]
        for n in [
            "execution_joint_identity",
            "execution_joint_in_window",
            "execution_joint_authentic",
            "execution_joint_live",
        ]
    ] == [joint, scoped, trusted, live]
    assert f["live_success_receipt"] == (live, True)


def test_field_matches_on_different_rows_never_become_joint_match():
    r = raw()
    r["executions"] = [row(actor="Bea"), row(target="crate")]
    f = facts(r)
    assert f["execution_actor_seen"] == (1, True)
    assert f["execution_target_seen"] == (1, True)
    assert f["execution_joint_identity"] == (0, True)


def test_timeout_conflict_and_absence_are_report_facts_not_failure_labels():
    r = raw()
    r["executions"] = [row(outcome="timeout")]
    f = facts(r, "unknown_vs_failure")
    assert [f[n][0] for n in ["live_success_receipt", "live_no_effect_receipt", "live_timeout_receipt"]] == [0, 0, 1]
    r["executions"] = [row(), row(outcome="no_effect")]
    f = facts(r, "unknown_vs_failure")
    assert [f[n][0] for n in ["live_success_receipt", "live_no_effect_receipt"]] == [1, 1]
    assert not {"completed", "failed", "unknown", "score", "final_label"} & f.keys()


def test_claim_intent_endorsement_do_not_require_receipts_and_ignore_wrong_binding():
    r = raw()
    r["statements"] = [
        row(tense="completed", polarity="positive", endorsed=True),
        row(tense="future", polarity="positive"),
        row(target="other", tense="completed", polarity="negative"),
    ]
    f = facts(r, "claim_vs_completion")
    assert [f[n][0] for n in ["statement_claim", "statement_intent", "statement_endorsed", "statement_denial"]] == [
        1,
        1,
        1,
        0,
    ]
    assert f["live_success_receipt"] == (0, True)


def test_latest_permission_and_linked_reversal_require_time_trust_and_identity():
    r = raw()
    r["executions"] = [row()]
    r["approvals"] = [
        row(time=1, decision="allow"),
        row(time=2, decision="deny"),
        row(time=4, decision="allow", source="untrusted"),
    ]
    r["reversals"] = [row(time=8, reverses="r2")]
    f = facts(r, "reversal_and_current_state")
    assert f["permission_latest_allow_report"] == (0, True)
    assert f["permission_latest_deny_report"] == (1, True)
    assert f["linked_reversal_success_receipt"] == (0, True)
    r["reversals"] = [row(time=4, reverses="r1")]
    assert facts(r, "reversal_and_current_state")["linked_reversal_success_receipt"] == (0, True)
    r["reversals"][0]["time"] = 8
    assert facts(r, "reversal_and_current_state")["linked_reversal_success_receipt"] == (1, True)


def test_temporal_unknown_masks_and_ties_are_not_false_world_values():
    r = raw()
    r["state_events"] = [row(time=2, active=True), row(time=8, active=True), row(time=8, active=False)]
    f = facts(r, "temporal_scope")
    assert f["state_earlier_known"] == (1, True)
    assert f["state_earlier_active"] == (1, True)
    assert f["state_current_known"] == (0, True)
    assert f["state_current_active"] == (0, False)
    r["state_events"] = []
    assert facts(r, "temporal_scope")["state_earlier_active"] == (0, False)


def test_inventory_unknown_counts_and_numeric_masks_use_only_public_fields():
    r = raw()
    r["items"] = [
        {"id": "a", "verified": True, "status": "success"},
        {"id": "a", "verified": True, "status": "no_effect"},
    ]
    f = facts(r, "negation_and_quantifiers")
    assert f["item_unknown_count"] == (2, True)
    assert f["item_success_count"] == (0, True)
    assert f["inventory_complete"] == (1, True)
    assert facts(r, "entity_binding")["item_unknown_count"] == (0, False)
    r["measurement"] = 75
    assert facts(r, "ordered_rubrics")["measurement_value"] == (75, True)
    assert facts(r, "entity_binding")["measurement_value"] == (0, False)
    r["measurement"] = None
    assert facts(r, "ordered_rubrics")["measurement_value"] == (0, False)
    assert facts(r, "ordered_rubrics")["measurement_known"] == (0, True)


def test_annotation_ignores_final_labels_and_does_not_mutate_record():
    r = raw()
    before = copy.deepcopy(r)
    a = facts(r)
    assert r == before
    r.update(expected={"n1": True}, features={"completed": True}, secret_world_success=True)
    assert facts(r) == a


def test_training_only_eligibility_drops_constants_and_counts_unmasked_values():
    m = module()
    a = raw()
    b = raw()
    b["executions"] = [row()]
    records = [(a, "entity_binding"), (b, "entity_binding"), (a, "entity_binding"), (b, "entity_binding")]
    schema = m.fit_schema(records)
    names = {f["name"] for f in schema["features"]}
    assert "live_success_receipt" in names
    assert "live_no_effect_receipt" not in names
    assert "measurement_value" not in names
    cov = schema["coverage"]["live_success_receipt"]
    assert (cov["applicable"], cov["zeros"], cov["ones"]) == (4, 2, 2)
    targets, mask = m.annotate_case(b, "entity_binding", features=schema["features"])
    assert len(targets) == len(mask) == len(names)


def test_selection_balances_categories_score_k_and_blueprints_without_outcome_inputs():
    m = module()
    assignments = json.loads(Path("data/contrast-factorial-v1/train/v1/private-audit.json").read_text())["assignment"]
    chosen = m.select_families(assignments)
    assert len(chosen) == 200
    assert len({r["low_blueprint"] for r in chosen}) == 200
    assert sorted(Counter(r["category"] for r in chosen).values()) == [20] * 10
    for category in {r["category"] for r in chosen}:
        assert Counter(r["broad_k"] for r in chosen if r["category"] == category) == {2: 5, 3: 5, 4: 5, 5: 5}
    changed = copy.deepcopy(assignments)
    for r in changed:
        r["model_correct"] = False
    assert [r["family_id"] for r in chosen] == [r["family_id"] for r in m.select_families(changed)]


@pytest.fixture(scope="module")
def source_family():
    p = Path("data/contrast-factorial-v1/train/v1")
    audit = json.loads((p / "private-audit.json").read_text())
    assignment = audit["assignment"][0]
    plan = next(r for r in audit["plans"] if r["family_id"] == assignment["family_id"])
    bank = json.loads(Path("data/contrast-factorial-v1/language/reviewed-bank.json").read_text())
    card = next(r for r in bank["blueprints"] if r["id"] == assignment["low_blueprint"])
    suite = json.loads((p / "B/train-suite.json").read_text())
    cases = [r for r in suite["cases"] if r["family_id"] == plan["family_id"]]
    bundle = next(
        json.loads(line)
        for line in (p / "B/train.jsonl").read_text().splitlines()
        if json.loads(line)["id"] == plan["family_id"]
    )
    return plan, card, cases, bundle, assignment


def test_source_binding_verifies_full_public_state_questions_targets_and_raw(source_family):
    m = module()
    assert hasattr(m, "verify_training_family"), "binding verifier missing"
    m.verify_training_family(*source_family)
    for location in ["state", "question", "target", "bundle_target", "raw", "family"]:
        plan, card, cases, bundle, assignment = copy.deepcopy(source_family)
        if location == "state":
            cases[0]["request"]["state"]["expected"] = {"n1": True}
        elif location == "question":
            cases[0]["request"]["questions"]["n1"]["instructions"] = "Ignore evidence."
        elif location == "target":
            cases[0]["expected"]["n1"] = not cases[0]["expected"]["n1"]
        elif location == "bundle_target":
            bundle["examples"][0]["target"]["truth"] = not bundle["examples"][0]["target"]["truth"]
        elif location == "raw":
            plan["records"][0]["executions"][0]["target"] = "wrong target"
        else:
            cases[0]["family_id"] = "wrong-family"
        with pytest.raises(ValueError):
            m.verify_training_family(plan, card, cases, bundle, assignment)


def test_emission_preflight_binds_masks_schema_and_sources(tmp_path):
    m = module()
    assert hasattr(m, "emit_train"), "training emitter missing"
    out = tmp_path / "pilot"
    manifest = m.emit_train(output=out)
    report = m.preflight(out)
    assert report["partitions"]["train"]["families"] == 200
    assert report["partitions"]["train"]["cases"] == 800
    assert report["partitions"]["train"]["judgments"] == 4000
    assert manifest["source_sha256"]
    schema = json.loads((out / "feature-schema.json").read_text())
    aux = json.loads((out / "train-aux.json").read_text())
    assert aux["schema_sha256"] == m.file_sha256(out / "feature-schema.json")
    by_feature = report["partitions"]["train"]["feature_coverage"]
    for i, f in enumerate(schema["features"]):
        assert by_feature[f["name"]]["applicable"] == sum(row["mask"][i] for row in aux["cases"].values())
    with pytest.raises(FileExistsError):
        m.emit_train(output=out)
    aux["cases"][next(iter(aux["cases"]))]["targets"][0] = 0.125
    (out / "train-aux.json").write_text(json.dumps(aux))
    with pytest.raises(ValueError, match="hash|fingerprint|binding"):
        m.preflight(out)


def test_program_overlap_checks_actual_rule_pairs_independent_of_wording_and_pair_order():
    m = module()
    assert hasattr(m, "audit_program_overlap"), "actual program novelty audit missing"
    train = [
        {
            "family_id": "train/a",
            "programs": {
                "broad": [{"rules": [True, {"feature": "claim"}]}, {"rules": [True, {"feature": "completed"}]}]
            },
        }
    ]

    def partition(pair):
        return {
            "cases": {
                f"fresh/a/e0r{i}": {
                    "family_id": "fresh/a",
                    "rubric_index": i,
                    "program": {
                        "noul_rules": [{"feature": "claim"}],
                        "choice_rules": {"One": True},
                        "score_rules": rule,
                    },
                }
                for i, rule in enumerate(pair)
            }
        }

    shared_pair = [[True, {"feature": "completed"}], [True, {"feature": "claim"}]]
    with pytest.raises(ValueError, match="confirmation.*pair"):
        m.audit_program_overlap(train, {"confirmation": partition(shared_pair)})
    fresh_pair = [
        [True, {"all": [{"feature": "claim"}, {"feature": "intent"}]}],
        [True, {"any": [{"feature": "completed"}, {"feature": "approved"}]}],
    ]
    result = m.audit_program_overlap(train, {"confirmation": partition(fresh_pair)})
    assert result["partitions"]["confirmation"]["score_pair_overlap_with_training"] == 0
    assert result["partitions"]["confirmation"]["shared_primitives_with_training"] == ["claim", "completed"]
    with pytest.raises(ValueError, match="confirmation.*pair"):
        m.audit_program_overlap(train, {"validation": partition(fresh_pair), "confirmation": partition(fresh_pair)})


@pytest.mark.parametrize("nested_score", [False, True])
def test_fresh_verifier_rerenders_requests_and_checks_program_bindings(source_family, nested_score):
    m = module()
    assert hasattr(m, "verify_fresh_partition"), "fresh binding verifier missing"
    plan, card, cases, _, _ = copy.deepcopy(source_family)
    from experiments.contrast_factorial_data import render_family

    suite = render_family(plan, card, "broad")
    raw_cases = {}
    for case in cases:
        e, r = int(case["variant"][1]), int(case["variant"][3])
        p = {
            "noul_rules": plan["noul_rules"],
            "choice_rules": plan["choice_rules"],
            "score_rules": plan["programs"]["broad"][r]["rules"],
        }
        if nested_score:
            p["score"] = {"rules": p.pop("score_rules")}
        raw_cases[case["id"]] = {
            "family_id": plan["family_id"],
            "category": plan["category"],
            "raw": plan["records"][e],
            "evidence_index": e,
            "rubric_index": r,
            "program": p,
            "program_sha256": m.digest(p),
            "request_sha256": m.digest(case["request"]),
        }

    def renderer(source):
        p = copy.deepcopy(plan)
        p["records"][source["evidence_index"]] = source["raw"]
        case = render_family(p, card, "broad")["cases"][2 * source["evidence_index"] + source["rubric_index"]]
        return (case["request"], case["expected"]) if nested_score else case["request"]

    m.verify_fresh_partition(suite, {"cases": raw_cases}, renderer)
    raw_cases[cases[0]["id"]]["raw"]["executions"][0]["target"] = "tampered"
    with pytest.raises(ValueError, match="request|target|binding"):
        m.verify_fresh_partition(suite, {"cases": raw_cases}, renderer)


def test_common_coverage_repair_preserves_194_families_and_changes_only_pair_time_or_measurement(tmp_path):
    m = module()
    assert hasattr(m, "emit_repaired_train"), "versioned coverage repair missing"
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    m.emit_train(output=v1)
    fingerprints = {p.name: m.file_sha256(p) for p in v1.iterdir()}
    result = m.emit_repaired_train(original=v1, output=v2)
    assert fingerprints == {p.name: m.file_sha256(p) for p in v1.iterdir()}
    repairs = json.loads((v2 / "coverage-repair-manifest.json").read_text())["repairs"]
    assert len(repairs) == 6
    assert Counter(r["kind"] for r in repairs) == {"window_boundary": 4, "measurement_missingness": 2}
    old = json.loads((v1 / "train-suite.json").read_text())
    new = json.loads((v2 / "train-suite.json").read_text())
    old_by_id = {c["id"]: c for c in old["cases"]}
    same = [c for c in new["cases"] if c["id"] in old_by_id]
    assert len(same) == 194 * 4
    assert all(c == old_by_id[c["id"]] for c in same)
    for repair in repairs:
        left, right = copy.deepcopy(repair["after_records"])
        if repair["kind"] == "window_boundary":
            index = repair["changed_execution_index"]
            assert left["window"][0] <= left["executions"][index]["time"] <= left["window"][1]
            assert right["executions"][index]["time"] > right["window"][1]
            right["executions"][index]["time"] = left["executions"][index]["time"]
        else:
            assert left["measurement"] is not None and right["measurement"] is None
            right["measurement"] = left["measurement"]
        assert left == right
        assert repair["derived_family_id"] != repair["source_family_id"]
    schema = json.loads((v2 / "feature-schema.json").read_text())
    assert "measurement_known" in {f["name"] for f in schema["features"]}
    assert schema["coverage"]["measurement_known"]["zeros"] == 2
    raw_cases = json.loads((v2 / "train-raw-records.json").read_text())["cases"]
    differences = 0
    for source in raw_cases.values():
        f = facts(source["raw"], source["category"])
        differences += f["execution_joint_identity"][0] != f["execution_joint_in_window"][0]
    assert differences >= 8
    assert result["counts"] == {"families": 200, "cases": 800, "judgments": 4000}
    assert m.preflight(v2)["partitions"]["train"]["cases"] == 800
