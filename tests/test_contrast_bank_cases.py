"""Literal semantic regressions for the frozen destination-change diagnostic."""

import copy
import json

import pytest

QUESTION_IDS = (
    "explicit_change_claim",
    "different_destination",
    "record_confirms_change",
    "use_permitted",
    "record_status",
    "operational_change_signal",
)


def cases_by_id():
    from experiments.contrast_bank_cases import build_bank_suite

    return {case["id"]: case for case in build_bank_suite()["cases"]}


@pytest.mark.parametrize("domain", ["payment", "shipping"])
@pytest.mark.parametrize(
    ("variant", "gold"),
    [
        ("unchanged_repeat", (False, False, False, True, "Unchanged", False)),
        ("explicit_unconfirmed", (True, True, False, False, "Unchanged", True)),
        ("explicit_same_destination", (True, False, False, True, "Unchanged", True)),
        ("denied_redirect", (False, True, False, False, "Unchanged", True)),
        ("change_question", (False, True, False, False, "Unchanged", True)),
        ("future_only", (False, False, False, True, "Unchanged", False)),
        ("same_destination", (False, False, False, True, "Unchanged", False)),
        ("different_destination", (False, True, False, False, "Unchanged", True)),
        ("confirmed_without_claim", (False, True, True, True, "Changed", True)),
        ("confirmed_with_claim", (True, True, True, True, "Changed", True)),
        ("unverified_record", (True, True, False, False, "Unknown", True)),
        ("conflicting_records", (True, True, False, False, "Unknown", True)),
        ("attributed_quote", (False, True, False, False, "Unchanged", True)),
        ("adopted_quote", (True, True, False, False, "Unchanged", True)),
        ("explicit_paraphrase", (True, True, False, False, "Unchanged", True)),
        ("explicit_distractor", (True, True, False, False, "Unchanged", True)),
        ("verified_but_policy_hold", (True, True, True, False, "Changed", True)),
    ],
)
def test_literal_semantic_expectations_keep_claim_fact_and_permission_separate(domain, variant, gold):
    case = cases_by_id()[f"{domain}.{variant}"]
    assert case["expected"] == dict(zip(QUESTION_IDS, gold, strict=True))


@pytest.mark.parametrize("domain", ["payment", "shipping"])
def test_evidence_pairs_change_only_the_record_or_transaction_hold(domain):
    cases = cases_by_id()
    unconfirmed = copy.deepcopy(cases[f"{domain}.explicit_unconfirmed"]["request"])
    confirmed = copy.deepcopy(cases[f"{domain}.confirmed_with_claim"]["request"])
    old_record = unconfirmed["state"].pop("current_record_snapshots")
    new_record = confirmed["state"].pop("current_record_snapshots")
    assert unconfirmed == confirmed
    assert old_record[0]["destination"] == unconfirmed["state"]["baseline"]["destination"]
    assert new_record[0]["destination"] == confirmed["state"]["requested_destination"]
    assert old_record[0]["destination"] != new_record[0]["destination"]

    held = copy.deepcopy(cases[f"{domain}.verified_but_policy_hold"]["request"])
    approved = copy.deepcopy(cases[f"{domain}.confirmed_with_claim"]["request"])
    assert held["state"]["use_policy"].pop("transaction_on_hold") is True
    assert approved["state"]["use_policy"].pop("transaction_on_hold") is False
    assert held == approved


@pytest.mark.parametrize("domain", ["payment", "shipping"])
def test_unknown_uses_unverified_or_conflicting_evidence_not_half_probability(domain):
    cases = cases_by_id()
    unverified = cases[f"{domain}.unverified_record"]
    assert all(record["authenticated"] is False for record in unverified["request"]["state"]["current_record_snapshots"])
    conflicting = cases[f"{domain}.conflicting_records"]
    records = conflicting["request"]["state"]["current_record_snapshots"]
    assert len(records) == 2
    assert all(record["authenticated"] is True for record in records)
    assert records[0]["effective_at"] == records[1]["effective_at"]
    assert records[0]["destination"] != records[1]["destination"]
    for case in (unverified, conflicting):
        assert case["expected"]["record_status"] == "Unknown"
        assert case["expected"]["record_confirms_change"] is False
        assert all(type(value) in (bool, str) for value in case["expected"].values())


def test_model_requests_have_no_gold_or_variant_oracle_and_are_independent():
    from experiments.contrast_bank_cases import build_bank_suite

    suite = build_bank_suite()
    assert len(suite["cases"]) == 34
    assert len({case["id"] for case in suite["cases"]}) == 34
    forbidden_keys = {"expected", "gold", "rationale", "variant", "family_id", "case_id", "record_status"}

    def walk(value):
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for case in suite["cases"]:
        assert set(case["request"]) == {"state", "questions"}
        walk(case["request"]["state"])
        assert len(case["request"]["questions"]) == 6
        assert set(case["expected"]) == set(case["rationale"]) == set(case["request"]["questions"])
        for question_id, question in case["request"]["questions"].items():
            answer = case["expected"][question_id]
            if question["type"] == "noul":
                assert type(answer) is bool
            else:
                assert question["type"] == "choice"
                assert answer in question["criteria"]
    assert json.loads(json.dumps(suite, allow_nan=False)) == suite
    original = build_bank_suite()
    suite["cases"][0]["request"]["state"]["message"]["body"] = "mutated"
    assert build_bank_suite() == original
    assert suite["cases"][1] == original["cases"][1]


def test_relations_resolve_with_literal_question_contrasts_and_no_false_complements():
    from experiments.contrast_bank_cases import build_bank_suite

    suite = build_bank_suite()
    cases = {case["id"]: case for case in suite["cases"]}
    assert len({relation["id"] for relation in suite["relations"]}) == len(suite["relations"])
    question_contrasts = set()
    for relation in suite["relations"]:
        left, right = relation["left"], relation["right"]
        a, b = cases[left["case_id"]], cases[right["case_id"]]
        left_gold, right_gold = a["expected"][left["question_id"]], b["expected"][right["question_id"]]
        assert a["family_id"] == b["family_id"]
        assert relation["reason"]
        if relation["kind"] == "invariant":
            assert left_gold == right_gold
            assert a["request"]["questions"][left["question_id"]] == b["request"]["questions"][right["question_id"]]
        else:
            assert relation["kind"] in {"flip", "question_contrast"}
            assert left_gold != right_gold
        if relation["kind"] == "question_contrast":
            assert left["case_id"] == right["case_id"]
            question_contrasts.add((left["case_id"], left["question_id"], right["question_id"]))
    assert ("payment.explicit_unconfirmed", "explicit_change_claim", "record_confirms_change") in question_contrasts
    assert ("shipping.denied_redirect", "explicit_change_claim", "operational_change_signal") in question_contrasts
    assert ("payment.verified_but_policy_hold", "record_confirms_change", "use_permitted") in question_contrasts


def test_predicate_pairs_cover_independent_truth_combinations():
    from experiments.contrast_bank_cases import build_bank_suite

    for domain in ("payment", "shipping"):
        answers = [case["expected"] for case in build_bank_suite()["cases"] if case["domain"] == domain]
        assert {(gold["explicit_change_claim"], gold["record_confirms_change"]) for gold in answers} == {
            (False, False), (False, True), (True, False), (True, True),
        }
        assert {(gold["explicit_change_claim"], gold["different_destination"]) for gold in answers} == {
            (False, False), (False, True), (True, False), (True, True),
        }
