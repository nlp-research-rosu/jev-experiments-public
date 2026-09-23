import json
import re
from collections import Counter, defaultdict

import pytest

from experiments.intermediate_fresh import (
    CATEGORIES,
    QUESTION_IDS,
    build_artifacts,
    canonical_sha256,
    render_request,
    validate_artifacts,
    validate_public_suite,
)


@pytest.fixture(scope="module")
def artifacts():
    return build_artifacts()


def test_authors_eighty_distinct_frames_and_hash_assigns_exact_splits(artifacts):
    frames = artifacts["manifest"]["frames"]
    assert len(frames) == len({row["frame_id"] for row in frames}) == 80
    assert Counter(row["category"] for row in frames) == Counter({name: 8 for name in CATEGORIES})
    assert Counter(row["split"] for row in frames) == Counter(
        {"validation": 20, "calibration": 20, "confirmation": 40}
    )
    by_category = defaultdict(Counter)
    for row in frames:
        by_category[row["category"]][row["split"]] += 1
    assert all(counts == Counter(validation=2, calibration=2, confirmation=4) for counts in by_category.values())
    assert len({row["language_sha256"] for row in frames}) == 80


def test_each_family_has_factorial_cases_and_all_semantic_contrasts(artifacts):
    for suite in artifacts["suites"].values():
        validate_public_suite(suite)
        grouped = defaultdict(list)
        for case in suite["cases"]:
            grouped[case["family_id"]].append(case)
            assert list(case["request"]["questions"]) == list(QUESTION_IDS)
            assert [q["type"] for q in case["request"]["questions"].values()] == [
                "noul", "noul", "noul", "choice", "score"
            ]
        for family_id, cases in grouped.items():
            assert len(cases) == 4
            assert {case["variant"] for case in cases} == {"e0r0", "e0r1", "e1r0", "e1r1"}
            axes = {row["contrast_axis"] for row in suite["relations"] if row["family_id"] == family_id}
            assert axes == {"evidence", "rubric", "question", "invariant"}


def test_private_records_rerender_exact_public_requests_and_gold(artifacts):
    raw_cases = artifacts["raw_records"]["cases"]
    public_cases = {
        case["id"]: case for suite in artifacts["suites"].values() for case in suite["cases"]
    }
    assert set(raw_cases) == set(public_cases)
    for case_id, private in raw_cases.items():
        request, expected = render_request(private)
        assert request == public_cases[case_id]["request"]
        assert expected == public_cases[case_id]["expected"]
        assert canonical_sha256(request) == private["request_sha256"]


def test_truth_coverage_unknown_false_and_score_ordinality(artifacts):
    cases = [case for suite in artifacts["suites"].values() for case in suite["cases"]]
    noul_values = [case["expected"][qid] for case in cases for qid in ("n1", "n2", "n3")]
    assert set(noul_values) == {False, True}
    raw_cases = artifacts["raw_records"]["cases"].values()
    assert any(private["derived_features"]["unknown"] for private in raw_cases)
    assert any(private["derived_features"]["failed"] for private in raw_cases)
    assert any(
        private["derived_features"]["unknown"] and not private["derived_features"]["failed"]
        for private in raw_cases
    )
    widths = Counter(len(case["request"]["questions"]["s1"]["criteria"]) for case in cases)
    assert set(widths) == {2, 3, 4, 5}
    for case in cases:
        levels = case["request"]["questions"]["s1"]["criteria"]
        assert all(level.startswith(f"Level {index}:") for index, level in enumerate(levels))
        assert 0 <= case["expected"]["s1"] < len(levels)


def test_raw_worlds_exercise_simulation_source_time_binding_and_reversal_edges(artifacts):
    records = [private["raw"] for private in artifacts["raw_records"]["cases"].values()]
    executions = [row for record in records for row in record["executions"]]
    assert any(row["mode"] == "simulation" for row in executions)
    assert any(row["source"] not in record["trusted_sources"] for record in records for row in record["executions"])
    assert any(not (record["window"][0] <= row["time"] <= record["window"][1]) for record in records for row in record["executions"])
    assert any(row["target"] != record["query"]["target"] for record in records for row in record["executions"])
    assert any(row["operation"] != record["query"]["operation"] for record in records for row in record["executions"])
    assert any(record["reversals"] for record in records)


def test_every_frame_has_a_unique_genuinely_different_score_rule_pair(artifacts):
    pair_hashes = {}
    split_pairs = defaultdict(set)
    raw_cases = artifacts["raw_records"]["cases"]
    for family_id, family in artifacts["raw_records"]["families"].items():
        rubric_rules = []
        for rubric_index in (0, 1):
            case_id = next(cid for cid in family["case_ids"] if cid.endswith(f"e0r{rubric_index}"))
            rubric_rules.append(raw_cases[case_id]["program"]["score"]["rules"])
        assert rubric_rules[0] != rubric_rules[1]
        pair_hash = canonical_sha256(rubric_rules)
        assert pair_hash not in pair_hashes, (family_id, pair_hashes.get(pair_hash))
        pair_hashes[pair_hash] = family_id
        split_pairs[family["split"]].add(pair_hash)
    assert len(pair_hashes) == 80
    assert split_pairs["confirmation"].isdisjoint(split_pairs["validation"] | split_pairs["calibration"])


def test_no_private_labels_or_oracle_feature_fields_enter_state(artifacts):
    forbidden_keys = {
        "expected", "rationale", "derived_features", "noul_rules", "choice_rules",
        "score_rules", "program_sha256", "request_sha256",
    }
    oracle_names = set(next(iter(artifacts["raw_records"]["cases"].values()))["derived_features"])
    for suite in artifacts["suites"].values():
        for case in suite["cases"]:
            state = case["request"]["state"]
            assert not (set(state) & forbidden_keys)
            assert not (set(state) & oracle_names)
            serialized = json.dumps(state, sort_keys=True).lower()
            assert not re.search(r"ground truth|gold label|expected answer|oracle feature", serialized)
            assert "false means the record does not establish" in serialized


def test_public_validator_rejects_any_mutated_gold_and_internal_validator_passes(artifacts):
    validate_artifacts(artifacts)
    suite = json.loads(json.dumps(artifacts["suites"]["validation"]))
    suite["cases"][0]["expected"]["n1"] = not suite["cases"][0]["expected"]["n1"]
    with pytest.raises(ValueError, match="gold|target|rerender"):
        validate_artifacts({
            **artifacts,
            "suites": {**artifacts["suites"], "validation": suite},
        })
