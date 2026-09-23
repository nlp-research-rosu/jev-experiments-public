"""Compact public-rubric invariance checks, without a model or GPU."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    assert importlib.util.find_spec("experiments.intermediate_compact") is not None, "compact renderer helper missing"
    from experiments import intermediate_compact

    return intermediate_compact


@pytest.fixture(scope="module")
def original():
    root = Path("data/intermediate-supervision-v1/fresh-draft")
    return (
        {
            split: json.loads((root / f"{split}-suite.json").read_text())
            for split in ["validation", "calibration", "confirmation"]
        },
        json.loads((root / "raw-records.json").read_text()),
    )


def test_compact_threshold_is_derived_from_rule_not_level_index(original):
    m = module()
    _, records = original
    score = copy.deepcopy(
        next(c["program"]["score"] for c in records["cases"].values() if len(c["program"]["score"]["rules"]) == 5)
    )
    # Keep the original four-requirement instruction but retain only thresholds 2 and 4.
    score["rules"] = [score["rules"][0], score["rules"][2], score["rules"][4]]
    score["levels"] = [score["levels"][0], score["levels"][2], score["levels"][4]]
    before = copy.deepcopy(score)
    compact = m.compact_score(score)
    assert "at least 2 of the four" in compact["levels"][1]
    assert "at least 4 of the four" in compact["levels"][2]
    assert "score_rule" in compact["levels"][1]
    assert compact["levels"][0] == before["levels"][0]
    assert {k: v for k, v in compact.items() if k != "levels"} == {k: v for k, v in before.items() if k != "levels"}
    assert score == before


def test_compact_rejects_ambiguous_or_changed_checklist_reference(original):
    m = module()
    _, records = original
    score = copy.deepcopy(
        next(c["program"]["score"] for c in records["cases"].values() if len(c["program"]["score"]["rules"]) == 5)
    )
    wrong_instruction = copy.deepcopy(score)
    wrong_instruction["instruction"] = "Use four unspecified requirements."
    with pytest.raises(ValueError, match="instruction|checklist"):
        m.compact_score(wrong_instruction)
    score["rules"][2]["at_least"]["of"][0] = {"feature": "has_measurement"}
    with pytest.raises(ValueError, match="checklist|requirements"):
        m.compact_score(score)


def test_every_raw_ast_target_identity_partition_and_non_score_question_is_preserved(original):
    m = module()
    suites, records = original
    before = copy.deepcopy((suites, records))
    compact_suites, compact_records, changes = m.compact_artifacts(suites, records)
    assert (suites, records) == before
    assert len(changes) == 320
    from experiments.intermediate_data import digest
    from experiments.intermediate_fresh import render_request

    for split, old_suite in suites.items():
        new_suite = compact_suites[split]
        assert new_suite["ordered_family_ids"] == old_suite["ordered_family_ids"]
        assert new_suite["relations"] == old_suite["relations"]
        assert new_suite["metadata"] == old_suite["metadata"]
        for old_case, new_case in zip(old_suite["cases"], new_suite["cases"], strict=True):
            assert old_case["id"] == new_case["id"]
            assert old_case["expected"] == new_case["expected"]
            old_private = records["cases"][old_case["id"]]
            new_private = compact_records["cases"][old_case["id"]]
            assert new_private["raw"] == old_private["raw"]
            assert new_private["raw_sha256"] == old_private["raw_sha256"]
            assert new_private["program"]["score"]["rules"] == old_private["program"]["score"]["rules"]
            assert new_private["program"]["noul_rules"] == old_private["program"]["noul_rules"]
            assert new_private["program"]["choice_rules"] == old_private["program"]["choice_rules"]
            assert new_private["language"] == old_private["language"]
            for q in ("n1", "n2", "n3", "c1"):
                assert new_case["request"]["questions"][q] == old_case["request"]["questions"][q]
            old_without_levels = copy.deepcopy(old_case)
            new_without_levels = copy.deepcopy(new_case)
            for row in (old_without_levels, new_without_levels):
                row["request"]["state"].pop("score_levels")
                row["request"]["questions"]["s1"].pop("criteria")
            assert new_without_levels == old_without_levels
            request, target = render_request(new_private)
            assert request == new_case["request"] and target == new_case["expected"]
            assert new_private["request_sha256"] == digest(request)
            assert new_private["program_sha256"] == digest(new_private["program"])
    for fid, family in compact_records["families"].items():
        old_family = records["families"][fid]
        assert family["case_ids"] == old_family["case_ids"]
        assert family["raw_hashes"] == old_family["raw_hashes"]
        for r in (0, 1):
            score = compact_records["cases"][f"{fid}/e0r{r}"]["program"]["score"]
            assert family["program_hashes"][r] == digest(score)


def test_emission_copies_training_bytes_and_preserves_originals(tmp_path):
    m = module()
    assert hasattr(m, "emit_compact"), "versioned compact emitter missing"
    old_prepared = Path("data/intermediate-supervision-v1/prepared-v2")
    old_draft = Path("data/intermediate-supervision-v1/fresh-draft")
    before = {str(p): m.file_sha256(p) for root in (old_prepared, old_draft) for p in root.rglob("*") if p.is_file()}
    draft, prepared = tmp_path / "fresh-draft-compact-v3", tmp_path / "prepared-v3"
    result = m.emit_compact(source_prepared=old_prepared, source_draft=old_draft, draft=draft, prepared=prepared)
    assert result["cases_changed"] == 320
    assert result["training_copied_byte_identical"]
    assert before == {p: m.file_sha256(p) for p in before}
    old_manifest = json.loads((old_prepared / "train-binding-manifest.json").read_text())
    for name in [*old_manifest["files_sha256"], "train-binding-manifest.json"]:
        assert (prepared / name).read_bytes() == (old_prepared / name).read_bytes()
    from experiments.intermediate_data import preflight

    assert set(preflight(prepared)["partitions"]) == {"train", "validation", "calibration", "confirmation"}
    with pytest.raises(FileExistsError):
        m.emit_compact(source_prepared=old_prepared, source_draft=old_draft, draft=draft, prepared=prepared)
