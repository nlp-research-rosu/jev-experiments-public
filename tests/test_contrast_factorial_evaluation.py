"""Calibration selection cannot see retained contexts or select by gold labels."""

import copy

import pytest


def row(identity, source, context, group=None):
    return {
        "id": identity,
        "group_id": group or identity,
        "provenance": {"assigned_split": "validation", "dataset": source},
        "examples": [
            {"state": {"text": context}, "question": {"type": "noul", "instructions": "P?"}, "target": {"truth": True}}
        ],
    }


def test_calibration_excludes_retained_ids_groups_and_duplicate_contexts():
    from experiments.contrast_factorial_evaluation import select_legacy_calibration

    keep = [row("held", "s1", "held context", "g-held")]
    rows = keep + [row("alias", "s1", "another rendering", "g-held"), row("copy", "s2", "held context")]
    rows += [row(f"{s}-{i}", s, f"{s} independent {i}") for s in ["s1", "s2"] for i in range(4)]
    out = select_legacy_calibration(rows, keep, per_source=2)
    assert len(out) == 4
    assert {r["id"] for r in out}.isdisjoint({"held", "alias", "copy"})
    assert sum(r["provenance"]["dataset"] == "s1" for r in out) == 2
    changed = copy.deepcopy(rows)
    for r in changed:
        r["examples"][0]["target"]["truth"] = False
    assert [r["id"] for r in out] == [r["id"] for r in select_legacy_calibration(changed, keep, per_source=2)]


def test_insufficient_disjoint_population_or_wrong_split_is_rejected():
    from experiments.contrast_factorial_evaluation import select_legacy_calibration

    rows = [row("a", "s", "same"), row("b", "s", "same")]
    with pytest.raises(ValueError):
        select_legacy_calibration(rows, [], per_source=2)
    rows[1]["provenance"]["assigned_split"] = "test"
    with pytest.raises(ValueError):
        select_legacy_calibration(rows, [], per_source=1)
