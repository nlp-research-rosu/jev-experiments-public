from pathlib import Path

import pytest


def test_diagnostic_cases_have_complete_valid_gold():
    from openjev.benchmark import load_cases
    from openjev.schema import parse_schema, valid_answer

    cases, provenance = load_cases(Path(__file__).resolve().parents[1] / "data", "diagnostic")
    assert len(cases) == 32
    assert len({c["id"] for c in cases}) == len(cases)
    assert all(valid_answer(c["expected"], parse_schema(c["schema"])) for c in cases)
    assert len(provenance["diagnostic.json"]) == 64


def test_upstream_cases_remain_unlabeled_and_include_255_choices():
    from openjev.benchmark import load_cases

    cases, _ = load_cases(Path(__file__).resolve().parents[1] / "data", "upstream")
    assert len(cases) == 4
    assert all("expected" not in c for c in cases)
    assert max(len(f.get("choices", [])) for c in cases for f in c["schema"].values()) == 255


def test_report_counts_quality_once_but_all_latency_repetitions():
    from openjev.benchmark import make_summary

    cases = [{"id": "a", "schema": {"ok": {"type": "boolean"}}, "expected": {"ok": True}}]
    rows = [
        {"id": "a", "repeat": 0, "values": {"ok": True}, "elapsed_ms": 10},
        {"id": "a", "repeat": 1, "values": {"ok": False}, "elapsed_ms": 30},
    ]
    result = make_summary(cases, rows)
    assert result["labeled_fields"] == 1
    assert result["field_accuracy"] == 1
    assert result["median_ms"] == 20
    assert result["timed_evaluations"] == 2
    assert result["repeat_value_disagreements"] == 1


def test_loader_rejects_duplicate_case_ids(tmp_path):
    import json

    from openjev.benchmark import load_cases

    case = {"id": "a", "context": "hi", "expected": {"ok": True}}
    (tmp_path / "diagnostic.json").write_text(
        json.dumps({"schema": {"ok": {"type": "boolean"}}, "cases": [case, case]})
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(tmp_path, "diagnostic")
