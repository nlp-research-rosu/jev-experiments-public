import json

import pytest


def test_snapshot_parser_accepts_only_json_callback_and_rejects_extra_code():
    from experiments.published_cases_replay import parse_snapshot

    assert parse_snapshot('__VIEWER_DATA__({"eval":{}});') == {"eval": {}}
    for raw in (
        "__VIEWER_DATA__({});alert(1)",
        '__VIEWER_DATA__({"x":NaN});',
        '__VIEWER_DATA__({"x":1,"x":2});',
        "executeSomething()",
    ):
        with pytest.raises(ValueError):
            parse_snapshot(raw)


def test_extraction_replays_only_reached_nodes_without_answer_leakage():
    from experiments.published_cases_replay import extract_bundles

    q = {"type": "noul", "instructions": "Is a refund requested?"}
    snapshot = {
        "eval": {
            "id": "test",
            "title": "Test",
            "n_cases": 100,
            "documents": [{"message": "Please refund me."}],
            "questions": [q],
            "examples": [{"case_id": "a", "name": "Sample", "label": "all agree"}],
            "cases": {
                "a": {
                    "models": {
                        "typesafe": {
                            "model": "historical",
                            "nodes": [
                                {
                                    "node": "triage",
                                    "ran": True,
                                    "doc": 0,
                                    "questions": {"refund": 0},
                                    "answers": {"refund": {"type": "noul", "noul": 0.8}},
                                },
                                {"node": "skipped", "ran": False, "doc": None, "questions": {}},
                            ],
                        }
                    },
                    "references": [],
                    "reference_answers": {},
                }
            },
        }
    }
    records = extract_bundles(snapshot)
    assert len(records) == 1
    assert records[0]["request"] == {"state": {"message": "Please refund me."}, "questions": {"refund": q}}
    assert "answers" not in json.dumps(records[0]["request"])
    assert records[0]["historical_answers"]["refund"]["noul"] == 0.8
    snapshot["eval"]["cases"]["a"]["models"]["typesafe"]["nodes"][0]["doc"] = -1
    with pytest.raises(ValueError):
        extract_bundles(snapshot)


def test_reference_null_label_does_not_discard_valid_probabilities():
    from experiments.published_cases_replay import reference_view

    q = {"type": "noul"}
    r = reference_view(
        q,
        {
            "type": "noul",
            "sets": [
                {"value": None, "probabilities": {"true": 0.5, "false": 0.5}},
                {"value": True, "probabilities": {"true": 0.9, "false": 0.1}},
            ],
        },
    )
    assert r["label"] == "true"
    assert r["probabilities"]["true"] == pytest.approx(0.7)
    assert r["probabilistic"]


def test_hard_label_votes_are_not_probabilities_and_ties_have_no_winner():
    from experiments.published_cases_replay import reference_view

    q = {"type": "choice", "criteria": {"a": "A", "b": "B"}}
    r = reference_view(
        q, {"type": "choice", "sets": [{"value": "a", "probabilities": None}, {"value": "b", "probabilities": None}]}
    )
    assert r["status"] == "tied"
    assert r["label"] is None
    assert not r["probabilistic"]
    assert r["rule"] == "label_votes"


def test_incompatible_reference_options_are_not_projected_onto_our_schema():
    from experiments.published_cases_replay import reference_view

    q = {"type": "choice", "criteria": {"cancel": "cancel", "billing": "billing"}}
    r = reference_view(
        q, {"type": "choice", "sets": [{"value": "cancel", "probabilities": {"cancel": 0.6, "refund": 0.4}}]}
    )
    assert r["status"] == "incompatible"
    assert r["label"] is None


def test_display_rounding_is_recorded_and_stored_score_is_preserved():
    from experiments.published_cases_replay import answer_view

    q = {"type": "score", "criteria": ["low", "mid", "high"]}
    r = answer_view(q, {"type": "score", "score": 1.01, "probabilities": {"0": 0.33, "1": 0.33, "2": 0.33}})
    assert r["original_probability_sum"] == pytest.approx(0.99)
    assert sum(r["probabilities"].values()) == pytest.approx(1)
    assert r["reported_score"] == 1.01
    assert r["expected_score"] == pytest.approx(1)
    assert r["label"] is None
