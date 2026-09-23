import math

import pytest


def test_latency_quantiles_count_every_sample_and_reject_invalid_timings():
    from experiments.judgment_latency import timing_summary

    result = timing_summary(list(range(1, 101)))
    assert result["samples"] == 100
    assert result["p50_ms"] == 50.5
    assert result["p95_ms"] == 95
    assert result["p99_ms"] == 99
    for values in ([], [math.nan], [-1], [math.inf]):
        with pytest.raises(ValueError):
            timing_summary(values)


def test_answer_comparison_detects_nested_decision_and_probability_changes():
    from experiments.judgment_latency import compare_answers

    a = {
        "x": [{"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.4}}, {}],
        "y": {"type": "noul", "noul": 0.49, "probabilities": {"false": 0.51, "true": 0.49}},
    }
    b = {
        "x": [{"type": "choice", "choice": "b", "probabilities": {"a": 0.4, "b": 0.6}}, {}],
        "y": {"type": "noul", "noul": 0.51, "probabilities": {"false": 0.49, "true": 0.51}},
    }
    result = compare_answers(a, b)
    assert result["questions"] == 2
    assert result["decision_changes"] == 2
    assert result["max_probability_difference"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        compare_answers(a, {"wrong": []})


def test_synthetic_scaling_requests_count_actual_units_and_preserve_nesting():
    from experiments.judgment_latency import make_request
    from openjev.judgments import compile_request

    request = make_request("history", 5)
    compiled = compile_request(request)
    assert len(compiled.questions) == 5
    assert len(compiled.units) == 13
    assert compiled.questions[0].path == ("decisions", 0, "answer")
    many = compile_request(make_request("history", 1, choices=128))
    assert len(many.units) == 128
