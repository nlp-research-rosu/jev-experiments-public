import copy
import math

import pytest
import torch
from test_consistent_cached_inference import wrapper
from torch import nn


def request():
    return {"state": {"record": "x" * 140}, "questions": {
        "n": {"type": "noul", "instructions": "Confirmed?"},
        "c": {"type": "choice", "instructions": "Select.", "criteria": {"a": "short", "b": "long " * 40}},
        "s": {"type": "score", "instructions": "Level?", "criteria": ["zero", "one", "two " * 30]},
    }}


def test_capture_reads_same_cached_hidden_units_without_changing_final_scores():
    from experiments.intermediate_evaluation import capture_case

    engine = wrapper()
    aux = nn.Linear(2, 2).eval()
    with torch.no_grad():
        aux.weight.copy_(torch.tensor([[.01, 1.], [.02, -1.]]))
        aux.bias.copy_(torch.tensor([.1, -.2]))
    req = request()
    before = copy.deepcopy(req)
    expected = engine.evaluate(req, details=True)
    original_method = engine.model._read.__func__
    capture = capture_case(engine, {"id": "case", "request": req}, aux)
    assert capture["response"]["answers"] == expected["answers"]
    assert capture["response"]["model"] == engine.model_id
    assert capture["stats"]["forward_calls"] == expected["usage"]["forward_calls"]
    assert req == before
    assert engine.model._read.__func__ is original_method and "_read" not in engine.model.__dict__
    compiled, prompts, _ = engine.prepare(req)
    prefixes = engine.prefix_lengths(compiled, prompts)
    for index, (prompt, prefix) in enumerate(zip(prompts, prefixes, strict=True)):
        bucket = ((len(prompt) - prefix + 64) // 64) * 64
        shift = (.1 + prefix / 1000 + .5 if prefix else 0.) + .4 + bucket / 1000
        hidden = torch.tensor([float(sum(prompt)), shift])
        wanted = aux(hidden).detach()
        torch.testing.assert_close(torch.tensor(capture["auxiliary"][index]), wanted, rtol=1e-6, atol=1e-5)


def test_capture_restores_readout_after_auxiliary_failure():
    from experiments.intermediate_evaluation import capture_case

    class Broken(nn.Module):
        def forward(self, value):
            raise RuntimeError("aux failure")
    engine = wrapper()
    original = engine.model._read.__func__
    with pytest.raises(RuntimeError, match="aux failure"):
        capture_case(engine, {"id": "case", "request": request()}, Broken())
    assert engine.model._read.__func__ is original and "_read" not in engine.model.__dict__
    assert engine.evaluate(request(), details=True)["answers"]


def test_fact_metrics_weight_questions_not_candidate_count_and_keep_units_visible():
    from experiments.intermediate_evaluation import fact_metrics
    from openjev.judgments import assemble_response, compile_request

    schema = {"features": [{"name": "receipt", "kind": "binary", "scale": 1},
                           {"name": "measurement", "kind": "regression", "scale": 100}]}
    suite = {"cases": [{"id": "c", "family_id": "f", "category": "test", "domain": "test",
                        "request": {"state": {}, "questions": {"n": {"type": "noul"},
                          "c": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}},
                        "expected": {"n": True, "c": "a"}}]}
    aux = {"cases": {"c": {"targets": [1., 20.], "mask": [True, True]}}}
    response = assemble_response(compile_request(suite["cases"][0]["request"]),
                                 [math.log(4), math.log(4), 0.], details=True)
    captured = {"c": {"auxiliary": [[0., .2], [10., .2], [-10., .4]], "response": response}}
    m = fact_metrics(suite, captured, aux, schema)
    assert m["features"]["receipt"]["questions"] == 2
    assert m["features"]["receipt"]["correct"] == 0  # both pooled probabilities tie at .5
    assert m['features']['receipt']['ambiguous_positive'] == 2
    assert m['features']['receipt']['ambiguous_negative'] == 0
    assert m['features']['receipt']['false_negatives'] == 0
    assert m["features"]["measurement"]["mean_absolute_error"] == pytest.approx(5.)
    assert m["joint_binary"]["questions"] == 2
    assert m["joint_binary"]["all_units_facts_correct"] == 0
    assert m["joint_binary"]["facts_and_final_correct"] == 0
    assert m["features"]["receipt"]["logloss"] == pytest.approx((math.log(2) + (math.log1p(math.exp(-10)) + math.log1p(math.exp(10))) / 2) / 2)


def test_masked_numeric_targets_do_not_become_false_world_facts():
    from experiments.intermediate_evaluation import fact_metrics
    schema = {"features": [{"name": "value", "kind": "regression", "scale": 100}]}
    suite = {"cases": [{"id": "c", "family_id": "f", "category": "t", "domain": "t", "request": {
        "state": {}, "questions": {"q": {"type": "noul"}}}, "expected": {"q": False}}]}
    m = fact_metrics(suite, {"c": {"auxiliary": [[999.]], "response": {"answers": {
        "q": {"type": "noul", "noul": .2}}}}}, {"cases": {"c": {"targets": [0.], "mask": [False]}}}, schema)
    assert m["features"]["value"]["questions"] == 0
    assert m["features"]["value"]["mean_absolute_error"] is None
    assert m["joint_binary"]["questions"] == 0


def test_calibration_uses_binary_log_odds_and_preserves_policy_identity():
    from experiments.intermediate_evaluation import calibrated_responses, calibration_records
    suite = {"cases": [{"id": "c", "family_id": "f", "request": {"state": {}, "questions": {
        "n": {"type": "noul"}, "s": {"type": "score", "criteria": ["low", "high"]}}},
        "expected": {"n": True, "s": 1}}]}
    captures = {"c": {"logits": [math.log(4), 100., 102.], "response": {
        "model": "checkpoint/policy", "execution_policy": {"id": "fixed"}}}}
    rows = calibration_records(suite, captures, origin="new")
    assert rows[0]["logits"] == [0., math.log(4)] and rows[0]["target_index"] == 1
    out = calibrated_responses(suite, captures, {"noul": 2., "score": 2.})["c"]
    assert out["model"] == "checkpoint/policy" and out["execution_policy"] == {"id": "fixed"}
    assert out['probability_transform']['temperatures'] == {'noul': 2., 'score': 2.}
    assert out["answers"]["n"]["noul"] == pytest.approx(2 / 3)
    assert out["answers"]["s"]["probabilities"]["1"] == pytest.approx(1 / (1 + math.exp(-1)))


def test_capture_scatter_across_multiple_b4_tiles_and_prefix_groups(monkeypatch):
    from experiments.intermediate_evaluation import capture_case
    engine = wrapper()
    req = {'state': {'text': 'record ' * 30}, 'questions': {
        'q': {'type': 'choice', 'criteria': {f'key{i}': f'Option {i}' for i in range(9)}},
        'n': {'type': 'noul'},
    }}
    # Different valid per-unit prefix lengths produce multiple groups while
    # keeping the underlying full prompts exactly unchanged.
    prefixes = engine.prefix_lengths
    def varied(compiled, prompts):
        values = prefixes(compiled, prompts)
        values[-1] = max(0, values[-1] - 32)
        return values
    monkeypatch.setattr(engine, 'prefix_lengths', varied)
    aux = nn.Linear(2, 1).eval()
    result = capture_case(engine, {'id': 'many', 'request': req}, aux)
    plain = engine.evaluate(req, details=True)
    assert result['response']['answers'] == plain['answers']
    assert len(result['auxiliary']) == 10 and all(len(row) == 1 for row in result['auxiliary'])
    assert result['stats']['suffix_forward_calls'] >= 3
    assert result['stats']['prefix_groups'] >= 2
