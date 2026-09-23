"""CPU-only tests for one post-hoc sampled draw on six greedy-capped questions."""

import copy
import hashlib
import importlib
import json
import math

import pytest
import torch
from test_native_reasoning_followup import CharacterTokenizer, scripted_scorer


def probe():
    return importlib.import_module("experiments.native_sampling_probe")


def test_presence_penalty_uses_generated_ids_once_and_leaves_raw_logits_unchanged():
    raw = torch.tensor([2., 2., 2., 2.])
    original = raw.clone()
    actual = probe().sampling_logits(raw, generated_ids=[1, 1, 3])
    assert torch.equal(raw, original)
    assert torch.equal(actual, torch.tensor([2., .5, 2., .5]))
    # Prompt-only token zero receives no penalty; the API accepts only generated history.
    assert actual[0] == 2 and actual[1] == .5


def test_top_p_keeps_the_threshold_crossing_token_after_top_k():
    raw = torch.tensor([math.log(.6), math.log(.3), math.log(.06), math.log(.04)] + [-100.] * 21)
    filtered = probe().sampling_logits(raw, generated_ids=[])
    assert torch.isfinite(filtered).nonzero().flatten().tolist() == [0, 1, 2]
    assert torch.equal(filtered[:3], raw[:3])
    uniform = probe().sampling_logits(torch.zeros(25), generated_ids=[])
    assert torch.isfinite(uniform).sum() <= 20


def test_fresh_per_case_generators_reproduce_the_same_draw_without_global_rng_dependence():
    raw = torch.tensor([0., .1, .2, .3])

    def draw(seed):
        generator = probe().case_generator("cpu", seed)
        generated = []
        for _ in range(20):
            generated.append(probe().sample_token(raw, generated, generator))
        return generated

    expected = draw(42)
    torch.manual_seed(990)
    torch.rand(19)
    assert draw(42) == expected
    assert draw(43) != expected


def parent_fixture():
    inputs, prepared, reasons, gold = [], [], {}, {}
    for primitive in ("noul", "choice", "score"):
        for offset in range(4):
            index = len(inputs)
            inputs.append({"case_id": f"case-{primitive}-{offset}", "question_id": "q", "primitive": primitive,
                           "category": "fixture", "input": {"state": "unchanged", "question": {"type": primitive}}})
            prepared.append({"reason": {"input_ids": [index, 91], "labels": ["a", "b"], "codes": ["A", "B"],
                                        "code_token_ids": [65, 66], "prompt_sha256": f"prompt-{index}"}})
            reasons[index] = {"index": index, "generation": {"stop_reason": "scratchpad_cap" if offset < 3 else "eos_after_think",
                                                              "scratchpad_tokens": 4096 if offset < 3 else 10,
                                                              "completed_thought": offset == 3},
                              "natural_answer": {"correct": True}}
            gold[str(index)] = "a"
    return {"inputs": inputs, "prepared": prepared}, reasons, gold


def test_selection_uses_only_greedy_termination_then_hash_and_preserves_exact_prompts():
    manifest, reasons, gold = parent_fixture()
    selected = probe().select_capped(manifest, reasons, gold)
    assert len(selected) == 6
    assert {kind: sum(row["primitive"] == kind for row in selected) for kind in ("noul", "choice", "score")} == {
        "noul": 2, "choice": 2, "score": 2}
    expected = sorted(range(3), key=lambda i: hashlib.sha256(f"sampled-native-v1:case-noul-{i}:q".encode()).hexdigest())[:2]
    assert [row["greedy_index"] for row in selected[:2]] == expected
    for row in selected:
        assert row["prepared"] == manifest["prepared"][row["greedy_index"]]["reason"]
        assert row["seed"] == 42 + row["greedy_index"]
    changed_gold = {key: "wrong" for key in gold}
    changed_reasons = copy.deepcopy(reasons)
    for value in changed_reasons.values():
        value["natural_answer"] = {"correct": False}
    assert [row["greedy_index"] for row in probe().select_capped(manifest, changed_reasons, changed_gold)] == [
        row["greedy_index"] for row in selected]


def test_sampled_generation_preserves_raw_tokens_and_uses_same_sampler_through_final_eos(monkeypatch):
    m = probe()
    original_sampler, calls = m.sample_token, []

    def tracked_sampler(logits, history, generator):
        calls.append(list(history))
        return original_sampler(logits, history, generator)

    monkeypatch.setattr(m, "sample_token", tracked_sampler)
    journal = []
    generated = m.generate_sampled(scripted_scorer("ok</think>\nB\x00extra"), [1, 2], CharacterTokenizer(),
                                   generator=m.case_generator("cpu", 42), deadline=100,
                                   clock=lambda: 0, on_token=journal.append)
    assert generated["generated_ids"] == list(b"ok</think>\nB\x00")
    assert generated["completed_thought"] and generated["natural_final_complete"]
    assert generated["natural_final_text"] == "\nB"
    assert [event["token_id"] for event in journal] == generated["generated_ids"]
    assert [event["phase"] for event in journal][-3:] == ["final"] * 3
    assert [len(history) for history in calls] == list(range(len(generated["generated_ids"])))
    assert calls[-1] == list(b"ok</think>\nB")


@pytest.mark.parametrize("kwargs,reason,count", [({"scratchpad_tokens": 3}, "scratchpad_cap", 3),
                                                ({"deadline": 0}, "walltime", 0)])
def test_sampled_generation_keeps_caps_unresolved_without_closing_thought(kwargs, reason, count):
    options = {"deadline": 100, **kwargs}
    generated = probe().generate_sampled(scripted_scorer("unfinished"), [1, 2], CharacterTokenizer(),
                                         generator=probe().case_generator("cpu", 42), clock=lambda: 0, **options)
    assert generated["stop_reason"] == reason
    assert len(generated["generated_ids"]) == count
    assert not generated["completed_thought"] and not generated["natural_final_complete"]


def test_archive_retains_draw_seed_raw_tokens_and_unmodified_conditional_readout(tmp_path, monkeypatch):
    m = probe()
    manifest, reasons, gold = parent_fixture()
    selected = m.select_capped(manifest, reasons, gold)
    for row in selected:
        row["prepared"].update(prompt="<think>", input_ids=list(b"<think>"))
    score_calls, draw_seeds = [], []

    class Scorer:
        device = "cpu"

        def score(self, input_ids, code_ids, labels):
            score_calls.append(input_ids)
            return {"raw_code_logits": [9., 3.], "semantic_probabilities": {"a": .9975273768, "b": .0024726232},
                    "valid_code_mass": .4, "greedy_code_valid": True}

    def generate(scorer, input_ids, tokenizer, **kwargs):
        draw_seeds.append(kwargs["generator"].initial_seed())
        for index, token in enumerate(b"done</think>A\x00"):
            kwargs["on_token"]({"index": index, "token_id": token, "phase": "thought" if index < 12 else "final"})
        return {"generated_ids": list(b"done</think>A\x00"), "generated_text": "done</think>A\x00",
                "thought_ids": list(b"done</think>"), "thought_text": "done</think>", "scratchpad_tokens": 12,
                "final_ids": [65, 0], "natural_final_text": "A", "final_tokens": 2, "completed_thought": True,
                "natural_final_complete": True, "stop_reason": "eos_after_think", "error": None,
                "elapsed_seconds": 1, "deadline_reached": False}

    monkeypatch.setattr(m, "generate_sampled", generate)
    result = m.collect(selected, CharacterTokenizer(), Scorer(), tmp_path / "sampled", {"fixed_parent": True}, clock=lambda: 0)
    assert result["selected_questions"] == 6 and result["natural"]["correct"] == 6
    assert "post-hoc" in result["interpretation"]
    first = json.loads((tmp_path / "sampled" / "question-000.json").read_text())
    assert first["seed"] == selected[0]["seed"]
    assert first["conditional_readout"]["raw_code_logits"] == [9., 3.]
    assert first["generation"]["generated_ids"] == list(b"done</think>A\x00")
    journal = [json.loads(line) for line in (tmp_path / "sampled" / "tokens-000.jsonl").read_text().splitlines()]
    assert [event["token_id"] for event in journal] == first["generation"]["generated_ids"]
    assert len(score_calls) == 6
    assert draw_seeds == [row["seed"] for row in selected]
    with pytest.raises(FileExistsError):
        m.collect(selected, CharacterTokenizer(), Scorer(), tmp_path / "sampled", {}, clock=lambda: 0)


def test_cli_help_is_read_only_and_probe_caps_cannot_extend():
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "experiments.native_sampling_probe", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "--preflight" in result.stdout
    with pytest.raises(ValueError, match="720|24576"):
        probe().probe_budget(max_seconds=721)
    with pytest.raises(ValueError, match="720|24576"):
        probe().probe_budget(total_tokens=24577)


def test_greedy_runtime_source_pins_reject_changed_dependency(tmp_path):
    dependency = tmp_path / "old_runtime.py"
    dependency.write_text("unchanged source")
    pins = {str(dependency): hashlib.sha256(dependency.read_bytes()).hexdigest()}
    probe().verify_source_pins(pins)
    dependency.write_text("changed source")
    with pytest.raises(ValueError, match="source.*changed"):
        probe().verify_source_pins(pins)


def test_sampled_final_token_cap_keeps_closed_thought_but_natural_answer_unresolved():
    result = probe().generate_sampled(scripted_scorer("ok</think>ABextra"), [1, 2], CharacterTokenizer(),
                                      generator=probe().case_generator("cpu", 42), deadline=100,
                                      final_tokens=2, clock=lambda: 0)
    assert result["completed_thought"]
    assert not result["natural_final_complete"]
    assert result["stop_reason"] == "final_token_cap"
    assert result["final_ids"] == list(b"AB")
