"""CPU-only contracts for completed original-model reasoning."""

import copy
import hashlib
import importlib
import json

import pytest


def probe():
    return importlib.import_module("experiments.native_reasoning_followup")


class CharacterTokenizer:
    chat_template = "native fixture"
    all_special_ids = [0]
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True
        end = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        return "\n".join(m["content"] for m in messages) + "\nASSISTANT:\n" + end


def suite():
    cases = []
    for category in range(10):
        for variant in range(2):
            cases.append({
                "id": f"case-{category}-{variant}", "category": f"category-{category}",
                "family_id": f"family-{category}", "rationale": "GOLD_RATIONALE_CANARY",
                "request": {"state": {"evidence": "fact", "rules": ["apply all definitions"]}, "questions": {
                    "n1": {"type": "noul", "instructions": "Has completion been established?"},
                    "n2": {"type": "noul", "instructions": "Was this authorized?"},
                    "n3": {"type": "noul", "instructions": "Was this attempted?"},
                    "c1": {"type": "choice", "instructions": "Choose exact status.",
                           "criteria": {"yes": {"rule": "completed"}, "no": "not completed"}},
                    "s1": {"type": "score", "instructions": "Choose applicable level.",
                           "criteria": ["absent", {"required": "all evidence"}]},
                }}, "expected": {"n1": True, "n2": False, "n3": True, "c1": "no", "s1": 1},
            })
    return {"cases": cases, "relations": []}


def test_fixed_subset_uses_only_input_hashes_and_preserves_separate_gold():
    original = suite()
    selected = probe().select_questions(original)
    assert len(selected) == 30
    assert {kind: sum(row["primitive"] == kind for row in selected) for kind in ("noul", "choice", "score")} == {
        "noul": 10, "choice": 10, "score": 10}
    chosen = min(("case-0-0", "case-0-1"), key=lambda x: hashlib.sha256(("completed-native-v1:" + x).encode()).hexdigest())
    assert {row["case_id"] for row in selected if row["category"] == "category-0"} == {chosen}
    chosen_qid = min(("n1", "n2", "n3"), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    assert {row["question_id"] for row in selected if row["primitive"] == "noul"} == {chosen_qid}
    changed = copy.deepcopy(original)
    changed["cases"].reverse()
    for case in changed["cases"]:
        case["expected"] = {key: "changed prediction-independent gold" for key in case["expected"]}
    assert [(r["case_id"], r["question_id"]) for r in probe().select_questions(changed)] == [
        (r["case_id"], r["question_id"]) for r in selected]
    assert all(set(row["input"]) == {"state", "question"} and "expected" not in row["input"] for row in selected)
    assert original == suite()


@pytest.mark.parametrize("primitive", ["noul", "choice", "score"])
def test_plain_prompt_keeps_verbatim_semantics_and_same_canonical_mapping(primitive):
    row = next(row for row in probe().select_questions(suite()) if row["primitive"] == primitive)
    direct = probe().prepare_prompt(CharacterTokenizer(), row["input"], mode="direct", order="canonical")
    reason = probe().prepare_prompt(CharacterTokenizer(), row["input"], mode="reason", order="canonical")
    reverse = probe().prepare_prompt(CharacterTokenizer(), row["input"], mode="direct", order="reversed")
    assert direct["messages"] == reason["messages"]
    assert direct["labels"] == reason["labels"] == list(reversed(reverse["labels"]))
    payload = json.loads(direct["messages"][1]["content"])
    assert payload["Evidence and state"] == row["input"]["state"]
    assert payload["Instructions"] == row["input"]["question"]["instructions"]
    assert "GOLD_RATIONALE_CANARY" not in direct["prompt"]
    assert row["case_id"] not in direct["prompt"] and row["question_id"] not in direct["prompt"]
    assert '"type"' not in direct["prompt"] and "noul" not in direct["prompt"]
    assert direct["prompt"].endswith(probe().FINAL_PREFIX)
    if primitive == "choice":
        assert [option["criterion"] for option in payload["Options"]] == ["not completed", {"rule": "completed"}]
    if primitive == "score":
        assert [option["criterion"] for option in payload["Options"]] == ["absent", {"required": "all evidence"}]


@pytest.mark.parametrize("text,valid,label", [(" A \n", True, "false"), ("B", True, "true"),
                                               ("Answer: A", False, None), ("AB", False, None), ("C", False, None)])
def test_natural_answer_parser_requires_only_one_allowed_code(text, valid, label):
    result = probe().parse_natural_answer(text, ["A", "B"], ["false", "true"])
    assert result["format_valid"] is valid
    assert result["label"] == label


def test_unclosed_or_capped_thought_never_becomes_forced_solution():
    row = probe().select_questions(suite())[0]
    prepared = probe().prepare_prompt(CharacterTokenizer(), row["input"], mode="reason", order="canonical")
    for reason in ("scratchpad_cap", "walltime", "eos_before_think_close", "error"):
        generation = {"completed_thought": False, "thought_ids": list(b"unfinished"), "stop_reason": reason}
        assert probe().prepare_completed_readout(CharacterTokenizer(), prepared, generation) is None


def test_completed_readout_uses_exact_same_prefix_and_actual_closed_tokens():
    row = probe().select_questions(suite())[0]
    tokenizer = CharacterTokenizer()
    direct = probe().prepare_prompt(tokenizer, row["input"], mode="direct", order="canonical")
    reason = probe().prepare_prompt(tokenizer, row["input"], mode="reason", order="canonical")
    thought = tokenizer.encode("done</think>")
    final = probe().prepare_completed_readout(tokenizer, reason, {"completed_thought": True, "thought_ids": thought})
    assert final["input_ids"] == reason["input_ids"] + thought + tokenizer.encode(probe().FINAL_PREFIX)
    assert final["prefix"] == direct["prefix"] == probe().FINAL_PREFIX
    assert final["labels"] == direct["labels"]


def test_budget_cannot_extend_and_uses_minimum_of_per_question_and_total_deadlines():
    m = probe()
    for kwargs in ({"max_seconds": 1801}, {"question_seconds": 121}, {"scratchpad_tokens": 4097},
                   {"total_tokens": 122881}, {"final_tokens": 33}, {"max_seconds": float("nan")}):
        with pytest.raises(ValueError):
            m.Budget(**kwargs)
    budget = m.Budget(started=10)
    assert budget.allowance(now=20, consumed=0) == {"deadline": 140, "scratchpad_tokens": 4096}
    assert budget.allowance(now=1800, consumed=122879) == {"deadline": 1810, "scratchpad_tokens": 1}
    assert budget.allowance(now=1810, consumed=0) is None
    assert budget.allowance(now=20, consumed=122880) is None


def scripted_scorer(text, *, after_step=None):
    """Tiny actual BF16 projection with a scripted body; no pretrained model is loaded."""
    from types import SimpleNamespace

    import torch

    from experiments.architecture_native import NativeScorer

    emitted = list(text.encode())

    class Body(torch.nn.Module):
        position = 0

        def forward(self, *, input_ids, attention_mask, use_cache, past_key_values=None):
            assert use_cache is True
            assert attention_mask.shape[1] == 2 + self.position
            if self.position:
                assert input_ids.shape[1] == 1 and past_key_values == "cache"
            hidden = torch.zeros((1, 1, 128), dtype=torch.bfloat16)
            hidden[0, 0, emitted[self.position]] = 10
            self.position += 1
            if after_step:
                after_step()
            return SimpleNamespace(last_hidden_state=hidden, past_key_values="cache")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()
            self.head = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                self.head.weight.copy_(torch.eye(128))
            self.config = SimpleNamespace(vocab_size=128, hidden_size=128)

        def get_output_embeddings(self):
            return self.head

    return NativeScorer(Model())


def test_generation_waits_for_natural_close_then_preserves_actual_final_and_all_raw_tokens():
    journal = []
    result = probe().generate_completed(scripted_scorer("ok</think>\nB\x00extra"), [1, 2], CharacterTokenizer(),
                                        deadline=100, scratchpad_tokens=4096, final_tokens=32,
                                        clock=lambda: 0, on_token=journal.append)
    assert result["completed_thought"] is True
    assert result["thought_ids"] == list(b"ok</think>")
    assert result["final_ids"] == list(b"\nB\x00")
    assert result["generated_ids"] == list(b"ok</think>\nB\x00")
    assert result["natural_final_text"] == "\nB"
    assert result["natural_final_complete"] is True
    assert result["stop_reason"] == "eos_after_think"
    assert [event["token_id"] for event in journal] == result["generated_ids"]


@pytest.mark.parametrize("text,kwargs,reason,closed,complete", [
    ("unfinished", {"scratchpad_tokens": 3}, "scratchpad_cap", False, False),
    ("bad\x00", {}, "eos_before_think_close", False, False),
    ("ok</think>abcdef", {"final_tokens": 2}, "final_token_cap", True, False),
])
def test_generation_caps_and_eos_never_invent_completion(text, kwargs, reason, closed, complete):
    result = probe().generate_completed(scripted_scorer(text), [1, 2], CharacterTokenizer(),
                                        deadline=100, clock=lambda: 0, **kwargs)
    assert result["completed_thought"] is closed
    assert result["natural_final_complete"] is complete
    assert result["stop_reason"] == reason
    assert "</think>" in result["generated_text"] if closed else "</think>" not in result["generated_text"]


def test_generation_deadline_stops_before_next_token_and_preserves_partial_text():
    now = [0]
    scorer = scripted_scorer("unfinished", after_step=lambda: now.__setitem__(0, now[0] + 1))
    result = probe().generate_completed(scorer, [1, 2], CharacterTokenizer(), deadline=2, clock=lambda: now[0])
    assert result["generated_ids"] == list(b"un")
    assert result["stop_reason"] == "walltime"
    assert not result["completed_thought"]


class FixtureScorer:
    def __init__(self):
        self.scored_inputs = []

    def score(self, input_ids, code_ids, labels):
        self.scored_inputs.append(list(input_ids))
        return {"raw_code_logits": [1., 0.], "semantic_probabilities": {labels[0]: .7310585786, labels[1]: .2689414214},
                "valid_code_mass": .2, "greedy_code_valid": False, "greedy_token_id": 127,
                "full_vocab_logsumexp": 3., "vocabulary_size": 128}


def test_collector_scores_only_closed_thoughts_and_preserves_natural_answer_separately(tmp_path, monkeypatch):
    m = probe()
    rows = m.select_questions(suite())[:2]
    calls = [0]

    def generate(scorer, input_ids, tokenizer, **kwargs):
        calls[0] += 1
        closed = calls[0] == 1
        text = "done</think>" if closed else "unfinished"
        return {"generated_ids": list((text + ("B\x00" if closed else "")).encode()),
                "generated_text": text + ("B\x00" if closed else ""),
                "thought_ids": list(text.encode()), "thought_text": text, "scratchpad_tokens": len(text),
                "final_ids": [66, 0] if closed else [], "natural_final_text": "B" if closed else "",
                "final_tokens": 2 if closed else 0, "completed_thought": closed,
                "natural_final_complete": closed, "stop_reason": "eos_after_think" if closed else "scratchpad_cap",
                "error": None, "elapsed_seconds": 1, "deadline_reached": False}

    monkeypatch.setattr(m, "generate_completed", generate)
    scorer = FixtureScorer()
    result = m.collect(rows, CharacterTokenizer(), scorer, tmp_path / "run", {"source": "fixed"}, clock=lambda: 0)
    assert len(scorer.scored_inputs) == 5  # two orders × two direct questions + one completed-thought readout
    records = [json.loads(path.read_text()) for path in sorted((tmp_path / "run").glob("reason-*.json"))]
    assert records[0]["natural_answer"]["code"] == "B"
    assert records[0]["conditional_readout"]["raw_code_logits"] == [1., 0.]
    assert records[1]["conditional_readout"] is None
    assert result["natural"]["resolved"] == 1 and result["natural"]["unresolved"] == 1
    assert result["conditional_reason"]["scored"] == 1
    with pytest.raises(FileExistsError):
        m.collect(rows, CharacterTokenizer(), scorer, tmp_path / "run", {"source": "fixed"}, clock=lambda: 0)


def test_collector_preserves_direct_canonical_if_reverse_fails(tmp_path):
    m = probe()

    class FailsSecond(FixtureScorer):
        def score(self, *args):
            if self.scored_inputs:
                raise RuntimeError("interrupted reverse")
            return super().score(*args)

    with pytest.raises(RuntimeError, match="interrupted reverse"):
        m.collect(m.select_questions(suite())[:1], CharacterTokenizer(), FailsSecond(), tmp_path / "run", {}, clock=lambda: 0)
    assert (tmp_path / "run" / "direct-000-canonical.json").exists()
    assert (tmp_path / "run" / "failure.json").exists()
    assert (tmp_path / "run" / "gold.json").exists()


def test_collector_total_scratchpad_budget_stops_remaining_questions_without_regeneration(tmp_path, monkeypatch):
    m = probe()
    generated = []

    def generate(scorer, input_ids, tokenizer, **kwargs):
        generated.append(kwargs["scratchpad_tokens"])
        return {"generated_ids": [120, 120], "generated_text": "xx", "thought_ids": [120, 120],
                "thought_text": "xx", "scratchpad_tokens": 2, "final_ids": [], "final_tokens": 0,
                "natural_final_text": "", "completed_thought": False, "natural_final_complete": False,
                "stop_reason": "scratchpad_cap", "error": None, "elapsed_seconds": 0, "deadline_reached": False}

    monkeypatch.setattr(m, "generate_completed", generate)
    result = m.collect(m.select_questions(suite())[:2], CharacterTokenizer(), FixtureScorer(), tmp_path / "run", {},
                       budget=m.Budget(total_tokens=2), clock=lambda: 0)
    assert generated == [2]
    assert result["scratchpad_tokens"] == 2
    assert result["reason_attempted"] == 1 and result["natural"]["unresolved"] == 2
    assert result["reason_stop"] == "total_scratchpad_cap"


def test_cli_help_and_cap_validation_do_not_load_native_weights():
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "experiments.native_reasoning_followup", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "--preflight" in result.stdout
    result = subprocess.run([sys.executable, "-m", "experiments.native_reasoning_followup", "--max-seconds", "1801"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "max_seconds" in result.stderr


def test_reason_prompt_requires_native_open_thinking_template():
    class NoThinking(CharacterTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return "ASSISTANT: "

    with pytest.raises(ValueError, match="thinking template"):
        probe().prepare_prompt(NoThinking(), probe().select_questions(suite())[0]["input"], mode="reason", order="canonical")


def test_total_wall_cap_preserves_late_natural_answer_as_unresolved_without_conditional_score(tmp_path, monkeypatch):
    m = probe()
    now = [0]

    def generate(scorer, input_ids, tokenizer, **kwargs):
        assert kwargs["deadline"] == 120
        now[0] = 121
        return {"generated_ids": list(b"ok</think>A\x00"), "generated_text": "ok</think>A\x00",
                "thought_ids": list(b"ok</think>"), "thought_text": "ok</think>", "scratchpad_tokens": 10,
                "final_ids": [65, 0], "final_tokens": 2, "natural_final_text": "A", "completed_thought": True,
                "natural_final_complete": True, "stop_reason": "eos_after_think", "error": None,
                "elapsed_seconds": 121, "deadline_reached": True}

    monkeypatch.setattr(m, "generate_completed", generate)
    scorer = FixtureScorer()
    result = m.collect(m.select_questions(suite())[:2], CharacterTokenizer(), scorer, tmp_path / "run", {},
                       budget=m.Budget(max_seconds=120), clock=lambda: now[0])
    assert len(scorer.scored_inputs) == 4
    assert result["natural"]["unresolved"] == 2
    assert result["conditional_reason"]["scored"] == 0
    assert result["reason_attempted"] == 1 and result["reason_stop"] == "total_walltime_cap"
    record = json.loads((tmp_path / "run" / "reason-000.json").read_text())
    assert record["natural_answer"]["code"] == "A" and not record["natural_answer"]["resolved"]


def test_probability_ties_get_no_correct_credit_and_confidence_denominators_stay_explicit():
    rows = [{"primitive": "noul", "category": "a", "gold": False},
            {"primitive": "choice", "category": "b", "gold": "x"},
            {"primitive": "noul", "category": "a", "gold": True}]
    result = probe().probability_metrics(rows, {
        0: {"false": .5000000000003, "true": .4999999999997}, 1: {"x": .05, "y": .95}})
    assert result["correct"] == 0
    assert result["ambiguous_top"] == 1 and result["resolved_predictions"] == 1
    assert result["scored"] == 2 and result["cohort_questions"] == 3
    assert result["coverage"] == pytest.approx(2 / 3)
    confidence = result["confidence"]["0.90"]
    assert confidence["covered"] == 1 and confidence["wrong"] == 1
    assert confidence["coverage"] == .5 and confidence["coverage_denominator_scored"] == 2
    assert confidence["coverage_of_full_cohort"] == pytest.approx(1 / 3)
    assert result["by_category"]["a"]["cohort_questions"] == 2
    assert result["by_category"]["a"]["ambiguous_top"] == 1


def test_completed_readout_preserves_generated_tokenization_even_when_text_could_reencode_differently():
    class AlternateTokenizer(CharacterTokenizer):
        def decode(self, ids, **kwargs):
            # Token 200 is an alternative native spelling of the same four characters.
            expanded = []
            for token in ids:
                expanded.extend(list(b"done") if token == 200 else [token])
            return bytes(expanded).decode()

    tokenizer = AlternateTokenizer()
    row = probe().select_questions(suite())[0]
    prepared = probe().prepare_prompt(tokenizer, row["input"], mode="reason", order="canonical")
    thought = [200] + list(b"</think>")
    result = probe().prepare_completed_readout(tokenizer, prepared, {"completed_thought": True, "thought_ids": thought})
    assert result["input_ids"] == prepared["input_ids"] + thought + tokenizer.encode(probe().FINAL_PREFIX)
    assert result["text_reencoding_matches"] is False
