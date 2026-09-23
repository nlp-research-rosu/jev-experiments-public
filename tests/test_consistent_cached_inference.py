"""Canonical state-prefix caching preserves shapes and cache isolation."""

import copy
from types import SimpleNamespace

import pytest
import torch
from test_consistent_inference import ToyJudgment
from torch import nn


class RunningCache:
    def __init__(self, totals, shift, tokens):
        self.totals, self.shift, self.tokens = totals, shift, tokens

    def reorder_cache(self, indices):
        self.totals = self.totals.index_select(0, indices)
        self.shift = self.shift.index_select(0, indices)
        self.tokens = [list(self.tokens[index]) for index in indices.tolist()]


class CachedShapeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0)
        self.calls, self.prefills, self.branches = [], [], []

    def forward(self, input_ids, *, use_cache, attention_mask=None, past_key_values=None):
        batch, length = input_ids.shape
        assert use_cache
        self.calls.append({"ids": input_ids.clone(), "mask": None if attention_mask is None else attention_mask.clone(),
                           "grad_enabled": torch.is_grad_enabled()})
        prefix_tokens = [[] for _ in range(batch)] if past_key_values is None else copy.deepcopy(past_key_values.tokens)
        base = torch.zeros(batch) if past_key_values is None else past_key_values.totals.clone()
        prior_shift = torch.zeros(batch) if past_key_values is None else past_key_values.shift.clone()
        current_shift = batch / 10 + length / 1000 + (.5 if attention_mask is None or attention_mask.all() else 0.)
        prefix = input_ids.float().cumsum(dim=1) + base[:, None]
        shifts = (prior_shift + current_shift)[:, None].expand_as(prefix)
        cache = past_key_values or RunningCache(torch.zeros(batch), torch.zeros(batch), prefix_tokens)
        cache.totals.copy_(prefix[:, -1])
        cache.shift.copy_(shifts[:, -1])
        cache.tokens = [prefix + suffix for prefix, suffix in zip(prefix_tokens, input_ids.tolist(), strict=True)]
        if attention_mask is None:
            self.prefills.append((cache, copy.deepcopy(cache)))
        else:
            self.branches.append(cache)
        return SimpleNamespace(last_hidden_state=torch.stack((prefix, shifts), dim=-1), past_key_values=cache)


class CachedToyJudgment(ToyJudgment):
    def __init__(self):
        super().__init__()
        self.backbone = CachedShapeBackbone()
        self.eval()

    def _read(self, hidden, kinds):
        return torch.where(kinds.bool(), self.binary(hidden).flatten(), self.compatibility(hidden).flatten())


class ChatTokenizer:
    def encode(self, text, **kwargs):
        return [32 if text == "A" else 33]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        rendered = "".join("<" + m["role"] + ">" + m["content"] + "</end>" for m in messages)
        if kwargs["add_generation_prompt"]:
            rendered += "<assistant>"
        return [ord(character) for character in rendered]


def wrapper():
    from openjev.consistent_cached_inference import ConsistentCachedJudgmentEngine
    from openjev.judgment_model import JudgmentEngine

    return ConsistentCachedJudgmentEngine(JudgmentEngine(CachedToyJudgment(), ChatTokenizer(), model_id="fixture"))


def test_prefix_boundary_is_state_derived_and_independent_of_question_population():
    engine = wrapper()
    state = {"text": "Literal \nQUESTION:\n inside JSON must remain escaped.", "history": "x" * 150}
    question = {"type": "choice", "instructions": "Choose.", "criteria": {"a": "first", "b": "second"}}
    single = {"state": state, "questions": {"q": question}}
    mixed = {"state": state, "questions": {"long": {"type": "noul", "instructions": "long " * 100}, "q": question}}
    before = copy.deepcopy(mixed)
    a, ap, _ = engine.prepare(single)
    b, bp, _ = engine.prepare(mixed)
    al, bl = engine.prefix_lengths(a, ap), engine.prefix_lengths(b, bp)
    assert al == bl[1:]
    assert al[0] == al[1] > 0
    assert all(length % 32 == 0 and length < len(prompt) for length, prompt in zip(bl, bp, strict=True))
    assert ap[0][:al[0]] == bp[1][:bl[1]]
    assert mixed == before


@pytest.mark.parametrize("count", [1, 2, 4, 5, 9])
@pytest.mark.parametrize("requested_batch", [1, 2, 4, 5])
def test_suffix_shapes_and_logits_ignore_logical_counts_and_batch_preferences(count, requested_batch):
    from openjev.consistent_cached_inference import score_canonical_cached

    model = CachedToyJudgment()
    prompt = [2] * 32 + [1, 3]
    scores, stats = score_canonical_cached(model, [prompt] * count, [0] * count, [32] * count,
                                           unit_batch_size=requested_batch)
    # Prefix: B1,L32,all-valid => .632; suffix: B4,L64,masked => .464.
    torch.testing.assert_close(scores, torch.full((count,), .68 + .632 + .464), atol=2e-7, rtol=0)
    assert model.backbone.calls[0]["ids"].shape == (1, 32)
    assert model.backbone.calls[0]["mask"] is None
    assert all(call["ids"].shape == (4, 64) for call in model.backbone.calls[1:])
    assert stats["prefill_forward_calls"] == 1
    assert stats["suffix_forward_calls"] == (count + 3) // 4
    assert stats["physical_tokens"] == 32 + 4 * 64 * ((count + 3) // 4)
    assert stats["requested_unit_batch_size"] == requested_batch


def test_distinct_prefixes_buckets_and_permutations_preserve_output_order():
    from openjev.consistent_cached_inference import score_canonical_cached

    prompts = [[2] * 32 + [3], [5] * 32 + [6] * 64, [7, 8], [2] * 32 + [9] * 128, [2] * 32 + [4]]
    prefixes, kinds = [32, 32, 0, 32, 32], [0, 1, 1, 0, 1]
    model = CachedToyJudgment()
    scores, stats = score_canonical_cached(model, prompts, kinds, prefixes)
    expected = torch.cat([score_canonical_cached(CachedToyJudgment(), [p], [k], [n])[0]
                          for p, k, n in zip(prompts, kinds, prefixes, strict=True)])
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)
    order = [3, 2, 0, 4, 1]
    shuffled, _ = score_canonical_cached(model, [prompts[i] for i in order], [kinds[i] for i in order],
                                         [prefixes[i] for i in order])
    torch.testing.assert_close(shuffled, scores[order], rtol=0, atol=0)
    assert stats["prefill_forward_calls"] == 2  # Shared first prefix across two suffix buckets.
    assert stats["suffix_forward_calls"] == 4
    assert stats["prefix_tokens"] == 64


@pytest.mark.parametrize("suffix,bucket", [(1, 64), (63, 64), (64, 128), (127, 128), (128, 192)])
def test_suffix_boundary_always_has_masked_pad_and_full_prefix_attention(suffix, bucket):
    from openjev.consistent_cached_inference import score_canonical_cached

    model = CachedToyJudgment()
    score_canonical_cached(model, [[2] * 32 + [3] * suffix], [0], [32])
    branch = model.backbone.calls[1]
    assert branch["ids"].shape == (4, bucket)
    assert branch["mask"].shape == (4, 32 + bucket)
    assert branch["mask"][:, :32 + suffix].all()
    assert not branch["mask"][:, 32 + suffix:].any()


def test_each_suffix_tile_forks_without_mutating_prefill_or_sharing_branch_storage():
    from openjev.consistent_cached_inference import score_canonical_cached

    model = CachedToyJudgment()
    prompts = [[2] * 32 + [value] for value in range(1, 6)]
    scores, stats = score_canonical_cached(model, prompts, [0] * 5, [32] * 5)
    cached, snapshot = model.backbone.prefills[0]
    torch.testing.assert_close(cached.totals, snapshot.totals, rtol=0, atol=0)
    torch.testing.assert_close(cached.shift, snapshot.shift, rtol=0, atol=0)
    assert cached.tokens == snapshot.tokens == [[2] * 32]
    assert model.backbone.branches[0] is not model.backbone.branches[1]
    assert model.backbone.branches[0].totals.data_ptr() != model.backbone.branches[1].totals.data_ptr()
    assert all(branch is not cached for branch in model.backbone.branches)
    for index, prompt in enumerate(prompts):
        branch = model.backbone.branches[index // 4]
        assert branch.tokens[index % 4][:len(prompt)] == prompt
    assert scores.shape == (5,)
    assert stats["dummy_rows"] == 3
    assert stats["dummy_input_tokens"] == 3
    assert stats["padding_tokens"] == 2 * 4 * 63
    assert stats["physical_tokens"] == stats["prefix_tokens"] + stats["real_suffix_tokens"] + stats["dummy_input_tokens"] + stats["padding_tokens"]


def test_zero_prefix_skips_prefill_and_preserves_all_prompt_tokens():
    from openjev.consistent_cached_inference import score_canonical_cached

    model = CachedToyJudgment()
    logits, stats = score_canonical_cached(model, [[0, 2, 3]], [1], [0])
    assert stats["prefill_forward_calls"] == stats["prefix_tokens"] == 0
    assert stats["suffix_forward_calls"] == 1
    assert model.backbone.branches[0].tokens[0][:3] == [0, 2, 3]
    assert logits.item() == pytest.approx(.1 + 2 * .464 + .1)


@pytest.mark.parametrize("prompts,kinds,prefixes,kwargs", [
    ([], [], [], {}), ([[1]], [], [0], {}), ([[1]], [0], [], {}), ([[]], [0], [0], {}),
    ([[1]], [0], [1], {}), ([[1] * 64], [0], [31], {}), ([[1]], [0], [-1], {}),
    ([[1]], [0], [True], {}), ([[True]], [0], [0], {}), ([[1]], [2], [0], {}),
    ([[1]], [0], [0], {"unit_batch_size": 0}), ([[1, 2]], [0], [0], {"max_input_tokens": 1}),
])
def test_invalid_inputs_fail_before_prefill(prompts, kinds, prefixes, kwargs):
    from openjev.consistent_cached_inference import score_canonical_cached

    model = CachedToyJudgment()
    with pytest.raises(ValueError):
        score_canonical_cached(model, prompts, kinds, prefixes, **kwargs)
    assert model.backbone.calls == []


def test_wrapper_is_raw_callable_state_preserving_and_has_distinct_cached_identity():
    engine = wrapper()
    request = {"state": "A preserved state " * 20,
               "questions": {"a": {"type": "noul"}, "b": {"type": "score", "criteria": ["low", "high"]}}}
    before = copy.deepcopy(request)
    result = engine(request, details=True)
    assert request == before
    assert result["model"] == engine.model_id
    assert "canonical-cache-v1-b4-k64-p32" in result["model"]
    assert result["execution_policy"]["prefix_block_tokens"] == 32
    assert result["execution_policy"]["device_type"] == "cpu"
    assert result["execution_policy"]["padding"] == (
        "suffix rows right-padded with at least one masked token; prefill unpadded"
    )
    assert result["answers"]["a"]["details"]["calibration_id"] == "identity-uncalibrated"
    assert result["elapsed_ms"] >= 0
    assert result["usage"]["physical_tokens"] > 0
    with pytest.raises(ValueError, match="cached"):
        engine.evaluate(request, cached=False)


def test_inference_preserves_parameters_flags_and_disallows_training():
    from openjev.consistent_cached_inference import score_canonical_cached

    model = CachedToyJudgment()
    before = copy.deepcopy(model.state_dict())
    flags = [p.requires_grad for p in model.parameters()]
    logits, _ = score_canonical_cached(model, [[1] * 32 + [2]], [0], [32])
    assert not logits.requires_grad
    assert all(not call["grad_enabled"] for call in model.backbone.calls)
    assert [p.requires_grad for p in model.parameters()] == flags
    assert all(p.grad is None for p in model.parameters())
    for name in before:
        torch.testing.assert_close(model.state_dict()[name], before[name], rtol=0, atol=0)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        score_canonical_cached(model, [[1] * 32 + [2]], [0], [32])
