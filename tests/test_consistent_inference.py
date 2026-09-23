"""Canonical physical shapes isolate inference from logical request packing."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn


class ShapeSensitiveBackbone(nn.Module):
    """Small real tensor model exposing shape and all-valid-mask dependence."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0)
        self.calls = []

    def forward(self, input_ids, attention_mask, use_cache=False):
        batch, length = input_ids.shape
        self.calls.append({"ids": input_ids.clone(), "mask": attention_mask.clone(),
                           "grad_enabled": torch.is_grad_enabled(), "training": self.training})
        prefix = input_ids.float().cumsum(dim=1)
        shift = batch / 10 + length / 1000 + (.5 if attention_mask.all() else 0.)
        return SimpleNamespace(last_hidden_state=torch.stack((prefix, torch.full_like(prefix, shift)), dim=-1))


class ToyJudgment(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ShapeSensitiveBackbone()
        self.compatibility = nn.Linear(2, 1, bias=False)
        self.binary = nn.Linear(2, 1, bias=True)
        with torch.no_grad():
            self.compatibility.weight.copy_(torch.tensor([[.01, 1.]]))
            self.binary.weight.copy_(torch.tensor([[.02, 2.]]))
            self.binary.bias.fill_(.1)
        self.eval()

    @property
    def device(self):
        return self.compatibility.weight.device

    def forward(self, input_ids, attention_mask, score_positions, readout_kind):
        hidden = self.backbone(input_ids, attention_mask, use_cache=False).last_hidden_state
        endpoints = hidden[torch.arange(len(input_ids)), score_positions]
        return torch.where(readout_kind.bool(), self.binary(endpoints).flatten(),
                           self.compatibility(endpoints).flatten())


def variable_score(model, prompts, kinds):
    length = max(map(len, prompts))
    inputs = torch.tensor([row + [0] * (length - len(row)) for row in prompts])
    lengths = torch.tensor(list(map(len, prompts)))
    return model(inputs, torch.arange(length)[None] < lengths[:, None], lengths - 1, torch.tensor(kinds))


class ToyTokenizer:
    def encode(self, text, **kwargs):
        return [32 if text == "A" else 33]

    def apply_chat_template(self, messages, **kwargs):
        return [ord(character) % 20 + 1 for message in messages for character in message["content"]]


def wrapped_engine():
    from openjev.consistent_inference import ConsistentJudgmentEngine
    from openjev.judgment_model import JudgmentEngine

    engine = JudgmentEngine(ToyJudgment(), ToyTokenizer(), model_id="test-checkpoint/sha256-fixture")
    return ConsistentJudgmentEngine(engine)


def test_variable_shapes_change_logits_but_canonical_shape_preserves_single_prompt():
    from openjev.consistent_inference import canonical_score_prompts

    model = ToyJudgment()
    prompt, companion = [1, 2, 3], [4] * 120
    old_single = variable_score(model, [prompt], [0])[0]
    old_mixed = variable_score(model, [prompt, companion], [0, 1])[0]
    assert old_single != old_mixed
    new_single, _ = canonical_score_prompts(model, [prompt], [0])
    new_mixed, _ = canonical_score_prompts(model, [prompt, companion], [0, 1])
    torch.testing.assert_close(new_single[0], new_mixed[0], rtol=0, atol=0)
    assert new_single[0].item() == pytest.approx(.06 + .4 + .128)


@pytest.mark.parametrize("count", [1, 2, 4, 5, 8, 9])
@pytest.mark.parametrize("requested_batch", [1, 2, 4, 5])
def test_logical_counts_and_requested_batch_sizes_keep_fixed_four_row_execution(count, requested_batch):
    from openjev.consistent_inference import canonical_score_prompts

    model = ToyJudgment()
    logits, stats = canonical_score_prompts(model, [[1, 2, 3]] * count, [0] * count,
                                            unit_batch_size=requested_batch)
    torch.testing.assert_close(logits, torch.full((count,), .588), rtol=0, atol=1e-7)
    assert all(call["ids"].shape == (4, 128) for call in model.backbone.calls)
    assert stats["forward_calls"] == (count + 3) // 4
    assert stats["dummy_rows"] == ((count + 3) // 4) * 4 - count
    assert stats["physical_tokens"] == ((count + 3) // 4) * 4 * 128
    assert stats["input_tokens"] == 3 * count
    assert stats["requested_unit_batch_size"] == requested_batch


def test_mixed_buckets_and_permutations_scatter_back_in_original_order():
    from openjev.consistent_inference import canonical_score_prompts

    prompts = [[3] * 128, [1, 2], [4] * 257, [5] * 127, [6] * 129]
    kinds = [0, 1, 0, 1, 0]
    model = ToyJudgment()
    original, _ = canonical_score_prompts(model, prompts, kinds)
    permutation = [3, 0, 4, 2, 1]
    shuffled, stats = canonical_score_prompts(model, [prompts[i] for i in permutation], [kinds[i] for i in permutation])
    torch.testing.assert_close(shuffled, original[permutation], rtol=0, atol=0)
    separate = torch.cat([canonical_score_prompts(model, [prompt], [kind])[0]
                          for prompt, kind in zip(prompts, kinds, strict=True)])
    torch.testing.assert_close(original, separate, rtol=0, atol=0)
    assert stats["forward_calls"] == 3
    assert stats["physical_tokens"] == 4 * (128 + 256 + 384)
    assert stats["dummy_rows"] == 7


@pytest.mark.parametrize("length,bucket", [(1, 128), (127, 128), (128, 256), (255, 256), (256, 384)])
def test_bucket_boundary_always_retains_a_masked_padding_token(length, bucket):
    from openjev.consistent_inference import canonical_score_prompts

    model = ToyJudgment()
    canonical_score_prompts(model, [[7] * length], [0])
    call = model.backbone.calls[0]
    assert call["ids"].shape == (4, bucket)
    assert (call["mask"].sum(dim=1) == length).all()
    assert not call["mask"][:, -1].any()
    assert (call["ids"][:, length:] == 0).all()


def test_dummy_fill_duplicates_real_rows_and_counters_include_duplicate_work():
    from openjev.consistent_inference import canonical_score_prompts

    model = ToyJudgment()
    _, stats = canonical_score_prompts(model, [[8, 9], [1, 2, 3]], [0, 1])
    call = model.backbone.calls[0]
    torch.testing.assert_close(call["ids"][2:], call["ids"][0].expand(2, -1), rtol=0, atol=0)
    torch.testing.assert_close(call["mask"][2:], call["mask"][0].expand(2, -1), rtol=0, atol=0)
    assert stats["physical_rows"] == 4
    assert stats["dummy_input_tokens"] == 4
    assert stats["padding_tokens"] == 512 - 5 - 4
    assert stats["prefix_tokens"] == 0


@pytest.mark.parametrize("prompts,kinds,kwargs", [
    ([], [], {}), ([[]], [0], {}), ([[1]], [], {}), ([[1]], [2], {}), ([[1]], [True], {}),
    ([[True]], [0], {}), ([[-1]], [0], {}), ([[1.5]], [0], {}), ([[1]], [0], {"unit_batch_size": 0}),
    ([[1]], [0], {"unit_batch_size": True}), ([[1, 2]], [0], {"max_input_tokens": 1}),
])
def test_invalid_inputs_are_rejected_before_forward(prompts, kinds, kwargs):
    from openjev.consistent_inference import canonical_score_prompts

    model = ToyJudgment()
    with pytest.raises(ValueError):
        canonical_score_prompts(model, prompts, kinds, **kwargs)
    assert model.backbone.calls == []


def test_inference_disables_gradients_preserves_parameters_and_rejects_training_mode():
    from openjev.consistent_inference import canonical_score_prompts

    model = ToyJudgment()
    weights = copy.deepcopy(model.state_dict())
    flags = [(p.requires_grad, p.dtype) for p in model.parameters()]
    logits, _ = canonical_score_prompts(model, [[1, 2]], [0])
    assert not logits.requires_grad
    assert not model.backbone.calls[0]["grad_enabled"]
    assert not model.backbone.calls[0]["training"]
    assert [(p.requires_grad, p.dtype) for p in model.parameters()] == flags
    assert all(p.grad is None for p in model.parameters())
    for name, value in weights.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        canonical_score_prompts(model, [[1, 2]], [0])
    assert model.training


def test_wrapper_retains_nested_answers_exposes_distinct_policy_identity_and_raw_calibration():
    wrapper = wrapped_engine()
    request = {"state": "A fact.", "questions": {"nested": [{"type": "choice", "criteria": {"left": "a", "right": "b"}}],
                                                    "binary": {"type": "noul"}}}
    before = copy.deepcopy(request)
    result = wrapper(request, details=True)
    assert request == before
    assert set(result["answers"]["nested"][0]["probabilities"]) == {"left", "right"}
    assert "noul" in result["answers"]["binary"]
    assert result["model"] == wrapper.model_id
    assert result["model"] != "test-checkpoint/sha256-fixture"
    assert "b4-k128" in result["model"]
    assert result["execution_policy"]["physical_batch_size"] == 4
    assert result["execution_policy"]["length_bucket_tokens"] == 128
    assert len(result["execution_policy"]["sha256"]) == 64
    assert result["answers"]["binary"]["details"]["calibration_id"] == "identity-uncalibrated"
    assert result["usage"]["model_units"] == 3
    replay = wrapper.evaluate({**request, "model": wrapper.model_id}, details=True)
    assert replay["answers"] == result["answers"]


def test_cached_mode_and_unknown_policy_identity_fail_without_forward():
    wrapper = wrapped_engine()
    request = {"state": {}, "questions": {"q": {"type": "noul"}}}
    with pytest.raises(ValueError, match="cached"):
        wrapper.evaluate(request, cached=True)
    with pytest.raises(ValueError, match="model"):
        wrapper.evaluate({**request, "model": "another-policy"})
    assert wrapper.model.backbone.calls == []


def test_policy_identity_is_stable_but_includes_underlying_checkpoint():
    from openjev.consistent_inference import ConsistentJudgmentEngine

    first, second = wrapped_engine(), wrapped_engine()
    assert first.model_id == second.model_id
    second.engine.model_id = "another-checkpoint"
    third = ConsistentJudgmentEngine(second.engine)
    assert third.model_id != first.model_id


def test_execution_identity_binds_backbone_head_dtypes_and_device_type():
    from openjev.consistent_inference import execution_policy

    model = ToyJudgment()
    model.backbone.register_parameter("fixture_weight", nn.Parameter(torch.ones(1)))
    fp32 = execution_policy(model)
    assert fp32["device_type"] == "cpu"
    assert fp32["backbone_dtypes"] == ["torch.float32"]
    assert fp32["head_dtypes"] == {"compatibility": ["torch.float32"], "binary": ["torch.float32"]}
    model.backbone.to(dtype=torch.bfloat16)
    bf16_body = execution_policy(model)
    assert bf16_body["backbone_dtypes"] == ["torch.bfloat16"]
    assert bf16_body["sha256"] != fp32["sha256"]
    model.compatibility.to(dtype=torch.float64)
    changed_head = execution_policy(model)
    assert changed_head["sha256"] != bf16_body["sha256"]
    # Meta tensors exercise actual device metadata without allocating a GPU.
    model.to("meta")
    changed_device = execution_policy(model)
    assert changed_device["device_type"] == "meta"
    assert changed_device["sha256"] != changed_head["sha256"]
