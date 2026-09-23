"""Parallel scoring with complete attention/recurrent cache isolation."""

import copy
import json
import time

import torch

from .prompts import option_codes, render_prompts
from .schema import parse_schema, valid_answer


def fork_cache(cache, batch_size):
    """Use beam reordering to copy all cache types, including DeltaNet states.

    DynamicCache.batch_repeat_interleave is not implemented for every recurrent
    layer. Reordering duplicate beam zero works for both attention and recurrence.
    Deepcopy prevents dictionary and metadata aliasing before the in-place reorder.
    """
    fork = copy.deepcopy(cache)
    fork.reorder_cache(torch.zeros(batch_size, dtype=torch.long))
    return fork


@torch.inference_mode()
def score_token_prompts(model, prompts, *, parallel=True, branch_batch_size=8, candidate_ids=None):
    """Score each prompt's next token, sharing a prefix and batching suffixes.

    Suffixes are right-padded; only their last real hidden states are scored.
    Extreme length outliers are isolated to keep padded work within twice the
    actual suffix tokens in each batch, even when the field count fits the limit.
    Branch caches include padding updates and must never be reused to continue
    a prompt. Each batch starts from a fresh copy of the common prefix cache.
    """
    if not prompts or any(not p for p in prompts) or branch_batch_size < 1:
        raise ValueError("need nonempty prompts and positive branch_batch_size")
    device = next(model.parameters()).device
    head = model.get_output_embeddings()
    weights = bias = None
    if candidate_ids is not None:
        indices = torch.tensor(candidate_ids, device=device)
        weights = head.weight.index_select(0, indices).float()
        if head.bias is not None:
            bias = head.bias.index_select(0, indices).float()

    def forward(inputs, *, use_cache, past_key_values=None, lengths=None, attention_mask=None):
        hidden = model.model(
            inputs, use_cache=use_cache, past_key_values=past_key_values, attention_mask=attention_mask
        ).last_hidden_state
        endpoints = hidden[:, -1] if lengths is None else hidden[torch.arange(len(inputs), device=device), lengths - 1]
        if weights is None:
            return head(endpoints)
        # BF16 output logits round nearby candidate scores noticeably. Project
        # just the needed vocabulary rows in FP32, rather than rounding first.
        return torch.nn.functional.linear(endpoints.float(), weights, bias)

    if not parallel:
        logits = [forward(torch.tensor([p], device=device), use_cache=False)[0] for p in prompts]
        return torch.stack(logits), {"forward_calls": len(prompts), "prefix_tokens": 0, "branch_groups": len(prompts)}
    common = 0
    for tokens in zip(*prompts):
        if len(set(tokens)) != 1:
            break
        common += 1
    common = min(common, min(map(len, prompts)) - 1)
    cache = None
    calls = 0
    if common:
        out = model.model(torch.tensor([prompts[0][:common]], device=device), use_cache=True)
        cache = out.past_key_values
        calls += 1
        del out
    lengths_by_prompt = [len(p) - common for p in prompts]
    batches = []
    for start in range(0, len(prompts), branch_batch_size):
        indices = list(range(start, min(start + branch_batch_size, len(prompts))))
        # One enormous enum query should not pad every short boolean query to
        # its length. Peel off outliers; ordinary mixed lengths stay together.
        while len(indices) * max(lengths_by_prompt[i] for i in indices) > 2 * sum(
            lengths_by_prompt[i] for i in indices
        ):
            longest = max(indices, key=lengths_by_prompt.__getitem__)
            batches.append([longest])
            indices.remove(longest)
        batches.append(indices)
    logits = [None] * len(prompts)
    branch_groups = 0
    pad_id = getattr(model.config, "pad_token_id", None) or 0
    for indices in batches:
        suffixes = [prompts[i][common:] for i in indices]
        max_length = max(map(len, suffixes))
        inputs = torch.tensor([p + [pad_id] * (max_length - len(p)) for p in suffixes], device=device)
        lengths = torch.tensor([len(p) for p in suffixes], device=device)
        # The prefix is real for every row. Position indices remain unchanged
        # because padding is strictly after each prompt's true endpoint.
        attention_mask = torch.arange(common + max_length, device=device)[None, :] < (common + lengths[:, None])
        branch = fork_cache(cache, len(suffixes)) if cache is not None else None
        scores = forward(inputs, past_key_values=branch, use_cache=True, lengths=lengths, attention_mask=attention_mask)
        for row, index in enumerate(indices):
            logits[index] = scores[row].clone()
        calls += 1
        branch_groups += 1
        del branch
    return torch.stack(logits), {"forward_calls": calls, "prefix_tokens": common, "branch_groups": branch_groups}


def parse_json_answer(text, fields):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(text, object_pairs_hook=unique_object)
    if not valid_answer(result, fields):
        raise ValueError("JSON does not match schema keys, types, or allowed values")
    return result


class DecisionEngine:
    def __init__(self, model, tokenizer, *, branch_batch_size=8, max_input_tokens=8192):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.branch_batch_size = branch_batch_size
        self.max_input_tokens = max_input_tokens
        self.device = next(model.parameters()).device
        self.codes, self.code_ids = option_codes(tokenizer)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _prepare(self, case, json_output=False):
        fields = parse_schema(case["schema"])
        prompts = render_prompts(self.tokenizer, case["context"], fields, self.codes, json_output=json_output)
        if max(map(len, prompts)) > self.max_input_tokens:
            raise ValueError(f"{case['id']}: input exceeds {self.max_input_tokens} tokens; no silent truncation")
        return fields, prompts

    @torch.inference_mode()
    def decide(self, case, *, parallel=True):
        self.synchronize()
        started = time.perf_counter()
        fields, prompts = self._prepare(case)
        logits, stats = score_token_prompts(
            self.model,
            prompts,
            parallel=parallel,
            branch_batch_size=self.branch_batch_size,
            candidate_ids=self.code_ids[: max(len(f.choices) for f in fields)],
        )
        values, probabilities = {}, {}
        for i, field in enumerate(fields):
            scores = logits[i, : len(field.choices)].float()
            probs = scores.softmax(-1).cpu().tolist()
            probabilities[field.name] = probs
            values[field.name] = field.choices[max(range(len(probs)), key=probs.__getitem__)]
        self.synchronize()
        return {
            "id": case["id"],
            "mode": "parallel" if parallel else "independent",
            "values": values,
            "probabilities": probabilities,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "input_tokens": max(map(len, prompts)),
            "total_prompt_tokens": sum(map(len, prompts)),
            "normalized_not_calibrated": True,
            **stats,
        }

    @torch.inference_mode()
    def generate_json(self, case, *, max_new_tokens=1024):
        self.synchronize()
        started = time.perf_counter()
        fields, prompts = self._prepare(case, json_output=True)
        inputs = torch.tensor([prompts[0]], device=self.device)
        calls = 0

        def count_forward(module, args):
            nonlocal calls
            calls += 1

        hook = self.model.register_forward_pre_hook(count_forward)
        try:
            output = self.model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        finally:
            hook.remove()
        generated = output[0, inputs.shape[1] :]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        error = None
        try:
            values = parse_json_answer(raw.strip(), fields)
        except (ValueError, TypeError) as exc:
            values, error = {}, str(exc)
        self.synchronize()
        return {
            "id": case["id"],
            "mode": "json",
            "values": values,
            "raw_text": raw,
            "error": error,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "input_tokens": inputs.shape[1],
            "output_tokens": len(generated),
            "hit_token_limit": len(generated) >= max_new_tokens,
            "forward_calls": calls,
        }
