"""Canonical state-prefix caching: B1 prefill, B4 suffix tiles, 64-token buckets."""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from .consistent_inference import ConsistentJudgmentEngine
from .consistent_inference import execution_policy as full_execution_policy
from .engine import fork_cache
from .judgments import assemble_response, render_unit_messages

POLICY_ID = "canonical-cache-v1-b4-k64-p32"
PHYSICAL_BATCH_SIZE = 4
SUFFIX_BUCKET_TOKENS = 64
PREFIX_BLOCK_TOKENS = 32


def execution_policy(model):
    policy = full_execution_policy(model)
    policy.pop("sha256")
    policy.update(id=POLICY_ID, length_bucket_tokens=SUFFIX_BUCKET_TOKENS,
                  bucket_rule="ceil((suffix_tokens+1)/64)*64", prefix_cache=True,
                  padding="suffix rows right-padded with at least one masked token; prefill unpadded",
                  prefill_batch_size=1, prefix_block_tokens=PREFIX_BLOCK_TOKENS,
                  prefix_boundary="per-unit state-only-chat/full-prompt common tokens; floor to 32; retain >=1 suffix token",
                  prefix_fork="fresh deepcopy and reorder_cache to four rows per suffix tile",
                  prefix_lifetime="one scorer call; never reuse a branch after suffix consumption")
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    return {**policy, "sha256": hashlib.sha256(encoded).hexdigest()}


@torch.inference_mode()
def score_canonical_cached(model, prompts, kinds, prefix_lengths, *, unit_batch_size=4, max_input_tokens=8192):
    """Consume exact prefixes at B1 and fresh forked suffixes at B4.

    Prefix lengths are explicit per-prompt boundaries, never inferred from other
    prompts in this call. The wrapper below derives them from each unit's state.
    Physical-token counters count computation, while cached_prefix_token_slots
    separately counts prefix positions available to suffix attention.
    """
    if type(unit_batch_size) is not int or unit_batch_size < 1:
        raise ValueError("unit_batch_size must be a positive integer")
    if type(max_input_tokens) is not int or max_input_tokens < 1:
        raise ValueError("max_input_tokens must be a positive integer")
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError("nonempty prompt sequence required")
    if not isinstance(kinds, (list, tuple)) or len(kinds) != len(prompts):
        raise ValueError("one readout kind is required per prompt")
    if not isinstance(prefix_lengths, (list, tuple)) or len(prefix_lengths) != len(prompts):
        raise ValueError("one prefix length is required per prompt")
    if any(type(kind) is not int or kind not in (0, 1) for kind in kinds):
        raise ValueError("readout kinds must be integers 0 or 1")
    if any(not isinstance(prompt, (list, tuple)) or not prompt or len(prompt) > max_input_tokens
           or any(type(token) is not int or token < 0 for token in prompt) for prompt in prompts):
        raise ValueError("nonempty nonnegative integer prompts within max_input_tokens required; no truncation")
    if any(type(prefix) is not int or prefix < 0 or prefix >= len(prompt) or prefix % PREFIX_BLOCK_TOKENS
           for prefix, prompt in zip(prefix_lengths, prompts, strict=True)):
        raise ValueError("prefix lengths must be nonnegative 32-token multiples and retain a suffix token")
    if any(module.training for module in model.modules()):
        raise ValueError("canonical cached inference requires every model module in eval mode")
    pad = getattr(model.backbone.config, "pad_token_id", None)
    pad = 0 if pad is None else pad
    if type(pad) is not int or pad < 0:
        raise ValueError("model pad token must be a nonnegative integer")
    groups = defaultdict(lambda: defaultdict(list))
    for index, (prompt, prefix) in enumerate(zip(prompts, prefix_lengths, strict=True)):
        suffix = len(prompt) - prefix
        bucket = ((suffix + SUFFIX_BUCKET_TOKENS) // SUFFIX_BUCKET_TOKENS) * SUFFIX_BUCKET_TOKENS
        groups[tuple(prompt[:prefix])][bucket].append(index)
    results = [None] * len(prompts)
    stats = {"input_tokens": sum(map(len, prompts)), "max_unit_tokens": max(map(len, prompts)),
             "model_units": len(prompts), "output_tokens": 0, "requested_unit_batch_size": unit_batch_size,
             "physical_batch_size": PHYSICAL_BATCH_SIZE, "prefix_tokens": 0, "logical_prefix_tokens": sum(prefix_lengths),
             "real_suffix_tokens": sum(len(prompt) - prefix for prompt, prefix in zip(prompts, prefix_lengths, strict=True)),
             "prefill_forward_calls": 0, "suffix_forward_calls": 0, "forward_calls": 0,
             "physical_rows": 0, "suffix_physical_rows": 0, "dummy_rows": 0, "physical_tokens": 0,
             "physical_suffix_tokens": 0, "dummy_input_tokens": 0, "padding_tokens": 0,
             "cached_prefix_token_slots": 0, "prefix_groups": len(groups), "groups": []}
    for prefix_tokens, buckets in sorted(groups.items()):
        prefix, cache = len(prefix_tokens), None
        if prefix:
            out = model.backbone(input_ids=torch.tensor([prefix_tokens], device=model.device, dtype=torch.long),
                                 use_cache=True)
            cache = out.past_key_values
            if cache is None:
                raise ValueError("backbone did not return the requested prefix cache")
            stats["prefill_forward_calls"] += 1
            stats["prefix_tokens"] += prefix
            stats["physical_tokens"] += prefix
            stats["physical_rows"] += 1
            del out
        for bucket, indices in sorted(buckets.items()):
            group_stats = {"prefix_tokens": prefix, "suffix_bucket_tokens": bucket, "real_rows": len(indices),
                           "forward_calls": 0, "dummy_rows": 0, "physical_suffix_tokens": 0}
            for start in range(0, len(indices), PHYSICAL_BATCH_SIZE):
                real = indices[start:start + PHYSICAL_BATCH_SIZE]
                dummy_count = PHYSICAL_BATCH_SIZE - len(real)
                tile = real + [real[0]] * dummy_count
                suffixes = [list(prompts[index][prefix:]) for index in tile]
                lengths = torch.tensor(list(map(len, suffixes)), device=model.device, dtype=torch.long)
                inputs = torch.tensor([row + [pad] * (bucket - len(row)) for row in suffixes],
                                      device=model.device, dtype=torch.long)
                mask = torch.arange(prefix + bucket, device=model.device)[None] < prefix + lengths[:, None]
                branch = fork_cache(cache, PHYSICAL_BATCH_SIZE) if cache is not None else None
                out = model.backbone(input_ids=inputs, attention_mask=mask, past_key_values=branch, use_cache=True)
                hidden = out.last_hidden_state[torch.arange(PHYSICAL_BATCH_SIZE, device=model.device), lengths - 1]
                logits = model._read(hidden, torch.tensor([kinds[index] for index in tile], device=model.device))
                if logits.shape != (PHYSICAL_BATCH_SIZE,) or not torch.isfinite(logits).all():
                    raise ValueError("model must return four finite scalar logits per canonical suffix tile")
                for row, index in enumerate(real):
                    results[index] = logits[row].clone()
                physical_tokens = PHYSICAL_BATCH_SIZE * bucket
                stats["suffix_forward_calls"] += 1
                stats["physical_rows"] += PHYSICAL_BATCH_SIZE
                stats["suffix_physical_rows"] += PHYSICAL_BATCH_SIZE
                stats["dummy_rows"] += dummy_count
                stats["physical_tokens"] += physical_tokens
                stats["physical_suffix_tokens"] += physical_tokens
                stats["dummy_input_tokens"] += dummy_count * len(suffixes[0])
                stats["padding_tokens"] += physical_tokens - sum(map(len, suffixes))
                stats["cached_prefix_token_slots"] += PHYSICAL_BATCH_SIZE * prefix
                group_stats["forward_calls"] += 1
                group_stats["dummy_rows"] += dummy_count
                group_stats["physical_suffix_tokens"] += physical_tokens
                del out, branch, hidden
            stats["groups"].append(group_stats)
        del cache
    stats["forward_calls"] = stats["prefill_forward_calls"] + stats["suffix_forward_calls"]
    return torch.stack(results), stats


class ConsistentCachedJudgmentEngine(ConsistentJudgmentEngine):
    """Raw judgment API with deterministic per-unit state-prefix boundaries."""

    def __init__(self, engine):
        super().__init__(engine)
        self.policy = execution_policy(self.model)
        self.model_id = f"{self.source_model_id}/{POLICY_ID}/sha256-{self.policy['sha256']}"

    def prefix_lengths(self, compiled, prompts):
        if len(compiled.units) != len(prompts) or any(not prompt for prompt in prompts):
            raise ValueError("one nonempty exact full prompt is required per compiled unit")
        lengths, state_tokens = [], {}
        for unit, prompt in zip(compiled.units, prompts, strict=True):
            messages = render_unit_messages(unit)
            before, separator, _ = messages[-1]["content"].partition("\nQUESTION:\n")
            if not separator:
                raise ValueError("renderer does not expose the pinned state/question boundary")
            messages[-1] = {**messages[-1], "content": before}
            key = tuple((message["role"], message["content"]) for message in messages)
            if key not in state_tokens:
                state_tokens[key] = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=False,
                )
            common = 0
            for state_token, full_token in zip(state_tokens[key], prompt):
                if state_token != full_token:
                    break
                common += 1
            prefix = (min(common, len(prompt) - 1) // PREFIX_BLOCK_TOKENS) * PREFIX_BLOCK_TOKENS
            if list(prompt[:prefix]) != list(state_tokens[key][:prefix]):
                raise ValueError("state-derived prefix does not match actual prompt tokens")
            lengths.append(prefix)
        return lengths

    @torch.inference_mode()
    def evaluate(self, request, *, details=False, cached=True, unit_batch_size=None):
        if cached is not True:
            raise ValueError("canonical cached identity requires cached=True; use the full policy for cached=False")
        self.synchronize()
        started = time.perf_counter()
        compiled, prompts, kinds = self.prepare(request)
        prefixes = self.prefix_lengths(compiled, prompts)
        logits, stats = score_canonical_cached(
            self.model, prompts, kinds, prefixes,
            unit_batch_size=self.unit_batch_size if unit_batch_size is None else unit_batch_size,
            max_input_tokens=self.max_input_tokens,
        )
        result = assemble_response(compiled, logits.cpu().tolist(), details=details)
        self.synchronize()
        result.update(usage=stats, elapsed_ms=(time.perf_counter() - started) * 1000,
                      execution_policy=dict(self.policy), source_model=self.source_model_id)
        return result

    __call__ = evaluate


def main(argv=None):
    from .judgment_cli import load_judgment_engine, read_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("reference", "fla"), default="reference")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--unit-batch-size", type=int, default=4, help="logical preference; suffix physical tiles remain B4")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.unit_batch_size < 1 or args.max_input_tokens < 1:
        parser.error("choose a new output and positive resource bounds")
    request = read_json(args.request)
    engine = load_judgment_engine(checkpoint=args.checkpoint, device=args.device, backend=args.backend,
                                  unit_batch_size=args.unit_batch_size, max_input_tokens=args.max_input_tokens,
                                  trainable=False)
    result = ConsistentCachedJudgmentEngine(engine).evaluate(request, details=args.details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {args.output}: {result['usage']['model_units']} logical units, "
          f"{result['usage']['physical_tokens']} physical tokens, {result['elapsed_ms']:.1f} ms")


if __name__ == "__main__":
    main()
