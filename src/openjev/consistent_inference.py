"""Opt-in canonical full-prompt inference with fixed four-row physical tiles.

Each prompt owns a 128-token length bucket including at least one masked pad.
There is no shared-prefix cache, training, parameter conversion or kernel change.
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from .judgments import DEFAULT_MODEL, assemble_response

PHYSICAL_BATCH_SIZE = 4
BUCKET_TOKENS = 128
POLICY_ID = "canonical-full-v1-b4-k128"


def execution_policy(model):
    def dtypes(module):
        return sorted({str(parameter.dtype) for parameter in module.parameters()})

    policy = {"id": POLICY_ID, "version": 1, "physical_batch_size": PHYSICAL_BATCH_SIZE,
              "length_bucket_tokens": BUCKET_TOKENS, "bucket_rule": "ceil((prompt_tokens+1)/128)*128",
              "padding": "right, always at least one masked token per row",
              "tile_fill": "duplicate first real row of each partial tile",
              "prefix_cache": False, "calibration": "raw identity-uncalibrated",
              "kernel_backend": getattr(model, "kernel_backend", "unspecified"),
              "device_type": model.device.type, "backbone_dtypes": dtypes(model.backbone),
              "head_dtypes": {"compatibility": dtypes(model.compatibility), "binary": dtypes(model.binary)}}
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    return {**policy, "sha256": hashlib.sha256(encoded).hexdigest()}


@torch.inference_mode()
def canonical_score_prompts(model, prompts, kinds, *, unit_batch_size=4, max_input_tokens=8192):
    """Score complete prompts at canonical shapes, returning logits and true costs.

    ``unit_batch_size`` records the caller's logical preference. Physical tiles
    are always four rows; a smaller requested value does not reduce GPU memory.
    The model must already be entirely in eval mode; its flags are not modified.
    """
    if type(unit_batch_size) is not int or unit_batch_size < 1:
        raise ValueError("unit_batch_size must be a positive integer")
    if type(max_input_tokens) is not int or max_input_tokens < 1:
        raise ValueError("max_input_tokens must be a positive integer")
    if not isinstance(prompts, (list, tuple)) or not prompts or not isinstance(kinds, (list, tuple)):
        raise ValueError("nonempty prompt and kind sequences required")
    if len(prompts) != len(kinds) or any(type(kind) is not int or kind not in (0, 1) for kind in kinds):
        raise ValueError("one integer readout kind 0 or 1 is required per prompt")
    if any(not isinstance(row, (list, tuple)) or not row or len(row) > max_input_tokens
           or any(type(token) is not int or token < 0 for token in row) for row in prompts):
        raise ValueError("prompts need nonempty nonnegative integer tokens within max_input_tokens; no truncation")
    if any(module.training for module in model.modules()):
        raise ValueError("canonical inference requires every model module in eval mode")
    pad = getattr(model.backbone.config, "pad_token_id", None)
    pad = 0 if pad is None else pad
    if type(pad) is not int or pad < 0:
        raise ValueError("model pad token must be a nonnegative integer")
    buckets = defaultdict(list)
    for index, row in enumerate(prompts):
        bucket = ((len(row) + 1 + BUCKET_TOKENS - 1) // BUCKET_TOKENS) * BUCKET_TOKENS
        buckets[bucket].append(index)
    results = [None] * len(prompts)
    stats = {"input_tokens": sum(map(len, prompts)), "max_unit_tokens": max(map(len, prompts)),
             "model_units": len(prompts), "output_tokens": 0, "prefix_tokens": 0, "forward_calls": 0,
             "physical_rows": 0, "dummy_rows": 0, "physical_tokens": 0, "dummy_input_tokens": 0,
             "padding_tokens": 0, "requested_unit_batch_size": unit_batch_size,
             "physical_batch_size": PHYSICAL_BATCH_SIZE, "buckets": {}}
    for bucket, indices in sorted(buckets.items()):
        bucket_stats = {"real_rows": len(indices), "forward_calls": 0, "dummy_rows": 0, "physical_tokens": 0}
        for start in range(0, len(indices), PHYSICAL_BATCH_SIZE):
            real = indices[start:start + PHYSICAL_BATCH_SIZE]
            dummy_count = PHYSICAL_BATCH_SIZE - len(real)
            tile = real + [real[0]] * dummy_count
            lengths = torch.tensor([len(prompts[index]) for index in tile], device=model.device)
            inputs = torch.tensor([list(prompts[index]) + [pad] * (bucket - len(prompts[index])) for index in tile],
                                  device=model.device, dtype=torch.long)
            mask = torch.arange(bucket, device=model.device)[None] < lengths[:, None]
            readout = torch.tensor([kinds[index] for index in tile], device=model.device, dtype=torch.long)
            logits = model(input_ids=inputs, attention_mask=mask, score_positions=lengths - 1, readout_kind=readout)
            if logits.shape != (PHYSICAL_BATCH_SIZE,) or not torch.isfinite(logits).all():
                raise ValueError("model must return four finite scalar logits per canonical tile")
            for row, index in enumerate(real):
                results[index] = logits[row]
            dummy_tokens = dummy_count * len(prompts[real[0]])
            physical_tokens = PHYSICAL_BATCH_SIZE * bucket
            stats["forward_calls"] += 1
            stats["physical_rows"] += PHYSICAL_BATCH_SIZE
            stats["dummy_rows"] += dummy_count
            stats["physical_tokens"] += physical_tokens
            stats["dummy_input_tokens"] += dummy_tokens
            stats["padding_tokens"] += physical_tokens - sum(len(prompts[index]) for index in tile)
            bucket_stats["forward_calls"] += 1
            bucket_stats["dummy_rows"] += dummy_count
            bucket_stats["physical_tokens"] += physical_tokens
        stats["buckets"][str(bucket)] = bucket_stats
    return torch.stack(results), stats


class ConsistentJudgmentEngine:
    """Wrap an already loaded JudgmentEngine without changing its model or loader."""

    def __init__(self, engine):
        self.engine = engine
        self.model, self.tokenizer = engine.model, engine.tokenizer
        self.max_input_tokens, self.unit_batch_size = engine.max_input_tokens, engine.unit_batch_size
        self.source_model_id = engine.model_id
        self.policy = execution_policy(self.model)
        self.model_id = f"{self.source_model_id}/{POLICY_ID}/sha256-{self.policy['sha256']}"

    def prepare(self, request):
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        if request.get("model", DEFAULT_MODEL) not in (DEFAULT_MODEL, self.source_model_id, self.model_id):
            raise ValueError("requested model does not match checkpoint and execution policy")
        compiled, prompts, kinds = self.engine.prepare({**request, "model": self.source_model_id})
        return replace(compiled, model=self.model_id), prompts, kinds

    def synchronize(self):
        self.engine.synchronize()

    @torch.inference_mode()
    def evaluate(self, request, *, details=False, cached=False, unit_batch_size=None):
        if cached is not False:
            raise ValueError("cached=True is incompatible with canonical full-prompt inference; use cached=False")
        self.synchronize()
        started = time.perf_counter()
        compiled, prompts, kinds = self.prepare(request)
        logits, stats = canonical_score_prompts(
            self.model, prompts, kinds,
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
    parser.add_argument("--unit-batch-size", type=int, default=4,
                        help="logical preference recorded in usage; physical tiles always have 4 rows")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.unit_batch_size < 1 or args.max_input_tokens < 1:
        parser.error("choose a new output and positive resource bounds")
    request = read_json(args.request)
    engine = load_judgment_engine(checkpoint=args.checkpoint, device=args.device, backend=args.backend,
                                  unit_batch_size=args.unit_batch_size, max_input_tokens=args.max_input_tokens,
                                  trainable=False)
    result = ConsistentJudgmentEngine(engine).evaluate(request, details=args.details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {args.output}: {result['usage']['model_units']} logical units, "
          f"{result['usage']['physical_tokens']} physical tokens, {result['elapsed_ms']:.1f} ms")


if __name__ == "__main__":
    main()
