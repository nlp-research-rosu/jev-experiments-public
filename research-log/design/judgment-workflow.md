# Nested judgment and training workflow

This implements the v0.2 contract alongside the original option-code experiment.
It uses the pinned Qwen3.5-2B text backbone, not Jev weights or an asserted copy of
Jev's undisclosed architecture. One numerical unit represents a Choice candidate,
Score level, or complete binary Noul judgment. Software groups logits, computes
probabilities, and rebuilds the caller's fixed nested dictionaries and arrays.

## Local request and output

`data/integration/nested-request.json` exercises nested input state, three primitive
types, dictionaries, arrays, empty containers and reserved-looking field names.
Run inference with the isolated training packages available:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.runtime-training:. \
  .venv/bin/python -m openjev.judgment_cli \
  --request data/integration/nested-request.json \
  --output reports/my-nested-response.json --backend fla --details
```

Add `--checkpoint /absolute/path/to/checkpoint` to load adapters and readouts.
Output files must be new. `answers` holds full distributions and optional raw
logits; `values` holds choice names, expected Score indices, and Noul probabilities.
There is no generated answer text. Nesting is software structure, not recursive
model reasoning or an ability to invent variable-length arrays. Leaves are
independent; cross-field logical constraints are not enforced.

The common token prefix is evaluated once for cached inference. Suffixes use
batches of four by default, with padding guards; requests needing more rows use
more batches. Both attention and recurrent cache state are forked. Training uses
full differentiable inputs with the identical semantic renderer, and complete
candidate groups contribute one joint categorical loss. Noul uses binary loss.

## Trainable weights

- Frozen: original embeddings, attention, recurrent, convolution, normalization
  and MLP weights. The full vocabulary output matrix is not retained as a head.
- Trainable: rank-8, alpha-16, dropout-0 LoRA matrices on the selected attention,
  DeltaNet and MLP linear projections, plus the compatibility and binary readouts.
- BF16 backbone; FP32 readouts and losses; activation checkpointing. No weight
  quantization. The exact trainable names/counts are recorded in each run report.
- Adapters start at learning rate `1e-4`, heads at `5e-5`, AdamW weight decay
  `0.01`; gradients are clipped at norm 1. Four original bundles are accumulated
  per update. Verified consistency pairs add weight `0.1` to supervised loss.

The readouts start at the original A-minus-B projection. This gives an untrained
reference, not an already calibrated classifier. Both supervised and consistency
losses are retained; agreeing at probability 0.5 cannot replace learning labels.

## Data and gates

See `reports/DATA_PREPARATION.md` for corpus construction, licenses and limitations.
Original cases and their augmented views stay together. Validation, calibration,
and test data are separate. Token limits reject whole oversized bundles and record
their IDs; prompts are never silently truncated. The initial training bound is
1,024 tokens per unit; inference accepts up to 8,192 by default.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.runtime-training:. \
  .venv/bin/python -m experiments.judgment_pipeline --stage smoke \
  --output reports/judgment-smoke-v2 --max-seconds 1800

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.runtime-training:. \
  .venv/bin/python -m experiments.judgment_pipeline --stage full \
  --output checkpoints/judgment-full-v0.2 \
  --gate reports/judgment-smoke-v2/report.json
```

The smoke run checks untrained nested cached/full inference, finite gradients,
tiny-sample learning, hashes of frozen weights, saved/reloaded predictions and
an optimizer update after reload. A passing gate is specific to source files,
dataset hashes, packages and training policy. A partial smoke run cannot authorize
full training. The first full run requests one corpus pass, with a three-hour
training-loop cap, periodic checkpoints and validation. Preparation and final
evaluation are outside that cap. A capped partial pass is reported as partial.

Those output paths name the recorded runs and must not be overwritten. Choose
new paths when repeating an experiment, and point `--gate` at the matching smoke.
The original `reports/judgment-smoke` is a preliminary diagnostic; the `-v2` run
includes the reviewed fixes and is the gate used for the first full run.

`--resume <checkpoint>` restores optimizer state, progress and random generators
and requires the same stage/data/source/runtime policy. Choose a new output
directory. Source changes require a new smoke gate. Keep the original pinned
backbone cache: saved artifacts contain adapters and readouts, not a duplicate
copy of all original model weights.

Calibration remains identity and is explicitly marked uncalibrated. This first
run does not prove generality, calibrated confidence, or a speedup. Score includes
a narrow tabular wine-rating task and synthetic rubrics; the held-out synthetic
rubric family is a limited transfer check. Tasksource currently contributes only
BANKING77. Public dataset overlap with backbone pretraining is unknown.

## Offline verification

The training overlay installs optional GPU kernels. Select the reference backend
before the CPU suite so Transformers does not dispatch CPU tensors to Triton:

```sh
PYTHONPATH=.runtime-training:. .venv/bin/python -c \
  'from openjev.judgment_model import configure_kernels; configure_kernels("reference"); import pytest; raise SystemExit(pytest.main(["-q"]))'
```

The base environment can run `.venv/bin/python -m pytest -q`; its optional PEFT
round-trip test skips when the training overlay is absent.

## Trained inference benchmark

[Results and reproduction](../reports/JUDGMENT_LATENCY.md) report median/p95 local
latency, cached/full scoring, unmerged/merged weights and scoring-batch changes.
The benchmark implementation is `experiments/judgment_latency.py`; it records
requests, raw samples, runtime hashes and quality regressions. Adapter merging is
an in-memory experimental variant and does not alter saved checkpoints or CLI
defaults. The observed BF16 probability differences should be considered before
using tight decision thresholds.
