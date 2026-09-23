# Consistent cached judgments

Use `ConsistentCachedJudgmentEngine` for new judgments that need stable results when requests are regrouped. The original model, checkpoint and numerical kernels are unchanged. This mode gives each scoring unit a deterministic state prefix and physical suffix layout, while preserving shared context computation.

On the pinned RTX 4080 runtime and H0 checkpoint, the GPU regression covered 43 requests, 215 judgments and 408 scoring units. Pooling, reversal, shuffling, singleton calls, changed question sets and bucket-boundary checks produced zero logit/probability differences and zero answer flips. This establishes the measured behavior for that runtime; it does not promise bitwise equality across hardware or library versions.

## Python API

Use the existing environment with `PYTHONPATH=.runtime-training:.`. The experiment's deterministic setup reproduces its precision settings before loading the model:

```python
from experiments.contrast_scaling_training import apply_determinism
from openjev.judgment_cli import load_judgment_engine
from openjev.consistent_cached_inference import ConsistentCachedJudgmentEngine

apply_determinism(42)
base = load_judgment_engine(
    checkpoint="checkpoints/calibrated-screen-v1/run-v1/H0/step-0400",
    device="cuda",
    backend="fla",
    max_input_tokens=1536,
    unit_batch_size=4,
)
judge = ConsistentCachedJudgmentEngine(base)
result = judge.evaluate(request, details=True)
```

The request/answer tree uses the existing nested Noul/Choice/Score schema. The response includes `execution_policy`, `source_model` and a policy-specific `model` identity. Keep those fields when saving or caching responses. Probabilities are raw and marked uncalibrated; a temperature fitted to a different execution policy should not silently be reused.

The standalone module also accepts the existing JSON-file boundary:

```sh
PYTHONPATH=.runtime-training:. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m openjev.consistent_cached_inference \
  --checkpoint checkpoints/calibrated-screen-v1/run-v1/H0/step-0400 \
  --request request.json --output new-response.json \
  --backend fla --device cuda --max-input-tokens 1536 --details
```

Use the Python setup above when reproducing the measured deterministic/precision configuration. Do not change process-wide kernel/precision settings between calls to a loaded engine. Historical runners retain their original implementation so their saved experiments remain reproducible.

## Computation and cost

The prefix boundary is derived independently for each unit from its own state-only chat tokens and full prompt, then rounded down to a 32-token boundary. Identical prefixes are computed once at batch size one. Suffixes use fixed four-row tiles and 64-token length buckets with at least one masked pad. Incomplete tiles duplicate a real row; only real outputs are returned. Each branch gets a fresh cache copy.

The caller's `unit_batch_size` is recorded as a logical preference; physical suffix tiles remain four rows. Reducing the logical value does not reduce this mode's physical batch memory requirement. Usage reports distinguish logical tokens, actual prefix work, suffix work, duplicate rows, padding and cached prefix slots. `cached=False` is rejected because it would denote a different policy; the separate `ConsistentJudgmentEngine` in `consistent_inference.py` provides a full-prompt reference.

In six warmed requests with four alternating repetitions per mode, median request times were 167 ms for legacy caching, 177 ms for consistent caching, and 575 ms for the consistent full-prompt reference. Median PyTorch allocated peaks were approximately 3.705 GiB and 3.735 GiB for legacy and consistent caching respectively. These are measurements on a shared desktop GPU, not production service latency or total device-memory guarantees.

Development accuracy was 152/200 and retention 55/60 for the consistent cached policy. Stabilizing execution does not solve the model's semantic or calibration errors. See the [diagnostic report](../reports/consistency-native-v1/RESULTS.md), raw cached validation and [root-cause analysis](../reports/consistency-native-v1/ROOT_CAUSE.md).
