# Parallel constrained decisions: first experiment

## Agreed objective

Recreate Harsha Gundala's shared-context, parallel categorical decision approach,
using local Qwen3.5-2B on the user's RTX 4080. Compare against ordinary JSON
generation and collect evidence before deciding whether to train. No training
is part of this first experiment. Text input first; vision is future work.

## Design

- A small Python package with a command-line benchmark and JSON/Markdown reports.
- Validate schemas containing categorical and boolean fields. Map answers to
  verified single-token option codes so multi-token labels cannot collide.
- Build one shared context/schema prefix, then one suffix per field. Prefill once,
  clone all attention and recurrent cache states into independent branches, and
  batch equal-length suffixes. Grouping by suffix length avoids recurrent-state
  corruption from padding. Use the model's non-thinking chat template.
- Compare parallel scoring with an uncached independent-field reference and
  greedy autoregressive JSON generation. Never call normalized scores calibrated.
- Record synchronized wall time, real forward-call counts, schema validity,
  field and exact-case accuracy, Brier score, log loss, calibration bins, errors,
  environment versions, model revision, and dataset hashes. Use warmups and
  repeated measurements; do not include model download/load time in inference.
- Preserve Harsha's original scenarios with attribution as unlabeled latency
  cases. Add a small clearly labeled synthetic diagnostic set. It is a smoke
  evaluation, not evidence of production/general capability.

## Implementation tasks

1. [x] Schema, option-code mapping, and metrics with tests for malformed schemas,
   shared-prefix label names, missing/wrongly typed answers, and hand-computed
   probability metrics.
2. [x] Model adapter, parallel scorer, independent reference, and JSON baseline.
   Test on tiny randomly initialized real Transformers models, including hybrid
   recurrent cache isolation; then check cached/reference equivalence on GPU.
3. [x] Attributed upstream scenarios, synthetic labeled evaluation, CLI and
   reproducible reports. Run actual RTX 4080 experiments and analyze failures.
4. [x] Fresh code review, targeted fixes, complete tests, and usage documentation.

## Decisions

- Work directly in this empty project directory; there is no existing Git branch
  or user code to isolate. Keep progress here and create no external project.
- Implement inline. The user has explicitly authorized implementation and tests;
  proceed without another design approval round.
- Keep dependencies and model downloads local or in standard package/model caches.
- Do not terminate existing GPU processes. Bound input lengths and branch batches.
- Default 2B model in BF16; do not claim performance before measuring.

## Review focus

Recurrent cache aliasing; candidate tokens at prompt boundaries; invalid JSON
being counted as correct; missing gold fields changing denominators; timing
asynchronous GPU work; model/dataset provenance and reproducibility.
