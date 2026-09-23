# Batching consistency investigation

The archived failure is reproducible with identical weights and input tokens. The new operator trace uses the museum-custody calibration case because it already exhibited a decision flip; this choice is for debugging, not estimating accuracy.

## Controlled interventions

For the identical target input, the Boolean logit is -0.25736 alone, +0.39115 with three identical copies, and +0.18126 in the original mixed batch. Replacing all companions with copies while retaining the same physical shape reproduces the mixed-batch result exactly. Moving the target to row three also leaves the result unchanged. Thus neighboring semantic content is not needed to reproduce the error.

The first differences appear in layer-zero linear projections. BF16 base projections differ by up to 0.03125 in the captured gate projection; FP32 LoRA matrix products differ at approximately 1e-6 and can alter BF16 rounding when combined. Keeping batch size one but adding only masked padding leaves the first three block outputs identical, with the first captured divergence at block three, a full-attention block. These differences propagate through the subsequent layers and scalar readout.

One-factor precision probes on the same input:

| Policy | Maximum Boolean probability spread across tested layouts |
|---|---:|
| Existing BF16/SDPA | 0.160546 |
| Disable reduced-precision BF16 GEMM reductions | 0.236703 |
| Math SDPA backend | 0.047959 |
| Fixed physical B=4 and padded length | 0.000000 |

Disabling reduced reductions did not solve this case. Math attention removed the padding-only difference but left the batch-size difference. The fixed-shape candidate preserved both current model weights and kernels and eliminated the tested layout changes. These are small numerical interventions, not training comparisons or proof of semantic correctness.

This is consistent with PyTorch's documented lack of bitwise equivalence between batched and sliced floating-point operations ([numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#batched-computations-or-slice-computations)). The project-specific evidence is the saved module-level trace and controlled operator/padding probes; the documentation does not by itself establish this model's cause or guarantee a fix.

## Fix selected for validation

Use four physical rows on every forward and length buckets determined by each prompt alone. Keep at least one masked padding token even at exact bucket boundaries, so another row cannot change an all-valid-mask optimization. Group prompts by their own bucket, duplicate a real row to fill incomplete tiles, then return only real scores in the original order. Validate against arbitrary caller grouping, permutations, companions and singleton requests.

This standardizes the numerical computation rather than claiming greater arithmetic accuracy. It retains four-row parallel execution but adds padding/dummy work and uses complete prompts; it does not claim equivalence to the old shared-prefix cache path. The new explicit policy must have a distinct response identity, honest compute counters, measured latency, and separate calibration provenance. The old source and experiment results remain unchanged.

Evidence: `trace-v1/report.json`, `precision-probe/report.json`, and their unmodified executable probes/logs. Broad GPU validation and performance measurements are recorded separately.
