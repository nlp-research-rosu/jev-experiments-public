# Post-pause investigation

The original scaling run is preserved at **1,692 complete family updates**. It has not resumed. This investigation uses training records, previously inspected validation records and disposable model copies; final test/Jev outcomes remain quarantined. The original 5,000-update endpoint and repeat-data control are unfinished.

## Main finding: separate probability scale from semantic accuracy

The strongest new result is that much of the apparent probability deterioration can be corrected without changing the model's selected answers. Four-fold calibration keeps each complete semantic family outside the temperature fit used to assess it. This is an exploratory validation diagnostic, not an untouched test or a deployed calibrator.

| Latest checkpoint, new validation | Original probabilities | One temperature | One temperature per primitive |
| --- | ---: | ---: | ---: |
| Correct answers | 629/800 | 629/800 | 629/800 |
| NLL, lower is better | 1.5498 | 0.5452 | 0.5072 |
| Brier, lower is better | 0.3848 | 0.2958 | 0.2849 |
| Answers assigned at least 90% probability | 732 | 137 | 469 |
| Wrong within that group | 128 | 9 | 22 |
| Actual accuracy within that group | 82.51% | 93.43% | 95.31% |
| Mean reported probability within that group | 99.58% | 94.76% | 95.72% |

The reduced confident-error count partly reflects reduced coverage: calibration does not fix any wrong class decision. Broad-task NLL also improves, from 0.3577 to 0.2626/0.2324. Broad accuracy remains 264/300. Positive temperature preserves the modal Score level but changes its expected numeric value; per-primitive calibration slightly worsens latest new-task Score MAE, 0.5888 to 0.5960. It is not an improvement in every metric.

Early training remains useful: accuracy rose from 555/800 to 631/800 by update 200. Later training reaches 629/800 at 1,692, with little evidence of further accuracy gain. Once both are calibrated, the six-component probability objective is nearly the same: 0.4498 at 200 versus 0.4469 at 1,692. That small difference is not evidence of a meaningful late-training win. It also weakens the claim that rising raw NLL alone proves the later checkpoint lost its underlying decision ability.

See [calibration results](calibration/RESULTS.md) and independent calibration review. All 112 fitting objectives and 23,100 derived predictions were independently reconstructed; no fitting-on-assessment leakage, weighting mismatch or class change was found.

## Training-pipeline audit

The independent audit found **no new reproducible error in the tested target mapping, loss weighting, candidate normalization, padding/endpoint logic, replay schedule, optimizer accounting or saved metrics**. It used independent hand derivatives and raw-logit reconstructions rather than merely comparing two calls to the same implementation.

- All 7,700 archived validation judgments and aggregate metrics reconstruct from saved logits and frozen gold labels.
- The paused state has 375 optimizer entries at update 1,692, a contiguous history and saved CPU/CUDA RNG. The separate production resume proof gives identical next-update parameters and probabilities.
- The current objective really is equal weighting of six means: new/old × Noul/Choice/Score. Every complete candidate group participates in its softmax.
- Relation metadata is used for coverage and evaluation; this scaling run **does not train an explicit pair/consistency loss**. That follows the frozen experiment recipe.

This audit does not certify every authored semantic label, prove generalization or cover every CUDA shape. See the scoped evidence and limitations in pipeline audit.

## A concrete data-coverage mismatch

The new contrast training corpus contains **20,000 Score questions: 10,400 with two levels and 9,600 with three; none has four or five**. All **160 new validation Score questions have four levels**. Original-task replay does contain other cardinalities, so the model has not literally never seen four-level questions.

This is a distribution gap that deserves a controlled follow-up. It does not establish that cardinality alone causes the Score errors: source admissibility, level definitions, semantic complexity and label balance also differ. In three-level training examples, the middle level appears only 724 times out of 9,600. Simply multiplying the current generator would preserve these weaknesses.

A useful next data pilot would deliberately vary the number of levels and their definitions while holding evidence fixed, and include same-evidence/different-rubric pairs. It should use a new versioned corpus and matched exposure; the existing frozen files should remain intact.

## Numerical and performance checks

The numerical investigation found a separate **reproducible PyTorch efficient-attention error** for broadcast-head K/V tensors at length 257. The analytical example should return 0.5 at its last query; inference returns 0 instead. Gradient-enabled execution and the math backend agree with the analytical answer. Making either K or V contiguous while preserving every value removes the failure; nearby lengths 256 and 258 pass. This is an operator/layout edge case, not merely BF16 rounding.

It has **not** been demonstrated on the production model: all tested real-checkpoint train/grad versus inference logits matched exactly. The production model has a different attention-head layout, whose installed repetition code materializes dense K/V in a source/shape reproduction. We have not found evidence that this edge case affected the archived validation results or caused the plateau. The reproducer and scope are retained in SDPA finding. No production backend was changed.

Ten disposable in-memory updates profiled the actual paused checkpoint, loss and optimizer, with original weights/Adam/RNG restored between workload conditions. Normal timing excludes warmup, loading, hashing and the separately traced update:

| Workload from a 23-family candidate pool | Padded tokens/update | Forwards/update | Mean update time | Peak allocated model memory |
| --- | ---: | ---: | ---: | ---: |
| Short | 30,211 | 6 | 11.39 s | 7.34 GiB |
| Median | 36,420 | 7 | 13.57 s | 7.62 GiB |
| Long | 42,084 | 6 | 16.65 s | 8.19 GiB |

Only two normal timings per workload were taken. These are diagnostic workloads selected by input length, not population quantiles or a stable speed benchmark. Memory excludes other desktop processes. Every update still includes 20 new judgments and three replay questions, complete candidate groups and the same six loss weights.

The median trace has 13.25 seconds of summed GPU kernel time, including approximately 0.253 seconds for reference convolution forward/backward: **1.9%**. Even eliminating that convolution time entirely would yield only about a 2% gain under this attribution. Matrix multiplication, copies and elementwise operations dominate. The large input-token count comes from repeated candidate contexts, and gradient checkpointing additionally recomputes layers. Sharing training context with correct gradient accumulation is a larger graph change, not a missing-package fix.

The profile completed with the original frozen-body digest and all checkpoint-file hashes unchanged. No model checkpoint was saved and the original training stream was not resumed.

An optimized convolution artifact matching the installed Torch/CUDA/ABI was downloaded into an isolated report directory and hash-verified, without installation into the original environment. Its BF16 forward and consumed input gradients pass two actual-size operator checks: relative L2 differences versus reference are approximately 0.29% and 0.27%, and three repeats per implementation are bit-identical. Convolution weights remain frozen; the extension's unused atomic weight-gradient reductions are not a claim of reproducible full-weight training. See operator gate.

The full-model comparison is now complete. The candidate is **rejected as an interchangeable replacement under the frozen numerical gates**: maximum logit difference was **0.2115** (limit 0.05), and full-gradient relative L2 difference was **5.87%** (limit 2%). Maximum probability difference was only 0.0000137, with no changed winning class across the 23 judgments. Total loss differed by 0.00000202; reference gradient norm was 0.00604. This is a near-saturated, previously trained workload, so small probability/loss differences do not establish general equivalence. Conversely, the failed fidelity gate does not establish a defective kernel or reduced model accuracy.

The optimized backend itself is reproducible on this check: after a fresh model load and optimizer/RNG restoration, logits, probabilities, every trainable gradient and next-update parameters all match **exactly**, passing the unchanged 1e-7 gate. All original checkpoint files and the frozen-body digest remain unchanged. The isolated process exited after **three disposable updates in 60.74 seconds**.

As predeclared, the failed cross-backend gate skipped the warmed speed benchmark. Instrumented correctness-capture durations are not speed measurements, and **no end-to-end speedup is claimed**. Including associated SiLU kernels, this candidate addresses roughly 2.8% of the earlier traced update. The original environment/backend remains intact. The bounded diagnostic pass is finished and the monitor is **paused**. The original scaling study remains paused and unfinished.

## What the papers imply for the next experiment

The reviewed confidence papers often train a generated answer plus a separate confidence response. Our numerical heads already expose the complete differentiable semantic probability vector, so their RL machinery is not automatically necessary or equivalent.

1. Keep calibration as a required comparator for future training claims. Beating uncalibrated CE alone is insufficient.
2. Prioritize the observed rubric/Score coverage gap and persistent question-contrast errors with a small, matched data intervention. Do not expand the present generator unchanged.
3. If testing the loss next, compare CE against CE plus full-vector Brier with identical starting weights, optimizer state, examples and update count. With hard targets, both still prefer the correct one-hot answer; Brier is not guaranteed to prevent overconfidence.
4. Genuine soft targets remain worth testing where probabilities have defensible provenance: known stochastic generators, validated annotator distributions or a separately checked teacher. Arbitrary teacher confidence is not ground truth; an `Unknown` answer can itself deserve high probability.
5. Add explicit consistency only for verified semantic equivalences/complements. A generic invariant penalty can reward consistently wrong predictions.

The [methods review](methods-review.md) gives primary-paper formulas, links, failure modes and a possible 200-update matched loss screen. No new training ablation, model change, RL, teacher generation or paid Jev request was launched by this investigation.

## Limits and preserved evidence

These results use one training seed and repeatedly inspected validation. The unfinished repeat-data arm means we cannot separate unique-data scale from optimizer exposure. There are 5,000 joint semantic configurations but only 200 independently authored language blueprints. Calibration has not been verified on the untouched final test. The original checkpoint, frozen datasets, baseline/default and staged reference repositories remain preserved. A recommendation for a small follow-up is not an automatic continuation of training.

The full CPU regression suite passes: **562 tests and 48 subtests**, with expected CPU-backend/deprecation warnings. Logs are retained in tests.log; GPU probes and profiler have their own reports and are not silently counted as CPU tests.
