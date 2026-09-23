# Integration and training readiness

The software, definitive smoke and first full training run completed successfully.
The final report was written at **2026-09-19 22:28:59 US/Central**. Training took
55 minutes 38 seconds; the complete job including preparation and evaluation took
58 minutes 41 seconds. Authoritative status is
`report.json`.

## Full-run results

All 5,000 updates completed over 19,999 original training bundles. The original
frozen-weight hashes are identical before/after training. Checkpoint reload gave
zero logit difference and identical decisions on the integration probe. Peak
allocated CUDA memory was 4.1034 GiB. The process has exited successfully.

The table reports original-view accuracy on 1,500 held-out cases, 250 per source.
Augmented views are not counted as extra independent test cases here.

| Source | Before training | After training |
|---|---:|---:|
| BoolQ | 62.0% | 83.2% |
| PAWS | 50.0% | 94.0% |
| CLINC restricted choices | 89.2% | 100.0% |
| BANKING77 restricted choices | 86.4% | 97.6% |
| Wine quality bands | 22.4% | 48.8% |
| Constructed rubrics | 82.8% | 100.0% |
| **Overall** | **65.47% (982/1,500)** | **87.27% (1,309/1,500)** |

Original-view test NLL improved from 0.94293 to 0.29215 and Brier score from
0.54811 to 0.16893 (lower is better). Final original-view validation accuracy was
88.0% (792/900). These are this curated corpus's results, not official benchmark
scores or proof of general performance on arbitrary schemas and tasks.

Consistency is mixed and remains unresolved:

| Test relationship / absolute probability residual | Before | After |
|---|---:|---:|
| Negation complement, mean across 500 pairs | 0.15155 | 0.05294 |
| Negation complement, 95th percentile | 0.27659 | 0.39533 |
| Sentence-swap invariance, mean across 250 pairs | 0.01565 | 0.04202 |
| Sentence-swap invariance, 95th percentile | 0.04037 | 0.34072 |

Average negation consistency improved, but tail negation error and sentence-swap
probability drift worsened. Better classification and aggregate probability loss
do not imply universal consistency or calibrated probabilities. No weight-zero
consistency ablation has been run, so the contribution of the extra penalty is
not established. The 900-case calibration partition remains unused.

Final artifacts: `checkpoint`,
`test predictions`,
`validation predictions`.
The [trained inference latency benchmark](JUDGMENT_LATENCY.md) is now complete.
With an in-memory adapter merge and scoring batches of 32, five short-state
questions take 80.2 ms median, ten take 142.5 ms and fifty take 601.1 ms.
These include input/output processing with the model loaded. The faster path
scores 87.00% on the same test set versus 87.27% for the original independent
reference; probability shifts and targeted precision checks are documented there.

## Delivered

- Nested `state` and nested fixed dictionary/list question trees; Choice, Score
  and Noul leaves; full probability answers plus reconstructed value trees.
- Shared-prefix cached inference, independent differentiable scoring, grouped
  categorical loss, direct binary loss and verified complement/invariance losses.
- Reproducible source-pinned preparation: 20,000 training, 900 validation,
  900 calibration and 1,500 test bundles. Token filtering retains 19,999 training
  bundles / 77,882 units; one 1,042-token BoolQ bundle is excluded, not truncated.
- Rank-8 LoRA and numerical readouts, checkpointing, optimizer/RNG resume,
  hash-identified model outputs, explicit frozen-parameter policy and launch gates.

The original untrained nested request has eight numerical units and requires
three model forwards at the default batch size of four (one shared prefill and
two suffix batches). This is a numerical-output interface, not JSON text generation.
The observed cached/full maximum probability difference was 0.021224 in BF16,
within the declared 0.03 gate. Float-valued results are consequently not exactly
identical; tiny-model FP32 tests use much tighter tolerances.

## Definitive smoke evidence

Source: `judgment-smoke-v2/report.json`.

| Check | Result |
|---|---|
| Offline implementation suite | 180 tests and 45 subtests passed |
| Owned-source lint | Passed |
| Total parameters held by training model | 1,890,238,785 |
| Trainable adapter parameters | 8,409,600 |
| Trainable readout parameters | 4,097 |
| Frozen original weights | Before/after hashes identical |
| Long-input backward stress | 776-token longest retained training unit passed |
| Large candidate-group backward stress | Eight-candidate group passed |
| Smoke updates | 60 on 12 original bundles / 18 views |
| Tiny-sample NLL | 0.869699 → approximately 1.21e-9 |
| Peak allocated CUDA memory | 4.0104 GiB |
| Save/reload maximum logit difference | 0.0 |
| Resume optimizer update | Finite and successful |

Peak allocated memory excludes the desktop, other processes and allocator-reserved
memory. The tiny sample is deliberately memorized to test learnability and the
optimizer path. Its near-perfect result says nothing about held-out generalization.
The initial smoke is retained as a preliminary diagnostic; it is not the final gate.

The independent review found two material reporting issues: NLL calculated from
rounded probabilities could saturate incorrectly, and basename-based checkpoint
identities could collide. Both were fixed, regression-tested and re-reviewed.
Evaluation also reports canonical original-view metrics separately from metrics
including augmented views. The full-run gate rejects incomplete smoke runs,
changed source/data/runtime policy, and resume checkpoints from another stage.

## Full run

Started 2026-09-20 at 02:30 UTC on the local RTX 4080. It begins from the original
base and fresh adapters, not the deliberately overfit smoke checkpoint. It uses
one pass over eligible training data, four original bundles per optimizer update,
5,000 planned updates, checkpoints every 250 updates, and a three-hour training
loop cap. Preparation and evaluation are additional. The same 1,500-case heldout
test is scored before and after training, with no test-driven hyperparameter
selection in this run; full validation is also evaluated at completion.

The untrained held-out reference completed: **982/1,500 original cases correct
(65.47%)**, NLL **0.94293**, Brier **0.54811**. These are canonical original-view
metrics on this curated task mix, not an estimate for arbitrary structured tasks.
Raw predictions are in
`untrained-test.json`.

See the live/final run report,
log, and process record.
The job ran independently of the chat. Outputs include intermediate/final adapters,
readouts, optimizer state, data/source fingerprints and raw predictions.

Calibration stays identity and is labeled uncalibrated. No latency improvement is
claimed for trained adapters. The corpus is an initial mix of tasks, not evidence
of Jev-level generality: Tasksource contributes BANKING77 only; real Score data
includes a narrow wine-quality task; broader rubric transfer still needs evaluation.
See [data preparation](DATA_PREPARATION.md) and [workflow](../design/judgment-workflow.md).
