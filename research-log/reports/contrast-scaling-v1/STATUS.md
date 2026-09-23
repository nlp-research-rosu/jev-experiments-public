# Contrast-data scaling experiment — paused; GPU validation complete

Updated 2026-09-21 16:05 UTC. Training remains **paused after 1,692 complete family updates**. All weights, optimizer/RNG state and exact resume checks are preserved. Full GPU validation of seven checkpoints is complete: initial accuracy gains plateau while NLL and high-confidence errors worsen with further updates. See [GPU results](mid-gpu-validation/RESULTS.md). Automatic training continuation is disabled. Final-test and Jev outcomes remain unopened.

## Completed interim CPU validation

See interim results. On the same 200-judgment slice, contrast accuracy is 71.0% / 79.5% / 79.0% for starting v0.2 / 200 families / 1,000 families. Errors at selected probability at least90% are7 /10 /31; NLL is0.6794 /0.6792 /1.1014. Original-task validation is54/60 /55/60 /54/60. This small CPU FP32 cached check is exploratory and does not replace the full GPU evaluation. Training was subsequently paused at the user’s request for full GPU validation.

The authorized follow-up performance investigation is recorded in PERFORMANCE_NEXT.md and will run after the current pipeline is safely complete, before another larger-scale study.

## Comparison

| Unique families | Cases | Judgments |
|---:|---:|---:|
| 200 | 800 | 4,000 |
| 1,000 | 4,000 | 20,000 |
| 5,000 | 20,000 | 100,000 |

The main run covers these strictly nested prefixes. A separate control repeats the first200families for1,000totalupdates; comparing it with1,000distinctfamilies holds update count fixed. The5,000endpoint includes both more data and more compute.

All runs start from the original v0.2 Qwen3.5-2B checkpoint, with the same hard-label BCE/CE loss and old-task replay. Every new family contributes all20judgments per update. No RL, soft targets, calibration adjustment or larger model is introduced in this experiment.

## Ready and verified

- Frozen training corpus:20,000cases across10categories, built from200authoredlanguageblueprints with varied evidence and supplied rules.
- Independent frozen evaluation:160validationcases/800judgments and320testcases/1,600judgments, with existing suites retained for regression checks.
- Software verification:537tests plus45subtests passed; data and reporting received independent review.
- GPU smoke:10completefamilies, finite gradients, frozen original weights unchanged, exact probability and next-update agreement after checkpoint restore at the unchanged1e-7tolerance.
- Deterministic GPU kernels and batch12 are recorded in all run fingerprints. The initial nondeterministic backward discrepancy was diagnosed and repaired before full training.
- Jev:all320responses saved locally; outcomes withheld from checkpoint selection.

## Original full pipeline — currently paused

1. Primary training:5,000families, saving the defined prefix checkpoints.
2. Compute control:restore the200checkpoint including optimizer/RNG, then800additionalrepeatupdates.
3. Validation:select checkpoints and write the selection lock.
4. Test:all unique endpoints/selections on new and retained suites.
5. Reporting:accuracy, contrast-pair correctness, NLL/Brier, high-confidence errors, Unknown precision/recall, Score error, retention, family-bootstrap intervals and Jev comparisons.

Smoke timing projects about19hours of training, plus evaluation. This is an estimate; live timing is in the training histories. Total unpadded candidate-input exposure is176,751,801tokens for the primary stream and28,305,620for the repeat extension.

The hourly follow-up is paused. Training will not restart automatically. The user will decide the next training experiment after the mid-run GPU validation results.

Live status: pipeline-status.json. Detailed audit: LEDGER.md. Final results will be written to `final-report/RESULTS.md` and `final-report/report.json` after the pipeline completes.

## Limits to interpretation

The5,000families do not constitute5,000independently authored language templates; there are200blueprints. The experiment uses one training seed. Family-bootstrap intervals measure held-out-case variation. Existing test suites have already been inspected in prior work; the newly authored suite provides the fresh evaluation. Jev is an external reference, not the ground truth. Nine cosmetic lint findings in generator/report test formatting remain recorded; runner/evaluator lint passed.
