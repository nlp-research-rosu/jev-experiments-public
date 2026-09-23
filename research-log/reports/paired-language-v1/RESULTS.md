# Paired language experiment: completed results

Natural wording offers a modest benefit over templates at equal training budget, while overconfidence remains a problem. On the 35 untouched test families, natural versus template training at 1,000 updates changes accuracy by **+1.14 percentage points**, with an exploratory family-bootstrap interval of **-0.43 to +2.86 points**. The accuracy advantage is uncertain at this sample size. Evidence-contrast pairs and NLL favor natural language more clearly, but both trained endpoints have worse NLL than the starting model.

The selected natural checkpoint is update 500. It improves semantic accuracy over v0.2, but fails the preset broad-task probability-loss retention limit. No checkpoint was promoted to the default.

## What was run

- Two 1,000-update runs from exactly the same v0.2 weights, one training seed, identical replay identities/order, targets, candidate ordering, learning rates and loss. Only record and question wording differs.
- Each arm has a pool of 800 requests / 4,000 judgments across ten contrast categories, generated from 20 workflows crossed with ten shared semantic patterns. These are not 200 independent reasoning templates.
- Fixed sampling presents 2,600 distinct new judgments: 1,000 binary, all 800 Choice and all 800 Score. All 800 cases appear, but only 71 have every question presented. There is no extra pair-consistency loss.
- Independent validation: 80 requests / 400 judgments plus 300 old-task judgments. Independent test: 160 requests / 800 judgments. Old-task tests: 1,500 judgments; original semantic suite: 764 judgments.
- Both smoke tests, gradients, frozen-weight checks and reload checks passed. Both full runs completed. Training time excluding evaluation was 25.1 minutes for templates and 27.3 for natural language on the local RTX 4080.

## A data-interface issue found during error analysis

Whole-request review missed that some questions depended on definitions located only in sibling questions or candidate criteria. Each local scoring unit sees its own question/candidate and shared state, so those premises were absent. A full audit identified 44 affected test judgments across five families, and the same rubric problem in two validation families. We retained every original score, label, request and checkpoint selection.

We then copied the exact existing definitions into shared state for all variants in the affected families. Questions, labels, factual records, ordering and weights stayed unchanged. These inputs were frozen before the new diagnostic calls. The clarified evaluation is **post-hoc sensitivity analysis**, not a new independent confirmatory test. Jev improved from 86/100 to 99/100 on the retested slice; single-sample API variation remains unmeasured.

For a conservative view, the following table excludes all five affected test families, leaving 35 families / 140 requests / 700 judgments. Exclusion followed a complete contract audit rather than filtering by whether a model was correct. It remains a post-hoc subset.

## Untouched test families

| System | Accuracy | Evidence pairs | Question pairs | Unknown recall | NLL | Brier | Wrong ≥.90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Starting v0.2 | 579/700 (82.7%) | 64.4% | 81.4% | 25/40 (62.5%) | 0.456 | 0.255 | 28 |
| Template, step 1000 | 601/700 (85.9%) | 72.2% | 87.1% | 32/40 (80.0%) | 0.805 | 0.252 | 69 |
| Natural, step 1000 | 609/700 (87.0%) | 77.8% | 88.6% | 33/40 (82.5%) | 0.632 | 0.217 | 59 |
| Natural, selected step 500 | 617/700 (88.1%) | 82.2% | 90.0% | 34/40 (85.0%) | 0.545 | 0.203 | 55 |
| Jev 1.13.0 | 678/700 (96.9%) | 94.4% | 98.6% | 35/40 (87.5%) | 0.094 | 0.042 | 1 |

Pair scores require both answers to be correct. Unknown is an explicit Choice label. NLL and Brier are lower-is-better; NLL uses a common 1e-12 probability floor for cross-system comparison, while raw local-logit losses are retained in JSON. The number of confidently wrong answers uses selected-answer probability, not Jev’s separate confidence field. Brier and NLL can disagree because NLL penalizes extremely unlikely correct labels especially strongly.

## Equal-budget language comparison

| Metric | Natural minus template | Exploratory 95% interval |
| --- | --- | --- |
| accuracy | 0.011 | [-0.004285714285714226, 0.02857142857142858] |
| nll | -0.172 | [-0.2729677290772745, -0.06987977113208876] |
| flip | 0.056 | [0.022222222222222143, 0.09999999999999998] |
| question_contrast | 0.014 | [-0.02857142857142858, 0.0714285714285714] |

This comparison uses both 1,000-update checkpoints. The selected-checkpoint comparison has different selected update counts (0 versus 500), so its larger difference is a result of the full selection recipe, not a pure equal-update language effect. Bootstrap draws whole families within category, 2,000 times; one training seed means it does not measure training-run variance.

## Full test with definitions exposed: post-hoc sensitivity

| System | Accuracy | Evidence pairs | Question pairs | Unknown recall | NLL | Brier | Wrong ≥.90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Starting v0.2 | 657/800 (82.1%) | 63.0% | 80.0% | 26/41 (63.4%) | 0.473 | 0.263 | 30 |
| Template, step 1000 | 673/800 (84.1%) | 68.0% | 82.5% | 33/41 (80.5%) | 0.816 | 0.272 | 78 |
| Natural, step 1000 | 684/800 (85.5%) | 73.0% | 83.8% | 34/41 (82.9%) | 0.650 | 0.233 | 65 |
| Natural, selected step 500 | 697/800 (87.1%) | 79.0% | 86.2% | 35/41 (85.4%) | 0.555 | 0.215 | 63 |
| Jev 1.13.0 | 777/800 (97.1%) | 95.0% | 98.8% | 36/41 (87.8%) | 0.090 | 0.040 | 1 |

All 700 unaffected predictions are reused exactly. The 100 judgments from the five changed families were rerun locally and on Jev, with every response retained. These results are kept separate from the original frozen scores.

## Category matrix on clarified inputs

| Specific category | General category | Questions | Starting v0.2 | Template, step 1000 | Natural, step 1000 | Natural, selected step 500 | Jev 1.13.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| action_binding | binding | 80 | 88.8% | 90.0% | 90.0% | 88.8% | 100.0% |
| attribution_and_endorsement | evidence | 80 | 78.8% | 72.5% | 75.0% | 77.5% | 92.5% |
| claim_vs_completion | evidence | 80 | 76.2% | 87.5% | 87.5% | 90.0% | 100.0% |
| entity_binding | binding | 80 | 76.2% | 71.2% | 75.0% | 80.0% | 95.0% |
| negation_and_quantifiers | logic | 80 | 78.8% | 90.0% | 91.2% | 88.8% | 98.8% |
| ordered_rubrics | rubric | 80 | 72.5% | 65.0% | 68.8% | 75.0% | 100.0% |
| permission_vs_execution | policy | 80 | 83.8% | 86.2% | 87.5% | 87.5% | 93.8% |
| reversal_and_current_state | temporal | 80 | 83.8% | 88.8% | 90.0% | 92.5% | 97.5% |
| temporal_scope | temporal | 80 | 90.0% | 91.2% | 91.2% | 95.0% | 98.8% |
| unknown_vs_failure | epistemic | 80 | 92.5% | 98.8% | 98.8% | 96.2% | 95.0% |

## General-category matrix on clarified inputs

| General category | Questions | Starting v0.2 | Template, step 1000 | Natural, step 1000 | Natural, selected step 500 | Jev 1.13.0 |
| --- | --- | --- | --- | --- | --- | --- |
| binding | 160 | 82.5% | 80.6% | 82.5% | 84.4% | 97.5% |
| epistemic | 80 | 92.5% | 98.8% | 98.8% | 96.2% | 95.0% |
| evidence | 160 | 77.5% | 80.0% | 81.2% | 83.8% | 96.2% |
| logic | 80 | 78.8% | 90.0% | 91.2% | 88.8% | 98.8% |
| policy | 80 | 83.8% | 86.2% | 87.5% | 87.5% | 93.8% |
| rubric | 80 | 72.5% | 65.0% | 68.8% | 75.0% | 100.0% |
| temporal | 160 | 86.9% | 90.0% | 90.6% | 93.8% | 98.1% |

## Original-task retention

| System | Broad correct / 1500 | Broad NLL | Old semantic correct / 764 | Retention checks |
| --- | --- | --- | --- | --- |
| Starting v0.2 | 1309/1500 | 0.293 | 463/764 | Reference |
| Natural, selected step 500 | 1319/1500 | 0.363 | 528/764 | FAIL |
| Template, step 1000 | 1326/1500 | 0.347 | 561/764 | FAIL |
| Natural, step 1000 | 1323/1500 | 0.347 | 554/764 | FAIL |

The selected natural checkpoint increases broad accuracy from 1309/1500 to 1319/1500 and old semantic accuracy from 463/764 to 528/764. Its broad NLL rises from 0.293 to 0.363, exceeding the allowed 0.05 increase. Broad accuracy and Score-source accuracy/MAE checks pass for that checkpoint; the NLL check fails. Therefore the original model remains the default.

## Selection sensitivity

| Arm | Original selected update | Clarified-validation selected update |
| --- | --- | --- |
| template | 0 | 0 |
| natural | 500 | 500 |

All saved validation checkpoints were reevaluated on the corrected eight-case slice; unaffected validation and original broad-validation predictions were preserved. The selection formula stayed fixed and alternative selections were locked before clarified local testing. Neither selected update changes.

## What to do next

Keep this as a bounded positive result for linguistic diversity, not justification for scaling the current generator by 100×. The next controlled experiment should test probability quality: a validation-fitted calibration baseline versus distribution-target training, while retaining hard contrast tests and original-task checks. Also vary the supplied rubric while holding the record fixed, and compare complete-family sampling so binary question contrasts are reliably seen. The export-stage regression in EXAMPLES.md is consistent with applying a familiar training rubric instead of the requested rubric; calibration alone cannot fix that class-ordering error. Review rendered model units before generating further data; full-request agreement alone is insufficient.

## Illustrative gains and regressions

See EXAMPLES.md and illustrative-errors.json for exact cases and full probability vectors. The natural checkpoint distinguishes an unsigned draft from signed verification much better, but can confidently miss an explicit approval-only stage or a resolved award for one person. These selected examples are explanatory, not additional benchmark estimates.

## Files

- `results-matrix.json`: full per-category, general-category, primitive, predicate and family metrics; confusion matrices; Unknown precision/recall; NLL/Brier/ECE; Score MAE; confidence-error counts; paired bootstrap; retention; audit provenance.
- `unaffected_families-matrix.json` and `post_hoc_clarified-matrix.json`: separate machine-readable views.
- `ORIGINAL_RESULTS.md` and `original-results-matrix.json`: preserved original frozen scores, with the input-contract caveat.
- `DATA.md`, `actual-training-exposure.json`, and `data/paired-language-v1/paired-manifest.json`: corpus and sampling details.
- `test-contract-audit.json`, `validation-contract-audit.json`, and `docs/paired-language-clarification-v1.md`: identified defects and diagnostic rules.
- `checkpoints/paired-language-v1/`: all checkpoints, histories, raw local responses and locked original selections.
- `reports/paired-language-clarification-v1/`: diagnostic predictions and selection-sensitivity record.
- `data/jev-responses/`: every paid response, including the 20 clarified requests. No source response was discarded.

Other limits: model-authored/reviewed synthetic examples; shared workflow and semantic scaffolds; only four original test families per category; unresolved minor wording caveats documented by the audit; one training seed and one Jev sample per request. These observations do not identify Jev’s internal architecture or establish universal judgment performance.
