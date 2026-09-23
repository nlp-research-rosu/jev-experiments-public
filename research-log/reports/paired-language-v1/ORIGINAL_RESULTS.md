# Original frozen paired-language experiment scores

**Evaluation audit:** Post-selection review found missing shared definitions in five test families and two validation families. These original scores are preserved for transparency. See RESULTS.md for unaffected-family and clarified-input sensitivity results; do not attribute every original error to model reasoning.

This comparison keeps semantic facts, labels, replay examples, optimizer recipe and update count fixed. Natural training changes both evidence wording and question wording. The independent test contains 160 requests / 800 judgments in 40 authored families across ten contrast categories.

## Validation-selected checkpoints

| Arm | Selected update | Validation objective |
| --- | --- | --- |
| template | 0 | 0.416 |
| natural | 500 | 0.372 |

| System | Accuracy | Flip pair | Question pair | Unknown recall | Probability NLL | Brier | Wrong ≥.90 | Score MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | 650/800 (81.2%) | 60.0% | 75.0% | 26/41 (63.4%) | 0.483 | 0.270 | 31 | 0.551 |
| template | 650/800 (81.2%) | 60.0% | 75.0% | 26/41 (63.4%) | 0.483 | 0.270 | 31 | 0.551 |
| natural | 696/800 (87.0%) | 78.0% | 83.8% | 35/41 (85.4%) | 0.571 | 0.222 | 65 | 0.355 |
| Jev | 764/800 (95.5%) | 90.0% | 92.5% | 36/41 (87.8%) | 0.116 | 0.058 | 2 | 0.075 |

All pair columns require both answers to be correct. Unknown is an explicit Choice label, not probability 0.5. Probability NLL uses the same 1e-12 floor across systems; local raw-logit NLL is retained separately. API probability rounding limits precision. Brier sums across answer classes and is not directly comparable between tasks with different class counts.

## Validation trajectory

| Arm | Update | Selection objective | New accuracy | New raw NLL | New wrong ≥.95 | Broad accuracy | Broad raw NLL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| template | 0 | 0.416 | 81.0% | 0.494 | 8 | 89.7% | 0.257 |
| template | 250 | 0.425 | 85.5% | 0.573 | 30 | 88.3% | 0.255 |
| template | 500 | 0.461 | 87.8% | 0.588 | 29 | 90.7% | 0.239 |
| template | 1000 | 0.539 | 85.0% | 0.746 | 40 | 87.7% | 0.291 |
| natural | 0 | 0.416 | 81.0% | 0.494 | 8 | 89.7% | 0.257 |
| natural | 250 | 0.418 | 85.5% | 0.587 | 34 | 87.7% | 0.278 |
| natural | 500 | 0.372 | 89.0% | 0.477 | 30 | 90.0% | 0.272 |
| natural | 1000 | 0.441 | 87.2% | 0.629 | 33 | 87.7% | 0.285 |

The selection objective equally weights new and broad validation, and equally weights the three primitive NLLs inside each set. It is therefore not the simple average of the two micro-NLL columns.

## Fixed 1,000-update comparison

| System | Accuracy | Flip pair | Question pair | Unknown recall | Probability NLL | Brier | Wrong ≥.90 | Score MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| template_1000 | 675/800 (84.4%) | 68.0% | 80.0% | 33/41 (80.5%) | 0.814 | 0.272 | 79 | 0.409 |
| natural_1000 | 682/800 (85.2%) | 73.0% | 81.2% | 34/41 (82.9%) | 0.657 | 0.237 | 69 | 0.352 |

Both arms received the same 1,000 optimizer updates. A selected checkpoint may be earlier; the fixed comparison separates equal-budget effects from validation-based checkpoint selection. No test result changes selection.

## Specific category accuracy

| Category | Questions | reference | template | natural | Jev |
| --- | --- | --- | --- | --- | --- |
| action_binding | 80 | 88.8% | 88.8% | 88.8% | 100.0% |
| attribution_and_endorsement | 80 | 78.8% | 78.8% | 77.5% | 92.5% |
| claim_vs_completion | 80 | 76.2% | 76.2% | 90.0% | 100.0% |
| entity_binding | 80 | 76.2% | 76.2% | 80.0% | 95.0% |
| negation_and_quantifiers | 80 | 78.8% | 78.8% | 88.8% | 98.8% |
| ordered_rubrics | 80 | 66.2% | 66.2% | 77.5% | 85.0% |
| permission_vs_execution | 80 | 83.8% | 83.8% | 87.5% | 93.8% |
| reversal_and_current_state | 80 | 83.8% | 83.8% | 92.5% | 97.5% |
| temporal_scope | 80 | 87.5% | 87.5% | 91.2% | 97.5% |
| unknown_vs_failure | 80 | 92.5% | 92.5% | 96.2% | 95.0% |

## General category accuracy

| Category | Questions | reference | template | natural | Jev |
| --- | --- | --- | --- | --- | --- |
| binding | 160 | 82.5% | 82.5% | 84.4% | 97.5% |
| epistemic | 80 | 92.5% | 92.5% | 96.2% | 95.0% |
| evidence | 160 | 77.5% | 77.5% | 83.8% | 96.2% |
| logic | 80 | 78.8% | 78.8% | 88.8% | 98.8% |
| policy | 80 | 83.8% | 83.8% | 87.5% | 93.8% |
| rubric | 80 | 66.2% | 66.2% | 77.5% | 85.0% |
| temporal | 160 | 85.6% | 85.6% | 91.9% | 97.5% |

## Contrast pairs by category

| Category | System | Flip both correct | Question both correct | Invariant both correct | Invariant probability drift |
| --- | --- | --- | --- | --- | --- |
| action_binding | reference | 75.0% | 50.0% | 100.0% | 0.081 |
| action_binding | template | 75.0% | 50.0% | 100.0% | 0.081 |
| action_binding | natural | 75.0% | 50.0% | 100.0% | 0.001 |
| action_binding | Jev | 100.0% | 100.0% | 100.0% | 0.023 |
| attribution_and_endorsement | reference | 75.0% | 100.0% | 50.0% | 0.154 |
| attribution_and_endorsement | template | 75.0% | 100.0% | 50.0% | 0.154 |
| attribution_and_endorsement | natural | 66.7% | 87.5% | 50.0% | 0.247 |
| attribution_and_endorsement | Jev | 91.7% | 100.0% | 75.0% | 0.260 |
| claim_vs_completion | reference | 41.7% | 62.5% | 100.0% | 0.021 |
| claim_vs_completion | template | 41.7% | 62.5% | 100.0% | 0.021 |
| claim_vs_completion | natural | 75.0% | 100.0% | 100.0% | 0.002 |
| claim_vs_completion | Jev | 100.0% | 100.0% | 100.0% | 0.007 |
| entity_binding | reference | 83.3% | 87.5% | 50.0% | 0.050 |
| entity_binding | template | 83.3% | 87.5% | 50.0% | 0.050 |
| entity_binding | natural | 91.7% | 87.5% | 50.0% | 0.002 |
| entity_binding | Jev | 91.7% | 100.0% | 100.0% | 0.010 |
| negation_and_quantifiers | reference | 50.0% | 100.0% | 100.0% | 0.027 |
| negation_and_quantifiers | template | 50.0% | 100.0% | 100.0% | 0.027 |
| negation_and_quantifiers | natural | 87.5% | 100.0% | 100.0% | 0.001 |
| negation_and_quantifiers | Jev | 87.5% | 87.5% | 100.0% | 0.003 |
| ordered_rubrics | reference | 12.5% | 25.0% | 75.0% | 0.130 |
| ordered_rubrics | template | 12.5% | 25.0% | 75.0% | 0.130 |
| ordered_rubrics | natural | 37.5% | 37.5% | 100.0% | 0.003 |
| ordered_rubrics | Jev | 50.0% | 37.5% | 75.0% | 0.072 |
| permission_vs_execution | reference | 58.3% | 87.5% | 100.0% | 0.008 |
| permission_vs_execution | template | 58.3% | 87.5% | 100.0% | 0.008 |
| permission_vs_execution | natural | 75.0% | 87.5% | 100.0% | 0.000 |
| permission_vs_execution | Jev | 91.7% | 100.0% | 100.0% | 0.008 |
| reversal_and_current_state | reference | 37.5% | 87.5% | 100.0% | 0.018 |
| reversal_and_current_state | template | 37.5% | 87.5% | 100.0% | 0.018 |
| reversal_and_current_state | natural | 87.5% | 100.0% | 100.0% | 0.001 |
| reversal_and_current_state | Jev | 87.5% | 100.0% | 100.0% | 0.057 |
| temporal_scope | reference | 75.0% | 75.0% | 75.0% | 0.245 |
| temporal_scope | template | 75.0% | 75.0% | 75.0% | 0.245 |
| temporal_scope | natural | 87.5% | 87.5% | 100.0% | 0.096 |
| temporal_scope | Jev | 87.5% | 100.0% | 100.0% | 0.060 |
| unknown_vs_failure | reference | 75.0% | 75.0% | 100.0% | 0.069 |
| unknown_vs_failure | template | 75.0% | 75.0% | 100.0% | 0.069 |
| unknown_vs_failure | natural | 91.7% | 100.0% | 100.0% | 0.003 |
| unknown_vs_failure | Jev | 100.0% | 100.0% | 100.0% | 0.037 |

## Unknown and probability quality by category

| Category | System | Unknown precision | Unknown recall | NLL | Brier | Wrong ≥.90 |
| --- | --- | --- | --- | --- | --- | --- |
| action_binding | reference | 100.0% | 100.0% | 0.345 | 0.197 | 6 |
| action_binding | template | 100.0% | 100.0% | 0.345 | 0.197 | 6 |
| action_binding | natural | 100.0% | 100.0% | 0.460 | 0.190 | 6 |
| action_binding | Jev | 100.0% | 100.0% | 0.044 | 0.010 | 0 |
| attribution_and_endorsement | reference | 57.1% | 100.0% | 0.555 | 0.327 | 4 |
| attribution_and_endorsement | template | 57.1% | 100.0% | 0.555 | 0.327 | 4 |
| attribution_and_endorsement | natural | 57.1% | 100.0% | 0.970 | 0.361 | 10 |
| attribution_and_endorsement | Jev | 80.0% | 100.0% | 0.184 | 0.099 | 0 |
| claim_vs_completion | reference | 100.0% | 25.0% | 0.720 | 0.376 | 6 |
| claim_vs_completion | template | 100.0% | 25.0% | 0.720 | 0.376 | 6 |
| claim_vs_completion | natural | 100.0% | 75.0% | 0.515 | 0.175 | 7 |
| claim_vs_completion | Jev | 100.0% | 100.0% | 0.062 | 0.015 | 0 |
| entity_binding | reference | 100.0% | 25.0% | 0.600 | 0.334 | 2 |
| entity_binding | template | 100.0% | 25.0% | 0.600 | 0.334 | 2 |
| entity_binding | natural | 100.0% | 100.0% | 1.047 | 0.357 | 11 |
| entity_binding | Jev | 100.0% | 75.0% | 0.108 | 0.050 | 0 |
| negation_and_quantifiers | reference | 33.3% | 25.0% | 0.411 | 0.246 | 2 |
| negation_and_quantifiers | template | 33.3% | 25.0% | 0.411 | 0.246 | 2 |
| negation_and_quantifiers | natural | 100.0% | 100.0% | 0.253 | 0.149 | 3 |
| negation_and_quantifiers | Jev | 100.0% | 100.0% | 0.053 | 0.020 | 0 |
| ordered_rubrics | reference | 0.0% | — | 0.784 | 0.436 | 3 |
| ordered_rubrics | template | 0.0% | — | 0.784 | 0.436 | 3 |
| ordered_rubrics | natural | — | — | 0.819 | 0.385 | 9 |
| ordered_rubrics | Jev | — | — | 0.266 | 0.175 | 1 |
| permission_vs_execution | reference | 100.0% | 25.0% | 0.352 | 0.203 | 1 |
| permission_vs_execution | template | 100.0% | 25.0% | 0.352 | 0.203 | 1 |
| permission_vs_execution | natural | 100.0% | 25.0% | 0.613 | 0.216 | 6 |
| permission_vs_execution | Jev | 75.0% | 75.0% | 0.157 | 0.084 | 1 |
| reversal_and_current_state | reference | 66.7% | 50.0% | 0.451 | 0.247 | 3 |
| reversal_and_current_state | template | 66.7% | 50.0% | 0.451 | 0.247 | 3 |
| reversal_and_current_state | natural | 80.0% | 100.0% | 0.510 | 0.151 | 5 |
| reversal_and_current_state | Jev | 100.0% | 75.0% | 0.088 | 0.037 | 0 |
| temporal_scope | reference | 80.0% | 100.0% | 0.364 | 0.195 | 4 |
| temporal_scope | template | 80.0% | 100.0% | 0.364 | 0.195 | 4 |
| temporal_scope | natural | 100.0% | 100.0% | 0.388 | 0.162 | 5 |
| temporal_scope | Jev | 80.0% | 100.0% | 0.110 | 0.045 | 0 |
| unknown_vs_failure | reference | 100.0% | 87.5% | 0.252 | 0.139 | 0 |
| unknown_vs_failure | template | 100.0% | 87.5% | 0.252 | 0.139 | 0 |
| unknown_vs_failure | natural | 100.0% | 75.0% | 0.130 | 0.072 | 3 |
| unknown_vs_failure | Jev | 100.0% | 75.0% | 0.088 | 0.044 | 0 |

## Retention on original tasks

| System | Broad correct / 1500 | Broad NLL | Original semantic correct | Retention gates |
| --- | --- | --- | --- | --- |
| reference | 1309/1500 | 0.293 | 463/764 | reference |
| template | 1309/1500 | 0.293 | 463/764 | PASS |
| natural | 1319/1500 | 0.363 | 528/764 | FAIL |

Gates fixed before training: broad accuracy and each Score-source accuracy within 2 percentage points of v0.2; each Score-source MAE within .05; broad NLL within .05. Full source metrics, gate decisions and deltas are in original-results-matrix.json.

## Paired uncertainty

| Comparison | Metric | Natural − template | Exploratory 95% family interval |
| --- | --- | --- | --- |
| validation_selected | accuracy | 0.057 | [0.03374999999999995, 0.08124999999999993] |
| validation_selected | nll | 0.087 | [-0.009493926728350921, 0.18750259515797685] |
| validation_selected | flip | 0.180 | [0.10999999999999999, 0.25] |
| validation_selected | question_contrast | 0.088 | [0.025000000000000022, 0.15000000000000002] |
| fixed_1000_updates | accuracy | 0.009 | [-0.006249999999999978, 0.025000000000000022] |
| fixed_1000_updates | nll | -0.156 | [-0.2467186933303347, -0.07249498140298838] |
| fixed_1000_updates | flip | 0.050 | [0.010000000000000009, 0.08999999999999997] |
| fixed_1000_updates | question_contrast | 0.012 | [-0.03750000000000009, 0.07499999999999996] |

Bootstrap uses 2,000 resamples of complete families within category, not individual correlated judgments. With four held-out families per category these intervals are exploratory and do not establish population performance. Both arms use one training seed; this bootstrap does not measure training-run variance.

## Scope and limitations

- Training has 20 workflow settings crossed with ten shared semantic patterns: 200 workflow/category families, 800 requests and 4,000 judgments per arm. This is not 200 independently designed reasoning templates. The six-slot schedule balances original/new and Noul/Choice/Score. Each arm sees 2,600 distinct new judgments: 1,000 of 2,400 Noul, all 800 Choice and all 800 Score. All 800 cases appear, but only 71 have all five questions presented. Contrast pairs are not jointly sampled and there is no pair-consistency loss. These coverage limits are shared by both arms.
- Natural records and questions were authored by Terra medium or Sol high, reviewed independently, and repaired where needed. These are synthetic/model-reviewed labels and language, not human adjudication. The original attempts and all review flags remain saved.
- Evaluation was independently authored and frozen before training language generation; conceptual categories and some broad domains overlap by design. This is a small controlled experiment, not a universal structured-reasoning benchmark.
- Jev is an external reference, not a training teacher or ground truth. Its complete paid responses remain archived. Similar behavior cannot identify its architecture or training method.
- Hard targets and the existing supervised loss were used. Distribution targets, calibration fitting, teacher distillation and RL were deliberately held out of this experiment.
- All original checkpoints remain unchanged. Neither arm is automatically promoted to the project default.

## Artifacts

- Frozen protocol: docs/paired-language-contract-v1.md
- Paired data and hashes: data/paired-language-v1/paired-manifest.json
- Complete metric matrix and confusion tables: reports/paired-language-v1/original-results-matrix.json
- Training histories, checkpoint selection and raw local predictions: checkpoints/paired-language-v1/
- Raw Jev observations: data/jev-responses/; replay mapping: reports/jev-paired-language-test-v1/
- Semantic review audit and original flagged drafts: reports/paired-language-v1/natural-review-audit.json and data/paired-language-v1/natural-candidates/
