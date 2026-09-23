# Architecture diagnostics v1 — results

The existing linear readout can fit every judgment and meaningful contrast pair in the small fitting set with the H0 backbone frozen. Larger or separate heads do not improve development correctness in this probe, and their raw probability errors worsen. This demonstrates finite fitting capacity, not abstract semantic understanding or generalization.

The real categorical cases already contain the full rubric in STATE. Explicitly repeating it is therefore a presentation intervention. The separate candidate-set sentinels expose the input-factorization boundary, but neither current scoring presentation solves their set-dependent contrasts.

All comparisons are exploratory. There are 10 fitting, 10 calibration and 10 development families, one per semantic category in each split: 200 judgments per split, plus 60 legacy retention judgments. H0 previously saw the fitting families. No original quarantined TEST/Jev outcome was opened and no paid calls were made.

## Fixed experiments

| Test | Intervention | Purpose |
|---|---|---|
| D0/D1 | Same H0 weights; original versus repeated full candidate bank | Test information/presentation |
| D2 | Original native vocabulary head: original and H0-adapted bodies; bounded original-model reasoning | Test accessible native capability under specified prompts |
| D3 | Current rank-8 adapters and heads, tiny fit-only run | Test fitting and transfer within 80 updates / 900 training seconds |
| D4 | Frozen H0 features; original, linear-refit, split Choice/Score and residual MLP heads | Isolate readout fitting and transfer |

## Development accuracy and raw probabilities

“Pair” is the equal-category mean of within-family both-correct contrast rates. Confident errors use maximum class probability ≥0.90, not the API entropy-based confidence field. Score accuracy uses the modal rubric level. All denominators and calibration modes are retained in `summary.json`.

| Condition | Accuracy | Pair | Raw NLL ↓ | Wrong / covered ≥.90 | Coverage ≥.90 |
|---|---:|---:|---:|---:|---:|
| H0 current presentation | 152/200 (76.0%) | 47.1% | 1.002 | 25/159 | 79.5% |
| H0 repeated full bank | 155/200 (77.5%) | 50.9% | 0.627 | 14/118 | 59.0% |
| linear-refit | 155/200 (77.5%) | 41.6% | 1.614 | 31/174 | 87.0% |
| split-choice-score | 155/200 (77.5%) | 41.6% | 1.923 | 31/174 | 87.0% |
| residual-mlp128 | 155/200 (77.5%) | 41.6% | 2.469 | 35/182 | 91.0% |
| D3 own step-zero baseline | 151/200 (75.5%) | 46.2% | 1.003 | 25/158 | 79.0% |
| D3 endpoint (80 updates) | 151/200 (75.5%) | 43.1% | 1.756 | 28/166 | 83.0% |

Repeating the full bank is the most encouraging simple presentation change in this probe: Choice accuracy rises 27→29/40 and Score 24→25/40, while Noul is exactly unchanged. Raw NLL falls 1.002→0.627; ≥90% errors fall 25→14 while coverage falls 79.5%→59%. It adds 8.4% input tokens and measured model time rises 20.7→22.4 seconds across the development suite. This small observed effect is not a confirmation study: the extra text, length and presentation all change, and the missing-set sentinels remain unsolved.

The refitted linear and split heads have identical development labels. The MLP changes four labels to other wrong labels; all three fitted heads have identical correctness vectors and pair counts. The fitting recipes differ between D3 and D4, so their relative scores are not a controlled LoRA-versus-head-training comparison.

## Fitting and retention

| Head probe | Fit judgments | Fit pairs | Legacy retention | Development NLL raw / global-T / per-primitive-T |
|---|---:|---:|---:|---:|
| frozen-original | 184/200 | 112/132 | 55/60 | 1.002 / 0.587 / 0.589 |
| linear-refit | 200/200 | 132/132 | 55/60 | 1.614 / 0.553 / 0.567 |
| split-choice-score | 200/200 | 132/132 | 55/60 | 1.923 / 0.579 / 0.584 |
| residual-mlp128 | 200/200 | 132/132 | 55/60 | 2.469 / 0.559 / 0.565 |

Temperatures were fitted only on the separate 200-judgment calibration split. They preserve ranking; improved NLL does not repair semantic decisions. Reduced confident-error counts must be read with coverage. For example, per-primitive calibration changes the frozen head from 25 wrong among 159 covered to 6 wrong among 87 covered; it changes the MLP from 35/182 to 9/78.

D3 ended with **INCONCLUSIVE**, reason `update_cap_inconclusive`, after 80 updates and 839.2 training-loop seconds (1038.0 seconds including its evaluations/checkpoints). Its fixed fitting milestones were:

| Update | Fit judgments | Fit pairs |
|---|---:|---:|
| 0 | 183/200 | 110/132 |
| 10 | 194/200 | 121/132 |
| 40 | 199/200 | 130/132 |
| 80 | 199/200 | 131/132 |

D3 legacy retention: 55/60 → 54/60. This deliberately fit-only diagnostic uses no replay. Any regression is part of its result, not a new deployment recommendation.

## Native interface

The direct mode averages canonical/reversed answer-code probabilities after mapping them back to meanings. The reasoning comparison instead uses canonical direct results, matching reasoning’s canonical ordering. These are constrained code probabilities at a forced answer prefix, not unrestricted chat-answer accuracy.

| Native condition | Accuracy | Pair | Raw NLL ↓ | Wrong / covered ≥.90 | Coverage ≥.90 |
|---|---:|---:|---:|---:|---:|
| native-original-development | 90/200 (45.0%) | 2.5% | 0.863 | 0/0 | 0.0% |
| native-H0-development | 94/200 (47.0%) | 8.0% | 0.918 | 0/0 | 0.0% |
| Original canonical direct | 94/200 (47.0%) | 1.5% | 0.894 | 0/0 | 0.0% |
| Original bounded reasoning | 89/200 (44.5%) | 3.5% | 0.912 | 0/0 | 0.0% |

Reasoning completed 200/200 questions, used 25600 generated tokens in 362.2 seconds, and stopped with `complete`. Scratchpad terminations: `{'token_cap': 200}`. The allowance was at most 128 tokens per question before forced final readout; a capped scratchpad is not a completed natural reasoning trace.

Direct native greedy outputs were valid answer codes on all captured direct requests. Native accuracy remains specific to this prompt, original vocabulary readout and inference budget. All 200 reasoning scratchpads reached the token cap, so this round does not adequately test completed native reasoning. A negative result cannot establish that Qwen lacks the capability under other prompts or longer reasoning.

## Candidate-set boundary

The sentinel task asks for the second-largest option, then adds a new larger option. For isolated candidate scoring, the relative A/B preference cannot change when neither branch sees a changed input. The mathematical limitation is independent of LoRA rank. Small floating-point changes due to batching do not supply the missing semantic information.

| Mode | Correct / 12 | Both-correct set-flip pairs / 6 |
|---|---:|---:|
| isolated | 4/12 | 0/6 |
| full_candidates | 2/12 | 0/6 |

All shared A/B orderings stayed unchanged in the isolated path. Complete options make set-relative rules representable, but exposing them zero-shot did not teach this scorer the rule. The full-bank real-case change was only 152→155 correct; all 80 categorical real development questions already supplied their criteria in STATE.

## Numerical and artifact checks

The same-instance batching check repeated both packings twice on four observed baseline-flip cases. Its within-packing ≤1e-7 repeat gates passed: **True**, with actual differences exactly zero. Switching packing reproduced all four earlier decision flips and each prior probability vector exactly. Frozen before/after hashes match; execution took 16.7 seconds with no cap overrun. Full raw logits, batch memberships and comparison receipts are in `batching-counterfactual/report.json` and `verification.json`. The evidence identifies batch-composition sensitivity, not its specific kernel cause.

D3’s step-zero batching differs from D0/D4: across the full cohorts, one fit and one development answer change, and the maximum calibration probability difference is 0.12744. Initial saved adapter/head tensors match H0 exactly. Use D3’s own 183/200 fit and 151/200 development baselines, rather than attributing that difference to training. A matched-packing repeat does not establish cross-packing numerical equivalence.

This body/batching sensitivity is distinct from the earlier FP32 CPU/GPU scalar-readout rounding. The latter was measured against an FP64 reference, retained both original GPU and CPU logits, passed probability difference ≤1e-6 with unchanged decision/tie sets, and preserved the CPU initial-head equivalence gate at 1e-6. Original checkpoint-resume gates were unchanged. Failed attempts and source versions remain archived.

Cohort/source/checkpoint hashes were rechecked. The complete H0 checkpoint and frozen backbone remain unchanged. All requested bounded jobs have saved observations, and D3 saves complete trainable tensors, AdamW and Python/CPU/CUDA RNG. No original experiment was resumed or promoted. The full CPU verification had 942 tests plus 48 subtests passing; the later relation-metadata alias repair separately passed its focused regression and all five metric-preflight suites.

## Interpretation and next decision

These results provide no evidence that simply enlarging the readout fixes nuanced generalization. Perfect small-set fitting is easy for the current linear readout, while development contrasts remain weak and larger heads sharpen mistakes. It would be premature to describe rank-8 LoRA as fundamentally incapable or to assume full fine-tuning would solve calibration.

Resolve numerical consistency across intended inference batch layouts before interpreting small accuracy changes. This changes only one development decision here and cannot by itself explain the much larger semantic error rate. The native-capability question also remains open: a smaller subset with enough budget to finish reasoning would be more informative than treating truncated traces as completed solutions. Then choose a separate controlled representation/adaptation experiment if needed: higher-rank LoRA or limited upper-block training, with matched data and explicit transfer/retention evaluation. No such follow-on training has been launched here.

There is only one training seed and one development family per category. No category-stratified uncertainty interval is presented: there is no within-category resampling population. These are exact observed counts, not significance claims. Native prompt and compute changes, frozen-feature fitting versus LoRA optimization differences, and previously inspected development data limit broader claims.

Artifacts: `PROTOCOL.md`, `summary.json`, `context/report.json`, `heads/results.json`, `native-*.json` and their complete `.observations/` directories, `review-results-context-heads.md`, `review-D3-baseline-batching.json`, and `batching-counterfactual/report.json`. Tiny-fit checkpoints and results are under `checkpoints/architecture-diagnostics-v1/tiny-fit/`.
