# Revised contrast data: fresh versus continued initialization

The two bounded runs and all frozen final tests are complete. This report compares **equal additional training budgets**, not equal lifetime compute. The original v0.2 checkpoint and previous experiment artifacts remain unchanged. No model was automatically promoted to the default.

**Recommendation:** continue from v0.2 for the next small experiment. It preserves
more broad capability within this additional budget, and its selected checkpoint
improves broad accuracy from 87.3% to 88.4%. Starting over reaches slightly higher
new-template accuracy (98.7% versus 98.4%) but only 83.1% on broad tasks. This is not
evidence that starting over is inherently worse: the fresh validation curve is
still improving and its lifetime training budget is smaller.

**Neither arm passes all readiness gates.** Continuation fails same-record
question contrasts (4/7) and broad NLL retention (.3504 versus .2927, exceeding the
allowed +.05). Its broad errors with answer probability at least .95 increase
from 28 to 42 despite fewer total mistakes. Fresh fails broad accuracy, broad NLL
and wine Score accuracy. The revised data and primitive balance address much of
the first pilot's brittleness, but do not finish the generalization or probability
quality work.

## Main comparison

| Initialization / reference | Revised contrast test | Original semantic test | Broad task test |
| --- | --- | --- | --- |
| Unchanged v0.2 | 579/1185 (48.9%) | 463/764 (60.6%) | 1309/1500 (87.3%) |
| Fresh original Qwen, step 1000 | 1170/1185 (98.7%) | 597/764 (78.1%) | 1246/1500 (83.1%) |
| Continue v0.2, step 250 | 1166/1185 (98.4%) | 638/764 (83.5%) | 1326/1500 (88.4%) |
| Jev 1.13.0 | 1128/1185 (95.2%) | 760/764 (99.5%), earlier archived run | Not run |

The revised test comprises **79 underlying scenarios**, each rendered in three layouts with five judgments: 237 requests and 1,185 correlated labels. Higher accuracy than Jev here does **not** establish broader superiority: our training directly covers these template families, while Jev is an external reference. The old semantic suite is an already-inspected regression test, not a new untouched benchmark.

## What changed and what stayed fixed

Training uses 220 scenarios × three layouts = 660 requests / 3,300 new judgments, plus 4,800 canonical original training questions (800 from each of six sources). Validation uses 48 scenarios / 720 judgments plus 300 canonical broad questions. Domain families and test question wording are held out, but generator templates are shared. The prose layout embeds JSON evidence and is not independently authored natural prose.

Each update averages six full-question losses: broad/new × Noul/Choice/Score. This gives each primitive one third of gradient weight and original/new data one half each. Both arms use the identical saved sampling schedule and prepared prompt hashes. Targets remain reviewed hard labels; this run does **not** test teacher distributions, RL, or an auxiliary consistency loss. The numerical head returns distributions, with Score interpreted as an ordered rubric and its probability-weighted mean.

Both runs use Qwen3.5-2B, rank-8/alpha-16 LoRA, frozen original backbone, fresh AdamW optimizers, adapter/head learning rates 5e-5/2.5e-5, BF16 backbone and FP32 heads/loss. The continuation arm begins with the original v0.2 adapters and heads; the fresh arm starts before our fine-tuning, not before Qwen’s own post-training. Continuation has 5,000 historical updates before this comparison. No input was truncated or excluded; longest sampled training unit: 700 tokens, ceiling: 1,536.

| Arm | Additional updates run | Gradient time | Selected update | Selected validation objective | Still improving at final interval? |
| --- | --- | --- | --- | --- | --- |
| Fresh original Qwen, step 1000 | 1000 | 21.9 min | 1000 | 0.2120 | True |
| Continue v0.2, step 250 | 1000 | 21.1 min | 250 | 0.1501 | False |

Gradient time excludes validation, final tests and loading. Selection was locked before any final test evaluation. The objective is the equal-weighted revised/broad mean NLL, with equal primitive weights inside each component. A lower objective wins; ties choose the earlier checkpoint. All 1,000 updates were run in both arms, even when the selected checkpoint is earlier. The fresh curve is still improving, so this study cannot rule out a better fresh-start recipe or a larger fresh budget. No new inference-speed benchmark was run.

## Validation curves

| Arm | Update | Selection objective | Revised validation accuracy | Broad validation accuracy |
| --- | --- | --- | --- | --- |
| fresh | 0 | 0.9952 | 373/720 (51.8%) | 205/300 (68.3%) |
| fresh | 250 | 0.3501 | 694/720 (96.4%) | 245/300 (81.7%) |
| fresh | 500 | 0.2338 | 701/720 (97.4%) | 247/300 (82.3%) |
| fresh | 1000 | 0.2120 | 720/720 (100.0%) | 252/300 (84.0%) |
| continued | 0 | 0.6928 | 359/720 (49.9%) | 269/300 (89.7%) |
| continued | 250 | 0.1501 | 713/720 (99.0%) | 271/300 (90.3%) |
| continued | 500 | 0.1714 | 718/720 (99.7%) | 262/300 (87.3%) |
| continued | 1000 | 0.1635 | 714/720 (99.2%) | 263/300 (87.7%) |

## Frozen readiness gates

**Fresh original Qwen, step 1000: FAIL** — 19/22 checks passed. Failed: `broad_accuracy_within_2pp`, `broad_nll_within_005`, `score_accuracy_wine_quality`.

**Continue v0.2, step 250: FAIL** — 20/22 checks passed. Failed: `broad_nll_within_005`, `pair_question_contrast`.

These are exploratory gates fixed before outputs, not statistical guarantees. Broad overall and each Score source must retain accuracy within two percentage points, each Score MAE within .05 level units, and broad NLL within .05. Revised accuracy, binary balance, positive and negative completion controls, separate Unknown recalls, pair correctness and layout gaps have their own thresholds. Passing one aggregate cannot hide a failed subgroup.

| Diagnostic | Unchanged v0.2 | Fresh | Continued | Jev |
| --- | --- | --- | --- | --- |
| Action Unknown recall | 0/90 | 90/90 | 90/90 | 89/90 |
| Registry Unknown recall | 20/54 | 54/54 | 54/54 | 45/54 |
| Macro binary balanced accuracy | 68.4% | 98.0% | 95.5% | 95.9% |
| Layout accuracy gap | 3.0% | 1.8% | 0.3% | 2.5% |

Completion controls below distinguish recognizing real success from correctly rejecting unsupported completion. Each layout has 18 positive and 33 negative completion cases.

| Model | Layout | Positive completion recall | Negative completion specificity |
| --- | --- | --- | --- |
| Unchanged v0.2 | flat | 18/18 | 3/33 |
| Unchanged v0.2 | nested | 18/18 | 3/33 |
| Unchanged v0.2 | prose | 18/18 | 1/33 |
| Fresh original Qwen, step 1000 | flat | 18/18 | 33/33 |
| Fresh original Qwen, step 1000 | nested | 18/18 | 33/33 |
| Fresh original Qwen, step 1000 | prose | 18/18 | 33/33 |
| Continue v0.2, step 250 | flat | 18/18 | 33/33 |
| Continue v0.2, step 250 | nested | 18/18 | 33/33 |
| Continue v0.2, step 250 | prose | 18/18 | 30/33 |
| Jev 1.13.0 | flat | 18/18 | 33/33 |
| Jev 1.13.0 | nested | 18/18 | 33/33 |
| Jev 1.13.0 | prose | 18/18 | 33/33 |

| Relation, both answers correct | Unchanged v0.2 | Fresh | Continued | Jev |
| --- | --- | --- | --- | --- |
| flip | 4/29 (13.8%) | 29/29 (100.0%) | 29/29 (100.0%) | 29/29 (100.0%) |
| question_contrast | 2/7 (28.6%) | 7/7 (100.0%) | 4/7 (57.1%) | 7/7 (100.0%) |
| invariant | 15/23 (65.2%) | 23/23 (100.0%) | 23/23 (100.0%) | 23/23 (100.0%) |
| layout_invariant | 348/790 (44.1%) | 776/790 (98.2%) | 769/790 (97.3%) | 742/790 (93.9%) |

The same-record question-contrast count is only seven pairs; evidence flips have 29 and semantic invariants 23. Layout invariants have 790 correlated pairs. Pair metrics therefore reveal specific behavior, not stable estimates of population accuracy. A both-correct rate measures answer correctness; it does not prove the complete probability vectors agree. Full distribution distances remain in the machine-readable summaries.

## Broad-task retention by source

| Source (250 cases each) | Unchanged v0.2 | Fresh | Continued |
| --- | --- | --- | --- |
| boolq | 208/250 (83.2%) | 196/250 (78.4%) | 214/250 (85.6%) |
| paws | 235/250 (94.0%) | 216/250 (86.4%) | 236/250 (94.4%) |
| tasksource/banking77 | 244/250 (97.6%) | 239/250 (95.6%) | 246/250 (98.4%) |
| clinc150 | 250/250 (100.0%) | 246/250 (98.4%) | 250/250 (100.0%) |
| oracle_policy | 250/250 (100.0%) | 250/250 (100.0%) | 250/250 (100.0%) |
| wine_quality | 122/250 (48.8%) | 99/250 (39.6%) | 130/250 (52.0%) |

| Score source mean absolute error, rubric units | Unchanged v0.2 | Fresh | Continued |
| --- | --- | --- | --- |
| oracle_policy | 0.0027 | 0.0003 | 4.69e-8 |
| wine_quality | 0.5817 | 0.6109 | 0.5765 |

Score accuracy uses the most probable level; MAE uses the distribution-weighted mean. These measure different behavior and are both retained. Original v0.2 was re-evaluated with the same canonical one-question protocol, rather than comparing differently packed old records.

## Probability quality and remaining errors

| Model | Revised NLL | Revised Brier | Revised errors at p ≥ .95 | Broad NLL | Broad Brier | Broad errors at p ≥ .95 |
| --- | --- | --- | --- | --- | --- | --- |
| Unchanged v0.2 | 1.1171 | 0.7036 | 20 | 0.2927 | 0.1691 | 28 |
| Fresh original Qwen, step 1000 | 0.0659 | 0.0242 | 12 | 0.5828 | 0.2388 | 80 |
| Continue v0.2, step 250 | 0.0887 | 0.0290 | 16 | 0.3504 | 0.1637 | 42 |
| Jev 1.13.0 | 0.1522 | 0.0799 | 0 | — | — | — |

Lower NLL and Brier are better. Brier is summed over all categories (binary is twice the scalar convention). Local NLL uses saved raw logits; Jev NLL uses returned probabilities floored at 1e-12, with zero-target counts separately recorded. No zero-target probability or vector normalization occurred in this Jev run. Its returned `confidence` is preserved but is not interpreted as the probability that its answer is correct. The high-probability-error count uses the largest answer probability, not that API confidence field. Good results on these hard labels do not establish calibration on genuinely ambiguous requests.

| Model | Revised Score accuracy | Revised Score mean absolute error |
| --- | --- | --- |
| Unchanged v0.2 | 99/237 (41.8%) | 0.6750 |
| Fresh original Qwen, step 1000 | 237/237 (100.0%) | 1.60e-5 |
| Continue v0.2, step 250 | 234/237 (98.7%) | 0.0115 |
| Jev 1.13.0 | 226/237 (95.4%) | 0.0675 |

**Fresh original Qwen, step 1000 error breakdown (15 judgments):**

| Variant | Question | Errors |
| --- | --- | --- |
| baseline_untrusted | policy_allows | 4 |
| intent_only | explicit_change_claim | 4 |
| stale_snapshot | policy_allows | 2 |
| claim_only | claimed_completion | 1 |
| conflicting_snapshots | policy_allows | 1 |
| intent_only | policy_allows | 1 |
| wrong_registry_snapshot | policy_allows | 1 |
| future_snapshot | policy_allows | 1 |

**Continue v0.2, step 250 error breakdown (19 judgments):**

| Variant | Question | Errors |
| --- | --- | --- |
| claim_only | claimed_completion | 6 |
| intent_only | explicit_change_claim | 4 |
| failed | completed | 3 |
| claimed_success | claimed_completion | 2 |
| conflicting | execution_stage | 2 |
| failed | execution_stage | 1 |
| future_snapshot | policy_allows | 1 |

The continuation arm's question-contrast failure is particularly revealing.
For the identical record “I have accepted the handoff for TRANSFER-73” with no
events, the correct answers are **claim present = true**, **completion confirmed
= false**. It assigned claim-present probability .000256 and completion-confirmed
probability .0000674: both answers became false. The earlier model often believed
a claim proved completion; this checkpoint sometimes fails to recognize the claim
itself. That is a different error, and it shows why both predicates need positive
controls. Inferring that training over-applied the evidence policy is plausible,
but this run does not isolate the cause.

## What Jev’s disagreements actually support

Jev matched 1,128/1,185 frozen labels (95.2%): 57 disagreements, independently reviewed after checkpoint selection. The gold remains unchanged. **Six disagreements are wording-sensitive:** the held-out phrase “explicit change assertion” lacks a past-completion qualifier, so “I will change…” can plausibly count as an assertion of a future change. This limitation was missed in the pre-freeze review. These six labels represent two scenarios in three layouts; they are not clean evidence that Jev confuses intent and completion. The same caveat applies when interpreting our models’ answers on these cases.

The other 51 disagreements comprise 32 temporal-scope judgments, 18 permission judgments (including one exact .5 tie), and one wrong-operation status. Under the explicit fixture contract, stale/future snapshots do not establish the requested current registry comparison. Jev failed 32 of 36 such temporal judgments, which come from just four scenarios across layouts. For policy_register/stale_snapshot/nested, a 09:00 snapshot was treated as evidence for a requested 10:00 state: change .91, changed status .88, stage 2 .92. This contract deliberately differs from “latest known state carries forward.”

Approval is explicitly independent of registry verification. Jev missed 18/84 policy approvals, 16 in policy_register and 17 in flat/nested layouts. Missing or unchanged evidence does not cancel recorded approval. This is observed domain/layout sensitivity; its cause is unproven. For consent/wrong_operation/flat, Jev selected no_effect (.56) over unknown (.42), although the rubric requires a matching attempt for no_effect and the only operation was an inspection.

These are useful teacher-audit cases, not reasons to treat every Jev distribution as a gold target. Any future soft-target experiment should separately preserve authored evidence rules, genuine annotation ambiguity, teacher disagreement and sampling variation. Repeated teacher samples measure teacher variability, not automatically the true probability of a proposition.

## Next bounded experiment

Keep the continuation checkpoint as a research candidate and preserve the current
default. Before scaling generation, author independent wording/template families
that separately vary claims, confirmation, negation, permission and evidence
availability; give the claim predicate its own positive and negative controls.
Carry both original broad data and all three primitives forward. Correct the
identified registry wording only in a new version, and create a new unseen outer
test before tuning to these findings.

Then compare a hard-target control with an otherwise identical audited
distribution-target run. Use held-out NLL/Brier, confident errors and both-correct
contrast pairs alongside accuracy. Jev/large-model distributions are fallible
observations; do not average them into gold or soften every clear label merely
because a teacher disagrees. Separate annotation ambiguity from model error.
A larger teacher or RL objective was not needed to demonstrate the data effect
in this run, and neither has been evaluated here.

## Verification and preserved artifacts

All 369 implementation tests and 45 subtests passed with the training overlay; targeted lint passed. Both initialization smoke runs had finite adapter/head gradients, no frozen gradients, unchanged frozen-weight hashes and exact save/reload probabilities. Both completed training arms retained the same frozen-weight digest. Source/data hashes match the frozen run record. Every final response and numerical prediction is saved, including all errors. The archive inventory verifies request and response hashes for all 451 Jev observations: 443 raw live captures and eight earlier imports with original artifact lines. No paid observation was discarded.

This revised comparison added 237 unique live observations (236,593 input tokens). Four Score preflight calls were reused in the 237-request final replay, leaving 233 additional calls there. At the previously recorded $0.042/M-input-token rate, this round’s list-price equivalent is **$0.00994**, not a verified account charge or balance. All 451 retained observations correspond to $0.02151 at that recorded rate.

- Frozen protocol
- [Corpus design and review](REVISED_DATASET.md)
- Full run record
- Locked selections
- Fresh selected checkpoint
- Continued selected checkpoint
- Both-arm smoke
- Jev replay and metric policy
- Jev archive integrity inventory
- Local artifact integrity checks
- Per-model errors and summaries
