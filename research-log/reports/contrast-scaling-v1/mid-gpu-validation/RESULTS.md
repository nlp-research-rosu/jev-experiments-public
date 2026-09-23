# Mid-run GPU validation — training paused

Training was intentionally paused after 1,692 complete family updates. The checkpoint preserves weights, optimizer and RNG state, and its production next-update equivalence check passed exactly. No training has resumed.

Every checkpoint was evaluated on the complete predeclared validation populations: 40 contrast families / 160 cases / 800 judgments, plus 300 original-task cases. The runtime matches the planned GPU evaluator: BF16 body, FP32 readouts, FLA, independent candidate scoring and batch four. Final-test and Jev outcomes remain unopened.

## Decision supported by this validation

Keep the current recipe paused. The initial 200-family adaptation improves accuracy by 9.5 percentage points (555/800 to 631/800) and improves Brier score, so the training did teach useful behavior. Continuing to 1,692 families gives 629/800, essentially the same accuracy as the 200-family checkpoint, while NLL rises from 0.7857 to 1.5498 and high-confidence errors rise from 71 to 128.

The last checkpoint makes 732 predictions at selected probability at least 90%. Their mean selected probability is 99.58%, but only 82.51% are correct. The concern is not solely that a larger high-confidence group contains more errors: its stated probabilities substantially exceed its observed accuracy.

The highest contrast accuracy observed is 637/800 at 800 families (3,200 cases), just 6 additional correct judgments over the 200-family checkpoint, with worse NLL and Brier. The predeclared probability-based combined validation objective still favors the starting checkpoint; among trained checkpoints it favors 200 families. No checkpoint is promoted.

Original-task accuracy changes from 269/300 initially to 264/300 at the pause point, while NLL rises from 0.2567 to 0.3577. This is a modest accuracy regression and a more noticeable probability-quality regression, not a wholesale loss of the original capability. New Score accuracy improves initially from 81/160 to 104/160, then drops to 93/160 at the pause point. Its NLL rises from 1.3833 at 200 families to 3.1228 at 1,692.

On the earlier CPU slice, GPU scoring changes zero predicted labels for all three matched checkpoints (600 matched judgments total). Individual probabilities can differ by up to roughly 0.117, so CPU and GPU probability values should not be treated as identical; the principal early warning survives the GPU check.

Paired resampling of whole validation families within category gives an exploratory 95% interval of [-1.625, +1.125] percentage points for the latest-minus200 accuracy change, versus [+0.674, +0.854] for its NLL increase. These intervals describe variation across this authored validation corpus, not training-seed variance or uncertainty from the user’s interim stopping decision. Detailed comparisons are in `paired-family-comparisons.json`.

This supports diagnosing/changing the recipe before more long training. It does not prove that reinforcement learning is necessary or that more data can never help: the repeat-data control was not run. A cheap calibration-only baseline can separate excessive sharpness from incorrect ranking; targeted Score/rubric/source-admissibility work addresses errors that calibration cannot repair. Further training awaits the user’s decision. Performance investigation remains documented in `../PERFORMANCE_NEXT.md` and can operate on disposable checkpoint copies.

## Overall comparison

| Checkpoint | Contrast accuracy | NLL ↓ | Brier ↓ | Wrong ≥.90 | Wrong ≥.95 | Broad correct | Broad NLL ↓ | Validation objective ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Starting v0.2 | 555/800 (69.38%) | 0.7287 | 0.4172 | 37 | 10 | 269/300 | 0.2567 | 0.5941 |
| 200 families / 800 cases | 631/800 (78.88%) | 0.7857 | 0.3357 | 71 | 50 | 271/300 | 0.2743 | 0.6757 |
| 400 families / 1,600 cases | 628/800 (78.50%) | 1.0166 | 0.3631 | 101 | 92 | 263/300 | 0.2852 | 0.8524 |
| 600 families / 2,400 cases | 610/800 (76.25%) | 1.4355 | 0.4094 | 119 | 106 | 264/300 | 0.2755 | 1.1621 |
| 800 families / 3,200 cases | 637/800 (79.62%) | 1.0036 | 0.3531 | 107 | 82 | 264/300 | 0.2438 | 0.7946 |
| 1,000 families / 4,000 cases | 626/800 (78.25%) | 1.2233 | 0.3791 | 125 | 109 | 268/300 | 0.3345 | 1.0284 |
| 1,692 families / 6,768 cases | 629/800 (78.62%) | 1.5498 | 0.3848 | 128 | 116 | 264/300 | 0.3577 | 1.2836 |

The lowest predeclared validation objective among these checkpoints occurs at step 0; the highest contrast accuracy occurs at step 800. These are descriptive validation findings, not a promotion or final-test selection lock. The objective gives equal weight to new and broad validation, with primitives weighted equally inside each.

## Primitive results

| Family updates | Primitive | Correct | NLL ↓ | Wrong ≥.90 |
| --- | --- | --- | --- | --- |
| 0 | noul | 397/480 | 0.4245 | 27 |
| 0 | choice | 77/160 | 1.2296 | 9 |
| 0 | score | 81/160 | 1.1403 | 1 |
| 200 | noul | 426/480 | 0.3487 | 30 |
| 200 | choice | 101/160 | 1.4993 | 29 |
| 200 | score | 104/160 | 1.3833 | 12 |
| 400 | noul | 428/480 | 0.4122 | 39 |
| 400 | choice | 97/160 | 2.0751 | 37 |
| 400 | score | 103/160 | 1.7714 | 25 |
| 600 | noul | 422/480 | 0.5160 | 41 |
| 600 | choice | 96/160 | 2.9779 | 44 |
| 600 | score | 92/160 | 2.6518 | 34 |
| 800 | noul | 424/480 | 0.4908 | 41 |
| 800 | choice | 112/160 | 1.7046 | 30 |
| 800 | score | 101/160 | 1.8409 | 36 |
| 1000 | noul | 428/480 | 0.4748 | 43 |
| 1000 | choice | 104/160 | 2.1639 | 37 |
| 1000 | score | 94/160 | 2.5280 | 45 |
| 1692 | noul | 428/480 | 0.5602 | 44 |
| 1692 | choice | 108/160 | 2.9453 | 40 |
| 1692 | score | 93/160 | 3.1228 | 44 |

## Complete-pair results

| Family updates | Relation | Both endpoints correct |
| --- | --- | --- |
| 0 | flip | 42/136 |
| 0 | layout_invariant | 12/20 |
| 0 | question_contrast | 36/60 |
| 0 | invariant | 40/40 |
| 0 | rubric flips only | 16/76 |
| 200 | flip | 56/136 |
| 200 | layout_invariant | 19/20 |
| 200 | question_contrast | 45/60 |
| 200 | invariant | 36/40 |
| 200 | rubric flips only | 20/76 |
| 400 | flip | 60/136 |
| 400 | layout_invariant | 18/20 |
| 400 | question_contrast | 45/60 |
| 400 | invariant | 36/40 |
| 400 | rubric flips only | 24/76 |
| 600 | flip | 51/136 |
| 600 | layout_invariant | 18/20 |
| 600 | question_contrast | 45/60 |
| 600 | invariant | 35/40 |
| 600 | rubric flips only | 19/76 |
| 800 | flip | 64/136 |
| 800 | layout_invariant | 19/20 |
| 800 | question_contrast | 43/60 |
| 800 | invariant | 36/40 |
| 800 | rubric flips only | 30/76 |
| 1000 | flip | 59/136 |
| 1000 | layout_invariant | 17/20 |
| 1000 | question_contrast | 45/60 |
| 1000 | invariant | 36/40 |
| 1000 | rubric flips only | 23/76 |
| 1692 | flip | 57/136 |
| 1692 | layout_invariant | 19/20 |
| 1692 | question_contrast | 45/60 |
| 1692 | invariant | 35/40 |
| 1692 | rubric flips only | 22/76 |

## High-confidence group

| Family updates | Predictions ≥.90 | Wrong | Observed accuracy | Mean stated probability |
| --- | --- | --- | --- | --- |
| 0 | 356 | 37 | 89.61% | 95.72% |
| 200 | 612 | 71 | 88.40% | 98.42% |
| 400 | 677 | 101 | 85.08% | 99.09% |
| 600 | 680 | 119 | 82.50% | 99.20% |
| 800 | 693 | 107 | 84.56% | 98.95% |
| 1000 | 722 | 125 | 82.69% | 99.27% |
| 1692 | 732 | 128 | 82.51% | 99.58% |

This table reports confidence coverage as well as error counts; a larger confidently-wrong count alone is not a complete calibration assessment.

## Category accuracy

| Category | 0 | 200 | 400 | 600 | 800 | 1000 | 1692 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| action_binding | 52/80 | 56/80 | 59/80 | 56/80 | 59/80 | 60/80 | 61/80 |
| attribution_and_endorsement | 74/80 | 74/80 | 74/80 | 70/80 | 73/80 | 73/80 | 72/80 |
| claim_vs_completion | 59/80 | 66/80 | 66/80 | 67/80 | 67/80 | 64/80 | 65/80 |
| entity_binding | 26/80 | 56/80 | 55/80 | 54/80 | 54/80 | 51/80 | 57/80 |
| negation_and_quantifiers | 58/80 | 55/80 | 55/80 | 48/80 | 54/80 | 51/80 | 51/80 |
| ordered_rubrics | 46/80 | 70/80 | 65/80 | 68/80 | 71/80 | 70/80 | 71/80 |
| permission_vs_execution | 54/80 | 70/80 | 72/80 | 69/80 | 71/80 | 71/80 | 71/80 |
| reversal_and_current_state | 64/80 | 62/80 | 61/80 | 57/80 | 61/80 | 60/80 | 61/80 |
| temporal_scope | 48/80 | 46/80 | 45/80 | 45/80 | 51/80 | 51/80 | 44/80 |
| unknown_vs_failure | 74/80 | 76/80 | 76/80 | 76/80 | 76/80 | 75/80 | 76/80 |

## Matched CPU slice

| Family updates | GPU correct | GPU NLL | GPU wrong ≥.90 | Changed labels vs CPU | Max probability difference |
| --- | --- | --- | --- | --- | --- |
| 0 | 142/200 | 0.6796 | 7 | 0 | 0.025441 |
| 200 | 159/200 | 0.6751 | 10 | 0 | 0.116887 |
| 1000 | 158/200 | 1.0994 | 30 | 0 | 0.111948 |

The matched slice uses precisely the same 200 judgments as the earlier CPU probe. It separates arithmetic/runtime differences from the larger validation population.

## Interpretation limits

This is validation data with one training seed. The user requested pausing after seeing an earlier subset of this validation population. The original 5,000-family endpoint and the update-matched repeat control are unfinished, so these observations do not isolate unique data quantity from additional optimization. The newest checkpoint is an intermediate 1,692-family point. The untouched final test is reserved for a later explicit selection decision. No learning objective, calibration, runtime package or dataset was changed for this evaluation.
