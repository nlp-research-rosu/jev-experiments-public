# Retrospective binary calibration diagnostic

This diagnostic finds a substantial confidence-scale problem, but a single global
monotone calibration does not repair the decision errors. It also finds useful
ordering signal: trained binary flip pairs have the correct direction in 77/80
cases, and per-family binary AUROC ranges from 0.764 to 0.843. These results do not
support a claim that discrimination is entirely absent. They show that global
confidence and threshold correction are insufficient, particularly across the
two generators' distinct question sets.

This is an analysis of previously inspected, authored templates. It is **not**
production calibration or an estimate of untouched held-out performance. No
calibration is deployed, no model is run, and no frozen input or primary report
is modified.

## Method fixed before fitting

- Inputs: frozen `data/semantic-contrasts-v1/suite.json`, its manifest, and the
  saved `reports/semantic-contrasts-v1/{untrained,trained}-responses.jsonl` files.
  Only 650 Noul fields from 114 cases are used; gold targets are hard booleans.
  All 114 categorical answers are excluded and remain untouched.
- Comparison: identity/raw logits; temperature-only `z' = a*z`; monotone affine
  `z' = a*z + b`. The scale is constrained to `[0.001, 20]`, the offset to
  `[-20, 20]`, and temperature fixes `b = 0`.
- Objective: stable float64 binary cross-entropy plus
  `0.5 * 1e-4 * ((a-1)^2 + b^2)`. Every original training case has equal total
  fitting weight, divided equally across its five or six binary fields. No
  per-predicate or held-out-family parameters are fit.
- CPU NumPy optimization uses convex coordinate minimization with 60 derivative
  bisection steps per free coordinate. Projected gradient tolerance is `1e-8`,
  with at most 2,000 cycles. All 32 fitted maps converged; the largest projected
  gradient was `9.5864e-9`. Several untrained maps reached the lower scale bound;
  no trained map reached either bound.
- Six family folds each fit on the other five families. The stricter two folds
  train on execution and predict destination, then reverse direction. The latter
  have disjoint question IDs as well as disjoint families. Every field receives
  exactly one out-of-fold prediction per method and split.
- Bounds, regularization, grouping, and optimization were fixed before scores
  were computed. No held-out scores were used to select a method or tune a map.
- Reported metrics below weight binary questions equally. Case-weighted
  NLL/Brier/accuracy are also retained in the machine-readable report. Brier is
  the primary report's two-class sum, `2*(p_true-y)^2`; the scalar convention is
  also stored. The decision rule is calibrated logit `>= 0` (probability `>= .5`);
  there are no exact-zero raw or out-of-fold logits in these results.

## Trained checkpoint

| Evaluation | Method | Correct / 650 | Accuracy | Balanced accuracy | NLL | Brier | Errors with confidence >= .95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Raw | Identity | 419 | 64.46% | 63.03% | 0.96261 | 0.57766 | 100 |
| Six family folds | Temperature | 419 | 64.46% | 63.03% | 0.63926 | 0.44821 | 0 |
| Six family folds | Affine | 422 | 64.92% | 64.47% | 0.59779 | 0.41917 | 0 |
| Two generator folds | Temperature | 419 | 64.46% | 63.03% | 0.76971 | 0.52674 | 0 |
| Two generator folds | Affine | 399 | 61.38% | 60.81% | 0.84734 | 0.56444 | 27 |

Temperature reduces confidence and improves aggregate loss while leaving every
binary decision unchanged. Removing all 100 highly confident errors therefore
does not mean correcting those errors. In the six-family affine result, false
positives fall from 227 to 146, while false negatives rise from 4 to 82: only
three additional answers become correct overall. Equal-case-weighted affine
accuracy is 64.65%, NLL 0.60007, and two-class Brier 0.42133.

The fold breakdown explains why a global offset transfers poorly:

| Held-out generator | Affine map trained on | a | b | Raw accuracy | Affine accuracy | Raw NLL | Affine NLL |
|---|---|---:|---:|---:|---:|---:|---:|
| Execution, 480 fields | Destination | 1.055918 | -0.919656 | 62.50% | 64.79% | 1.09800 | 0.86614 |
| Destination, 170 fields | Execution | 0.900437 | -2.461241 | 70.00% | 51.76% | 0.58032 | 0.79425 |

The execution-trained map creates 82 false negatives on destination fields,
compared with three raw false negatives. The reverse transfer retains 27 highly
confident errors. Even temperature alone worsens destination NLL from 0.58032
to 0.61683; the aggregate benefit is driven by execution fields.

## Binary contrast pairs

Eligible pairs have two Noul endpoints in the same held-out family, so both
endpoints use the same fitted map. Of 260 frozen relations, 226 are eligible:
80 flips, 42 different-question contrasts, and 104 invariance pairs. The other
34 are categorical. Counts below require **both answers to be correct**.

| Trained method | Flips / 80 | Question contrasts / 42 | Invariance pairs / 104 |
|---|---:|---:|---:|
| Raw | 26 | 6 | 83 |
| Family temperature | 26 | 6 | 83 |
| Family affine | 25 | 4 | 67 |
| Generator temperature | 26 | 6 | 83 |
| Generator affine | 17 | 3 | 72 |

All positive-scale maps preserve within-family AUROC and contrast direction.
The trained model orders 77/80 flips and 33/42 question contrasts correctly under
every map. The large gap between correct direction and both-correct counts shows
that useful relative responses coexist with poorly placed decision thresholds.
The tested global affine correction reduces rather than improves these
both-correct contrast counts. Lower total variation after shrinking confidence
should likewise not be mistaken for improved semantic invariance.

## Untrained reference

| Evaluation | Method | Accuracy | Balanced accuracy | NLL | Brier |
|---|---|---:|---:|---:|---:|
| Raw | Identity | 56.92% | 57.52% | 0.74292 | 0.53237 |
| Six family folds | Temperature | 56.92% | 57.52% | 0.69987 | 0.50661 |
| Six family folds | Affine | 50.15% | 48.94% | 0.69822 | 0.50498 |
| Two generator folds | Temperature | 56.92% | 57.52% | 0.77179 | 0.55464 |
| Two generator folds | Affine | 50.46% | 49.51% | 0.76905 | 0.55314 |

None of the untrained conditions has an error with confidence >= .95. Its family
affine improvement in probability loss comes with worse binary accuracy;
generator-transfer probability losses also worsen. This reinforces the need to
separate probability loss, decisions, and contrast behavior.

## Scope and reproducibility

These are correlated views from six authored domain families and only two
generators. Family folds retain closely related templates in their training
portion, so their apparent transfer is weaker evidence than the generator split.
Neither split undoes prior inspection of the suite. No confidence intervals or
population-generalization claims are made. This tests only two simple global
monotone map classes with fixed bounds and regularization. It cannot identify
the internal cause of semantic mistakes, prove no richer calibration would
help, or replace fresh evaluation data. Ranking and both-correct metrics are
needed alongside improved NLL to avoid calling confidence shrinkage a semantic
fix. Pooled out-of-fold AUROC is intentionally omitted because different folds
use different maps; within-family and within-fold AUROC are recorded instead.

Reproduce with `.venv/bin/python -m experiments.contrast_calibration_probe
--output reports/<new-directory>`; the script refuses to overwrite an existing
output directory. The default output is `reports/contrast-calibration-v1/`:

- `summary.json`: every fit's coefficients, objectives, convergence, bounds,
  training/test membership, metrics, and input/source hashes.
- `predictions.jsonl`: all raw and out-of-fold binary predictions, including
  original logits, transformed logits, targets, probabilities, and fold IDs.
- `relations.jsonl`: every eligible relation's both-correct and direction result.
- `manifest.json`: input/source hashes and hashes of all three result files.

Verification: 11 new tests pass, covering held-out exclusion, held-out-label
perturbation, monotonicity, temperature decisions, saturated logits, equal case
weighting, optimizer convergence, metric definitions, and relation eligibility.
The full CPU/offline suite completed with **250 passed, 1 skipped, and 45
subtests passed**. Lint passed for the two new Python files. An independent
post-run check confirmed family disjointness and unchanged within-family AUROC
and contrast direction under every positive-scale map. The runner verifies
frozen suite integrity and checks that all four input hashes are unchanged at
completion.
