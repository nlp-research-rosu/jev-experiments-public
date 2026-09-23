# Cross-fitted calibration diagnostic

Training stayed paused. This used saved raw validation logits, with no model inference, weight update, new teacher request or final-test access. Every assessed family was excluded from the temperature fit applied to it. The same folds and fit objective were used for all seven checkpoints.

## Main finding

For the latest checkpoint, new-task NLL improves from **1.5498 to 0.5452** with one global temperature and to **0.5072** with one temperature per primitive. Broad-task NLL improves from **0.3577 to 0.2626/0.2324** respectively. New-task accuracy stays **629/800**, and broad accuracy stays **264/300**. Thus much of the raw probability deterioration is adjustable through scale; the semantic classification mistakes remain.

Per-primitive calibration leaves 469/800 new predictions above 90% probability, with 22 wrong (95.31% correct) and mean selected probability 95.72%. Raw predictions had732/800above 90%,128 wrong (82.51% correct), and mean probability 99.58%. Coverage reduction is reported explicitly; the fewer high-confidence errors are not an accuracy gain.

This is stronger evidence than fitting and scoring a temperature on the same rows, but it is still exploratory validation evidence: the corpus was previously inspected, multiple checkpoints/variants are compared, and only one model-training seed exists. The four fitted transforms form an out-of-fold evaluation procedure, not one deployed calibrator.

## Protocol

Four folds preserve 40 contrast families and 295 original-task context groups. Each new-task assessment fold has 200 judgments. Broad fold sizes are 80/76/72/72. Families are assigned by a fixed metadata-only hash within category/source. Fits use the other three folds and the predeclared objective: equal new/broad weight and equal primitive weight within each. Positive temperatures are searched over 0.05–100. An independent reviewer reconstructed all 112 fits and 23,100 predictions without refitting and found no leakage, weighting mismatch or decision change.

## All checkpoints

| Family updates | Variant | New correct | New NLL ↓ | New Brier ↓ | Wrong ≥.90 | Broad NLL ↓ | Objective ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | identity | 555/800 | 0.7287 | 0.4172 | 37 | 0.2567 | 0.5941 |
| 0 | global | 555/800 | 0.6824 | 0.3966 | 2 | 0.2446 | 0.5577 |
| 0 | per_primitive | 555/800 | 0.6788 | 0.3966 | 12 | 0.2457 | 0.5554 |
| 200 | identity | 631/800 | 0.7857 | 0.3357 | 71 | 0.2743 | 0.6757 |
| 200 | global | 631/800 | 0.5424 | 0.2954 | 5 | 0.2376 | 0.4661 |
| 200 | per_primitive | 631/800 | 0.5205 | 0.2900 | 14 | 0.2227 | 0.4498 |
| 400 | identity | 628/800 | 1.0166 | 0.3631 | 101 | 0.2852 | 0.8524 |
| 400 | global | 628/800 | 0.5488 | 0.2993 | 7 | 0.2538 | 0.4806 |
| 400 | per_primitive | 628/800 | 0.5202 | 0.2920 | 17 | 0.2413 | 0.4619 |
| 600 | identity | 610/800 | 1.4355 | 0.4094 | 119 | 0.2755 | 1.1621 |
| 600 | global | 610/800 | 0.6214 | 0.3392 | 8 | 0.2781 | 0.5324 |
| 600 | per_primitive | 610/800 | 0.5763 | 0.3203 | 30 | 0.2457 | 0.4961 |
| 800 | identity | 637/800 | 1.0036 | 0.3531 | 107 | 0.2438 | 0.7946 |
| 800 | global | 637/800 | 0.5427 | 0.2956 | 14 | 0.2502 | 0.4681 |
| 800 | per_primitive | 637/800 | 0.5286 | 0.2907 | 29 | 0.2347 | 0.4522 |
| 1000 | identity | 626/800 | 1.2233 | 0.3791 | 125 | 0.3345 | 1.0284 |
| 1000 | global | 626/800 | 0.5577 | 0.3032 | 6 | 0.2695 | 0.4946 |
| 1000 | per_primitive | 626/800 | 0.5295 | 0.2946 | 22 | 0.2436 | 0.4684 |
| 1692 | identity | 629/800 | 1.5498 | 0.3848 | 128 | 0.3577 | 1.2836 |
| 1692 | global | 629/800 | 0.5452 | 0.2958 | 9 | 0.2626 | 0.4788 |
| 1692 | per_primitive | 629/800 | 0.5072 | 0.2849 | 22 | 0.2324 | 0.4469 |

## High-confidence coverage

| Family updates | Variant | New predictions ≥.90 | Wrong | Observed accuracy | Mean stated probability |
| --- | --- | --- | --- | --- | --- |
| 0 | identity | 356 | 37 | 89.61% | 95.72% |
| 0 | global | 136 | 2 | 98.53% | 92.22% |
| 0 | per_primitive | 233 | 12 | 94.85% | 93.90% |
| 200 | identity | 612 | 71 | 88.40% | 98.42% |
| 200 | global | 181 | 5 | 97.24% | 93.48% |
| 200 | per_primitive | 378 | 14 | 96.30% | 95.87% |
| 400 | identity | 677 | 101 | 85.08% | 99.09% |
| 400 | global | 132 | 7 | 94.70% | 94.30% |
| 400 | per_primitive | 413 | 17 | 95.88% | 95.46% |
| 600 | identity | 680 | 119 | 82.50% | 99.20% |
| 600 | global | 75 | 8 | 89.33% | 96.55% |
| 600 | per_primitive | 422 | 30 | 92.89% | 95.25% |
| 800 | identity | 693 | 107 | 84.56% | 98.95% |
| 800 | global | 249 | 14 | 94.38% | 93.63% |
| 800 | per_primitive | 440 | 29 | 93.41% | 95.28% |
| 1000 | identity | 722 | 125 | 82.69% | 99.27% |
| 1000 | global | 105 | 6 | 94.29% | 94.52% |
| 1000 | per_primitive | 454 | 22 | 95.15% | 95.41% |
| 1692 | identity | 732 | 128 | 82.51% | 99.58% |
| 1692 | global | 137 | 9 | 93.43% | 94.76% |
| 1692 | per_primitive | 469 | 22 | 95.31% | 95.72% |

## Fitted temperatures

| Family updates | Variant | Primitive | Mean T | Fold range | Boundary optimum |
| --- | --- | --- | --- | --- | --- |
| 0 | global | all | 1.627 | 1.590–1.696 | False |
| 0 | per_primitive | choice | 1.952 | 1.890–2.032 | False |
| 0 | per_primitive | noul | 1.321 | 1.258–1.360 | False |
| 0 | per_primitive | score | 1.664 | 1.532–1.937 | False |
| 200 | global | all | 2.780 | 2.731–2.834 | False |
| 200 | per_primitive | choice | 3.437 | 3.381–3.539 | False |
| 200 | per_primitive | noul | 1.641 | 1.599–1.698 | False |
| 200 | per_primitive | score | 3.443 | 3.233–3.616 | False |
| 400 | global | all | 3.641 | 3.608–3.658 | False |
| 400 | per_primitive | choice | 4.568 | 4.475–4.688 | False |
| 400 | per_primitive | noul | 2.133 | 2.076–2.209 | False |
| 400 | per_primitive | score | 4.449 | 4.186–4.585 | False |
| 600 | global | all | 4.862 | 4.757–4.940 | False |
| 600 | per_primitive | choice | 6.097 | 6.045–6.186 | False |
| 600 | per_primitive | noul | 2.200 | 2.143–2.251 | False |
| 600 | per_primitive | score | 6.333 | 5.881–6.503 | False |
| 800 | global | all | 3.377 | 3.310–3.424 | False |
| 800 | per_primitive | choice | 3.933 | 3.839–4.002 | False |
| 800 | per_primitive | noul | 2.102 | 2.088–2.115 | False |
| 800 | per_primitive | score | 4.353 | 4.052–4.589 | False |
| 1000 | global | all | 4.057 | 4.009–4.102 | False |
| 1000 | per_primitive | choice | 4.339 | 4.229–4.417 | False |
| 1000 | per_primitive | noul | 2.223 | 2.195–2.280 | False |
| 1000 | per_primitive | score | 5.789 | 5.506–5.959 | False |
| 1692 | global | all | 5.632 | 5.569–5.673 | False |
| 1692 | per_primitive | choice | 6.722 | 6.631–6.817 | False |
| 1692 | per_primitive | noul | 2.721 | 2.649–2.786 | False |
| 1692 | per_primitive | score | 7.397 | 7.096–7.585 | False |

## Score expected-value error

| Family updates | Variant | New Score MAE ↓ | Broad Score MAE ↓ |
| --- | --- | --- | --- |
| 0 | identity | 0.7916 | 0.3774 |
| 0 | global | 0.8202 | 0.3698 |
| 0 | per_primitive | 0.8200 | 0.3717 |
| 200 | identity | 0.5455 | 0.3688 |
| 200 | global | 0.5979 | 0.3582 |
| 200 | per_primitive | 0.6212 | 0.3562 |
| 400 | identity | 0.5360 | 0.3776 |
| 400 | global | 0.5860 | 0.3555 |
| 400 | per_primitive | 0.6086 | 0.3538 |
| 600 | identity | 0.5869 | 0.3700 |
| 600 | global | 0.6286 | 0.3480 |
| 600 | per_primitive | 0.6573 | 0.3498 |
| 800 | identity | 0.5368 | 0.3458 |
| 800 | global | 0.5642 | 0.3416 |
| 800 | per_primitive | 0.5930 | 0.3417 |
| 1000 | identity | 0.5703 | 0.3499 |
| 1000 | global | 0.5934 | 0.3504 |
| 1000 | per_primitive | 0.6355 | 0.3508 |
| 1692 | identity | 0.5888 | 0.3766 |
| 1692 | global | 0.5745 | 0.3482 |
| 1692 | per_primitive | 0.5960 | 0.3494 |

Temperature preserves the most-probable Score level, but changes the returned expected numeric Score and can change downstream threshold decisions. For the latest checkpoint, new Score MAE is 0.5888 raw, 0.5745 global, and 0.5960 per-primitive. The lower-NLL per-primitive option slightly worsens this MAE. Calibration must not be described as fixing rubric compliance or improving every metric.

## Verification and provenance

protocol.json freezes folds, source hashes, bounds and comparisons. Per-checkpoint files retain every derived prediction and fit. Identity reconstruction matches the archived probabilities/losses. Tests cover analytic temperature optima, macro weighting, class-group cardinality, Score mean changes, family leakage, class invariance and large-offset stability. The numerical tie guard fails closed if the existing probability-tolerance winner changes. See ../calibration-review.md for independent verification. Original outputs/checkpoints and held-out test/Jev artifacts were unchanged.
