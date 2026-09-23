# Paired-language test: data and saved predictions

This folder holds everything needed to check the headline comparison without a GPU or an API key.

## The test

The frozen suite is [`data/paired-language-v1/test/suite.json`](../../data/paired-language-v1/test/suite.json): 160 requests and 800 judgments in 40 families, four per semantic category. Each family has four variants of one record and the same five questions, covering Noul, Choice and Score. Only each case's `request` is model input; labels, rationales and relation metadata stay outside the prompt. The [freeze record](../../data/paired-language-v1/evaluation-freeze.json) fixes its hash before any training language was generated.

The suite was written independently of the training data. The training corpus covers the same ten categories with different workflows and families, so the categories are familiar to our model but the families are not. Jev saw none of it.

## Predictions

| File | System |
|---|---|
| `predictions/public-data-base-v0.2.json` | Our model after training on public datasets only |
| `predictions/template-step-1000.json` | Continued for 1,000 updates on templated contrast data |
| `predictions/natural-step-1000.json` | Continued for 1,000 updates on the same facts in natural language |
| `predictions/natural-step-500.json` | The natural-language checkpoint selected on validation (update 500) |
| `predictions/jev-1.13.0.json` | Jev 1.13.0 through TypeSafe's API, one call per request |

Each row names the family, case and question and gives the expected answer, the predicted answer, the full probability distribution, the selected-answer probability, and per-judgment log loss and Brier score. The selection was locked before test outcomes were read ([`selection-locked.json`](selection-locked.json)).

## The five excluded families

After selection, an audit of every compiled model input found five families in which some questions relied on definitions written only in sibling questions or candidate criteria. Each scoring unit sees the shared state and its own question, so those definitions were invisible to it. [`test-contract-audit.json`](test-contract-audit.json) records the finding and lists the 35 untouched families. The exclusion followed a complete audit, not model outcomes, but it is still post hoc; both views are reported.

## Recompute

```sh
python evaluation/paired-language-v1/score_predictions.py
```

Expected output:

| System | All 800 | Wrong at ≥ 0.90 | 35 untouched | Wrong at ≥ 0.90 |
|---|---:|---:|---:|---:|
| Public-data base (v0.2) | 650 (81.25%) | 31 | 579 (82.71%) | 28 |
| Templated language, 1,000 updates | 675 (84.38%) | 79 | 601 (85.86%) | 69 |
| Natural language, 1,000 updates | 682 (85.25%) | 69 | 609 (87.00%) | 59 |
| Natural language, update 500 (selected) | 696 (87.00%) | 65 | 617 (88.14%) | 55 |
| Jev 1.13.0 | 764 (95.50%) | 2 | 678 (96.86%) | 1 |

Pair metrics, per-category results, calibration curves and the post-hoc run with definitions exposed are in the [full report](../../research-log/reports/paired-language-v1/RESULTS.md). The metrics code used for the report is [`experiments/paired_language_metrics.py`](../../experiments/paired_language_metrics.py).
