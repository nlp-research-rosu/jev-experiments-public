# Results

| File | Contents |
|---|---|
| [training-runs.csv](training-runs.csv) | Every training arm: study, date, initialization, data, objective, updates, training minutes on one RTX 4080, outcome and report |
| [jev-comparisons.csv](jev-comparisons.csv) | Every test on which we compared our models with Jev 1.13.0, with counts and sources |
| [llm-judge-audit.csv](llm-judge-audit.csv) | The 45-case KleverBench proof-spec audit: Jev, Sonnet and Luna; [case files and saved judgments](../data/kleverbench-judge-v1/README.md) |
| [figures/](figures/) | Earlier research figures, as PNG and editable SVG; the [current thread figures](../thread/tweets.md) live with the posts |

Training minutes exclude data preparation, smoke tests and evaluation. They come from each study's report or its training history. The calibrated screen reports 5.06 GPU-hours for its six arms, listed here as 50.6 minutes each.
