# Research log

These are the original reports and design notes from our research archive, copied unchanged except for links. Links to files that are in this repository were rewritten to point to them. Links to artifacts that are not published here (raw JSON results, logs, checkpoints, source snapshots) are shown as plain text. The reports keep their original voice, including notes about authorization, review and scheduling.

Read them in the order below. [What we tried](../docs/what-we-tried.md) summarizes each study.

## 19 September: the untrained scorer and Jev's contract

- [RESULTS.md](reports/RESULTS.md): first measurement of shared-context scoring against JSON generation, with [diagnostic](reports/diagnostic-qwen35-2b.md) and [upstream](reports/upstream-qwen35-2b.md) tables
- [INVESTIGATION.md](reports/INVESTIGATION.md): answer encoding, option order and numerical precision
- [FEWSHOT.md](reports/FEWSHOT.md), [ALIGNED_FEWSHOT.md](reports/ALIGNED_FEWSHOT.md), [COMPACT_FEWSHOT.md](reports/COMPACT_FEWSHOT.md): prompting with examples
- [BATCHING.md](reports/BATCHING.md): shared prefill with batched suffixes
- [JEV_CAPABILITIES.md](reports/JEV_CAPABILITIES.md), [JEV_API_MODEL_BOUNDARY.md](reports/JEV_API_MODEL_BOUNDARY.md): what Jev's documentation discloses
- [PROBABILITY_DESIGN.md](reports/PROBABILITY_DESIGN.md), [CONSISTENCY_DATA_AUDIT.md](reports/CONSISTENCY_DATA_AUDIT.md): probability semantics and consistency data
- Design: [experiment plan](design/experiment-plan.md), [general training direction](design/general-training-direction.md), [training contract v0.2](design/training-contract-v0.2.md), [judgment workflow](design/judgment-workflow.md)

## 20 September: base training and the first contrasts

- [DATA_PREPARATION.md](reports/DATA_PREPARATION.md), [TRAINING_READINESS.md](reports/TRAINING_READINESS.md): the public-data corpus and the v0.2 run
- [JUDGMENT_LATENCY.md](reports/JUDGMENT_LATENCY.md): trained-model latency
- [PUBLISHED_CASES.md](reports/PUBLISHED_CASES.md): replay of TypeSafe's published example cases
- [SEMANTIC_CONTRASTS.md](reports/SEMANTIC_CONTRASTS.md) with [examples](reports/semantic-contrasts-v1/examples.md), [CONTRAST_DIAGNOSIS.md](reports/CONTRAST_DIAGNOSIS.md), [CONTRAST_CALIBRATION.md](reports/CONTRAST_CALIBRATION.md)
- [JEV_LIVE_SMOKE.md](reports/JEV_LIVE_SMOKE.md) and the [comparison guide](design/jev-comparison.md): first live Jev calls
- [CONTRAST_PILOT.md](reports/CONTRAST_PILOT.md): 300 contrast scenarios
- [REFERENCE_STRATEGIES.md](reports/REFERENCE_STRATEGIES.md): the eight community reproductions
- [REVISED_DATASET.md](reports/REVISED_DATASET.md), [REVISED_INITIALIZATION.md](reports/REVISED_INITIALIZATION.md): fresh versus continued initialization
- [CONTRAST_GENERATION_RESEARCH.md](reports/CONTRAST_GENERATION_RESEARCH.md), [GENERATION_PILOT.md](reports/GENERATION_PILOT.md), [TOOL_EFFORT_PROBE.md](reports/TOOL_EFFORT_PROBE.md): generating contrasts
- Design: [contrast taxonomy](design/contrast-taxonomy.md)

## 21 September: language, scale and calibration

- [Paired language](reports/paired-language-v1/RESULTS.md), with [original scores](reports/paired-language-v1/ORIGINAL_RESULTS.md), [data](reports/paired-language-v1/DATA.md), [examples](reports/paired-language-v1/EXAMPLES.md) and [Jev error notes](reports/paired-language-v1/jev-error-notes.md); design: [contract](design/paired-language-contract-v1.md), [clarification](design/paired-language-clarification-v1.md)
- [CALIBRATED_TRAINING_RESEARCH.md](reports/CALIBRATED_TRAINING_RESEARCH.md): literature on calibrated training
- Contrast scaling: [status](reports/contrast-scaling-v1/STATUS.md), [GPU validation](reports/contrast-scaling-v1/mid-gpu-validation/RESULTS.md), [pause](reports/contrast-scaling-v1/PAUSE.md); design: [contract](design/contrast-scaling-contract-v1.md), [generator](design/contrast-scaling-generator-v1.md)
- Post-pause audit: [results](reports/post-pause-audit-v1/RESULTS.md), [calibration](reports/post-pause-audit-v1/calibration/RESULTS.md), [methods review](reports/post-pause-audit-v1/methods-review.md)

## 22 September: rubrics, losses, teachers and architecture

- [Contrast factorial](reports/contrast-factorial-v1/RESULTS.md); design: [contract](design/contrast-factorial-contract-v1.md), [generator](design/contrast-factorial-generator-v1.md)
- [Calibrated screen](reports/calibrated-screen-v1/RESULTS.md); design: [roadmap](design/calibrated-training-roadmap-v1.md)
- [Architecture diagnostics](reports/architecture-diagnostics-v1/RESULTS.md); design: [options](design/architecture-options-v1.md)
- [Consistency and native reasoning](reports/consistency-native-v1/RESULTS.md), [root cause](reports/consistency-native-v1/ROOT_CAUSE.md); design: [consistent inference](design/consistent-inference.md)

## 23 September: running

- [Intermediate supervision proposal](design/intermediate-supervision-pilot-proposal.md)
