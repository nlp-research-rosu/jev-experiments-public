# Trained judgment latency on RTX 4080

Completed benchmark of the first trained Qwen3.5-2B judgment checkpoint. Small requests reach the initial latency target with merged adapters and larger scoring batches; latency still grows substantially with question count. This is a local warm-inference measurement, not a Jev API or autoregressive JSON-generation comparison.

## Main results

Every number includes JSON parsing, request compilation, prompt rendering/tokenization, synchronized model scoring, answer assembly and JSON serialization. Model loading, network and file I/O are excluded. The model stays loaded, but each request recomputes its shared context; there is no cross-request prompt-cache reuse.

Current means unmerged adapters with scoring batches of four and the FLA backend used for this experiment. The CLI needs `--backend fla` to select that backend. Merged means an in-memory BF16 merge with batches of up to 32. No checkpoint or production default was changed.

| Case | State tokens | Questions | Scoring rows | Current median | Merged median | Merged p95 |
|---|---:|---:|---:|---:|---:|---:|
| Nested support request | 77 | 4 | 8 | 91.7 ms | 60.2 ms | 78.1 ms |
| Short state, 5 questions | 179 | 5 | 13 | 145.4 ms | 80.2 ms | 93.3 ms |
| Short state, 10 questions | 179 | 10 | 28 | 214.9 ms | 142.5 ms | 176.5 ms |
| Short state, 25 questions | 179 | 25 | 68 | 552.9 ms | 323.0 ms | 356.3 ms |
| Short state, 50 questions | 179 | 50 | 133 | 974.8 ms | 601.1 ms | 668.1 ms |
| Medium state, 10 questions | 563 | 10 | 28 | 281.6 ms | 154.3 ms | 157.8 ms |
| Long state, 10 questions | 2099 | 10 | 28 | 487.3 ms | 282.7 ms | 302.4 ms |
| Longer state, 10 questions | 4147 | 10 | 28 | 729.5 ms | 477.6 ms | 529.1 ms |
| One 32-option Choice | 50 | 1 | 32 | 254.1 ms | 93.2 ms | 95.7 ms |

The short scaling cases contain a 179-token serialized state (including the synthetic history); the complete scoring inputs are about 300 tokens each. Question types cycle through four-option Choice, binary Noul and three-level Score. One Choice requires one scoring row per candidate; a Noul requires one row. Scaling cases repeat an explicit support policy with distinct check instructions. They are performance fixtures, not new gold-labeled evaluations. Longer states add archived-note text; the 2,099- and 4,147-token states exceed the retained training length range.

The 5-question case meets the proposed 100 ms median target. The 10-question case meets the proposed 200 ms p95 threshold but misses 100 ms median. The 25- and 50-question cases miss the proposed 250 ms median target, and 50 questions also exceed 500 ms p95. There is no evidence here of question-count-independent latency.

## What batching and merging contribute

| Short-state workload | Adapters, batch 4 | Adapters, batch 32 | Merged, batch 4 | Merged, batch 32 |
|---|---:|---:|---:|---:|
| Short state, 5 questions | 145.4 ms | 102.1 ms | 115.4 ms | 80.2 ms |
| Short state, 10 questions | 214.9 ms | 163.7 ms | 199.5 ms | 142.5 ms |
| Short state, 25 questions | 552.9 ms | 446.0 ms | 451.8 ms | 323.0 ms |
| Short state, 50 questions | 974.8 ms | 719.7 ms | 876.2 ms | 601.1 ms |

Combining merging and batching gives about 1.5–1.8× lower median latency on these mixed short-state workloads. The 10-question request has 28 rows and uses two forwards at batch 32; 50 questions have 133 rows and use six forwards. Increasing the batch to cover every row at once was explored in the separate two-repeat pilot and was not consistently faster. Those pilot results are exploratory and excluded from the main table.

## Cached versus independent scoring

A separate matched run rotates cached and independent scoring of identical requests. Independent scoring repeats the whole prompt per candidate and uses batches of four. This is the same trained judgment model, not a text-generating LLM baseline.

| Representation / case | Independent median | Cached batch-32 median | Ratio |
|---|---:|---:|---:|
| adapters, Nested support request | 157.6 ms | 82.3 ms | 1.91× |
| adapters, Short state, 10 questions | 871.9 ms | 206.4 ms | 4.22× |
| merged, Nested support request | 116.9 ms | 61.9 ms | 1.89× |
| merged, Short state, 10 questions | 504.6 ms | 122.4 ms | 4.12× |

Absolute times differ between runs on the active desktop. Use within-run comparisons; do not combine the fastest samples from different runs.

## Accuracy and probability regression checks

All 1,500 previously evaluated original test cases were checked; these are execution regression checks, not an additional untouched test set. The cached-all quality path is identical to cached-32 on this test corpus because each bundle has at most eight scoring rows.

| Execution | Original-case accuracy | Changed predictions versus original independent | Largest probability shift |
|---|---:|---:|---:|
| Original adapters, independent | 87.27% (1,309/1,500) | — | — |
| Adapters, cached batch 4 | 87.20% (1308/1,500) | 29 | 11.12 percentage points |
| Merged, independent | 87.27% (1309/1,500) | 42 | 8.43 percentage points |
| Merged, cached | 87.00% (1305/1,500) | 35 | 11.00 percentage points |

The fastest tested configuration has four fewer correct original cases than the original independent reference. Its 35 changed predictions comprise 34 wine-band classifications and one BoolQ answer. A Score classification here means its most probable level; the API normally returns the expected level as a continuous number. Wine expected-score changes averaged 0.0064 levels and reached 0.0252 levels on the 0–2 scale. Small changes can flip a nearly tied most-probable level.

Probability changes also matter without a class flip. Seven original cases in the merged cached comparison differed by more than three percentage points; the maximum was about eleven points. Thus similar aggregate accuracy is not numerical equivalence. Probabilities remain uncalibrated.

All main timing fixtures retained their categorical/threshold decisions and were exactly repeatable within each fixed configuration across 30 samples. Their maximum cross-configuration probability shift was 0.02876. That easy synthetic check alone would have missed the larger differences on the real test cases.

### Targeted precision diagnosis

The two largest unmerged cached/full probability discrepancies and one changed wine case were rerun with BF16 FLA, BF16 reference arithmetic, and FP32 reference arithmetic. Reference comparisons used the same math attention backend with TF32 disabled. This separates a precision change from a recurrent-backend change.

| Case | BF16 FLA difference | BF16 reference difference | FP32 reference difference |
|---|---:|---:|---:|
| paws/train/40154 | 0.105827 | 0.075316 | 0.00000727 |
| wine_quality/unsplit/white/3730 | 0.031451 | 0.010771 | 0.00000408 |
| boolq/train/9226 | 0.111156 | 0.016509 | 0.00000599 |

FP32 reduced the cached/full differences below 0.00001 in these probes. This points to arithmetic precision and kernel execution differences, rather than a cache mapping defect in those cases. It is a targeted diagnosis, not a full FP32 deployment evaluation or a universal guarantee.

## Profiling and next optimization targets

For three merged 10-question requests, the profiler attributes roughly 188 ms of self CUDA time to matrix multiplication (about 63% of its reported 300 ms total), versus about 15 ms to the reference depthwise convolution. Operator and kernel entries overlap in the profile; their rows must not be added together. Profiling overhead means these durations are not substitute latency measurements.

The evidence favors optimizing the bulk linear algebra and reducing repeated scoring work before expecting a large gain from replacing convolution alone. A larger GPU may help this workload, but no H100 speedup is measured here. Tokenization/rendering also grows with context: around 10 ms for the short 10-question case and 100 ms for the 4,147-token state, because the current renderer tokenizes every unit.

## Reproduction and limits

- RTX 4080 16 GB; original desktop workloads left running. Single request at a time, not a concurrent-service throughput benchmark.
- Three warmups and 30 measured samples per case/configuration; mode order rotates. Main sweep: 1,080 timed calls. Separate reference run: 240 timed calls. Representation order is adapters first, then merged.
- Median is the sample median; p95/p99 use the nearest-rank rule. With 30 samples, p99 is the observed maximum, and tail estimates are limited—not a production SLA.
- BF16 backbone, FP32 readouts/adapters before merging, no quantization. CUDA matrix-multiply TF32 disabled. FLA 0.5.2 with SDPA; causal convolution uses the PyTorch fallback.
- GPU-event durations include stream idle gaps caused by host launches. They are recorded as CUDA-stream elapsed time, not pure kernel computation.
- Request JSON, raw samples, predictions, package versions, source/data hashes and GPU status are retained. Loading and initial calls are recorded separately.
- All 183 offline tests and 45 subtests passed; benchmark lint passed. A read-only review found no material timing issue. Saved checkpoint identity was verified unchanged after the experiments.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.runtime-training:. \
  .venv/bin/python -m experiments.judgment_latency \
  --output reports/my-latency-run --repeats 30 --warmups 3 \
  --modes cached-4 cached-32 \
  --case-ids nested-realistic context-128-questions-5 \
    context-128-questions-10 context-128-questions-25 context-128-questions-50 \
    context-512-questions-10 context-2048-questions-10 \
    context-4096-questions-10 choices-32
```

Artifacts: main report, requests, matched independent comparison, precision diagnosis, profiler table, profiler trace, exploratory pilot.
