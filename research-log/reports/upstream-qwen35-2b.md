# Parallel decision experiment

Model: `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc`. GPU: NVIDIA GeForce RTX 4080.

Suite: **upstream**, 4 unique cases; 3 timing repetitions per case.
Quality metrics use the first repetition only. Timing excludes download/loading and includes prompt preparation, GPU execution, and answer assembly.

| Method | Field accuracy | Exact cases | Valid schema | Median ms | p95 ms | Brier |
|---|---:|---:|---:|---:|---:|---:|
| parallel | unlabeled | unlabeled | 100.0% | 757.3 | 1145.9 | — |
| json | unlabeled | unlabeled | 75.0% | 5889.7 | 6466.9 | — |

Median paired JSON/parallel latency ratio: **8.08×**.

## Cached versus independent reference

Configured maximum absolute probability difference: 0.030.

- code_security: max probability difference 0.025156; choice disagreements 0.
- fintech_fraud: max probability difference 0.019980; choice disagreements 2.
- high_cardinality_255: max probability difference 0.020299; choice disagreements 0.
- support_triage: max probability difference 0.019715; choice disagreements 0.

## Errors

### parallel

No gold labels; accuracy is not measured.

### json

No gold labels; accuracy is not measured.

## Interpretation limits

The diagnostic set is small, synthetic, and hand-authored. Upstream scenarios have no gold labels. Neither establishes production quality or parity with Jev.
Candidate probabilities are normalized scores, not calibrated confidence. The Brier score is the sum over all classes, averaged over labeled fields; ECE uses ten equal-width confidence bins.
The parallel path uses verified single-token option codes and independently answers each field. JSON generation uses original answer values and can condition later fields on earlier ones, so these are different inference objectives and prompts.
The JSON baseline is greedy ordinary generation, without grammar constraints. This is not a comparison against an optimized structured-output serving engine.
Reported speed is for this implementation, model, hardware, input sizes, and current GPU load. All forward calls, including prefill, are counted in the raw report.
