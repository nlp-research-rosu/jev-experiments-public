# First local experiment — 19 September 2026

Follow-up: [answer encoding, fresh holdout and completed precision checks](INVESTIGATION.md).
The measurements below remain the record of the first experiment.

The shared-context constrained-decision approach works locally without additional
training. On our small labeled test, it is **3.67× faster** than ordinary JSON
generation, but its answers are less accurate. This reproduces the useful mechanism
behind Harsha's demonstration; it does not establish how TypeSafe Jev works.

## Labeled diagnostic results

Qwen3.5-2B, RTX 4080 (16 GB), 32 synthetic text cases, five fields each.
Accuracy uses one evaluation per case; latency uses three repetitions per case.

| Measure | Parallel constrained decisions | Ordinary JSON generation |
|---|---:|---:|
| Median time per case | 206.3 ms | 758.5 ms |
| 95th percentile time | 230.1 ms | 845.4 ms |
| Correct fields | 126/160 (78.8%) | 138/160 (86.2%) |
| Completely correct cases | 10/32 (31.2%) | 17/32 (53.1%) |
| Schema-valid outputs | 32/32 | 32/32 |
| Answer changes across repeats | 0 | 0 |

The median paired speed ratio is 3.67×. Schema validity measures keys, types and
allowed values; it does not mean an answer is correct. The parallel implementation
constructs the final object from permitted values, guaranteeing that narrow form
of validity. JSON generation happened to satisfy the schema throughout this test.

The parallel method made 34 field errors, compared with 22 for JSON. Nineteen of
the parallel errors were normal-priority cases classified as urgent. It also made
seven department-routing errors, five missed phone requests, two outage errors,
and one refund error. The full error list and raw scores are retained in the
[diagnostic report](diagnostic-qwen35-2b.md) and raw JSON.

The five diagnostic questions have different suffix lengths, so they run as five
separate branches after one shared prefill: six model calls per case. JSON needed
35–48 calls. Thus this measured gain primarily comes from choosing answers directly
and avoiding token-by-token generation; it does not isolate a gain from batching.
The larger upstream schemas also exercise branches batched at equal lengths.

## Harsha's original scenarios

Four unmodified upstream examples, three timed repetitions each. Times below
are per-scenario medians; ratios are medians of paired measurements.

| Scenario | Fields | Parallel | JSON generation | Speed ratio |
|---|---:|---:|---:|---:|
| Code security | 28 | 757 ms | 6,416 ms | 8.47× |
| Fraud detection | 28 | 762 ms | 6,102 ms | 8.11× |
| Support triage | 28 | 715 ms | 5,810 ms | 8.06× |
| High cardinality, up to 255 choices | 4 | 1,137 ms | 1,886 ms | 1.65× |

There are no gold answers, so these results establish latency and format behavior,
not accuracy. Parallel outputs matched the schema for all four examples. JSON
matched three: in fraud detection it invented `UNUSUAL_VELOCITY` as a
`secondary_anomaly`, which is not an allowed value. This recurred in all three
repetitions. Neither suite hit the JSON output-token limit.

The median paired ratio across all upstream measurements was 8.08×. Reporting
individual scenarios matters: the long 255-choice case gains much less than the
28-field cases. Maximum PyTorch allocation was 4.67 GiB in this run, excluding
other processes and some non-PyTorch GPU allocations. See the
[upstream report](upstream-qwen35-2b.md) and raw results.

## What we implemented

The implementation describes the permitted answers, maps them to verified
single-token codes, processes common context once, then scores each question's
allowed codes. It copies attention caches and recurrent states correctly for
Qwen3.5's hybrid architecture and constructs typed output values in Python.
No weights were updated, and there is no training pipeline in this experiment.

This is an adaptation of [Harsha's published approach](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD),
using [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), explicit option codes,
and recurrent-cache support. Harsha's published benchmark supplies latency and
format examples, without gold labels. The labeled diagnostic set is our addition.

## Correctness checks and numerical precision

All 18 implementation tests (plus 16 subtests) pass, including actual tiny Qwen2
and hybrid Qwen3.5 models. They compare cache reuse against full independent
forward passes, check recurrent-state isolation, and exercise different branch
lengths and batch sizes. Independent review found no critical or important issues.

The three diagnostic GPU reference checks chose identical values, with maximum
absolute probability differences of 0.0089–0.0134. Upstream BF16 differences
reached 0.02516, initially stopping the benchmark at its default 0.02 threshold.
Separate FP32 checks on the code-security and fraud cases agreed within 0.000005
and chose identical answers. This supports a reduced-precision explanation for
the observed discrepancy on those cases.

The upstream timed run explicitly uses a 0.03 tolerance, recorded in its metadata.
All four BF16 reference checks fit that bound. However, two near-tie fraud answers
change between cached and independent execution (`secondary_anomaly` and
`auto_reverse_transfer`). Small probability differences can matter for a decision;
the BF16 paths are not claimed to be bitwise or universally choice-equivalent.

The 255-choice scenario needs 10,031 input tokens, so its run raises the limit to
16,384 without truncating the original data. Its full FP32 reference exhausted
available GPU memory alongside existing desktop workloads, even with a
memory-efficient attention kernel. Its BF16 reference completed, with a maximum
difference of 0.02030 and no choice disagreements. No full FP32 result is claimed
for that case or for support triage.

Retained evidence: BF16 reference outputs,
the two completed FP32 checks, and
the initial failed threshold check.
The precision probe can be rerun from the project root:

```sh
.venv/bin/python reports/check_upstream_precision.py --dtype float32 --cases code_security fintech_fraud
.venv/bin/python reports/check_upstream_precision.py --dtype bfloat16
```

## Interpretation and next experiment

We have evidence for a speed advantage in this implementation, with an accuracy
cost on this task. We do **not** yet have evidence that additional training is
necessary. The comparison changes prompting, answer representation and dependence
between fields, as well as execution strategy.

Before training, isolate the cause: compare answer codes with direct answer-token
scoring where possible, test sensitivity to option order, and compare question
wording and model size. Those would be new experiments. Once we use these errors
to guide changes, reserve a fresh labeled holdout before claiming an improvement.
If failures persist on the intended workload, supervised fine-tuning becomes a
reasonable next step; reinforcement learning is not required merely to reproduce
this mechanism.

## Measurement limits

- The diagnostic set is small, synthetic and manually labeled. It is not a
  production benchmark, a general intelligence test or a comparison with Jev.
- Candidate probabilities are normalized scores, not calibrated confidence.
  Their diagnostic Brier score is 0.2769; the raw report includes confidence bins.
- Both methods use the same BF16 backbone. Candidate output-head projection uses
  FP32. JSON generation is greedy, uses original answer values, and has no grammar.
- The runtime uses reference PyTorch recurrent kernels; optional optimized
  `causal_conv1d` and `flash-linear-attention` kernels are absent. These timings
  do not represent an optimized serving-engine comparison.
- Timings include prompt preparation, GPU execution and output construction,
  with synchronization. Downloads, loading and warmup are excluded. Existing
  desktop GPU workloads remained active, so absolute times can vary.
- Raw reports record the exact model revision, source/dataset hashes, package
  versions, forward-call counts, input lengths and GPU memory measurements.
