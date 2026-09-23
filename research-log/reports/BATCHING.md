# Shared prefill and padded field scoring

The inference pipeline now follows the shared-prefix / batched-suffix execution
pattern in Harsha's published implementation. Five unequal-length field prompts
fit into one batch after prefill: **two backbone calls instead of six**.
The Qwen3.5-2B checkpoint, prompt renderers, example requests and answer codes
are unchanged. No training or dependency changes were made.

## Execution change

Previously we grouped suffixes by exact length. On this dataset all five lengths
differed, so the implementation performed five separate branch calls. The new
path right-pads each batch, masks its padding, and gathers each row's last real
hidden state before projecting the allowed answer tokens. It copies Qwen3.5's
attention, convolution and recurrent state from the original prefix for every
batch. It never continues generation from a padding-updated branch cache.

The batch limit remains eight. More than eight fields require multiple suffix
batches; a missing shared prefix requires no separate prefill. An added guard
isolates the longest suffix while a batch would process more than twice its
actual suffix-token count. Normal five-field prompts still share one batch.
Independent field
decisions still do not enforce consistency constraints between fields.

Harsha's [MLX implementation](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/2af86848be75847ccb3553b0941cc51d6ef7e4e9/core/engine_mlx.py)
also gathers `suffix_lengths[i] - 1` after its batch. Our adaptation retains
tokenizer-verified answer codes and handles the newer model's recurrent cache.

## Verification and measurement protocol

- Tiny actual Qwen2 and hybrid Qwen3.5 models compare padded scores against full
  independent FP32 forward passes. Tests cover selected/full vocabulary scores,
  reversed prompt order, bounded batches, a one-token suffix beside a 65-token
  suffix, no prefix, identical prompts and cache isolation. The call-count test
  was observed failing on the old implementation before the change.
- The old engine is retained in `sources/engine_length_grouped_v1.py`. The
  comparison script runs old and new scoring in one process with identical
  prompt hashes, model weights, precision and examples. Actual backbone hooks
  verify reported call counts. Job order rotates and timings synchronize CUDA.
- All 32 previously inspected v4 cases are reused for execution and speed checks,
  with three timing repetitions and the identity option layout. This is not a new
  accuracy holdout or a new option-layout stability study. Quality uses only the
  first repetition, avoiding repeated-label inflation.
- End-to-end time includes tokenization, answer-code validation, GPU scoring and
  answer assembly. A second timing isolates scoring. Warmups and model load are
  excluded. Ordinary greedy JSON with the same four examples is a separately
  prompted control, not an optimized structured-output server.
- Upstream scenarios retain their published schemas and have no gold labels.
  Their measurements establish speed and validity, not semantic correctness.
- BF16 arithmetic can change with batch shape. Any changed decisions require a
  targeted FP32 check before attributing them to a padding/cache defect.

## Five-field results

RTX 4080, pinned Qwen3.5-2B, BF16 backbone, selected output rows projected in FP32.
These are paired measurements from this run, not comparisons against timings
collected before a restart or under a different desktop load.

| Prompt format | Old median | Batched median | Calls, old → batched | Correct fields, old → batched |
|---|---:|---:|---:|---:|
| No examples | 141.6 ms | 94.8 ms | 6 → 2 | 128/160 → 129/160 |
| Four shorter examples | 423.6 ms | 452.7 ms | 6 → 2 | 141/160 → 141/160 |
| Four long examples | 859.9 ms | 966.6 ms | 6 → 2 | 142/160 → 142/160 |
| Ordinary JSON, four examples | 499.0 ms | — | 35–49 | 143/160 |

Median paired speed ratios, old/batched: **1.498×**, **0.937×**, **0.890×**.
The short no-example path cuts latency by about one third. Batching the longer
prompts is slower in this environment. Four shorter examples reach only about
1.10× the JSON control's speed by the ratio of medians; this does not meet the
desired low-latency few-shot result.

All outputs were schema-valid and repeated decisions were stable. Both example
formats preserved every decision. The no-example run changed one near-tie:
`v4_a02 / phone_call_requested`. The old scores favored false at 0.50224 versus
0.49776; the batched scores favored true at 0.50328 versus 0.49672. These are
normalized candidate scores, not calibrated probabilities. The one-label accuracy
increase is not evidence of better reasoning or a prompt improvement.

The four targeted FP32 case/condition checks include that flip and each format's
largest probability discrepancy. Old, batched and independent execution all made
the same decisions. Largest old/batched probability difference: **0.000004023**;
largest batched/independent difference: **0.000002235**. This supports numerical
sensitivity as the source of the BF16 flip, rather than incorrect padding or cache
reuse. Full-precision peak allocation was **10.00 GiB**.

The full 32-case measurement preceded the extreme-padding guard. Its inactive
five-field scheduling path is unchanged; the final guarded implementation is also
checked on two cases and all upstream scenarios in a separate retained run.

## Larger schemas and the padding guard

The original batching comparison gave these per-scenario medians:

| Upstream scenario | Old | Unrestricted padding | Calls, old → batched |
|---|---:|---:|---:|
| Code security, 28 fields | 499.3 ms | 373.8 ms | 18 → 5 |
| Fraud, 28 fields | 502.6 ms | 357.3 ms | 19 → 5 |
| Support, 28 fields | 489.3 ms | 364.0 ms | 17 → 5 |
| 255-choice scenario, 4 fields | 1083.0 ms | 2557.1 ms | 5 → 2 |

All 88 upstream field decisions agreed with the old implementation. There are no
gold labels, so this agreement does not establish accuracy. BF16 peak allocation
across the full initial comparison was **6.59 GiB** (reserved: 8.64 GiB).

The last scenario exposed an avoidable regression: its suffix lengths are 4846,
43, 46 and 59 tokens. Padding every query to 4846 more than triples suffix work.
The final guard evaluates the long query separately and the three short queries
together, requiring three calls including prefill. A real-model regression test
first failed on the unrestricted implementation, then passed after this fix.

The final guarded rerun measured **1058.2 ms** versus **1082.9 ms** for the old
255-choice path, with shapes `[1, 5185]`, `[1, 4846]`, `[3, 59]`: one prefix,
one long query, one three-query batch. This removes the 2.56-second padding
regression without claiming a substantial speedup on that long prompt. The other
three upstream cases measured 374.0, 349.6 and 367.6 ms, with five calls each.
All 88 upstream decisions still agree; peak BF16 allocation is **4.95 GiB**.

The two-case, three-repeat final control also confirms the earlier five-field
pattern: 143.8 → 96.4 ms without examples, 426.2 → 456.5 ms with shorter examples,
863.9 → 972.1 ms with long examples. All decisions agree on those two cases.
Across the full earlier v4 run, the largest padded/real suffix-work factors were
1.435, 1.192 and 1.087 respectively, so none would activate the new 2× guard.
The completed offline suite passes **70 tests and 30 subtests**; lint and format
checks pass. A fresh review of the initial batching change found no actionable
issues; the later guard has its own failing-then-passing real-model regression
test and GPU rerun.

## What this establishes

The missing batched-scoring technique is implemented. It helps short field
queries; it does not eliminate the cost of long field-specific demonstrations.
Neither the model weights nor the amount of useful suffix-token work shrinks
because five queries are placed in a batch. Harsha's terse field suffixes,
different model, quantization and MLX runtime remain important differences.

A one-case operator profile identifies the recurrent
DeltaNet layers as the largest of the three instrumented module groups, followed
by dense feed-forward layers, then full attention. Dense matrix multiplication
remains substantial in both execution paths. The profile adds instrumentation
overhead and includes overlapping CPU/CUDA range records; do not sum those records
or use their durations as end-to-end latency. Its source
is retained. This is a direction for investigation, not proof that optimized
recurrent kernels alone will meet the latency target.

The next controlled experiment should test the GPU runtime, including the missing
optimized recurrent kernels, before deciding whether to change prompts again or
train. No optimized kernels were installed during this comparison.

Artifacts: full paired run,
targeted FP32 checks,
final guard check, and the archived
original scorer and
unrestricted padded scorer.

Reproduce from the project directory:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m experiments.padded_batching \
  --include-json --include-upstream --output reports/my-batching-comparison.json
```

The model must already be cached for this offline command. Reports refuse to
overwrite existing outputs. The optional optimized convolution and DeltaNet
kernels remain uninstalled, preserving the previous execution environment.
