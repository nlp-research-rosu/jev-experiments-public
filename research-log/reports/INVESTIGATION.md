# Investigation: answer encoding and precision

Follow-up: [few-shot prompts and a fresh permutation-stability holdout](FEWSHOT.md).

The speed benefit is reproducible, but answer quality is sensitive to how choices are presented. A change that improved the original examples did not improve a fresh holdout. No model training was performed and the default decision engine remains unchanged.

## Comparison with Harsha

[Harsha’s pinned README](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/2af86848be75847ccb3553b0941cc51d6ef7e4e9/README.md) reports 5.6–7.0× speedups on an M4 Max using 4-bit Qwen2.5-1.5B-Instruct and MLX. Our BF16 Qwen3.5-2B / RTX 4080 implementation achieved 8.06–8.47× on the three 28-field examples, 1.65× on the four-field 255-choice example, and 3.67× on the original labeled diagnostic set.

This reproduces the mechanism and general speed benefit, not his exact benchmark. The model, precision, hardware, runtime, prompting, and some published table field counts differ. His released examples have no gold answers; schema validity does not establish semantic accuracy or calibrated confidence. We cannot claim comparable accuracy to Harsha or equivalence to Jev.

## Protocol

- The original 32 cases / 160 labels became the development set for this investigation.
- Before evaluating variants, we froze 24 new synthetic cases / 120 labels in `data/holdout-v1.json`: six cases per department, 12 urgent and 12 normal.
- Six variants changed option order, code vocabulary, or schema scope. The original wording, model weights and field definitions otherwise stayed fixed. The combined word/single-field variant can be compared to each corresponding one-factor variant.
- Token codes were verified to be unique single tokens, including at the actual prompt boundary. Reversed answers and probabilities are mapped back to canonical typed values.
- Selection used maximum development field accuracy, then whole-case accuracy. This selected reversed option order before opening holdout results.
- The holdout evaluated only the selected variant, the original prompt and ordinary JSON generation, with three timing repetitions. Quality counts each case once; no further variant was chosen or tuned using holdout results.
- The model revision remains `15852e8c16360a2fea060d615a32b45270f8a8fc`. Reports retain hashes, versions, raw outputs and errors.

Frozen protocol · Frozen selection

## Development results

| Variant | Field accuracy | Entire case correct | False urgent | Missed urgent |
|---|---:|---:|---:|---:|
| Original letters | 78.8% | 31.2% | 19 | 0 |
| Reversed option order | 87.5% | 50.0% | 3 | 3 |
| Numeric answer codes | 48.1% | 3.1% | 24 | 0 |
| Meaningful answer words | 75.6% | 18.8% | 22 | 0 |
| Only relevant field, letters | 85.6% | 43.8% | 0 | 5 |
| Only relevant field, words | 85.6% | 50.0% | 7 | 2 |

The control reproduced every original decision after the restart. Reversing option order changed 30/160 decisions despite unchanged meanings. Numeric codes changed 66/160 decisions and labeled all 32 cases urgent. Showing only the relevant schema reduced false urgency but missed five truly urgent cases. Meaningful answer words helped some fields and harmed others; they were not a general fix.

On six selected development examples (30 field decisions per variant), the unrestricted next-token winner was an allowed answer for all six variants. The mean total probability assigned to allowed answers was about 99% for letters and numbers. These probes use ordinary BF16 output logits rather than the FP32 selected-head path. They suggest that the observed errors are not simply the scorer forcing tokens the model was unwilling to emit. They do not establish calibration.

The experiment establishes sensitivity to presentation, not a complete explanation of the internal mechanism. Reversal changes both display order and letter-to-value assignment; it does not isolate position bias from letter bias.

Raw development results

## Fresh holdout results

| Method | Correct fields | Entire case correct | Valid schema | Median latency |
|---|---:|---:|---:|---:|
| Original letters | 99/120 (82.5%) | 10/24 (41.7%) | 100% | 126.2 ms |
| Reversed option order | 95/120 (79.2%) | 6/24 (25.0%) | 100% | 126.9 ms |
| Ordinary JSON | 98/120 (81.7%) | 11/24 (45.8%) | 100% | 436.4 ms |

The original parallel method remained **3.42× faster** than JSON by median paired latency. Its one-field accuracy advantage over JSON is too small on this synthetic sample to support a general accuracy claim. JSON produced one more completely correct case.

Reversed order, selected at 87.5% on development, fell to 79.2% on the holdout and lost four correct field answers compared with the original prompt. It removed false urgency on this holdout but missed eight genuinely urgent cases; the original prompt had seven false-urgent cases and no missed urgency. The original development set was 75% normal-priority, whereas this holdout is balanced. The changed class balance helps expose the tradeoff, though it is not the only difference between the sets.

All three methods were deterministic across the measured repeats and schema-valid. No JSON output hit the token limit. Absolute times improved after the restart; compare methods within this run, not old and new timings as if they measured a code optimization.

Raw holdout results and per-field errors

## Precision and memory after restart

The GPU initially had 13,253 MiB (12.94 GiB) free, versus roughly 8.6 GiB before the restart. Full FP32 storage still ran out of memory on the longest scenario, so the reference probe also tested lossless compact storage: it verifies that each source weight is exactly BF16-representable, stores it that way, and converts it to FP32 before computation. This is not approximate weight quantization or training.

All four scenarios now have completed FP32-compute cached/independent checks, obtained across separate runs:

| Scenario | Maximum probability difference | Different chosen values |
|---|---:|---:|
| code_security | 0.00000501 | 0 |
| fintech_fraud | 0.00000328 | 0 |
| high_cardinality_255 | 0.00000403 | 0 |
| support_triage | 0.00000322 | 0 |

The first three checks completed with compact storage, peaking at 10.98 GiB allocated. That process subsequently exhausted memory when starting support triage; support triage then passed in a fresh process with ordinary FP32 storage. We retain the partial-run evidence rather than claiming the entire compact-storage process succeeded. Comparisons with earlier ordinary-FP32 results on security and fraud also differed by less than 0.000006.

These results support numerical rounding as the source of the larger BF16 cached/independent discrepancies. They do not mean every BF16 near-tie decision is stable: two fraud choices had flipped earlier. This precision issue is distinct from the much larger prompt-format sensitivity.

Completed compact-storage checks · Support-triage FP32 check

## Decision

Keep the current default. Do not promote reversed order or numeric encoding, and do not interpret valid JSON or normalized scores as reliable decisions. The next quality gate should include option-permutation stability and separate false-urgent/missed-urgent rates, using a larger task-specific dataset with a fresh holdout. These two sets now both inform development and should not be reused as untouched final tests.

A sensible next controlled comparison is a few explicit input/answer demonstrations against the existing zero-shot prompt. If the required accuracy and option-order robustness remain inadequate, task-specific supervised fine-tuning is justified to investigate. This experiment neither demonstrates that training is necessary nor that it can be avoided for production quality. No reinforcement-learning claim follows from the speedup.

## Reproduce

From the project root:

```sh
.venv/bin/python -m experiments.answer_ablation --output reports/new-development.json --token-probes
.venv/bin/python -m experiments.answer_ablation --data data/holdout-v1.json --variants codes_original codes_reversed --include-json --repeats 3 --output reports/new-holdout.json
.venv/bin/python -m pytest -q
```

The runner refuses to overwrite an existing result. The experiments remain separate from the default engine. All 22 implementation/experiment tests and 16 subtests passed, including exact control-prompt equality, canonical probability mapping after reversal, single-field scope, and semantic-code type preservation.
