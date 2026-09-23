# Few-shot prompts and option stability

The same four examples helped ordinary JSON generation but harmed this parallel, single-code prompt format. On 40 fresh synthetic cases, JSON field accuracy rose from 79.5% to 88.0%; parallel accuracy in the original layout fell from 78.5% to 60.0%. Its sensitivity to option order also increased. No model weights were updated, and no prompt change was promoted to the default engine.

## Controlled design

- Model: Qwen3.5-2B at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, BF16 backbone with FP32 candidate projection on the RTX 4080.
- Froze eight worked examples from previously inspected cases. Conditions use no examples, the first four, or all eight. All eight demonstration cases are excluded from every development condition.
- Development: 48 previously inspected cases / 240 unique labels. Fresh holdout: 40 new cases / 200 unique labels, with ten cases per department, twenty urgent requests, twenty phone requests, and four cross-department refund requests.
- Eight layouts cross four cyclic letter-to-value assignments with forward/reversed option display. Display reversal preserves each letter’s meaning. This is a structured sample, not exhaustive permutations.
- Demonstration answers are remapped consistently for each layout. All code tokens are verified at the actual prompt boundary. The no-example, original-layout prompt is exactly the existing implementation; all 48 development control predictions reproduced previous results.
- The selected example count maximizes development mean accuracy over layouts, then stability, then worst-layout accuracy, with fewer examples breaking remaining ties. Four beat eight, but both lost to the zero-example control. Selection was frozen before holdout evaluation.
- Holdout evaluates both parallel conditions over all layouts once, plus two additional timing repeats for the original layout. JSON gets the same examples with original typed answers and three repetitions, evaluated only in the original schema order.

Frozen protocol · Frozen selection · Actual remapped prompt example

The parallel demonstrations show code objects for all fields in a shared prefix; the query asks for a single field’s code. JSON demonstrations show complete typed objects matching its output format. This output-format difference is a material limitation: the experiment tests this concrete shared-example design, not every way of doing few-shot prompting.

## Development: selecting the example count

| Examples | Mean field accuracy across layouts | Worst layout | Fields invariant across all layouts |
|---|---:|---:|---:|
| 0 | 79.1% | 71.2% | 55.0% |
| 4 | 66.5% | 43.3% | 11.7% |
| 8 | 58.9% | 39.6% | 10.4% |

Adding more examples did not monotonically help. Four examples were the less-bad few-shot condition, not an improvement to promote. The examples were selected for task coverage before evaluation and were not edited after seeing results.

Raw development results

## Fresh holdout: accuracy and latency in the original layout

| Method | Correct fields | Entire case correct | Median time | p95 time |
|---|---:|---:|---:|---:|
| Parallel, no examples | 157/200 (78.5%) | 13/40 (32.5%) | 126.6 ms | 131.9 ms |
| Parallel, four examples | 120/200 (60.0%) | 2/40 (5.0%) | 147.2 ms | 154.9 ms |
| JSON, no examples | 159/200 (79.5%) | 14/40 (35.0%) | 431.5 ms | 449.8 ms |
| JSON, four examples | 176/200 (88.0%) | 21/40 (52.5%) | 454.4 ms | 610.8 ms |

Every output matched its schema. Outputs were identical across timing repeats for the same input. No JSON run reached its output-token limit. Timing excludes loading and warmup, includes prompt construction/tokenization, and synchronizes GPU work. These timings use the same unoptimized recurrent-kernel runtime as earlier runs.

The median paired speed advantage over the corresponding JSON condition was 3.40× with no examples and 3.08× with four examples. The latter speed advantage comes with substantially worse answers; it is not a quality-equivalent speedup. Peak PyTorch allocation was 3.63 GiB.

JSON with examples corrected 17 additional field answers. Its phone-request errors fell from 13 to 3 and refund errors from 7 to 2. Priority classification remained weak: 13 of its remaining 24 errors concern urgency. Only 21 of 40 complete JSON answers were entirely correct.

## Fresh holdout: option stability

| Parallel condition | Mean accuracy across layouts | Worst layout | Invariant field decisions | Pairwise agreement |
|---|---:|---:|---:|---:|
| 0 examples | 78.2% | 75.5% | 108/200 (54.0%) | 78.4% |
| 4 examples | 66.6% | 51.5% | 27/200 (13.5%) | 59.9% |

An invariant decision gives the same typed answer for that case/field in all eight layouts; this does not imply it is correct. Mean accuracy averages layout scores, not independent extra test cases. There are still only 40 unique cases and 200 unique labels. Permutation robustness was measured for parallel scoring; no such claim is made for JSON.

| Isolated order effect on holdout | No examples | Four examples |
|---|---:|---:|
| Answer changes when display alone reverses | 25.6% | 53.1% |
| Answer changes between code assignments, fixed display | 17.2% | 22.1% |
| Selects last displayed option on binary fields | 59.8% | 77.9% |

The last-option tendency is a concrete failure pattern. With a binary question whose answer is invariant to display reversal, the answer appears last in exactly half of the paired displays. The much higher rate with examples, together with isolated display flips, indicates strong presentation sensitivity. It does not establish a specific internal neural mechanism.

Across layouts, false-urgent rates were 45.0% without examples and 51.2% with four. Missed-urgent rates were 17.5% and 13.8%. Examples changed that tradeoff rather than resolving it.

Raw holdout answers, probabilities and errors · Separated order-effect counts

## What this changes about the training decision

These results give us a better quality reference: ordinary JSON with four examples reached 88.0% field accuracy on this holdout. The model can benefit from these examples without weight updates. At the same time, the current way of showing full code objects before asking for a single code regresses badly.

The next focused prompt experiment should make each demonstration match the target response format: one field question followed by its one-code answer, with mappings kept consistent. This is a hypothesis to test on development data and a newly reserved holdout, not a claimed fix. It may cost more prefill work because less context can be shared between fields. If such format-aligned prompts remain inaccurate or unstable on a larger realistic workload, task-specific supervised fine-tuning becomes a more justified experiment.

The samples are hand-authored and synthetic. They support diagnosing this implementation, not production accuracy, universal few-shot behavior, or parity with Jev. Normalized token probabilities remain uncalibrated. The v2 holdout is now inspected and must be treated as development data in future tuning.

## Reproduce

From the project root, use new output filenames:

```sh
.venv/bin/python -m experiments.fewshot_stability --output reports/new-fewshot-development.json
.venv/bin/python -m experiments.fewshot_stability --data data/holdout-v2.json --shots 0 4 --identity-repeats 3 --include-json --output reports/new-fewshot-holdout.json
.venv/bin/python -m pytest -q
```

All 29 tests and 16 subtests passed. The new tests verify exact control prompts, independent display/mapping changes, remapped demonstrations, typed JSON examples, separation from evaluation cases, and stability counts unaffected by timing repeats. Reports preserve model/data/source hashes and complete predictions. The default engine and original experiment files remain unchanged.
