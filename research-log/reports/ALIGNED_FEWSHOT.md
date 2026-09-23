# Matching demonstrations to one-code answers

Four demonstrations presented as one-field question / one-code answer exchanges
improved constrained field accuracy from **79.4% to 90.0%** on a fresh 32-request
synthetic holdout. The previous full-object demonstration format scored 57.5%.
Decisions invariant to all eight option layouts increased from **51.9% to 88.8%**.
No model weights changed.

The improvement costs speed: this straightforward aligned format takes **729 ms**
per request versus **128 ms** without examples and **475 ms** for ordinary JSON
with examples. It repeats much more context separately for each field. This is
evidence that better demonstration formatting helps the constrained method, not
yet an accurate replacement that preserves its original speed advantage.

## Controlled experiment

- Same Qwen3.5-2B checkpoint, revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`, RTX 4080, BF16 backbone and FP32
  candidate projection. Same unoptimized reference recurrent kernels.
- Same first four frozen example requests as the preceding experiment, in the
  same order. The new format has four native user/assistant exchanges for each
  queried field. Each assistant message contains exactly one correct code.
- Each demonstration question uses the existing full-schema prompt. The final
  target question, system message, schema, choices and code scoring are unchanged.
  The native non-thinking template is used throughout. Its actual rendering is
  retained in the rendered example.
- One new format was fixed in advance. No candidate selection or prompt tuning
  followed the smoke or holdout scores. Eight previously inspected development
  cases served as an infrastructure check; both old controls exactly reproduced
  previous predictions. Aligned smoke accuracy was 95%, but that was not treated
  as the final result.
- The new holdout was written and hashed before model evaluation. It has 32 cases,
  160 labels, eight requests per department, sixteen urgent and sixteen phone
  requests. It includes nine refund requests and three current production outages.
  IDs and normalized request text are disjoint from every earlier dataset.
- As before, eight layouts cross four cyclic code assignments with two display
  orders; example answers are remapped consistently. These are structured layouts,
  not every possible permutation. JSON is evaluated only in the original order.
- All three code conditions run over all layouts once, with two extra identity
  timing repeats. JSON zero/four-example controls have three repeats. Identical
  identity-layout code prompts are also scored serially once per case. Token-ID
  hashes verify the serial and cached prompts match.
- Quality uses repetition zero. Run order rotates by case. Timing includes prompt
  construction, tokenization, model execution and answer assembly; excludes model
  loading and warmup; GPU work is synchronized. The run retains 1,248 evaluations.

Frozen plan · Frozen protocol and input hashes
· Raw smoke results · Raw holdout results

## Fresh holdout, original layout

| Method | Correct fields | Entire request correct | Median time | p95 time |
|---|---:|---:|---:|---:|
| Constrained, no examples | 127/160 (79.4%) | 12/32 (37.5%) | 128.4 ms | 134.5 ms |
| Constrained, four old object examples | 92/160 (57.5%) | 4/32 (12.5%) | 149.0 ms | 155.1 ms |
| Constrained, four aligned chat examples | 144/160 (90.0%) | 20/32 (62.5%) | 729.1 ms | 770.1 ms |
| Ordinary JSON, no examples | 125/160 (78.1%) | 11/32 (34.4%) | 434.8 ms | 454.3 ms |
| Ordinary JSON, four examples | 138/160 (86.2%) | 17/32 (53.1%) | 474.9 ms | 626.2 ms |

All outputs obeyed the schema, all timing repeats gave identical answers, and
neither JSON condition hit its output limit (maximum 36 and 49 generated tokens).
Aligned examples corrected 22 previously wrong fields and introduced five new
errors, for a net gain of 17 correct fields. They still got at least one field
wrong in 12 of 32 requests. Peak BF16-run allocation was 3.81 GiB.

The 90% versus 86.2% comparison is only a six-field difference on this small
synthetic set; it does not establish general superiority over JSON generation.

## Stability under option changes

| Constrained prompt | Mean field accuracy over layouts | Worst layout | Invariant field decisions | Pairwise agreement |
|---|---:|---:|---:|---:|
| No examples | 76.3% | 73.1% | 83/160 (51.9%) | 77.4% |
| Old object examples | 66.1% | 50.6% | 24/160 (15.0%) | 60.8% |
| Aligned chat examples | 89.1% | 85.6% | 142/160 (88.8%) | 95.0% |

An invariant decision returns the same typed value in all eight layouts; it need
not be correct. Layouts and timing repeats do not create extra independent cases.

| Additional diagnostic | No examples | Old object examples | Aligned chat examples |
|---|---:|---:|---:|
| Answer changes when display alone reverses | 26.9% | 53.0% | 5.9% |
| Chooses last displayed option on binary fields | 59.5% | 78.2% | 49.8% |
| False urgent among normal requests, across layouts | 43.0% | 50.8% | 5.5% |
| Missed urgent among urgent requests, across layouts | 21.1% | 19.5% | 25.0% |

The earlier last-option tendency largely disappears. Urgency classification
improves overall, but the error tradeoff matters: fewer false alarms accompany
slightly more missed urgent requests across layouts. The original layout alone
has only two priority errors with aligned examples, so it would conceal some of
that sensitivity.

In the original layout, remaining aligned errors are seven missed phone requests,
four missed refund requests, three routing errors and two priority errors. For
example, “Please ring me next Tuesday” was not recognized as a phone request.
Mixed requests about a product fix or account access plus reimbursement sometimes
route to billing despite the explicitly stated main request. These are semantic
mistakes despite valid output types.

The four frozen demonstrations also have a coverage limitation: both phone-request
examples are urgent, and both non-phone examples are normal. There is no example
of a phone request without urgency. This could contribute to missed future calls;
the present experiment does not establish that causal explanation.

Order diagnostics, per-field error counts and paired corrections

## Parallel execution versus prompt format

Serial scoring here still asks one code question per field. It is distinct from
ordinary autoregressive JSON generation, which uses a different output format and
can condition later fields on earlier generated fields.

| Identical identity-layout code prompts | Cached accuracy | Serial accuracy | Different answers | Largest BF16 probability difference |
|---|---:|---:|---:|---:|
| No examples | 79.4% | 80.0% | 1/160 | 0.022551 |
| Old object examples | 57.5% | 58.8% | 2/160 | 0.032013 |
| Aligned chat examples | 90.0% | 90.0% | 0/160 | 0.022195 |

The aligned improvement persists under serial execution. To examine numerical
differences, a separate FP32 run repeated every case/condition with a BF16 answer
disagreement, plus the largest probability discrepancy in each condition: five
case/condition checks total. All FP32 answers agreed; the maximum probability
difference was **0.000003368**. This supports reduced-precision arithmetic as the
source of those observed execution differences, rather than a cache correctness
failure. The full FP32 reference peaked at 8.36 GiB allocation.

This reference covers the selected checks, not every case and layout. It does not
claim exact BF16 arithmetic equivalence. Raw precision checks

## Why it became slower

The original prompt has a median 506 tokens in its longest field question, with
405 shared prefix tokens. The aligned version has 2,336 tokens in its longest
question, but only 400 shared prefix tokens. Total prompt tokens across five fields
increase from 2,377 to 10,915 because each field receives its own demonstration
history and repeated full schema.

This implementation groups equal-length suffixes. For these five-field prompts,
all five suffix lengths differ, so the cached path uses one shared prefill and
five separate branch calls. Its “parallel” label denotes the general independent
branch algorithm; these specific prompts do not execute all five fields in one
simultaneous batch. The serial comparison uses five complete forward passes.

Aligned cached scoring has a 729 ms median versus 813 ms serially: some reuse
remains, but most processing is field-specific. The median paired slowdown versus
the original cached prompt is 5.68×; versus JSON with four examples it is 1.55×.
Thus this aligned version loses the original speed advantage.

## Interpretation and next step

The previous negative few-shot result was specific to the tested presentation.
The same stock model and four example requests can improve constrained decisions
substantially when demonstrations answer the queried field directly. This change
also uses separate native chat turns and repeats the schema; the experiment does
not isolate single-code output alignment as the sole causal factor.

Before training, the next useful experiment is to preserve this accuracy and
stability while shortening the demonstrations or restoring more shared context.
That needs a new controlled comparison; these results do not establish that a
shorter format will retain the gain. The default engine remains unchanged.

All cases are authored synthetic support requests from one author and domain.
Only three contain current production outages. There is no production-quality,
calibration, or Jev-parity claim. The v3 holdout is now inspected and must be
treated as development data if used for further tuning. No model training occurred.

## Reproduce

Use new output filenames from the project root:

```sh
.venv/bin/python -m experiments.aligned_fewshot --smoke --output reports/new-aligned-smoke.json
.venv/bin/python -m experiments.aligned_fewshot --output reports/new-aligned-holdout.json
.venv/bin/python -m pytest -q
```

The separate `experiments/aligned_precision.py` script targets the retained
`reports/aligned-holdout.json` and refuses to overwrite its precision report.
It records the input report hash and explicitly states its selected-check scope.
All 40 tests and 16 subtests passed. New tests cover unchanged final questions,
single-code demonstrations, all eight remappings, invalid demonstration labels,
and execution comparisons that reject mismatched tokenized prompts. Owned Python
sources pass lint and formatting. Core inference sources and the preceding
few-shot experiment remain byte-for-byte unchanged from their recorded hashes.
