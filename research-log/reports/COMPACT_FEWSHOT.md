# Shorter examples: about twice as fast, with a quality tradeoff

Removing repeated full schemas from the demonstrations, while retaining their
complete field questions and answer choices, reduced median latency from **766 to
377 ms** on a fresh 32-request holdout. Original-layout field accuracy changed from
**142/160 (88.8%) to 141/160 (88.1%)**. Both methods got every field correct on
16 of 32 requests. No model training occurred.

The gain is useful, but it is not a demonstrated lossless optimization. Accuracy
averaged across eight option layouts fell from **89.9% to 87.8%**, and invariant
field decisions fell from **90.6% to 88.1%**. The candidate had also failed the
development accuracy-preservation rule. The default engine remains unchanged.

## What changed

All conditions use Qwen3.5-2B at revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, the RTX 4080, the same four example
requests in the same order, the same schema and choices, and unchanged final user
questions. The backbone is BF16 with FP32 candidate projection; the runtime still
uses reference recurrent kernels. Target labels never enter the prompt.

Three compact formats were investigated:

1. **Short field histories:** one full schema in the system message, then four
   request / field-name question / one-code answer exchanges per queried field.
2. **Shared short history:** the same compact exchanges for all five fields,
   ordered by example then field: twenty exchanges in one history shared by all
   target questions. This still uses only four source requests, with twenty field
   answers; each assistant turn contains one code.
3. **Short histories with full questions:** four field-specific exchanges, keeping
   each original question description and option list. This removes the full
   schema from each demonstration and supplies it in the system message instead.

The final target user turn retains its original full schema in all conditions.
Thus the selected format has two full-schema copies per field history—one in the
system and one in the target—instead of five, rather than literally one schema
copy in the whole prompt. Demonstration codes track every option remapping.

Initial prompt renderings ·
Full-question rendering

## Development, selection, and the adaptive refinement

The previous v3 holdout is now development data: 32 requests and 160 labels. Its
previous long-prompt results supply the quality baseline. Before any new scoring,
a separate v4 holdout was authored and hashed. It has 32 new requests, eight per
department, sixteen urgent requests, sixteen phone requests, seven refund requests,
and three current production outages. IDs and normalized text are disjoint from
all preceding sets and the demonstrations.

The initial two compact formats and selection rule were frozen first. Eligibility
requires development mean accuracy within one percentage point of the long
baseline, invariant fields within two points, worst-layout accuracy within two
points, and at least 25% lower median latency. Choose the fastest eligible format.
If none qualifies, select the highest mean accuracy, then stability, then speed,
and label it an exploratory fallback. These are descriptive selection margins,
not a statistical non-inferiority test.

Both initial formats failed the quality rules. Before any v4 scoring, one adaptive
development refinement was recorded: restore the complete demonstration questions
and choices while retaining schema compression. Initial results and the initial
selection remain untouched. The revised selection compares all three candidates
under the same rule, and was frozen before the final holdout run. There was no
further prompt tuning after holdout outputs were seen.

| Development format | Mean accuracy over layouts | Worst layout | Invariant fields | Original-layout accuracy | Median time |
|---|---:|---:|---:|---:|---:|
| Previous long aligned prompt | 89.1% | 85.6% | 88.8% | 90.0% | 729.1 ms |
| Short field histories | 83.7% | 81.9% | 76.2% | 81.9% | 305.9 ms |
| Shared short history | 85.6% | 70.0% | 58.1% | 89.4% | 244.5 ms |
| Short histories, full questions | 85.6% | 84.4% | 89.4% | 85.0% | 360.8 ms |

The shared history illustrates why the ordinary option order alone is insufficient:
its 89.4% accuracy looked promising, while another layout scored only 70.0%.
The full-question revision restored stability but still missed the accuracy margin.
It tied the shared format on mean accuracy and won the stability tiebreaker, so it
was selected as the exploratory fallback, not as a quality-equivalent replacement.

Baseline timing above comes from the preceding run. Eight control prompt/answer
checks passed in each of the two development runs. The final holdout reruns every
control in the same process as the selected candidate for the latency comparison.

Initial protocol · Initial development
· Initial selection · Adaptive protocol
· Refinement development
· Final frozen selection

## Fresh holdout: quality and speed

| Method, original layout | Correct fields | Entire request correct | Median latency | p95 latency |
|---|---:|---:|---:|---:|
| Constrained, no examples | 128/160 (80.0%) | 11/32 (34.4%) | 130.5 ms | 141.2 ms |
| Constrained, long aligned examples | 142/160 (88.8%) | 16/32 (50.0%) | 766.4 ms | 850.8 ms |
| Constrained, shorter full-question examples | 141/160 (88.1%) | 16/32 (50.0%) | 376.9 ms | 426.1 ms |
| Ordinary JSON with four examples | 143/160 (89.4%) | 18/32 (56.2%) | 496.7 ms | 647.4 ms |

Median paired speedups are **2.03× versus the long prompt** and **1.31× versus
JSON with examples**. The shorter prompt is still 2.88× slower than the original
no-example method. These are speed/quality tradeoffs, not equal-quality speedups.

Compared with the long prompt, the shorter one corrected four fields and made five
new mistakes in the original layout. Thus the one-field net accuracy difference
does not mean the methods differed on just one answer. JSON's two-field advantage
over the shorter prompt is also too small and narrow a sample for a broad ranking.

All 1,088 holdout evaluations obeyed the schema. Repeated evaluations produced the
same answers, and JSON never hit its token limit (maximum 49 generated tokens).
The two development stages contain 960 additional timed evaluations. Accuracy
counts the first repetition once; layouts and timing repeats are not independent
new requests. Timings include preparation, tokenization, execution, and answer
assembly, with synchronized GPU work and excluded warmup/loading. Conditions rotate
by case. Peak allocation in the main holdout run was 3.81 GiB.

Raw holdout results · Derived diagnostics

## Option stability and remaining errors

| Constrained format | Mean accuracy over layouts | Worst layout | Invariant field decisions | Display-only answer flips |
|---|---:|---:|---:|---:|
| No examples | 79.6% | 75.6% | 88/160 (55.0%) | 24.4% |
| Long aligned examples | 89.9% | 86.9% | 145/160 (90.6%) | 5.0% |
| Shorter full-question examples | 87.8% | 86.2% | 141/160 (88.1%) | 6.9% |

The shorter prompt preserves most of the stability gain relative to zero-shot,
but loses some accuracy and stability relative to the long version. An invariant
answer can still be wrong. JSON's option-order robustness was not measured.

The shorter prompt's 19 original-layout errors comprise nine missed phone
requests, five routing errors, three missed refunds and two priority errors.
For example, it missed explicit requests to telephone next week and misrouted some
account-access requests that mentioned refunds. Compared with the long prompt,
the new errors concern three phone requests and two routing decisions.

Across layouts, false-urgent rates are 35.9% without examples, 1.6% for the long
prompt and 2.3% for the shorter one. Missed-urgent rates are 7.8%, 14.8% and 20.3%,
respectively. Reduced false alarms do not eliminate the urgency tradeoff.

The fixed demonstrations still have a coverage weakness: both phone-request
examples are urgent, and both non-phone examples are normal. That is a plausible
contributor to missed future calls, not a demonstrated causal explanation. The
examples were deliberately unchanged here to focus on prompt compression.

## Computation and numerical checks

Median maximum prompt length falls from 2,335 to 1,360 tokens. Total prompt tokens
across five fields fall from 10,910 to 6,035. The selected format reduces duplicated
work; it does not restore the large shared history of the faster rejected format.
Its common prefix is 403 tokens, versus 400 for the long prompt.

All five suffix lengths differ in these prompts, so the shared-cache implementation
uses one common prefill and five separate branch calls, not one simultaneous batch
of five fields. The selected prompt takes 377 ms with reuse versus 449 ms for
independent serial scoring of the same questions.

Token-ID hashes confirm identical serial/cached prompts. All 160 original-layout
field answers agree between those execution paths in BF16, though probability
differences reach 0.035215. Because that numerical difference warranted checking,
the largest-discrepancy case (`v4_a02`) was rerun in FP32. All five answers agreed;
the maximum probability difference fell to **0.000003517**. This supports rounding
as the explanation for the observed discrepancy. It is a targeted check, not an
exhaustive FP32 evaluation. Peak FP32 allocation was 7.49 GiB.

FP32 reference

## Interpretation and reproducibility

The experiment recovered about half the latency while keeping the example format
recognizable and retaining most measured quality. It did **not** meet the frozen
accuracy-preservation criterion. The aggressive shared-history variant was faster
but substantially more sensitive to option presentation. The shorter full-question
format is a useful alternative to keep for further evaluation, not an automatic
replacement for the long prompt or the original engine.

These are small, synthetic datasets from one author/domain, with only three live
outage cases in the new set. They do not establish production accuracy, confidence
calibration or Jev parity. The v4 holdout is now inspected and becomes development
data if used in future tuning. No training, extra model download, optimized-kernel
installation, or default-engine prompt change was performed.
The JSON baseline uses ordinary greedy generation without grammar constraints;
the constrained methods score fields independently and assemble JSON in code.
This does not compare against an optimized structured-output serving engine.

To rerun confirmation, use a new output filename:

```sh
.venv/bin/python -m experiments.compact_fewshot --stage holdout --protocol reports/compact-refinement-protocol.json --selection reports/compact-refinement-selection.json --output reports/new-compact-holdout.json
.venv/bin/python -m pytest -q
```

The precision script targets the retained holdout report and refuses to overwrite
its reference output. Reports store model revision, data/source hashes, selection
hashes, full answers, normalized scores and timings. Archived
original aligned source and
first compact source match their earlier report
hashes. Core inference sources remain unchanged; the experimental scorer gained
an optional prompt renderer so all formats use the same scoring code.

All 68 tests and 16 subtests passed. New tests cover code remapping for every
layout, unchanged final questions, shared-history identity, retained full question
text/options, invalid labels, and candidate selection that rejects faster but
inferior results. Owned Python sources pass lint and formatting checks.
