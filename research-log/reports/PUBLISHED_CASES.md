# Qualitative replay of TypeSafe's public cases

The first trained model was run on all 20 published examples, alongside its
untrained baseline. Fine-tuning improved agreement with the supplied references,
but the stored Jev run remains stronger overall. The most useful findings concern
descriptive scoring, evidence of completed actions, and overly broad positive
answers—not a single headline accuracy number.

No training, calibration fitting, prompt tuning, live Jev calls or paid API calls
were performed. The source snapshot and trained checkpoint were held fixed.

## What was replayed

The public website contains 46 reached TypeSafe **display bundles**, holding 408
questions: 264 Noul, 117 Choice and 27 Score. We replayed their published states
and question definitions through the existing local interface. Thirty skipped
nodes were left skipped. This keeps the recorded Jev path fixed; it does not
execute a newly branching workflow using our answers.

The website lists 711 cases in its full evaluation but exports five selected
examples per workflow. These selections deliberately illustrate agreement and
disagreement. Its historical model identifier is
`typesafe:v13_snowy_elephant`; these are stored results, not a measurement of the
current live Jev API. Invoice's display merges many questions into one semantic
bundle even though its description discusses seven rounds. Original request
batching and the complete executable workflow are not recoverable from this export.

Of the 408 questions, 336 have a determinate compatible reference label. The
comparison explicitly excludes 54 missing references, 17 ties and one incompatible
answer space. For Choice we compare labels; for Noul the side of 0.5; for Score
the most probable level. Score expectation comparisons are also given below.

## Descriptive results

These are question-level **reference agreement**, not human-gold accuracy and not
the official workflow benchmark score. The pooled row is question-weighted;
invoice questions account for 167 of its 336 observations. Questions within a case
are correlated, and the 20 selected cases are not a representative sample.

| Workflow | Comparable questions | Untrained | Trained | Stored Jev |
|---|---:|---:|---:|---:|
| Security incidents | 26 | 15/26 (57.7%) | 18/26 (69.2%) | 21/26 (80.8%) |
| Agent-trace review | 52 | 25/52 (48.1%) | 33/52 (63.5%) | 41/52 (78.8%) |
| Invoice processing | 167 | 133/167 (79.6%) | 148/167 (88.6%) | 162/167 (97.0%) |
| Customer service | 91 | 68/91 (74.7%) | 73/91 (80.2%) | 82/91 (90.1%) |
| **Question-weighted total** | **336** | **241 (71.7%)** | **272 (81.0%)** | **306 (91.1%)** |

Fine-tuning changed 51 reference disagreements into agreements and 20 agreements
into disagreements. Our trained model matched the reference where Jev did not on
18 questions; the reverse happened on 52. These counts describe agreement with
the supplied reference and do not independently establish which answer is correct.

| Primitive | Comparable questions | Untrained | Trained | Stored Jev |
|---|---:|---:|---:|---:|
| Noul | 220 | 183 (83.2%) | 193 (87.7%) | 205 (93.2%) |
| Choice | 89 | 48 (53.9%) | 70 (78.7%) | 81 (91.0%) |
| Score, modal level | 27 | 10 (37.0%) | 9 (33.3%) | 20 (74.1%) |

There are 169 questions with actual reference distributions. Mean total-variation
distance to those distributions is 0.412 untrained, 0.258 trained, and 0.139 for
stored Jev. This is distribution agreement, not a calibration measurement.
Invoice's references supply hard labels only; their vote fractions are excluded
from this probability-distance metric.

## Findings worth following up

### 1. Choice transfers better than descriptive Score

Choice agreement rises from 48/89 to 70/89. Descriptive Score remains a clear gap.
Its expected-score error improves modestly: mean absolute error, divided by each
rubric's full scale, falls from 0.269 to 0.238; Jev's is 0.103. Thus the one-question
decline in modal agreement should not be described as deterioration on every
Score measure, but neither measure suggests the problem is solved.

In the sarcastic cancellation case, the customer uses profanity and attacks the
quality of the service. The frustration rubric spans 0–4. Reference expectation:
**3.595**; stored Jev: **3.79**; ours: **1.96**. Our model gets the cancellation
intent right while understating the intensity. Our Score training used wine
quality and arithmetic policy bands; broad emotional and semantic rubrics were
not represented by those sources. That is a plausible data-coverage explanation,
not a demonstrated causal diagnosis.

Locator: `customer_service / retention-sarcastic-sure-whatever/t0 / triage / frustration`.

### 2. Saying an action happened is not evidence that it happened

In the stale database-grant case, the sole recorded tool call lists grants.
The assistant announces an escalation, but no tool result confirms an escalation
record. The question explicitly requires tool confirmation.

| P(handoff completed) | Untrained | Trained | Stored Jev | Pooled reference |
|---|---:|---:|---:|---:|
| Stale-grant case | 0.222 | **0.940** | 0.040 | 0.015 |

This is a clear fine-tuning regression on the visible evidence, in an unclipped
case. The trained model also correctly recognizes that a handoff is required,
but fails to distinguish that obligation from confirmed execution. It suggests
that our training needs examples separating requests, promises and recorded actions.

Locator: `agent_trace_observability / trace_ebba36efa92c88a555ad4be8eabf149d / outcome / handed_off`.

### 3. An apparent bank-change success hides an overly broad response

For one invoice, a vendor message calls its payment details unchanged while
supplying a different account ending than the vendor record. Our trained model
detects the claim and matches both reference labels; stored Jev falls below 0.5.
However, our model answers yes to this same question in **all five invoices**.
Only one has a positive reference label.

| Invoice | Reference label | Untrained P(yes) | Trained P(yes) | Stored Jev P(yes) |
|---|---|---:|---:|---:|
| ap_00149 | no | 0.284 | **0.748** | 0.200 |
| ap_00198 | no | 0.246 | **0.570** | 0.190 |
| ap_00074 | yes | 0.249 | **0.693** | 0.430 |
| ap_00246 | no | 0.248 | **0.540** | 0.360 |
| ap_00225 | no | 0.242 | **0.670** | 0.310 |

For ap_00149, the emails concern overdue payment and timing, without a changed
payment destination or a claim of prior bank verification. The positive answer
there is unsupported. The five-case pattern warrants targeted negative examples;
it does not reveal the model's internal heuristic.

Locator: each invoice's `semantic / bank_change_claimed_in_comms`.

### 4. Mentioning evidence can be enough

The repeatedly charged customer explicitly says they have a transcript of the
earlier cancellation conversation. The question counts mentioning evidence;
an uploaded attachment is unnecessary. P(evidence mentioned) is **0.100** from
our trained model, versus **0.920** for Jev and **0.965** for the reference.
Untrained was 0.421, so training increases confidence in the wrong direction.
This short-context error cannot be attributed merely to long input length.

Locator: `customer_service / retention-charged-after-cancel/t0 / money / offers_evidence`.

### 5. Agreement can still hide legitimate ambiguity

The espresso customer accepts a $148 return/refund proposal and stresses a
financial deadline. On whether that changes the proposed terms, our P(yes) is
**0.015**, Jev's is **0.760**, and the pooled reference is **0.240**. Reading the
deadline as urgency rather than a new condition is reasonable, but not the only
possible reading. Matching the reference does not justify our near-certainty.

Locator: `customer_service / refunds-espresso-impulse-148/t2 / consent / changes_terms`.

## Execution checks and source limitations

Eight review fields—the five bank-change questions, the unclipped handoff failure,
the evidence question and frustration Score—were rerun through independent
full-input scoring. None changed its modal/binary decision; maximum probability
movement was 0.00635. The highlighted interpretations are therefore not explained
by the cache-versus-independent discrepancies observed in earlier benchmarking.

Two public states in the frozen-schedule case contain an explicit marker that
534 characters of a tool result were omitted. We replayed the published text
unchanged, but cannot claim it is the full original input. These two nodes cover
seven questions. Removing them gives trained **270/329**, untrained **238/329**,
and stored Jev **302/329** agreement. Absence of a marker elsewhere is not proof
that the website export preserved every original detail.

Other limitations retained in the artifacts:

- Reference pools vary. The first agent-trace case uses Astra alone; the other
  four use Astra and Fable. Security describes substituting Opus for Fable on
  some full-dataset documents without identifying each substitution in the excerpt.
  Individual reference sets lack their own provider/input records; comparison
  uses the export's node/question mapping rather than independently verified
  input parity for every reference call.
- Invoice has 184 covered questions with hard labels and null probability fields.
  Votes are not uncertainty probabilities. Tied votes are left unresolved rather
  than broken using option order.
- A customer `intent_pair` has different candidate sets across providers. Its
  reference puts mass on an option absent from the Jev question, so it is excluded.
  This limits interpreting the public display; it does not establish that the
  unpublished full evaluator made the same mistake.
- Stored Jev probabilities are rounded to two decimal places. One vector sums to
  0.99 and is normalized with its original total retained. Ten stored Score means
  differ slightly from means recomputed from their rounded bins; both are kept.
- Some published questions are conditional or speculative and may not affect a
  workflow's eventual action. Per-question agreement cannot substitute for final
  action correctness. References are model judgments and may also be debatable.

## What this suggests for the next experiment

The first fine-tune transferred beyond its own training mix, especially for
Choice, but is not yet a generally reliable structured-judgment model. The next
data expansion should target diverse descriptive rubrics, tool-result grounding,
and hard negative examples for claims absent from a record. Keep this inspected
subset as a diagnostic regression set and use fresh cases for any subsequent
generality claim. Calibration and confidence thresholds should be assessed on
those task families; rescaling probabilities alone cannot repair wrong readings.

## Reproduction and artifacts

Qwen3.5-2B pinned backbone; trained checkpoint
`openjev-judgment-v0.2/sha256-4f88417678a33bc14f47ad97e9f0ffc96bf426e84d12a900eb99c2f97800ff49`.
Unmerged BF16 inference with FLA, FP32 heads, batches of four, and an experimental
12,288-token bound. The largest actual unit was 9,264 tokens. No input was
truncated by our replay. Peak model allocation was about 4.59 GiB. These long
inputs exceed the retained training maximum of 776 tokens.
The untrained control uses the same numerical interface with initial A-minus-B
readouts; it is not a conventional chat-generation evaluation of the base model.

The parser and comparison rules were tested, including executable-suffix rejection,
duplicate JSON keys, index validity, answer leakage, null labels with valid
probabilities, label-only ties, incompatible options and rounded distributions.
The full offline suite passed **189 tests and 45 subtests**; lint passed. Independent
source and code reviews verified the counts, exclusions and input boundary.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.runtime-training:. \
  .venv/bin/python -m experiments.published_cases_replay \
  --output reports/my-published-cases-run
```

- All 20 cases, every question side by side
- Raw comparisons and exact published input bundles
- Run report, post-hoc analysis, and independent execution checks
- Source lock and hashes
- Official pages: [security](https://evals.typesafe.ai/security_incidents), [agent trace](https://evals.typesafe.ai/agent_trace_observability), [invoices](https://evals.typesafe.ai/invoice_processing), [customer service](https://evals.typesafe.ai/customer_service)
