# Small contrastive-training pilot with live Jev references

The small contrast corpus materially improves targeted judgments, but the current
recipe is **not ready to scale**. Fresh-test accuracy rises from 64.1% to 95.3%,
versus 71.4% after original-data continuation alone. However, broad-task accuracy
falls from 87.3% to 83.7%, exceeding the predeclared two-percentage-point retention
limit. Some positive completion judgments also regress under the held-out input
format. The new checkpoint remains experimental; the original is preserved.

All paid Jev observations are stored locally and reused. There are **214 unique
observations**: eight imported earlier responses and 206 new calls in this round.
The new calls reported 264,494 input tokens, equivalent to **$0.01111** at the
published $0.042 per million input tokens. This is a list-price calculation, not
an account charge or balance reading. No API calls were used for training labels.

## Data and controlled comparison

- 300 training scenarios across 12 domains, 50 validation scenarios across two
  other domains, and 100 fresh test scenarios across four further domains.
- Each scenario contains seven judgments: plan, completion claim, invocation,
  historical completion, current effect, authorization and categorical status.
  There are 2,100 training judgments and 700 fresh test judgments.
- Each domain instantiates the same 25 authored evidence variants. These include
  pending, failed, timed-out, simulated, real and reversed actions; wrong targets,
  operations and times; unverified/conflicting evidence; independent policy and
  statement changes. Literal labels were reviewed before model evaluation.
- Related scenarios stay together by domain. The fresh test also changes question
  wording and moves structured task fields into a prose brief with a nested
  record. Thus domain, wording and layout shift together. Repeated templates,
  four test domains and class imbalance make this limited transfer evidence.
- Both trained arms begin from the same v0.2 final checkpoint, with a fresh
  optimizer. The frozen base, rank-8 LoRA policy and numerical heads are unchanged.
  Adapter LR is 1e-4; head LR 5e-5; AdamW decay .01; gradient clip 1; BF16 body,
  FP32 heads/losses; FLA; scoring-row batch size four.
- Each arm receives 300 updates and four equally weighted source bundles per
  update. The control uses only original training data. The contrast arm replaces
  two slots with new scenarios; its other two original-data slots exactly match
  the control's common slots. Each new training scenario is seen twice.
- Original replay uses 100 canonical first examples per source, across six
  original training sources. No original test/validation/calibration data enter
  replay. All seven contrast questions share their scenario's bundle weight.
  The existing binary/categorical probability losses are used; this comparison
  adds no auxiliary relation loss and no teacher distributions.

Budgets match **optimizer updates and source-bundle weights**, not compute or
numbers of judgments. The control processes 1,200 questions / 3,752 neural rows;
the contrast arm processes 4,800 questions / 7,908 rows. Training takes 194 seconds
and 806 seconds respectively. Consequently this is evidence about the specified
data mixture, not a compute-matched estimate of contrastive supervision alone.

## Main results

| Evaluation | Starting checkpoint | Original-data control | Contrast mixture | Live Jev 1.13.0 |
|---|---:|---:|---:|---:|
| Fresh test | 449/700 (64.1%) | 500/700 (71.4%) | **667/700 (95.3%)** | 696/700 (99.4%) |
| Original contrast diagnostic | 463/764 (60.6%) | 491/764 (64.3%) | **623/764 (81.5%)** | 760/764 (99.5%) |
| Broad canonical test | 1,309/1,500 (87.3%) | 1,316/1,500 (87.7%) | **1,256/1,500 (83.7%)** | Not run |

The starting checkpoint reproduces every probability from the prior original
contrast run exactly. Every arm's final evaluation was performed once; no
checkpoint selection or hyperparameter change used these outcomes.

The frozen scaling gate required a fresh-test gain of at least 10 points over
control, improved macro binary balanced accuracy, improved mean correctness of
the three relation kinds, and no more than two points of broad accuracy loss.
The first three pass; broad retention fails with a **3.53-point loss**.

Probability quality also needs separate reporting:

| Mean negative log likelihood; lower is better | Starting | Control | Contrast |
|---|---:|---:|---:|
| Fresh test | .7492 | .7029 | .1373 |
| Original contrast diagnostic | .9731 | 1.3943 | .8623 |
| Broad canonical test | .2921 | .4728 | .5101 |

Original-data continuation itself increases confidence in some mistakes: its
broad accuracy improves slightly while probability loss worsens. Probability
quality is not established by a correct winning label.

## The headline hides a completion regression

On the fresh test, plan, claim and authorization judgments each reach 100/100;
invocation reaches 92/100, current effect 97/100 and categorical status 98/100.
Historical completion is only 80/100, versus 79/100 initially. Its error mix flips:

| Completion judgment | Starting | Control | Contrast |
|---|---:|---:|---:|
| Positive cases correctly confirmed | 32/32 | 27/32 | **12/32** |
| Negative cases correctly rejected | 47/68 | 62/68 | **68/68** |
| Balanced binary accuracy | 84.6% | 87.8% | **68.8%** |

The contrast model removes all 21 initial false positives but introduces 20 false
negatives. In the plain successful-execution variant it rejects completion in all
four test domains, assigning P(yes) approximately .008–.039. This is the reason
positive controls and both-members-correct scores matter.

| Fresh relation; both endpoints must be correct | Starting | Control | Contrast | Jev |
|---|---:|---:|---:|---:|
| Evidence-change pairs | 25/80 | 35/80 | 40/80 | 80/80 |
| Same-record question contrasts | 70/140 | 90/140 | 120/140 | 140/140 |
| Invariance pairs | 73/76 | 69/76 | **52/76** | 76/76 |

Their unweighted mean improves from 59.1% to 68.0%, only modestly above control's
66.3%. Repeated positive endpoints amplify the completion failure in relation
metrics; high field accuracy does not establish robust contrast behavior.

On the older action diagnostic, completion confirmation improves to **77/80**,
from 33/80. That apparent discrepancy motivated the format probe below. Meanwhile,
the four older registry Unknown cases still score **0/4** after contrast training:
learning action uncertainty did not establish transfer to registry uncertainty.

## Follow-up: layout and wording

After inspecting the frozen test results, an inference-only probe takes 16 of its
scenarios: success, pending, wrong target and reversed, in all four test domains.
It varies task representation and question wording separately, retaining the
same facts, criteria and labels. This is retrospective diagnosis, not a new test
for model selection; no weights or production defaults change.

| Task representation | Question wording | All judgments | Completion judgments |
|---|---|---:|---:|
| Structured task fields | Training wording | 112/112 | 16/16 |
| Structured task fields | Held-out wording | 112/112 | 16/16 |
| Nested record + prose task brief | Training wording | 106/112 | 14/16 |
| Nested record + prose task brief | Held-out wording | 106/112 | 12/16 |

The last condition exactly reproduces the saved predictions. Under the structured
representation the model handles the unfamiliar question wording. The nested
representation is more brittle, with a further completion-specific wording
interaction. This does **not** isolate JSON depth alone: the intervention also
changes structured task attributes into prose. Software accepts and preserves
both representations; the model's behavior is what changes.

## Retention losses are concentrated in Score tasks

The contrast model loses 53 net correct broad judgments relative to its starting
checkpoint. **46 of those net losses come from the two Score sources**:

| Broad source; 250 examples each | Starting | Control | Contrast |
|---|---:|---:|---:|
| BoolQ | 208 | 209 | 208 |
| PAWS | 235 | 233 | 230 |
| BANKING77 subset | 244 | 242 | 242 |
| Synthetic policy Score | 250 | 250 | **228** |
| Wine-quality Score | 122 | 133 | **98** |
| CLINC150 subset | 250 | 249 | 250 |

The new contrast corpus has Noul and Choice but no Score examples; Score is present
only in the replay portion. This is consistent with inadequate retention under
the chosen mixture, but does not uniquely identify head drift, body changes or
data balance as the cause. The unchanged original checkpoint remains available.

## What live Jev adds

Jev serves as a fallible comparison model, not a label oracle. Its four fresh-test
disagreements all concern a stale successful event and `current_effect`:
P(yes) is .56–.57 despite the intended in-window evidence rule. Independent review
found a plausible alternative reading: yesterday's success may still have an
effect today, and this question's wording does not repeat the in-window
restriction as explicitly as the completion question. These are four instances
of one template issue. Labels and frozen scores remain unchanged; a clearer
contract belongs in a future version.

The four original-suite Jev disagreements are clearer distinctions:

- Success for `REQUEST-204-OTHER` is counted as completion/effect for `REQUEST-204`
  at P(yes) .52/.55.
- A refund plan is treated as an assistant completion claim at P(yes) .53.
- A shipping transaction's policy hold interferes with registry verification:
  P(record confirms change) is .22 while P(categorical Changed) is .59, despite
  the equivalent event meaning of those two predicates in that contract.

None is assigned at least 95% winning probability. Jev passes all declared fresh
pairs, but nine of its 76 invariance pairs drift by more than the diagnostic .05
probability-distance tolerance. These observations neither establish general
calibration nor reveal Jev's architecture or training method.

## Preserve and reuse every observation

Each call has an immutable request, raw response body, timestamp, response hash,
HTTP status, safe response headers, requested/returned model and validation record
under `data/jev-responses/`. It is persistent experiment data, not disposable
cache storage. Old JSONL observations retain their exact original artifact lines;
their unavailable historical HTTP bytes are explicitly distinguished from new raw
captures. Literal credential bytes are redacted; authorization headers are not
stored. The archive does not decode arbitrary escaped credential echoes.

Exact semantic requests against the same pinned returned version reuse the
earliest eligible observation, never the highest-scoring draw. A historical alias
can satisfy a pin only when the actual returned version matches. `--fresh`
explicitly creates another independent sample without replacing the old one.

The first fresh-test run stopped after one returned Choice vector summed to .99.
The paid body was already archived. Analysis policy v2 permits renormalization only
for positive totals within .020000001 of one and probabilities on the .01 grid;
raw values, the reported sum and a repair flag remain visible. Other malformed
vectors are rejected. The original invalid verdict remains immutable, with a
separate revalidation record. Exactly one vector needed this treatment.

The resumed run reused all 72 earlier observations and made only the remaining
28 calls. The full original-suite run reused the original eight and made 106
new calls. No 429/529 responses occurred. All 214 response-body hashes and request
hashes were checked. Timing from a cache read is not Jev network latency.

## Consequences for the four stages

1. **Small contrast pilot: completed; scaling gate failed.** Next revise input
   representation coverage and protect Score retention in another bounded pilot.
   Preserve positive completion controls and unseen registry tests alongside
   aggregate accuracy. More copies of the same 25 templates are insufficient.
2. **Distribution targets: still a separate experiment.** Use Jev observations
   with explicit teacher provenance, audited disagreements and independently
   justified labels. Compare against hard-target training at a matched budget;
   do not silently turn the held-out Jev cases into training data. Retention
   targets from the starting model are another distinct control, not truth labels.
3. **Luna/Terra generation: gated, not launched at scale.** Prioritize varied
   state layouts, question phrasings, registry/event schemas and positive/negative
   matched families. Jev can help triage questionable cases, while rules and
   independent review establish labels. Keep fresh whole families out of training.
4. **Larger Qwen teacher: remains a candidate.** Compare it with Jev and the
   reviewed labels before using its probabilities. Open weights add access to
   raw logits and controlled scoring protocols; token likelihood and stated
   probability remain different quantities. No larger model download or rental
   was needed for this pilot.

## Artifacts and verification

- Frozen protocol and execution plan
- Training cases, fresh test
- Completed local run and scaling gate
- Experimental checkpoint metadata
- Layout/wording probe
- Jev fresh-test comparison
- Jev original-suite comparison
- Preserved-response inventory
- [API access and archival guide](../design/jev-comparison.md)

The source-specific two-update smoke passed finite-gradient, frozen-backbone and
save/reload checks; maximum reload probability difference was zero. Both full
arms completed all 300 updates and preserved the frozen base hash. Their full
1,500-bundle evaluations also streamed raw logits so interrupted work would remain
available. Evaluations and training have explicit time caps. Independent reviewers
checked the data semantics, runner and archive; findings were fixed and tested.
The final source/data/checkpoint hashes and complete artifact counts are verified
separately from the reported accuracy values.

Final verification: **318 tests and 45 subtests passed**, with the existing harmless
pytest warning about the already imported `anyio` module. Lint passed for all new
and changed experiment/test files. Content-addressed source snapshots are retained
under `reference/runners/` in addition to the recorded hashes.
