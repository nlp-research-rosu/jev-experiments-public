# Actual tool and reasoning-effort probe

**For this precise-edit workload, Terra medium is the provisional quality/time
compromise: 63/64 admitted edits, with a timed batch near Terra low. Sol high
produced the cleanest batch at 64/64, with longer elapsed time. Higher effort was
not consistently better within a model.** The released Polyjuice and Tailor
generator cores ran locally, but yielded few usable edits in the tested setups.
Their older full pipelines were not reproduced, so these results must not be
presented as reproductions of their papers or universal generator rankings.

## Results

| Writer / setting | Usable edits | Meaning changes | Meaning-preserving rewrites | Distinct usable task/text pairs | Timed 32-output repeat |
|---|---:|---:|---:|---:|---:|
| Terra low | 59/64 (92.2%) | 30/32 | 29/32 | 38 | 24.5 s |
| Terra medium | **63/64 (98.4%)** | **32/32** | 31/32 | 44 | **25.8 s** |
| Terra high | 59/64 (92.2%) | 31/32 | 28/32 | 37 | 36.2 s |
| Sol low | 61/64 (95.3%) | 30/32 | 31/32 | 44 | 29.3 s |
| Sol medium | 62/64 (96.9%) | 30/32 | **32/32** | 46 | 33.3 s |
| Sol high | **64/64 (100%)** | **32/32** | **32/32** | **45** | 46.2 s |
| Polyjuice core | 3/32 (9.4%) | 3/16 | 0/16 | 2 | Not comparable |
| Tailor initial adapter + native parser | 6/32 (18.8%) | 5/16 | 1/16 | 5 | Not measured |
| Tailor corrected context + native parser | 3/32 (9.4%) | 3/16 | 0/16 | 2 | Not measured |

All six instruction-model settings produced at least one usable candidate for
each of the 16 tasks. Polyjuice covered two tasks, initial Tailor four, and the
corrected Tailor setup two. Duplicates remain in per-attempt results and are
separately identified; 64 outputs are not 64 independent semantic problems.

The model/effort settings each ran twice in fresh contexts on the same 16 tasks,
with two candidates per task per run. The two admission counts were Terra low
28/32 and 31/32; Terra high 30/32 and 29/32; Sol low 30/32 and 31/32; Sol high
32/32 twice. The adaptive middle-effort extension yielded Terra medium 32/32 and
31/32, and Sol medium 32/32 and 30/32. The middle settings were added after seeing
the low/high results, with unchanged tasks, instructions and acceptance policy.
This is exploratory selection, not an untouched confirmatory test. This small
replication does not establish statistical superiority
across arbitrary generation workloads.

Timing measures one correctly instrumented repeat from immediately before agent
dispatch to final output-file write. It includes scheduling, model reasoning,
tool calls and serialization. It is not pure inference latency, a stable speed
benchmark or token cost. The initial timing attempt started after asynchronous
dispatch and is explicitly excluded. No billed-token usage was exposed, so no
dollar-cost optimum is claimed.

Both middle-effort calls have valid timestamps: Terra medium took 37.1 and 25.8
seconds, and Sol medium took 53.9 and 33.3 seconds. The table consistently shows
the second call, not the fastest arbitrarily selected trial. This variation is
another reason to treat timing as exploratory rather than a stable service rate.

## What the task actually tested

Eight families each supply a controlled meaning change and a paraphrase:
completed claim versus future intention; successful refund versus explicit
no-effect rejection; authentication; target identity; before versus after a time
boundary; revocation; universal versus not-all completion; and authorizer versus
authorized-person roles. Every method starts from the same original sentence.
Surrounding context stays fixed in software; the method can return only a
replacement sentence.

Instruction models receive the precise natural-language goal. The specialist
tools receive their supported native control tags/headers and manually supplied
spans. Those interfaces are **not equally expressive or compute-matched**.
Polyjuice's broad edit category often does not encode the exact desired new
meaning, while some Tailor transformations are explicitly unsupported by the
tested native-control construction. This is a practical fit/coverage comparison,
not proof of what the weights could learn with a different training method.

The frozen file has 16 tasks, not a new broad held-out benchmark. The 480 original
generation attempts comprise 384 instruction-model outputs, 32 Polyjuice outputs
and two separate 32-output Tailor runs. Native parsing adds 64 derived views of
the existing Tailor text, **not** 64 new generations.

## The errors are relevant to our training problem

| Candidate transformation | What changed unintentionally |
|---|---|
| “The executor successfully restored snapshot A.” → “According to the executor, snapshot A was successfully restored.” | An asserted event becomes a report; the restorer is no longer identified. This was a Sol-low candidate. |
| A source reports a successful refund → the source reports that **it personally refunded** the order | Reporting source is silently made the event actor. This appeared in Terra outputs. |
| “is not authenticated” → “remains unauthenticated” | Adds a claim about earlier authentication status. This appeared in Sol-low outputs. |
| “was not revoked” → “remained in effect and was not revoked” | Adds continued validity, which need not follow solely from absence of revocation. This appeared in Terra-low output. |
| A written assistant assertion → an assertion without written-medium information | Loses part of the original evidence description. This appeared in Terra paraphrases. |

These are small changes that can survive an embedding-similarity check while
altering a relevant distinction. Admission used the frozen requirement to
preserve the full requested meaning, including attribution and temporal scope.
Some language-equivalence boundaries remain judgment calls; the ratings are
model-based audits, not human-annotated truth. A production task that deliberately
ignores written versus spoken medium could use a different stated contract.

No candidate was silently repaired or relabeled to improve the table. The primary
agent inspected all candidate strings, all rejected instruction-model examples
and the native-tool accepted examples; no blinded judgments were overridden.
An added deterministic check rejects unchanged word sequences, including mere
punctuation changes.

## What we learned from running the paper tools

### Polyjuice: fast native edits, coarse semantic control

The actual released `uw-hai/polyjuice` checkpoint ran on the RTX 4080. The adapter
uses the source's unchanged prompt construction and blank-reconstruction parser,
with frozen manual blank spans. Automatic spaCy span selection and the wrapper's
perplexity/edit-type filters are omitted. Every raw sample is retained and judged
by the common audit. [Official implementation](https://github.com/tongshuangwu/polyjuice).

It produced valid simple negations, such as changing an authenticated record to
one that is not authenticated. But the requested future-intention edit to “I sent
invoice 17” produced “I received invoice 17” or left the original unchanged. A
requested before→after edit produced “on 10:30.” Whole-sentence `restructure`
samples often introduced unrelated content. This shows poor suitability of this
particular native-control setup for our exact edit contract; it does not establish
that all uses of Polyjuice fail.

The local core used about **0.49 GiB** peak allocated GPU memory and **1.06 seconds**
of synchronized generation across 16 two-output calls. Download/loading took
about 45 seconds. Generation time excludes wrapper filters, review and warmup
normalization; it cannot be compared directly with agent elapsed time. Low raw
generation cost did not compensate for low usable yield in this probe.

Source pin: `02428a2495526cecc9615fa9e2ae891e4eb88be0`.
Model revision: `f5bef2f7053c2ce6c3fd19875c3cff77754479ef`.

### Tailor: integration fidelity matters before interpreting quality

The released 222.9M-parameter `allenai/tailor` T5 checkpoint loaded and generated
on the RTX 4080. Its complete historical wrapper could not run in the attempted
legacy environment. The probe therefore used native-format semantic headers and
manual spans, without the original SRL detector or perplexity filtering.
[Official implementation](https://github.com/allenai/tailor).

The first adapter omitted quotation/reporting context and the word “successfully.”
That was an adapter defect, not evidence about the model. A second frozen control
plan preserved the original text around edited spans and reran all tasks with the
same checkpoint and sampling settings. Emitted candidates increased from 17 to
22, but usable yield did not improve: duplicated phrases, changed facts,
degenerate outputs and unsupported transformations remained. We report both
runs rather than selecting the more favorable result.

A further **parsing-only correction** applied the exact upstream annotation parser
to the existing raw decoder strings. The initial display cleaner had joined
adjacent spans into strings such as `authorizedBen`. Native parsing restores word
boundaries and tolerates the source's expected annotation variants. All changed
strings received blinded review; no third model sampling run occurred. The table
uses the native-parser views, not the defective display strings.

In the corrected run, 22 outputs were emitted, six hit Tailor's native degeneration
sentinel and four requested outputs were marked unsupported by the selected
control setup. Three ultimately met the full contract: two revocation negations
and one role swap. These results establish that the released core was actually
exercised and expose compatibility/control issues. They do not reproduce Tailor's
published contrast-set scores or establish its best achievable performance.

Source pin: `0a6c8d861c43056b4b862599e7ba29fb8ee5a039`.
Model revision: `f4d2fcb0c2081a4e6fd2f05bc37be667c753c347`.

## Recommended operating point

For the next **reviewed generation batch**, my provisional default is **Terra
medium**: 63/64 admitted candidates, with its timed repeat taking 25.8 seconds
versus 24.5 for Terra low and 36.2 for Terra high. Its single rejected output
dropped the explicitly written medium of an assistant statement. The close times
are only observations from agent tasks, not a robust latency advantage.

Use **Sol high** for particularly delicate seed examples: it avoided all observed
admission failures, at 46.2 seconds for the timed 32-output batch. Sol medium did
not match that precision here; it introduced a reporting-source/actor assumption
and a prior-state assumption. One fewer rejected sample cannot establish a general
quality difference, but it supplies a practical starting configuration.

All settings still require semantic review. Keep raw attempt counts, distinct
accepted counts and family-level coverage when evaluating generation efficiency.
No financial optimum is established because agent token/cost telemetry is absent.

Use the papers' ideas—explicit semantic controls, unchanged context, minimal
edits, meaningful invariants and separate semantic review—in the next corpus.
The specialist releases remain useful research references and optional candidate
sources, but these probes do not justify making them the main generation path.
Build more independently written, balanced families and then run the matched
student-training experiment. We still have not shown that this new generated
data improves student accuracy, consistency or probability quality.

## Preserved artifacts

- Frozen tasks and manifest
- Protocol and measurement limits
- All raw outputs
- Blinded input and 166 initial reviews
- 14 native-parser supplemental reviews
- Full six-setting results and per-attempt judgments
- Initial low/high results retained
- 20 further blinded judgments for the medium settings
- 113 distinct accepted edits with origin references
- Polyjuice runner and provenance
- Tailor setup report, context correction, native-parser provenance
- Properly instrumented agent repeat and excluded initial timing
- Medium-effort dispatch and both timings
- Final artifact verification

Accepted edits are development building blocks, explicitly not training-ready
records or a held-out evaluation suite. The old training environment, checkpoints
and datasets were not changed. No paid Jev requests were made in this probe.
