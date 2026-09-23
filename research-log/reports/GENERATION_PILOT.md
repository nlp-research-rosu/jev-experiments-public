# First controlled language-generation pilot

The small generation comparison is complete: **96 candidate requests / 480
judgments**, with all raw text and reviews preserved. **48 Terra requests / 240
judgments**, forming eight complete contrast families, are admitted to the
development corpus. The 48 Luna candidates are quarantined for unmet writing
requirements, not counted as 48 semantic-label failures. No student was trained.

This is eight family specifications and 32 core semantic states, not 96
independently discovered scenarios. It is a feasibility/quality pilot rather than
large-scale dataset construction or a general generator ranking.

**Paper-tool status:** this experiment used prompted Luna/Terra writers and our
own semantic specifications/review scripts. It did not run the released Tailor,
Polyjuice, MiCE or DIPPER generators, nor reproduce CoBA's pipeline. Semantic
control and reviewed contrast families were design inspiration; no result here
establishes the quality of those published implementations or their effect on
our student model.

## What we compared

The same unlabeled semantic facts and predicate contracts went to isolated Luna
and Terra writers, both at high reasoning effort. Each wrote four core states,
a same-writer paraphrase and an appended irrelevant distractor for each family.
Questions cover three Noul judgments, one Choice and one ordinal Score. Full
interpretation policies and fixed criteria accompany the generated record;
the task was to vary record language and accurately reword question instructions.

| Family | Main controlled distinctions |
|---|---|
| Bulletin publication | All four combinations of completed-action claim and confirmed execution |
| Certificate reissue | Permission and execution vary independently |
| Collection reservation | Timeout, explicit no-effect rejection, success, equally authoritative conflict |
| Snapshot restore | Correct versus wrong target, wrong operation, two targets with different outcomes |
| Service-key rotation | Inside, before, after and exactly at the inclusive window boundary |
| Dataset sharing | Historical grant, linked reversal, unrelated reversal, unresolved reversal |
| Delivery-address registry | All four combinations of completed-change claim and verified change |
| Custodian registry | Authentication, stale observation and missing trusted baseline |

The full question definition now distinguishes **a claim that a change already
happened** from a future plan. Both writers receive this explicit definition.
These fixtures do not yet cover every item in the [contrast inventory](../design/contrast-taxonomy.md).

## First-pass results

| Measure | Luna | Terra |
|---|---:|---:|
| Candidate requests | 48 | 48 |
| Blind reviewer labels agreeing with intended labels | 240/240 | 240/240 |
| Ambiguity flags from blind reviewer | 0 | 0 |
| Schema/identity/duplicate/append checks | Pass | Pass |
| Question instructions copied exactly from seed | **240/240** | **0/240** |
| Mean token edit distance along designated close-edit edges | **72.6%** | **12.1%** |
| Mean token edit distance for paraphrase siblings | 78.9% | 65.5% |
| Mean record length | 76.0 words | 53.8 words |
| Records below the suggested 75-word minimum | 22/48 | 47/48 |
| Admitted to this requested development corpus | **0/48** | **48/48** |
| Complete admitted families | 0/8 | 8/8 |

Token distance is token-level Levenshtein distance divided by the longer sequence,
after lowercasing. It is a surface measure: no post-hoc numerical distance cutoff
was used to reject a record, and a large distance alone is not semantic drift.
Length was guidance rather than a semantic acceptance gate; Terra preserved the
specified facts with shorter records, but produced much less prose than requested.

Luna failed the explicit instruction to paraphrase the question instructions;
it copied all of them. Its records also frequently explain the very distinction
being tested. For example, `luna/f01/v1` adds **“This is an explicit past-completion
assertion.”** Other examples explain that a statement is a future plan, that a
target matches, or that a registry comparison is unchanged. Some records omit
the absolute calendar date present in the semantic specification. These findings
and case notes are retained in the compliance and adjudication artifacts.

The primary review therefore quarantines Luna's first pass under the frozen
writing instructions. This is **generation eligibility**, distinct from whether
a model can recover the intended answers. Many Luna records may be salvageable
after removing explanatory hints, restoring scope details, writing actual
question paraphrases and applying constrained edits. No such repair was silently
substituted into the first-pass comparison. One attempt does not establish an
intrinsic limitation of Luna or a general quality advantage for Terra.

Terra follows the close-edit structure much better: changing a promise to a
past-completion statement can leave nearly the entire record untouched, and a
time-scope contrast can change just the event timestamp. Its paraphrase variants
change substantially more wording. Primary review checked every Terra record
against the intended facts and question meanings, including the timeout/conflict,
wrong-target, inclusive-boundary and missing-baseline cases.

## Why blind agreement was insufficient

A fresh reviewer saw 96 randomly ordered anonymous requests, with no writer IDs,
latent specifications or intended labels. Its 480 answers and evidence excerpts
were frozen before comparison. All exact evidence excerpts were verified against
the corresponding record. The primary agent then reviewed all 96 records,
including apparent agreements, against the specification and writing instructions.

The blind reviewer successfully interpreted the generated text, but that task
does not establish that the generator obeyed the construction rules. In
particular, an inserted answer explanation can make label agreement **easier**.
We need separate checks for label preservation, fact fidelity, question wording,
unwanted answer hints and controlled edit size. These are model-based reviews,
not human annotations or proofs of semantic equivalence.

## The admitted corpus

The complete-family export contains **48 requests, 240 judgments and 262 declared
relations**: 55 evidence flips, 160 invariants and 47 same-record question
contrasts. The relation count includes correlated views and predicates; it is not
262 independent scenarios. All relation endpoints and label relationships pass
the existing suite validator.

| Coverage | Count in the admitted 48 requests |
|---|---|
| Completed-action/change claim | 12 true, 36 false |
| Confirmed execution/change | 24 true, 24 false |
| Permission | **45 true, 3 false** |
| Choice | 20 unknown, 15 effective, 3 no_effect, 2 reversed, 6 changed, 2 unchanged |
| Score levels | 13 at level 0, 9 at level 1, 26 at level 2 |
| Primitive judgments | 144 Noul, 48 Choice, 48 Score |

This is not a balanced ready-to-scale training mixture. Permission negatives,
reversal/no-effect outcomes and question-language diversity are particularly
sparse. There is only one new action-question formulation and one new registry
formulation in the admitted writer output, reused across families. The evidence
is still concise synthetic audit prose under shared policies, not a broad corpus
of naturally occurring agent traces. Old and new domains/templates have not been
tested for student transfer in this pilot.

The exported bundles use the existing training input/target format, but are
explicitly marked **development**. All siblings and both writers share their
semantic-family group so future splits cannot scatter related variants across
train and test. No record from this inspected pilot should later be presented as
an untouched outer test.

## Recommendation

Use Terra as the provisional close-edit writer for the next small batch, while
testing independently written question formulations in a separate pass. Add
automatic exact-copy checks before review, and explicitly audit answer-explaining
sentences and missing scope details. Preserve both raw and rejected candidates.

Before scaling, extend the accepted pattern into more independent families and
discourse styles, balance false permission and other rare outcomes, and inspect
writer diversity at the semantic-family level. The next training comparison
should hold the training budget and primitive mixture fixed and test whether
these language variations improve an independently written outer suite. This
pilot establishes a workable data format and exposes generation failures; it
does not yet establish better student accuracy or probability calibration.

## Artifacts

- Frozen protocol
- Unlabeled writer specification
- Intended semantic labels
- Writer instructions
- Raw Luna candidates
- Raw Terra candidates
- Blind review packet
- Frozen blind answers and exact evidence
- Blind comparison
- Writing compliance
- Primary adjudication
- Accepted complete families and relations
- Development bundles
- Quarantined candidates and reasons
- Machine-readable results
- Dispatch/instruction provenance
- Schema, targets, loss and artifact verification

The artifact scripts in the report directory are one-off research scaffolding,
not production pipeline changes. Raw hashes, review hashes and accepted-artifact
hashes are recorded. Agent token usage and monetary cost were not exposed, so
no generation-cost estimate is asserted. No paid Jev calls were needed for this
generation-quality comparison; its existing response archive remains unchanged.
