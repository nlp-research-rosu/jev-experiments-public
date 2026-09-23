# Contrast inventory for the next data-generation study

Compiled 2026-09-20 from project artifacts and the discussions supplied in the
task. This is a consolidated inventory, not a new dataset or a claim that every
capability is solved. Existing partial tables are in
[contrast diagnosis](../reports/CONTRAST_DIAGNOSIS.md),
[training consistency semantics](training-contract-v0.2.md), and
[the original data audit](../reports/CONSISTENCY_DATA_AUDIT.md).

## The transformations must be distinguished

| Relation | What changes | What should happen | Example |
|---|---|---|---|
| Evidence contrast | A relevant fact; question held fixed | The appropriate label changes | Refund result succeeded → explicitly failed |
| Question contrast | Predicate; evidence held fixed | Answers differ when the record warrants it | “Claimed completion?” yes; “Completion confirmed?” no |
| Meaning-preserving variant | Expression, layout, or an irrelevant fact | Correct answer remains; measure probability drift separately | Active → passive voice with roles preserved |
| Exact complement | Negate the complete binary proposition | Complementary labels and probabilities | “The record establishes success” ↔ “The record does not establish success” |
| Directional/ordinal contrast | A rubric-relevant degree or evidence stage | Specified ordered movement, not necessarily a binary flip | No invocation → invoked → verified execution |

“Similar words” and “similar meaning” are different constraints. We need high
word-overlap examples whose labels differ **and** diverse wording whose labels
stay the same. Neither word overlap nor embedding similarity certifies the label.
Claim and confirmation are independent predicates, not exact complements.

## Semantic distinctions identified so far

Status: **trained/tested** means some authored examples entered a completed local
training run and an evaluation; it does not mean broad mastery. **Tested** means
explicit diagnostic cases exist. **Proposed** means discussed or found in a
reference, without a dedicated local contrast-family evaluation. Several rows
describe overlapping dimensions of the same cases; do not add their case counts.

| ID | Distinction | Controlled example / expected relation | Local coverage and remaining gap | Evidence |
|---|---|---|---|---|
| S01 | Request, intention, proposal, invocation | Request to refund ≠ promise to refund ≠ recorded tool invocation | Tested in the original suite; execution-stage supervision in revised training. Intention has no separate output in the revised suite. | [Original suite](../../data/semantic-contrasts-v1/suite.json), [revised generator](../../experiments/contrast_revision_data.py) |
| S02 | Invocation, pending result, explicit failure, real completion | Same call; result pending / no-effect failure / authenticated success | Trained/tested. A call alone does not prove success; timeout does not prove physical failure. | [Revised corpus](../reports/REVISED_DATASET.md) |
| S03 | Claim versus verified completion | “I completed X”; no execution evidence: claimed=yes, confirmed=no | Trained/tested with all four claim/confirmation combinations. Continuation still fails three claim/confirmation pairs in the latest outer test. | [Latest results](../reports/REVISED_INITIALIZATION.md), [original diagnostic](../reports/SEMANTIC_CONTRASTS.md) |
| S04 | Claim of completed change versus future plan or explicit denial | “I changed X” / “I will change X” / “X did not change” | Trained/tested, but six revised-test registry labels have ambiguous “change assertion” wording. Define completed-change assertions explicitly in the next version. | [Wording limitation](../reports/REVISED_DATASET.md) |
| S05 | Statement attribution and quotation | Quoting another speaker's assertion ≠ adopting it as one's own assertion | Explicit attributed/adopted-quote cases in the original diagnostic; no dedicated independent generation family yet. | [Bank cases](../../experiments/contrast_bank_cases.py) |
| S06 | Permission versus execution | Approved but never executed; executed despite denial | Tested in action and registry domains; revised training has registry permission targets and unauthorized-action examples, but no action-permission output. | [Diagnosis](../reports/CONTRAST_DIAGNOSIS.md), [revised generator](../../experiments/contrast_revision_data.py) |
| S07 | Claim versus verified registry fact | Sender claims a change while trusted records are unchanged, and the converse | Trained/tested; require all four claim/fact combinations. | [Original diagnostic](../reports/SEMANTIC_CONTRASTS.md) |
| S08 | Requested destination versus asserted/verified change versus operational rule | A request uses a different destination despite denying a change; a defined rule may still trigger | Tested with `different_destination`, `explicit_change_claim`, `record_confirms_change`, `operational_change_signal`. Revised registry tasks cover a narrower subset. | [Bank generator](../../experiments/contrast_bank_cases.py) |
| S09 | Positive evidence, contrary evidence, absent evidence | Success record / explicit no-effect failure / missing or timed-out result | Trained/tested with explicit Unknown choices. “No proof of X” is not “proof of not-X.” Binary labels concern the specified evidence predicate, not hidden world truth. | [Diagnosis](../reports/CONTRAST_DIAGNOSIS.md) |
| S10 | Evidence conflict versus simple absence | Equally authoritative success/failure records versus no result | Trained/tested with conflict mapped to Unknown under stated rules. A separate four-way conflict label remains proposed. | [Revised evidence contracts](../../experiments/contrast_revision_data.py) |
| S11 | Trusted/authenticated source versus unverified assertion | Identical apparent value from an authenticated registry versus an unverified message | Trained/tested. Source authority must be part of the supplied contract; avoid treating any particular tool name as universally authoritative. | [Revised corpus](../reports/REVISED_DATASET.md) |
| S12 | Correct entity/target/record binding | Successful action for ticket A does not confirm action for ticket B | Trained/tested with wrong-target and wrong-registry cases. Several targets with mixed outcomes in one record remain a proposed expansion. | [Original suite](../../data/semantic-contrasts-v1/suite.json), [revised corpus](../reports/REVISED_DATASET.md) |
| S13 | Correct operation binding | Inspect/prepare/validate succeeds; issue/delete/update is not thereby confirmed | Wrong-operation contrasts trained/tested. More subtle operation pairs and role-swaps need independent examples. | [Latest results](../reports/REVISED_INITIALIZATION.md) |
| S14 | Time window and snapshot eligibility | Same successful record inside the requested interval versus older/future; current versus stale snapshot | Trained/tested under explicit timestamp scope. Whether old effects persist depends on the question; do not impose a universal stale=false rule. | [Revised contracts](../../experiments/contrast_revision_data.py), [Jev disagreements](../reports/REVISED_INITIALIZATION.md) |
| S15 | Historical completion versus current effect and reversal | Grant then revoke: grant occurred=yes; access remains established=no | Trained/tested, including unresolved and unrelated reversals. General event histories are still narrow. | [Revised corpus](../reports/REVISED_DATASET.md) |
| S16 | Real execution versus dry run/simulation | Same apparent successful operation in real mode versus “would execute” | Trained/tested; simulation can establish invocation without a real effect under the explicit fixture contract. | [Original suite](../../data/semantic-contrasts-v1/suite.json) |
| S17 | Necessary prerequisites and evidence sufficiency | Trusted baseline absent / baseline only / baseline plus eligible current evidence | Revised Score trained/tested as a sequential rubric. Removing one of multiple independently necessary evidence sentences is an upstream Nimble technique, not yet our dedicated local family. | [Revised corpus](../reports/REVISED_DATASET.md), [Nimble audit](../reports/REFERENCE_STRATEGIES.md) |
| S18 | Rubric-defined degree and boundary | Adjacent Score levels; greater evidence changes a declared stage but may not change permission | Original wine/numeric-policy and revised execution/verification Score trained/tested. Broad descriptive rubrics and targeted near-threshold pairs remain limited. | [Data preparation](../reports/DATA_PREPARATION.md), [latest results](../reports/REVISED_INITIALIZATION.md) |
| S19 | Same facts, different decision rule | Same request is urgent under a today-deadline rule but not an outages-only rule | Proposed general-schema test; supplied policies exist locally, but no dedicated matched rule-change study has established transfer. | [General direction](general-training-direction.md) |
| S20 | Logical scope/composition | AND/OR/NOT/UNLESS; change one prerequisite or exception | Discussed; Kev has executable compositional reference code. Dedicated local held-out rule-structure contrast training is not implemented. | [Community review](../reports/REFERENCE_STRATEGIES.md) |
| S21 | Multi-entity/multi-event interaction, partial completion, corrections | A succeeds, B fails, C times out; “both completed?” differs from “any completed?” | Proposed expansion. Existing single-focus wrong-target/reversal cases do not establish general compositional binding or quantifier handling. | Supplied discussions; [diagnostic limits](../reports/CONTRAST_DIAGNOSIS.md) |

## Invariance and logical consistency already considered

| ID | Relation | Required guard | Local status / evidence |
|---|---|---|---|
| I01 | Equivalent question/criterion wording | Preserve negation, tense, actor, scope and evidence threshold | Hand-authored paraphrases tested; broad independently generated valid paraphrases are missing. [Original suite](../reports/SEMANTIC_CONTRASTS.md) |
| I02 | Equivalent evidence wording | Preserve every relevant fact and attribution | Hand-authored paraphrases tested. No independent prose-generation corpus yet. [Original suite](../reports/SEMANTIC_CONTRASTS.md) |
| I03 | Irrelevant distractor insertion/removal | Irrelevant to this particular question, not merely to the scenario theme | Tested, including unrelated reversal. Evidence filtering was a diagnosis, not proof all removed text is generally irrelevant. [Diagnosis](../reports/CONTRAST_DIAGNOSIS.md) |
| I04 | Flat/nested/prose-framed serialization | Same facts and complete contract must survive rendering | Trained/tested in all revised splits. Three renderings are one underlying scenario, not three independent facts. [Revised corpus](../reports/REVISED_DATASET.md) |
| I05 | Sentence-pair symmetry | Only for a symmetric task such as equivalence, not directional entailment | PAWS sentence swapping trained with 4,000 invariant relations in original data. [Preparation](../reports/DATA_PREPARATION.md) |
| I06 | Choice ordering, field ordering, aliases and score-level reversal | Align outcomes by meaning; changed candidate sets are a different task; arbitrary level permutation destroys ordinality | Ordering tested in early inference work; schema handling specified/tested in v0.2. No blanket arbitrary-renaming or event-order invariance. [Contract](training-contract-v0.2.md), [early investigation](../reports/INVESTIGATION.md) |
| L01 | Exact binary complement | Negate the entire judged proposition; do not replace Unknown by False or treat claim/confirmation as complements | 8,000 complement relations in original training; supervised plus complement consistency supported. Revised pilot uses single-question hard losses, without that auxiliary term. [Preparation](../reports/DATA_PREPARATION.md), [loss](../../src/openjev/judgment_training.py) |
| L02 | Implication/exclusivity relationships | Apply only when declared semantics actually entail the relation | General logical constraint graphs identified as a gap. Soft consistency alone can reward consistently wrong predictions. [Consistency design](training-contract-v0.2.md), [data audit](../reports/CONSISTENCY_DATA_AUDIT.md) |
| L03 | Ordered probability response | Stage/degree changes may imply direction, not calibrated target values | Score distributions and means are implemented; special ordinal/contrast losses were not tested in the revised pilot. [Reference strategies](../reports/REFERENCE_STRATEGIES.md) |

## Existing evidence and priority

The original diagnostic has 114 cases, 764 judgments and 260 relations (90 evidence
flips, 42 question contrasts, 128 invariants). The revised training corpus has
220 scenarios / 660 renderings / 3,300 judgments. Its training suite declares
80 flips, 20 question contrasts, 64 semantic invariants and 2,200 layout invariants;
these are evaluation metadata, not all extra loss terms. The revised outer test
has 79 scenarios / 237 requests / 1,185 judgments and only **seven** question
contrast pairs. Thus a high aggregate score does not mean broad question-contrast
coverage.

Priority for the next corpus is S03/S04 with exact predicate definitions, varied
independent expressions for I01/I02, and stronger multi-record/role/time interactions
(S12–S15/S21). Preserve S09 Unknown, positive controls, all three primitives and
original broad-task retention. The existing train/validation/test artifacts stay
frozen; any corrected wording receives a new dataset version.

## Generation record to aim for

For each family retain: semantic facts and evidence policy; the changed atomic
fact or changed predicate; unchanged facts; natural-language realizations and
their evidence spans; complete per-question labels; relation type; generator and
reviewer provenance; review disagreements; and a family ID binding every sibling
to one split. A latent-state label remains valid only if the realized text still
expresses that state. A generator can omit a necessary qualifier or add a claim;
format validation cannot catch that by itself.

See [generation research and candidate tools](../reports/CONTRAST_GENERATION_RESEARCH.md)
for primary papers, released models, mapping to these contrasts, and a proposed
small generation comparison.

The [first generation pilot](../reports/GENERATION_PILOT.md) now supplies an
audited development batch spanning eight of these family specifications,
including a two-target binding case. This adds generated examples, not evidence
that student training has solved the listed capabilities.
