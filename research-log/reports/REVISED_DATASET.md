# Revised corpus and frozen initialization comparison

This revision addresses two failures from the first contrast pilot: representation
brittleness and loss of broad Score performance. It was reviewed and frozen before
new model outputs. It is a small synthetic study, not a general production benchmark.

| Split | Domain families | Underlying scenarios | Flat/nested/prose-framed requests | Judgments |
|---|---:|---:|---:|---:|
| Train | 14 | 220 | 660 | 3,300 |
| Validation | 3 | 48 | 144 | 720 |
| Test | 5 | 79 | 237 | 1,185 |

Every request has three Noul questions, one Choice and one Score. The action and
registry generators share reviewed templates across splits, but domains stay
separate, all test question wordings are absent from training, and all three
representations appear in every split. Prose is prose-framed evidence with JSON
event/policy material, not an independently authored natural-language corpus.
No exact request is duplicated within a split, and train/test domains do not overlap.

## What is supervised

Action cases distinguish claims from authenticated in-window completion, current
effect from historical completion, scoped calls from wrong-target/operation/time
events, and definite no-effect outcomes from pending, unverified or conflicting
evidence. An older success alone never establishes this scoped operation's
current effect, even if its physical effect might persist. Timestamps and
equal-authority conflicts have explicit fixture semantics.

Action Score is ordered execution evidence: no matching in-window invocation;
matching invocation without established real completion; established real
completion, including completion later reversed. These are levels of a declared
rubric, not a probability or a generic “confidence” score.

Registry cases separate explicit claims, verified baseline/current comparison,
and policy approval. Missing, unverified, conflicting and out-of-scope records
produce Unknown where the Choice contract requires it. Registry Score is a
sequential verification stage: no trusted baseline; trusted baseline without a
usable current consensus; trusted baseline plus unambiguous authenticated current
consensus. It is not a count of independent checks.

All labels are literal reviewed targets. Jev does not assign or overwrite them.
Score descriptions stand alone without gratuitous numeric index prefixes; the
software retains their order and computes probability-weighted means.

## Review findings resolved before freeze

- A registry intent-only case had a trusted baseline but originally received the
  wrong verification stage; its literal target was corrected to match the rubric.
- A wrong-target event's result text still named the requested target; the text
  was corrected to identify the actual other target.
- An unchanged-claim variant now contains an explicit no-change assertion, removing
  a duplicate request.
- Prose-framed versions preserve the complete policy, action description and
  evidence contract. Conflict and timestamp conventions are explicit.

Positive and negative controls, all four claim/confirmation combinations, Score
targets, layout preservation, split boundaries and relation endpoints were checked
independently before inference. Gold labels, IDs, rationales and relation metadata
are evaluator-only; only state and question definitions reach either model.

## Training weight rather than row-count balance

The corpus has more Noul judgments per scenario, so simply sampling scenario rows
would recreate the previous imbalance. Each update instead averages six complete
question losses:

| Primitive | Original broad source | Revised corpus | Total loss weight |
|---|---:|---:|---:|
| Noul | 1/6 | 1/6 | 1/3 |
| Choice | 1/6 | 1/6 | 1/3 |
| Score | 1/6 | 1/6 | 1/3 |

Candidate rows remain grouped within each question. Both arms use exactly the same
schedule and tokenized questions. The broad pool contains 800 original **training**
records from each of six sources: 4,800 canonical questions. Validation and test
records are excluded. The revised pool contains all 3,300 new training judgments.
Sampling cycles without replacement within each primitive/source stream.

Fresh initialization uses the original Qwen3.5-2B checkpoint, new LoRA and initial
numerical heads. Continued initialization uses our original v0.2 checkpoint and
a new optimizer. Both use 1,000 additional updates, adapter/head learning rates
5e-5/2.5e-5 and the same supervised probability losses. The continuation arm's
earlier 5,000 updates remain an explicit historical advantage; this is a practical
additional-budget comparison, not equal lifetime compute.

Validation is recorded at steps 0, 250, 500 and 1,000. Checkpoint selection uses
only the equal-weighted new/broad macro-primitive NLL, with earlier checkpoints
breaking ties. The full protocol defines separate positive/negative completion,
Unknown, pair, layout, broad accuracy, Score accuracy/MAE and probability-loss
gates. Test scores cannot choose checkpoints. Jev test observations are archived
separately and remain outside training.

## Files

- Frozen protocol
- Training suite and manifest
- Validation suite
- Test suite
- Actual training populations
- Both-initialization smoke checks

Both GPU smoke runs passed adapter/head gradient, frozen-base and save/reload
checks. Reloaded probabilities match exactly. The longest sampled training unit
is 700 tokens, below the explicit 1,536-token ceiling; no sampled record was
excluded or silently truncated. These checks establish pipeline readiness, not
trained quality. See the [completed comparison](REVISED_INITIALIZATION.md) for
final quality results and the remaining failed gates.

## Wording limitation identified after selection

The Jev disagreement audit found that the held-out registry question “Is an explicit
change assertion present in the assistant text?” omits the completed/past-change
qualifier intended by the label definition. A future promise can plausibly qualify
as a change assertion. This affects six intent-only judgments (two underlying test
scenarios across three layouts). The pre-freeze audit missed this ambiguity.
The frozen labels and reported scores remain unchanged; these judgments must not
be presented as clean evidence of a model confusing intent with completed change.
Any wording correction belongs in a separately versioned future corpus.
