# Comparable scores and calibrated probabilities

Research/design note, 2026-09-19. The user asked to resolve probability semantics
before adopting the draft criterion-scoring contract. No training or model
implementation was performed for this review. Mathematical examples below are
illustrations, not measurements of Qwen or Jev.

## Main finding

Independent forward evaluations are compatible with comparable scores, but only
when their meaning and training objective make them comparable. Sharing a model
and readout is useful; it is not, by itself, a guarantee. A normalized distribution
is not necessarily empirically calibrated. These are separate questions:

1. Are candidate scores on a useful common scale within one decision?
2. Does their transformation define the intended probability distribution?
3. Do those probabilities agree with outcome frequencies on the deployment task?

The current v0.1 draft already proposes a grouped categorical loss. That is a
defensible way to address the first two questions, not an automatic answer to the
third. Its independent-utility assumption, universal Noul treatment and scalar
readout must remain proposals until tested.

## 1. Specify the event before the number

| Primitive | Intended probability semantics | Not interchangeable with |
|---|---|---|
| Choice | Distribution over the answer selected under the supplied question, candidate set and annotation policy | Independent probabilities that each description is true |
| Noul | Probability that one defined proposition is true, given the supplied evidence and criteria | Relevance of an option relative to a changing candidate list |
| Score | Distribution over the levels of a particular rubric | A universally comparable measure of severity, skill or uncertainty |

Choice requires an explicit rule for ambiguous, overlapping or missing answers.
Independent attributes should be multiple Noul questions, not falsely exclusive
classes. A `none`/`other` category is part of the task definition, not an automatic
fix for unsupported inputs. Class probability is conditional on the modeled
answer space; forcing one candidate cannot establish that the candidate is right.

An NLI label such as `neutral` should not be converted into falsehood of the
hypothesis. It can be a negative answer to the precisely different question
"Does the passage establish the hypothesis?" Data conversion must preserve this
distinction. Likewise, absence of evidence is not universally a 0.5 target.

## 2. How separate candidate evaluations can be trained together

Let x contain state and instructions, and c_i be a candidate description. A
shared scorer can produce:

```text
z_i = f_theta(x, c_i)
p_i = exp(z_i) / sum_j exp(z_j)
loss = -log(p_correct)
```

Although the model reads candidates in separate rows, the loss uses the full
candidate group. At temperature one its derivative with respect to a candidate
logit is `p_i - y_i`. Thus all candidates receive coupled supervision in a common
competition. This does not require the forward input for one candidate to contain
its neighbours. It does require retaining groups, labels and normalization during
training; separately requesting arbitrary confidence numbers is not equivalent.

A shared, separately applied scoring network with a list-level objective has a
clear precedent in **ListNet**: [Cao et al., 2007, Learning to Rank: From Pairwise
Approach to Listwise Approach](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/).
Its exponential top-one distribution and list-level cross-entropy are relevant
architecture/training precedents. Its retrieval-ranking results are not evidence
that our probabilities will be calibrated or that Jev uses that method.

For our classifier, the target is a known categorical label or a sourced label
distribution, not ListNet's transformed relevance grades. Log loss and Brier loss
are proper probability losses. In the ideal expected-loss setting, their optimum
is the true distribution when it is representable. Finite data, an unsuitable
model family, optimization and distribution shift can prevent reaching that
optimum. [Gneiting and Raftery, 2007, Strictly Proper Scoring Rules, Prediction,
and Estimation](https://doi.org/10.1198/016214506000001437)

Raw logits remain relative. Adding a question-dependent constant to every
candidate score leaves its softmax unchanged. A score of 8 on one request is
therefore not intrinsically more certain than a score of 3 on another request.
Their probability distributions, evaluated for calibration, are the relevant
objects to compare.

## 3. Normalization alone does not determine uncertainty

Both `[1, 0]` and `[10, 0]` select the first option. Their softmax winning
probabilities are approximately 0.7311 and 0.99995. The winner is identical, but
the uncertainty is radically different. Softmax supplies arithmetic, not evidence
for which scale is appropriate.

Even one ordinary LLM forward pass does not solve this automatically: next-token
likelihood depends on answer wording and prompt construction. Prompt-format and
answer-position biases have been documented by
[Zhao et al., 2021, Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c.html).
Their contextual calibration addresses prompt bias; it should not be conflated
with proof that emitted probabilities equal semantic correctness frequencies.

For a trained classifier, held-out temperature scaling is a reasonable baseline.
A positive global temperature changes sharpness without changing the winner. It
cannot repair a wrong ranking or compensate for every wording/domain bias.
[Guo et al., 2017, On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)

Calibration measured on familiar data can deteriorate under a new input
distribution. That is particularly important for our target of unfamiliar schemas
and domains; one fitted temperature is not a universal guarantee.
[Ovadia et al., 2019, Can You Trust Your Model's Uncertainty?](https://papers.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)

## 4. A structural limitation of isolated scores plus softmax

If z_A and z_B do not depend on the other candidates, then:

```text
p_A / p_B = exp(z_A - z_B)
```

Adding another option changes the denominator but cannot change these odds. This
is the familiar independence-of-irrelevant-alternatives restriction of the Luce /
multinomial-logit family. See
[Seshadri and Ugander, Fundamental Limits of Testing the Independence of
Irrelevant Alternatives in Discrete Choice](https://arxiv.org/abs/2001.07042),
especially its choice-system definition.

This can be appropriate for fixed, mutually exclusive events whose relative
probabilities are independent of which alternatives are displayed. It is not a
universal model of "best among these descriptions," where overlaps, near-duplicates
or relative wording may matter. If two equal-score alternatives are replaced by
three with one semantic duplicate, plain softmax gives each one third; the two
similar answers together receive two thirds. Whether that is wrong depends on the
target event and annotation policy, but it exposes a constraint we must test.

An isolated `none of the above` utility can act as a competing default. It cannot
be interpreted as an independently computed truth probability for the proposition
"none of this changing list applies" when it cannot see that list.

TypeSafe explicitly documents Score level isolation. It does not disclose that
its normalizer is our scalar softmax, or mandate isolated Choice options on the
pages reviewed. We should keep a Choice baseline that sees the complete candidate
set, rather than treating independent Choice utilities as an established fact.

## 5. Noul: do not mix Bernoulli logits and categorical utilities

A single binary logit has a direct interpretation:

```text
z = binary_model(state, proposition, true_rubric, false_rubric)
p_true = sigmoid(z)
p_false = 1 - p_true
loss = -y*log(p_true) - (1-y)*log(1-p_true)
```

This is a proposed simpler Noul baseline, consistent with its documented
yes/no output contract. Its true/false criteria may be in one input; TypeSafe's
Score isolation requirement does not forbid that for Noul.

A useful counterexample: suppose separate evaluations yield genuine Bernoulli
log-odds for P(true)=0.8 and P(false)=0.2, namely `log(4)` and `-log(4)`.
Applying another two-way softmax to those logits produces **0.941176**, not 0.8.
Softmax of `log(0.8)` and `log(0.2)` would preserve 0.8 instead. The distinction
between log probabilities, binary log-odds and relative utilities matters.

This does not invalidate a two-criterion model trained end-to-end with the grouped
loss: its logits would be learned as categorical utilities, not standalone
Bernoulli logits. It does mean the v0.1 A/B-difference initialization cannot be
treated as an already justified probability scale. Direct binary Noul should be
compared with that two-unit variant; one common interface should not be forced
merely for implementation symmetry.

## 6. Score: comparable levels, a declared numerical scale

Define the rubric's target outcome first and train its categorical distribution
with a grouped proper loss. The network can still evaluate level descriptions
separately. Then compute `sum_i i*p_i` in software, as the API specifies.

That mean is meaningful relative to the declared index scale. Adjacent indices
have equal arithmetic spacing; their natural-language meanings do not establish
equal real-world differences. Normalizing a severity scale and a satisfaction
scale to [0,1] does not turn them into the same quantity.

The distributions `[0,1,0]` and `[0.5,0,0.5]` both have mean 1. The first is certain
about the middle level; the second is split between extremes. Do not train only
the mean and then invent a distribution or confidence. Retain the full probability
vector. A ranked probability score, based on cumulative probabilities, can be an
additional ordinal evaluation measure; its order comes from the rubric metadata,
not from exposing neighbouring levels in each neural input.

## 7. Concrete recommendation before the full training run

Freeze the outcome meanings above first. Keep complete question groups and
provenance in the dataset so the same labels can support competing encodings.
Treat the following as a proposed pilot, not a training job launched by this note:

| Primitive | Main probability candidate | Required comparator |
|---|---|---|
| Choice | Shared isolated scorer with grouped categorical log loss | Whole-candidate-set scorer using the same labels and evaluation sets |
| Noul | One binary judgment with binary log loss | v0.1 two-criterion categorical model |
| Score | Isolated descriptive levels with grouped categorical log loss | Current whole-rubric selector as a diagnostic control, not a proposed replacement for documented isolation |

Use the same initialization budget, dataset partitions and comparable tuning
budget. Include the untrained baseline. Freeze model-selection criteria before
seeing the final test outcomes. Fit post-hoc calibration on its own partition,
then report raw and calibrated predictions on untouched tests.

Measure accuracy alongside log loss, Brier score, reliability plots and
error/coverage when abstaining. For Noul, assess P(true), not only confidence in
the winning binary answer. For Score, assess the distribution and mean/ranking
under the same rubric. A low ECE alone is insufficient: a constant prior can be
calibrated yet unhelpful, and global averages can hide bad task-specific behavior.

Stratify by task, primitive, candidate count and seen/unseen schema. Test changes
to candidate wording, candidate sets, overlapping options, missing answers and
base rates. Sampling changes class prevalences; a calibrator fitted on an
artificially balanced mixture is not automatically correct for another mixture.
Keep all expansions of a source context together and use grouped uncertainty
estimates rather than pretending that its candidate rows are independent samples.

Software's derived `confidence` statistic is a separate API quantity. Evaluating
calibration of the probability vector does not license interpreting that statistic
as P(correct). No result in these papers establishes Jev's exact training method.

## Decision

The useful part of v0.1 is preserved: explicit model/software boundaries and
complete-group supervision. Its probability interpretation, universal two-unit
Noul and isolated-Choice default are **under review**. Do not freeze the final
renderer/readout or convert a large corpus until this probability design is
agreed and a bounded pilot is specified.
