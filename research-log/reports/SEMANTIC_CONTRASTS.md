# Semantic contrasts: first frozen diagnostic

The user's proposed distinctions are now explicit tests. The first run shows a
mixed result: fine-tuning improves many individual decisions and evidence-change
contrasts, but remains poor at changing answers when the *question's meaning*
changes. It also makes errors much more confident. Correct negative predictions
from the base model did not consistently reflect a robust underlying capability:
several base predicates simply answered no on every positive and negative case.

## Test design

Frozen before inference: **114 scenarios, 764 judgments, 260 relations**, grouped
into six families. The four execution domains are support escalation, refunds,
appointment booking and access revocation. The two destination domains are payment
bank details and shipping contacts. These are authored, explicit synthetic
contracts, not independently sampled production cases or a general benchmark.

- **90 evidence changes:** the same question should change answer when relevant
  evidence changes. Examples include planned versus invoked, pending versus
  completed, live versus simulated, right versus wrong target, authenticated versus
  unverified records, and matching versus unrelated reversals.
- **42 question changes:** hold the record fixed while asking different predicates,
  such as claimed versus confirmed, confirmed versus permitted, or explicit assertion
  versus the broader operational rule. These predicates are not complements.
- **128 invariance checks:** an irrelevant detail, equivalent wording, or a change
  irrelevant to the particular predicate should preserve its label. A predeclared
  total-variation tolerance of 0.05 measures probability drift separately.

All four combinations of claimed/not-claimed and confirmed/not-confirmed occur.
Evidence absence is not labeled physical failure: binary questions ask whether the
record establishes a fact, and categorical questions distinguish confirmed outcomes
from not-established/Unknown. Definite no-effect rejection, uncertain timeout,
simulation and reversal are separate conditions. No arbitrary 0.5 probability gold
was invented. Permission and execution are judged independently.

The methodology follows behavioral testing and semantic perturbation principles
from [CheckList](https://aclanthology.org/2020.acl-main.442/) and
[Contrast Sets](https://aclanthology.org/2020.findings-emnlp.117/). This is our own
fixture set and evaluator, not a reproduction of those papers' datasets.

## Results

Both models used the same numerical interface, independent full-input scoring,
unmerged BF16 weights, FP32 readouts, FLA and batches of four. There was no training,
calibration fitting or prompt tuning during this experiment. The untrained control
is the initial A-minus-B readout interface, not an ordinary chat evaluation of all
capabilities in the base Qwen model.
The longest scoring input was 690 tokens, within the first training run's retained
maximum of 776; these failures do not require a long-context stress condition.

| Metric | Before fine-tuning | Current trained model |
|---|---:|---:|
| Individual judgments correct | 423/764 (55.4%) | 463/764 (60.6%) |
| Complete scenarios, every field correct | 0/114 | 10/114 |
| Evidence-change pairs, both answers correct | 5/90 (5.6%) | 26/90 (28.9%) |
| Same-record question contrasts, both correct | 8/42 (19.0%) | 6/42 (14.3%) |
| Invariance pairs, both answers correct | 68/128 (53.1%) | 100/128 (78.1%) |
| Invariance pairs within 0.05 probability-distance tolerance | 118/128 | 87/128 |
| Mean NLL, lower is better | 0.801 | 0.973 |
| Mean categorical Brier score, lower is better | 0.548 | 0.584 |
| Wrong predictions assigned at least 95% probability | 0 | 100 |

The individual total combines uneven class mixes and correlated template variants;
it should not be read as population accuracy. Per-predicate class counts and
balanced binary accuracy are in the raw report. Brier is the sum over the full
distribution, so the binary value is twice the single-probability convention.

Training fixes 208 individual errors and introduces 168. Among pairs that the base
gets entirely right, 47 are retained and 34 regress. Eighty-five formerly failed
pairs become entirely correct, and 94 fail under both models. These counts concern
the defined cases and relations, not independent statistical samples.

## What the contrasts reveal

### A negative answer alone is weak evidence of retained skill

For `claimed_done`, the base says no on **all 80 action scenarios**. That yields
85% ordinary accuracy because only 12 cases contain a completion assertion, but
only 50% balanced binary accuracy. It also says no on every one of the 34
`record_confirms_change` cases. Consequently, its correct answer to a missing-
handoff example does not by itself show that it can distinguish missing evidence
from a real successful handoff. The positive control is essential.

The copied conversation's proposed causal story—learning semantic association
instead of evidential support—is a hypothesis. Our original BoolQ/PAWS Noul data
already contained balanced positive and negative originals plus constructed
complements. These results identify behavior gaps, not which particular training
examples, readout changes or adapter updates caused them.

### The trained model often gives affirmative answers to distinct propositions

In the support scenario with an unsupported completion assertion:

| Question about the same record | Gold | Base P(yes) | Trained P(yes) |
|---|---|---:|---:|
| Did the assistant claim completion? | yes | 0.446 | 0.967 |
| Does an operational result confirm completion? | no | 0.378 | 0.965 |

Training learns to recognize the assertion but still treats confirmation almost
the same way. In a promise-only support case, trained P(confirmed completion) is
0.960. A simulation with explicit no-effect output produces 0.954, and success for
the wrong subject produces 0.983. Real matching success produces 0.991. The model
responds to evidence changes, but the affirmative bias makes many absolute answers
wrong.

The same pattern occurs when permission is withheld: the action really completed,
so confirmation should be yes, while permission should be no. The trained model
assigns approximately 0.990 and 0.962 respectively. Successful execution does not
establish authorization under the supplied contract.

### Claim, registry fact and permission need separate supervision

In a payment case, a sender explicitly claims new details, but the authenticated
current registry still contains the baseline. Trained P(explicit claim) is 0.865,
which is appropriate, while P(registry confirms change) is also 0.737, which is not.
In the converse case—confirmed registry change, but no explicit assertion—the
model still assigns 0.867 to an explicit claim. All four truth combinations are
included precisely to prevent an accidental “one is the opposite of the other”
shortcut.

The broad operational rule is a different predicate: a changed requested
destination can trigger it even when the sender denies a change. The suite tests
that distinction directly rather than imposing a universal rule for the phrase
“bank details changed.”

### Better ordering does not guarantee correct probabilities or decisions

The trained model moves probabilities in the right direction on **82/90** evidence
contrasts, compared with 69/90 before training, but only gets both endpoint labels
right on 26/90. Likewise, direction improves on question contrasts from 22/42 to
33/42 while both-correct falls from 8/42 to 6/42. This is evidence of both improved
sensitivity and inadequate absolute decision behavior. It does not establish that
all failures are representational, nor that calibration alone would fix them.

Directional checks use stored probabilities and require movement toward both
expected classes for categorical flips. Saturation can theoretically hide a
strict ordering; no binary probability was exactly 0 or 1 in this run. A model can
rank a positive above a negative while labeling both yes, so directional success
is deliberately reported separately from pair correctness.

Invariance has the opposite trap: a model can consistently return the same wrong
answer. The base's high tolerance-pass count is partly compatible with its hesitant,
often nearly constant predictions. The trained model gets more invariant pairs
correct but exhibits greater probability drift. Neither statistic substitutes for
the other. The 0.05 tolerance is an explicit diagnostic choice, not a universal
mathematical requirement for semantic equivalence.

## Implications for the next training experiment

Use this suite as a behavioral regression check, not as the next training set.
Generate separate training scenarios and reserve new domains/wordings for a fresh
transfer test. Include the full claim/evidence truth table, wrong-target and
wrong-operation records, uncertain outcomes, policy changes without outcome
changes, and contrastive question definitions.

Evaluate jointly: positive and negative accuracy, both-members-correct contrast
scores, probability loss, high-probability errors, and retention of prior correct
pairs. Also preserve the existing broad task and descriptive-Score evaluations.
Do not optimize only the new suite or reward consistency without correctness.
The cause of the regression would require controlled data/readout/adapter
ablations; neither this diagnostic nor one Jev example reveals a private training
method or proves that a handful of new examples will be sufficient.

## Artifacts and verification

- [Frozen suite and rationales](../../data/semantic-contrasts-v1/suite.json)
- [Freeze manifest](../../data/semantic-contrasts-v1/manifest.json), including the pre-inference hash and metric policy
- [Concrete scenario examples with before/after answers](semantic-contrasts-v1/examples.md)
- Run report, changed judgments, and per-model prediction/relation files in that directory

The full repository suite passed **240 tests and 45 subtests**. Labels and relation
semantics were independently reviewed before inference; the evaluator was reviewed
separately. Only state and question definitions enter the model. No frozen label,
checkpoint or training corpus was altered in response to model outputs.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.runtime-training:. \
  .venv/bin/python -m experiments.semantic_contrasts \
  --stage evaluate --suite data/semantic-contrasts-v1 \
  --output reports/my-contrast-evaluation
```

The 114 cases share six authored families and many repeated definitions. Some
predicates are mechanically decidable from the structured record and would often
belong in application code. Here they are controlled probes of instruction scope,
evidence binding and semantic distinctions. Once inspected and used to guide
development, this frozen diagnostic is no longer an untouched generalization test.
