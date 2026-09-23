# Diagnosing the semantic contrast failures

The follow-up separates several problems that an overall accuracy number hides.
Selecting the evidence sources relevant to each question improves this diagnostic
from **459/764 (60.1%) to 559/764 (73.2%)** against a matched grouping control.
Registry verification and permission benefit, but completion confirmation barely
changes and all four Unknown registry cases still fail. Confidence scaling is
another problem: reducing confidence improves probability loss without fixing any
decisions. No additional neural training or production changes were made.

These are retrospective interventions on the already inspected, frozen
[semantic contrast suite](SEMANTIC_CONTRASTS.md): 114 scenarios, 764 judgments,
260 relations, six domain families and two shared generators. Repeated templates
and uneven class counts make these diagnostic results, not estimates of accuracy
on general structured queries. This round covers Noul and Choice; it adds no new
descriptive Score measurements.

## Classes that need separate tests

| Class | Controlled distinction | Evidence from this round / remaining question |
|---|---|---|
| Claims and authoritative evidence | “I completed it” versus a matching successful event; a sender's change claim versus a registry change | Restricting registry questions to registry evidence improves confirmation from 14/34 to 33/34. Completion confirmation remains poor, so source selection alone is insufficient. |
| Execution stage and current effect | Intended, proposed, invoked, pending, failed, succeeded, reversed | Completion confirmation changes from 33/80 to 32/80. Current-effect accuracy improves from 33/80 to 39/80. The distinctions still need attention. |
| Correct target, operation and time | Success for this entity/action/time versus an unrelated success; relevant versus unrelated reversal | The frozen suite includes these contrasts. The current interventions do not establish a general repair; a separate breakdown and fresh examples should isolate each dimension. |
| Permission and outcome | Authorized but not executed; executed without authorization | Policy-only evidence improves action permission from 72/80 to 80/80. Permission to use a destination improves from 15/34 to 29/34. These are distinct predicates with different evidence needs. |
| Missing or conflicting evidence | Known unchanged versus unavailable, unauthenticated or conflicting records | Filtering gets all 30 known registry states right but **0/4 Unknown**. Unknown needs explicit positive controls; it must not be conflated with a false binary proposition. |
| Question scope and wording | A claim was made; a fact was verified; a broader operational rule was triggered | Same-record question contrasts improve from 6/42 to 17/42 with both answers correct, but remain weak. Extracting explicit intent and explicit change claims gets slightly worse. |
| Descriptive Score | Rubric-defined degree, ordinal ordering, numerical meaning | Previously weak in the public-case replay; not tested anew here. It needs a separate rubric-grounded suite, with justified targets rather than invented probability labels. |

For every class, distinguish three checks: change relevant evidence and require
the appropriate answer change; change only the question and require the correct
semantic distinction; change irrelevant wording and require stable, correct
answers. Claim and confirmation require all four truth combinations, rather than
assuming they are complements. Also report positive and negative performance:
the base model's always-no behavior on some predicates can make isolated correct
negatives look like retained reasoning skill.

## Evidence selection: the largest measured gain

The selector is a reviewed mapping for these known question definitions. It reads
no expected answers. For example, a completion question retains the request,
audit scope and events; a permission question retains the request and policy;
a registry-state question retains evaluation time, authority, baseline and current
snapshots. Required temporal and conflict information remains available.

We run the same question groups and order with full evidence as a control.
Grouping alone changes five labels relative to the original run, with maximum
probability difference 0.092. This records numerical sensitivity under BF16
batching; the comparison below uses the matched groups.

| Metric | Full evidence, grouped | Relevant sources, grouped |
|---|---:|---:|
| Correct judgments | 459/764 | 559/764 |
| Mean negative log likelihood, lower is better | 0.9733 | 0.6399 |
| Errors assigned at least 95% probability | 101 | 22 |
| Evidence-change pairs, both answers correct | 25/90 | 48/90 |
| Same-record question contrasts, both correct | 6/42 | 17/42 |
| Invariance pairs, both answers correct | 99/128 | 124/128 |
| Complete scenarios | 9/114 | 30/114 |

Filtering corrects 123 judgments and breaks 23, for a net gain of 100. The full
per-question breakdown makes the uneven improvement visible:

| Question | Full evidence correct | Relevant sources correct |
|---|---:|---:|
| Expressed intent | 77/80 | 68/80 |
| Claimed completion | 31/80 | 52/80 |
| Invocation shown | 52/80 | 69/80 |
| Completion confirmed | 33/80 | 32/80 |
| Effective now | 33/80 | 39/80 |
| Permitted | 72/80 | 80/80 |
| Outcome status | 32/80 | 36/80 |
| Explicit change claim | 27/34 | 25/34 |
| Different requested destination | 32/34 | 34/34 |
| Record confirms change | 14/34 | 33/34 |
| Destination use permitted | 15/34 | 29/34 |
| Record status | 12/34 | 30/34 |
| Operational change signal | 29/34 | 32/34 |

In the payment case with an explicit change claim but an unchanged authenticated
registry, P(record confirms change) falls from 0.737 in the original full-input
run to 0.374 after filtering, correcting the decision. In the support intent-only
case, P(completion confirmed) merely falls from 0.960 to 0.946: the model still
incorrectly confirms completion despite seeing only the relevant evidence.

The apparent invariance improvement needs particular care. Filtering makes the
per-question inputs identical for **110/128 invariance pairs**, versus 24/128
before filtering. Much of that benefit is software making inputs equivalent,
not the model acquiring an ability to ignore distractors. No flip or
question-contrast endpoints collapse to identical inputs. This audit omits routing
IDs and batch packing; identical semantic inputs can still encounter numerical
differences.

This intervention removes irrelevant content **and** shortens context, so it does
not isolate those explanations. It also depends on reviewed predicate semantics.
It does not establish that an automatic selector would work for arbitrary user
questions, or that filtering should become the default. Some predicates in these
fixtures are mechanically decidable and could simply be handled in application
code; here they serve as controlled tests of the model interface.

## Model body versus numerical readout

The trained system contains LoRA updates to the body and trained numerical output
heads. We temporarily combine initial/trained heads with adapters enabled/disabled
in memory. Both original combinations reproduce all saved probabilities exactly.
The checkpoint on disk is unchanged, and heads and adapters are restored after
each intervention.

| Body | Output heads | Correct / 764 | Mean NLL | Errors with probability >= .95 |
|---|---|---:|---:|---:|
| Base | Initial | 423 (55.4%) | 0.8015 | 0 |
| Base | Trained | 378 (49.5%) | 0.8685 | 0 |
| Adapted | Initial | 456 (59.7%) | 0.8667 | 5 |
| Adapted | Trained | 463 (60.6%) | 0.9731 | 100 |

Much of the changed decision behavior persists with the adapted body and initial
heads. The trained heads substantially increase the count of highly confident
errors when combined with that body. Resetting heads does not restore completion
semantics: the support intent-only case still has P(confirmed) = 0.922.

The components were trained together, so mismatched combinations are probes of
their interaction, not unique causal attribution to a layer. The base control
uses our initial numerical readout, not ordinary chat generation. We have not
tested head resets against the broad original task suite and do not adopt them.

## Confidence and decisions are separate problems

The [calibration diagnostic](CONTRAST_CALIBRATION.md) uses the saved logits for
**650 binary judgments**, excluding the 114 categorical judgments above. It fits
positive temperature scaling and a positive-scale affine map on other families,
then checks held-out families. A stricter check transfers between the execution
and destination generators, which also have different question definitions.

| Trained binary condition | Correct / 650 | Mean NLL | Errors with probability >= .95 |
|---|---:|---:|---:|
| Raw | 419 | 0.9626 | 100 |
| Family-held-out temperature | 419 | 0.6393 | 0 |
| Family-held-out affine map | 422 | 0.5978 | 0 |
| Generator-held-out affine map | 399 | 0.8473 | 27 |

Temperature scaling changes no decisions. Its removal of highly confident errors
means those errors become less confident, not correct. The family affine map
reduces false positives from 227 to 146 but increases false negatives from 4 to
82: a net gain of only three. Transferring an execution-fitted map to destination
questions reduces accuracy from 70.0% to 51.8%.

Useful relative discrimination exists: the trained model orders 77/80 binary
evidence-flip pairs in the correct direction. These simple global maps cannot
turn that into reliable absolute decisions across predicates. This does not rule
out richer calibration. Family folds share templates, and all fixtures were
previously inspected; neither split establishes production calibration.

## Representation sensitivity

Replacing the structured state object with a JSON string containing exactly the
same object changes 34/764 decisions. Seventeen errors are corrected, 14 formerly
correct answers regress and three wrong answers change to other wrong answers.
Overall accuracy barely changes, from 60.6% to 61.0%, while the largest probability
shift is 0.689. Aggregate accuracy conceals substantial individual instability.

The JSON string round-trips exactly, but its prompt representation adds escaping
and changes whitespace and token length. This is a serialization robustness probe,
not a pure whitespace test or evidence of a preferred universal representation.

## Implications for the next experiment

Prioritize completion semantics, Unknown/conflicting evidence and question scope.
Keep source selection as an explicit experimental condition. Isolate context
length from source relevance, and test new wording and domains before adopting a
selector. For target/time binding, vary one entity, operation or timestamp at a
time while preserving the surrounding vocabulary. Descriptive Score requires its
own suite rather than extrapolating these binary findings.

Build separate training examples and fresh transfer sets. Keep this inspected
suite as a regression check. Original Noul training already included balanced
positive/negative examples and complements; the next requirement is coverage of
these distinctions, not merely adding generic negatives. Preserve broad-task
performance alongside pair correctness, class balance, probability loss and
representation stability. Retention should be measured on both members of a
contrast, since one correct base-model negative does not establish a robust skill.

No inference-only intervention here is sufficient evidence to change the freezing
policy or launch a new full training run. The results provide specific hypotheses
and measurable acceptance checks for that next design.

## Artifacts and verification

- Component/context/representation report
- Transitions, input-equivalence audit and Unknown cases
- [Calibration method, folds and results](CONTRAST_CALIBRATION.md)
- Raw responses, per-question predictions and all 260 relations for each of seven
  conditions are in `contrast-components-v1/<condition>/`.

Both diagnostics completed. The full repository check passed **254 tests and 45
subtests** with the reference CPU kernels; the new experiment files pass lint.
The frozen suite, saved original responses, relevant source hashes and checkpoint
identity were verified unchanged. Tests cover required evidence preservation,
temporary component restoration and calibration fold separation/convergence.

Frozen suite SHA-256:
`acdfa86007eb167ba5716d61d22b8567a7cfc4f0b28cfa42daf2c9d90960ad93`.
Checkpoint:
`openjev-judgment-v0.2/sha256-4f88417678a33bc14f47ad97e9f0ffc96bf426e84d12a900eb99c2f97800ff49`.

GPU probes use Qwen3.5-2B on the local RTX 4080, independent full-input scoring,
FLA, BF16 body weights, FP32 numerical heads and a scoring batch size of four.
They do not benchmark latency. Reproduce into a new output directory:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.runtime-training:. \
  .venv/bin/python -m experiments.contrast_component_probe \
  --output reports/my-component-probe
.venv/bin/python -m experiments.contrast_calibration_probe \
  --output reports/my-calibration-probe
```
