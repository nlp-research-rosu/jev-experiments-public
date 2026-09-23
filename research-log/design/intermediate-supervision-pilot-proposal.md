# Proposed next experiment: intermediate supervision at fixed inference cost

Status: proposal for discussion, 2026-09-22. No training, generation, paid calls or monitor started by this note. Completed experiment artifacts and the original test/Jev quarantine stay unchanged.

## Question and evidence

Test whether supervising the evidence distinctions used to make a decision improves transfer to new semantic families while preserving the current cached inference path.

The existing linear readout fits the small fitting set perfectly; more expressive heads did not improve development correctness. Tiny LoRA continuation improved fitting but not development accuracy and worsened raw probability quality. Completed native reasoning solved 15/15 questions, compared with 9/15 for direct native scoring and 13/15 for H0 on those same questions, but half the full 30-question cohort failed to finish. These observations motivate an intermediate-supervision pilot. They do not prove that all reasoning can be compressed into a fixed inference budget, that LoRA is adequate for every task, or that the original model is a reliable oracle.

## First comparison

| Arm | Training signal | Serving behavior |
|---|---|---|
| A: control | Existing final-answer BCE/grouped CE, with matched old-task replay | Current scalar heads and stable cached inference |
| B: verified facts | Same final-answer loss and replay, plus supervised intermediate evidence features | Same serving path; training-only auxiliary heads removed |

Use the same H0 starting checkpoint, rank-8 adapter placement, final heads, inputs, family ordering, label loss weights, replay and fixed update budget. Do not combine this with a fresh initialization, larger backbone, higher LoRA rank or a probability-objective change. Instantiate equivalent auxiliary machinery in the control where needed to separate supervision from bookkeeping differences. Normalize auxiliary losses by eligible facts/logical questions rather than inadvertently weighting larger candidate sets more. Declare the coefficient from training/validation diagnostics, record gradient norms/clipping and actual training cost, and do not tune it on confirmation outcomes.

B supervises the hidden representations consumed by final scoring branches. It is not simply a new batch of separately prompted Boolean questions: existing training already includes several such questions per state. Auxiliary targets are labels outside the input, and must never be copied into STATE or supplied at serving time. Their purpose is to shape shared representations, not to provide the answer to the model.

An illustrative target family:

- Assistant claims that order 17 was refunded.
- Successful tool receipt is for order 18.
- Intermediate targets distinguish the claim, successful receipt, matching action and mismatching target.
- The final question asks whether operational evidence confirms the refund of order 17: no.

Use evidence-status predicates with precise definitions. Missing evidence is not a false world fact; unknown, conflict and inapplicability must not be flattened into arbitrary Boolean targets. Vary claims, outcomes, identity, authority, time and rubric separately so intermediate labels cannot be predicted merely from category or formatting.

## Data and annotation

Start with approximately 200 balanced training families from the existing training partition: about 800 cases / 4,000 final judgments under the current four-case, five-question family structure. This is a proposed pilot budget, not a statistical sufficiency claim. Preserve whole contrast families and report the number of distinct authoring blueprints separately from row count.

The generator already computes scoped features from raw observations in `experiments/contrast_scaling_logic.py::derive_features`, with final rule evaluation in the factorial renderer. This provides a source for exact annotations. Audit that every selected target follows from the rendered public evidence and supplied rules; hidden generator facts are not permissible evidence. Some finer record-level matching targets may require instrumenting the oracle rather than merely exporting existing aggregate features. Current generic rationale strings are not substantive explanations.

Do not train on the 30 native-probe questions or their inspected evaluation traces. Freeze an independently authored confirmation set before training, holding out authoring/scenario families and rule combinations; reserve separate checkpoint/weight-selection and probability-calibration partitions. Existing inspected suites remain regression tests. Use more than one family per category if family-stratified uncertainty intervals are desired.

A fresh teacher is not required for arm B. Larger Qwen models can later help phrase concise explanations, and Jev can remain an external comparison or disagreement signal. Neither should override independently verified gold simply because its final answer is confident. Correct final answers do not certify every statement in a generated explanation.

## Paper connection and staged alternatives

- **[Distilling Step-by-Step](https://arxiv.org/html/2305.02301):** jointly trains label and rationale prediction and allows label-only inference. It supplies the motivating principle of additional training supervision without required rationale generation at serving. Arm B is our structured-feature adaptation, not a literal reproduction of its T5 text-to-text recipe.
- **Optional next arm C: verified short explanations.** Apply the paper's multitask rationale idea more directly: a separate training-only explanation objective shares the adapted backbone with the direct decision objective. Use concise, fact-checked explanations, not copied 2B repetition loops. Keep a direct-answer training branch that never receives gold explanations, and evaluate with no generated explanation. Account for the extra training tokens/compute. This is conditional on the A/B result, not an automatic third run.
- **[From Explicit CoT to Implicit CoT](https://arxiv.org/html/2405.14838):** a related reference checked for this proposal. It gradually removes reasoning tokens during training. This offers a later compression curriculum; its demonstrated math settings and additional training/stability requirements differ from our judgment task.
- **[RLCR](https://arxiv.org/abs/2507.16806), Rewarding Doubt and ConfTuner:** retain as probability/teacher-training options. They do not establish that rewarding confidence will teach the missing evidence distinctions. Our six-arm screen already motivates comparing against ordinary CE plus separately fitted calibration.
- **[Coconut](https://arxiv.org/html/2412.06769):** a related reference checked for this proposal. Continuous latent reasoning uses additional sequential model computation. Consider a small explicit accuracy/latency trade-off only if training-only supervision fails to transfer; latent steps are not free inference.

## Evaluation and decisions

First perform a smoke check of target alignment, gradients and training/evaluation numerical agreement. The new canonical cached implementation is inference-only; do not assume its cache-fork implementation supplies correct training gradients. Any training-forward changes need explicit checks and must be identical between A and B. Then use a bounded matched pilot, provisionally 200 updates per arm, with a declared endpoint/validation rule. A capped negative result is not proof of an architectural impossibility.

Primary: both-correct contrast-pair accuracy on held-out families. Also report per-category and rubric/option-change behavior, Unknown precision/recall, intermediate-feature accuracy, retention, raw and independently calibrated NLL/Brier, and confident-error risk at matched coverage. Measure the actual serving path to verify that improvement did not require extra inference steps. Inspect joint intermediate-plus-final correctness rather than assuming a decodable feature is used by the final decision.

| Outcome | Interpretation / next decision |
|---|---|
| B improves held-out contrasts with retained probability quality and latency | Evidence that richer supervision transfers useful distinctions; replicate before scaling |
| Intermediate facts transfer, final answers do not | Investigate applying the question/rubric and composing the facts |
| Facts fit training but fail new families | Investigate evidence grounding, linguistic coverage and representation adaptation |
| Facts/answers still fail even a bounded fitting check | Audit targets/optimization; a matched rank or upper-block adaptation test remains reasonable |
| Only extra reasoning steps help on well-controlled cases | Consider a measured adaptive/latent-compute path, without treating one negative pilot as proof that extra compute is necessary |

Inference speed, retention and probability quality remain guardrails. Improvement in training accuracy or confidence alone does not justify scaling the corpus or promoting a checkpoint.
