# Calibrated training: literature and proposed next experiment

Research checked 2026-09-21. This is a research/design note, not a new training run. Source snapshots and model metadata are in `reports/calibrated-training-research-v1/sources/`.

## Judgment

Probability-aware objectives and distribution-valued supervision are worth testing. Our current evidence does not establish that supervised learning has a fundamental ceiling, that more diverse data cannot help, or that Jev's advantage comes primarily from RL. Data coverage, initialization, output semantics, capacity, regularization and optimization remain plausible causes.

The paired-language experiment established a useful but limited result: at the same 1,000-update budget, natural language modestly improved several metrics relative to templates; all trained variants nevertheless failed at least the broad probability-loss retention check. Eight of the ten new training categories shared one action-stage rubric. A model applying that familiar rubric to a different supplied rubric is consistent with a data shortcut. It is not evidence that small models cannot obey arbitrary rubrics.

## What TypeSafe actually discloses

TypeSafe's [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer) depicts RLCD as a post-training path from pretrained language models. It criticizes preference-driven probability concentration and says its objective targets calibrated decisions. Its [launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev) additionally claims architecture and sampling changes. Neither inspected source specifies Jev's parameter count, backbone identity, reward formula, optimizer or training dataset. The public evidence supports the broad pretrained-model hypothesis, not a particular small backbone or a reproducible RLCD recipe.

The [confidence documentation](https://docs.typesafe.ai/confidence) distinguishes returned class probabilities from a derived concentration statistic. Our evaluation should continue to use selected-answer probability when counting confidently wrong answers; the separate API `confidence` field is not that probability.

## Primary research most relevant to this project

| Work | Verified contribution | Transfer to our experiment |
| --- | --- | --- |
| [Beyond Binary Rewards / RLCR](https://arxiv.org/abs/2507.16806), 2025; revised May 2026 | Jointly rewards generated-answer correctness and calibrated verbal confidence using correctness minus squared confidence error. The paper provides incentive analysis and reports calibration gains including Qwen3-8B. | Closest RL reference. Its answer-plus-confidence setting differs from our directly differentiable class vector. [Code](https://github.com/damanimehul/RLCR). |
| [Rewarding Doubt](https://arxiv.org/abs/2503.02623), 2025; revised February 2026 | Uses a logarithmic scoring reward and PPO to train verbal confidence. Answers are generated before the confidence update; the method targets confidence rather than jointly changing answer selection. Includes 8B-scale experiments. | Useful confidence-training precedent, but not an interchangeable joint decision reward. [Code](https://github.com/pasta99/RewardingDoubt). |
| [ConfTuner](https://arxiv.org/abs/2508.18847), 2025 | Introduces a differentiable tokenized Brier objective for verbal confidence without supplied gold confidence percentages; evaluates 7–8B models. | Evidence that calibrated-confidence training need not use an RL optimizer. Our numerical output head avoids its confidence-token machinery. [Code](https://github.com/liushiliushi/ConfTuner). |
| [On the effectiveness of reward functions…](https://arxiv.org/abs/2607.04332), July 2026 preprint | Characterizes confidence-reward incentives and demonstrates practical reward hacking on small LMs: a model can favor wrong answers with low reported confidence. | Audit the joint decision/confidence reward before implementation. Calibration-only success is insufficient. Treat this recent preprint as supporting evidence, not a settled universal result. |
| [Just Ask for Calibration](https://arxiv.org/abs/2305.14975), 2023 | Finds verbalized confidence can be better calibrated than conditional output-token probabilities in tested RLHF models. | A teacher's token likelihood, verbal confidence and empirical correctness are different quantities; none is automatically our gold label distribution. |
| [Adaptive Temperature Scaling](https://arxiv.org/abs/2409.19817), 2024 | Fits input-dependent temperature adjustments to address calibration shifts following RLHF. | A relevant calibration control; its token-level method needs adaptation to our grouped semantic logits. |
| [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), 2017 | Studies modern-network miscalibration and temperature scaling. | Start with validation-fitted positive temperatures as an inexpensive baseline; these preserve each question's winning class. |
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), 2015 | Transfers information through a teacher's softened output distribution. | Supports full-vector distillation, not an assumption that teacher distributions are calibrated truth. |
| [When Does Label Smoothing Help?](https://research.google/pubs/when-does-label-smoothing-help/), 2019 | Reports generalization/calibration benefits and a potential loss of information useful for distillation. | Uniform smoothing is a control, not equivalent to evidence-dependent target probabilities. |

## Why the distinction between an objective and an optimizer matters

Our current system already predicts a probability distribution. Binary BCE and categorical cross-entropy are proper probability losses in the ideal expected-loss setting; hard observations can train a distribution. They do not guarantee finite-model generalization or calibration after distribution shift. See the prior [probability-design note](PROBABILITY_DESIGN.md) and [proper scoring rules](https://doi.org/10.1198/016214506000001437).

An illustrative outcome has probability 0.6 under the visible information. A correctness-only sampled-action reward is maximized by always choosing the more likely class. It does not require reporting 0.6. Probability scoring instead evaluates the reported distribution against outcomes. This explains why optimizing accuracy and estimating uncertainty are different tasks; it does not imply an unavoidable accuracy/calibration tradeoff.

For a fixed finite class set, we can compute every candidate score and differentiate the entire probability loss. Switching the name to RL does not add missing evidence or make the gradient more informative. RL becomes substantively different when training involves sampled reasoning, external outcomes, non-differentiable judges, or action-dependent feedback. A larger reasoning teacher could use such training and then supply distributions to a fast student.

For RLCR's separate generated answer and confidence, the published reward is `correct - (confidence-correct)^2`. Its bounded calibration term is paired with a correctness incentive. A confidence-only objective can prefer an answer known to be wrong reported with confidence zero; that differs from proper multiclass scoring of a fixed gold outcome. Do not blindly transplant a generated-confidence reward onto the student's class probabilities. Also do not conclude that ordinary categorical cross-entropy has this reward-hacking pathology.

## What a target distribution should mean

Three target sources should be tagged separately:

1. **Known evidence outcome.** A verified, fully specified deterministic record can retain a hard label. `Unknown` may itself be the correct, high-probability Choice label. It does not automatically mean a 50/50 Noul target.
2. **Known conditional uncertainty.** A controlled latent-state generator can provide true probabilities when prior frequencies and noisy observation rules are explicitly supplied. For example, a balanced hidden state and a sensor with 80% sensitivity and 20% false-positive rate imply an 80% posterior after a positive reading. This is different from asking whether the reading conclusively proves the hidden state.
3. **Teacher or annotator estimates.** A larger model's vector or reviewer distribution is useful supervision with provenance, not an oracle. Agreement under repeated sampling is affected by shared biases, prompt framing and sampling temperature. Preserve raw responses, candidate order mappings, evidence and disagreement flags; evaluate teacher distributions on disjoint, verifiable examples before assigning weight to them.

For an audited target vector q, the candidate-group loss can be `-sum(q_i log p_i)`, equivalent to minimizing forward KL up to a q-dependent constant. For Noul use the corresponding two-outcome distribution; for Score retain the entire level distribution. Fitting only the Score mean cannot distinguish a concentrated middle level from a mixture of extremes.

The existing `target_vector` and `bundle_loss` already accept Noul `probability_true`, Choice probability maps and Score probability vectors. The missing work is primarily target construction, objective comparisons and evaluation, not a new numerical API.

A positive combination of full-vector cross-entropy and Brier loss remains a proper expected-loss objective. That does not prove it will outperform cross-entropy here; it is an empirical ablation. Brier alone can have small gradients on saturated wrong softmax outputs, another reason to retain a CE control or mixture. Uniformly replacing every 1/0 label with 0.9/0.1 merely imposes fixed uncertainty; it cannot teach which particular cases are difficult.

## Fresh initialization has two meanings

Our earlier fresh-start experiment removed OpenJev's learned adapters/heads, but still used Qwen's post-trained release. It achieved strong generator-aligned accuracy while regressing on broad probability quality; broad NLL was 0.5828 versus 0.3504 for continuation, with unequal lifetime training and a still-improving fresh curve. Thus it neither establishes that restarting solves calibration nor rules out a better fresh recipe. See [REVISED_INITIALIZATION.md](REVISED_INITIALIZATION.md).

The official [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B) identifies those weights as post-trained. Qwen also publishes genuinely pretraining-only [2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) and [4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) releases. Their cards say chat control tokens support LoRA with the official template. This makes a pretrained-only comparison practical; it does not establish model quality or exact local memory requirements.

Metadata checked without downloading weights:

- 2B-Base revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`.
- 4B-Base revision `1001bb4d826a52d1f399e183466143f4da7b741b`.

## Proposed bounded experiment before large-scale generation

**Data:** Use richer semantic families and complete contrast bundles. Vary the supplied rubric itself while keeping evidence fixed; vary evidence while keeping the rubric fixed; include genuine invariances, action/source/entity/time binding and explicit missing-information cases. Inspect each rendered scoring unit, including its candidate definition, before accepting data. Required definitions must be visible in that unit. Hold out whole rule systems and authoring patterns, not only names.

**Evaluation:** The existing inspected suites remain regression tests. Create a new untouched test set, plus separate checkpoint-selection and calibration partitions. Evaluate accuracy, whole-pair correctness, NLL/Brier, Unknown precision/recall, rubric-change behavior, error-detection AUROC and risk versus coverage. Low confidence everywhere is not success. Keep probability thresholds, retention limits and selection rules fixed before test outcomes. Report family uncertainty and training-seed variation separately.

**Initialization:** At 2B, compare pretraining-only Base with the fresh post-trained Qwen release; reset our adapters and scoring heads consistently. Keep v0.2 as an external historical reference. Check tokenizer anchors, prompt rendering, score-head initialization and the complete API before training. Give fresh models sufficient task-interface training rather than silently assuming v0.2's warm start.

**Objective ablation:** Hold the richer corpus, independent test, sampling and training exposure fixed:

| Arm | Objective | Question answered |
| --- | --- | --- |
| A | Hard-label BCE/CE | Does the new data and initialization suffice? |
| B | Hard-label CE plus full-vector Brier | Does an altered proper-loss emphasis help? |
| C | Audited distribution targets, optionally mixed with hard labels | Does evidence-dependent target shape improve accuracy/probability quality? |

Fit a validation-only temperature baseline for each arm without changing its argmax predictions. A uniform-label-smoothing control is useful if Arm C improves: it distinguishes target-specific information from generic softening. Keep explicit consistency-loss additions separate initially; complete-family sampling is shared across arms. Repeated data presentations, not only optimizer updates, must be matched and reported.

**Scale:** First identify whether the loss/initialization change helps. Then compare 4B with 2B under the same chosen recipe and new held-out rules. A sub-10B model is plausible given the cited 3B/7B/8B experiments, but those tasks and inference paths differ from ours. Profile 4B on the 4080; 9B BF16 language weights alone are roughly 18GB before other memory, so that size needs a different memory plan. Larger capacity also has an inference-latency cost that must be measured.

**RL branch:** If the direct-loss student still fails, use RLCR as the first open reference for a reasoning teacher or a separate sampled-answer experiment. Specify the action, reported probability, correctness verifier, reward and reference-policy regularizer. An arbitrary overconfidence penalty or KL weight is not a guarantee of honest probabilities. Judge quality and reward hacking must be evaluated. Distill useful teacher behavior back into the fast, non-generating student only after validating the teacher.

Do not change initialization, model size, data volume, target type and optimizer all at once. That might yield a better checkpoint while leaving the cause of improvement unknown.

## Status

Research and source inspection complete. No new model weights downloaded, GPU training started, external teacher requests made, package installed, or original experiment artifact changed for this research turn. The experiment above is a proposal for the next implementation step.
