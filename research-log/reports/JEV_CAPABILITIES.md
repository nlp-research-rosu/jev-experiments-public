# Jev's documented output contract and Harsha's implementation

Checked 2026-09-19 against TypeSafe's official documentation, retrieved directly
as Markdown, and the current Harsha repository revision. This is a source review,
not an independent Jev API evaluation.

## Jev primitives

| Primitive | Documented result | Limits and interpretation |
|---|---|---|
| [Choice](https://docs.typesafe.ai/primitives/choice) | Selected option, full probability distribution and confidence | Up to 255 caller-defined options. |
| [Score](https://docs.typesafe.ai/primitives/score) | Probability-weighted mean of level indices, distribution, legend and confidence | 2–10 descriptive levels indexed from zero; the score can fall between levels. |
| [Noul](https://docs.typesafe.ai/primitives/noul) | Probability that a yes/no question is true | A number in [0,1]; no separate confidence property. Values near zero express a strong no, not low confidence. |

[Confidence](https://docs.typesafe.ai/confidence) is described as a statistic of
the distribution's concentration. The page does not specify its exact formula.
It is not simply the highest probability: the Choice documentation includes a
winning probability of 0.60 with confidence 0.39. It should not automatically be
interpreted as a probability of correctness.

[TypeSafe's AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)
states that RLCD trains calibrated outcome probabilities. Calibration is a
population property: events given probability 0.8 should occur about 80% of the
time across appropriate sets of predictions. It is separate from merely producing
normalized numbers. We have not independently verified Jev's calibration.

## Rich structure and extraction

The [advanced structure page](https://docs.typesafe.ai/primitives/advanced) allows
objects and arrays in instructions and descriptions. Its examples still return
Choice, Score and Noul answers; a field description saying `type: string` or
`type: number` is not an arbitrary-output generation endpoint.

The official [pre-parsed extraction cookbook](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook)
finds candidate email addresses, phone numbers and amounts with regular expressions;
Jev selects among them and code copies/normalizes the selected value. The
[date cookbook](https://docs.typesafe.ai/cookbooks/date_extraction_cookbook) selects
date components and resolves/validates them in code. The
[structured extraction cascade](https://docs.typesafe.ai/cookbooks/sde_cascade)
uses an LLM to extract, Jev to verify fields, and a stronger LLM for escalations.

Thus applications can return richer records while the primitive model interface
remains constrained. A supplied complete address can be a Choice candidate; Jev
does not have to generate that address token by token. Candidate discovery is an
additional component, and a missing candidate cannot be recovered by selection.

The [System One page](https://docs.typesafe.ai/concepts/system-one) explicitly says
Jev does not generate text. Its [known weaknesses](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
also acknowledge incorrect decisions and lack of guaranteed structural consistency
between related questions. Type safety is not semantic correctness.

## Harsha's published implementation

The current Hugging Face API reports revision
`2af86848be75847ccb3553b0941cc51d6ef7e4e9`, last modified 2026-09-16. This is the
same revision previously inspected for this project.

- The [MLX engine](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/2af86848be75847ccb3553b0941cc51d6ef7e4e9/core/engine_mlx.py)
  loads stock `mlx-community/Qwen2.5-1.5B-Instruct-4bit`. The Torch engine defaults
  to stock `Qwen/Qwen2.5-1.5B-Instruct`. The published repository has no new model
  weights, adapters or training pipeline. It demonstrates inference engineering;
  the underlying Instruct checkpoint already had its original training.
- The [schema implementation](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/2af86848be75847ccb3553b0941cc51d6ef7e4e9/core/schema.py)
  accepts booleans and enums (with choice/selection aliases), capped at 255 enum
  choices. It rejects other field types. There is no distinct Score primitive or
  arbitrary free-form output path in the constrained engine.
- Prefix caching, batched field suffixes and code-side JSON assembly avoid
  sequential JSON generation. Field telemetry includes the selected value,
  confidence and up to five top-choice probabilities. MLX can run extra decoding
  steps when candidate tokens collide.
- On the ordinary path, its confidence is the winning candidate's softmax score.
  Softmax normalization alone does not establish calibration. In the MLX
  collision fallback, it clamps the selected value's probability to at least
  0.75 and distributes the remaining mass uniformly. This heuristic is not
  empirical calibration, despite the repository's calibration wording.
- Its published speed/validity benchmarks demonstrate a useful inference
  technique. They do not establish Jev-equivalent semantic accuracy, calibration,
  training or underlying architecture.

## Implications for this experiment

Our scorer already retains per-option probability vectors, beyond the selected
answer codes shown in the walkthrough. Noul-like outputs could expose P(true);
Score-like outputs could compute a weighted mean over ordered levels. Both are
compatible with batched scoring. Matching TypeSafe's exact confidence statistic
requires a specified definition; matching its claimed calibration requires
separate evaluation, not just a similarly shaped response.

The proposed general training experiment should measure held-out probability
quality as well as accuracy, option-order stability and latency. Calibration data
must remain separate from the final test set. These are design implications;
no new primitives, calibration stage or training run were implemented in this review.
