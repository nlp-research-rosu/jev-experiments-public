# Consistency and native reasoning — results

The new canonical cached inference mode removes the observed request-packing dependence in the pinned GPU tests while retaining shared context computation. Median request latency was **177 ms versus 167 ms** for legacy caching (6.4% overhead in this sample). Development accuracy remains 152/200 and legacy retention 55/60.

The original Qwen3.5-2B checkpoint can solve some of these nuanced judgments when it finishes reasoning. In the fixed greedy probe, all 15 naturally completed answers were correct, but another 15 hit the 4,096-token thought cap. On the same 15 completed questions, direct native scoring got 9 right and the trained H0 judge got 13 right. This is evidence of useful additional computation, not 100% accuracy on the full 30-question population.

## Numerical consistency

The first divergence was located in shape-sensitive layer-zero projections; padding introduced another difference at a full-attention block. Identical neighboring copies reproduced the problem, so differing neighbor semantics were not required. Precision-switch probes did not eliminate it. This agrees with the general floating-point caveat in [PyTorch’s documentation](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#batched-computations-or-slice-computations); the project-specific evidence is in ROOT_CAUSE.md and its saved traces.

The fix derives the cache prefix independently from each unit’s state, computes equal prefixes once at B1, and evaluates suffixes in fixed B4 tiles with per-unit length buckets and masked padding. Every branch gets a fresh cache fork. The policy has a distinct identity, including device/dtypes/backend, and reports raw, uncalibrated probabilities and actual physical work. It standardizes arithmetic layout; it does not claim closer agreement with ideal exact arithmetic.

| Check | Observed result |
|---|---|
| 43 cases / 215 judgments / 408 units: separate requests vs pooled, reversed and shuffled | Exact zero logit and probability differences; zero answer flips |
| 39 independently called units from the four original failure cases | Exact zero differences |
| Removing other questions from the same state | 20 checks; exact zero differences |
| Six suffix-length boundary fixtures, including cached and uncached prefixes | Exact zero differences |
| H0 checkpoint and frozen model hashes | Unchanged |

These results apply to the pinned runtime and tested populations. Hardware, kernel or precision changes need their own validation. The legacy runners are preserved; new inference uses the explicit consistent policy described in [the usage guide](../../design/consistent-inference.md).

## Speed and memory

Six fixed requests, four alternating warmed repetitions per method on the shared RTX 4080. Each request contains five judgments. Memory below is PyTorch allocated memory for this process, not total device usage.

| Method | Median request time | Median peak allocated |
|---|---:|---:|
| Legacy shared cache | 166.6 ms | 3.705 GiB |
| Consistent shared cache | 177.2 ms | 3.735 GiB |
| Legacy full independent | 347.8 ms | 3.725 GiB |
| Consistent full-prompt reference | 574.7 ms | 3.757 GiB |

The first full-prompt fix was substantially slower, which motivated the shared-cache implementation. The final cached mode retains almost all of the measured speed. These small, warmed desktop timings are not production latency guarantees.

Numerical stability does not solve semantic accuracy or calibration. The consistent cached policy has raw development NLL 1.010, with 26 mistakes among 161 judgments whose maximum class probability is at least .90. Coverage is 80.5%. Those probabilities are not automatically calibrated by stabilizing their execution.

## Original-model capability: fixed greedy probe

Selection used one hashed case per existing development category and one question per primitive: 30 judgments over ten states. Inputs contained the same evidence/rules/criteria; presentation used ordinary task language. Expected labels stayed out of prompts. Original pinned Qwen3.5-2B means the released post-trained checkpoint, not Qwen3.5-2B-Base. No model parameters were trained.

| Evaluation | Correct | Denominator / interpretation |
|---|---:|---|
| Native canonical direct code scoring | 13/30 | Entire fixed cohort |
| Native two-order averaged direct scoring | 13/30 | Entire fixed cohort |
| Greedy reasoning, natural completed answers | 15/15 | Completion-conditioned accuracy; 15 other cases unresolved |
| Greedy reasoning, successful answers over the full cohort | 15/30 | 50% successful coverage |
| Direct canonical on that same completed subset | 9/15 | Matched questions |
| Trained H0 consistent cached on that same completed subset | 13/15 | Matched questions |
| Trained H0 consistent cached over the full cohort | 23/30 | All questions answered |

| Primitive | Naturally finished | Correct finished answers | Capped |
|---|---:|---:|---:|
| Boolean | 4/10 | 4/4 | 6 |
| Choice | 5/10 | 5/5 | 5 |
| Score | 6/10 | 6/6 | 4 |

The run used 90,289 thought tokens and 1217.6 seconds. All unfinished cases reached the token cap; completed answers had valid natural code/EOS output. No capped thought was forcibly closed or scored as a completed solution. The completed sample is selected by termination behavior and cannot stand in for the unresolved half.

Two completed examples corrected H0 errors: an action-binding Score judgment (correct level 2, H0 chose 1), and an ordered-rubric Choice judgment (correct `rejected`, H0 chose `unknown`). Exact states, questions, predictions and token counts are linked in NATIVE_ANALYSIS.json. The other completed cases include distinctions H0 already handled, so this is not a hidden uniformly superior native model.

## Follow-up on thinking loops

Capped traces contained repeated paragraphs, sometimes repeatedly stating an intended answer. The [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B#best-practices) explicitly warns that this model can enter thinking loops. It recommends sampling and a presence penalty for thinking-text generation. This motivated a separate decoding check; it does not prove that every unresolved trace has the same cause.

We selected two capped questions per primitive by a fixed hash, independent of correctness. Each received one seeded attempt with temperature 1.0, top-p .95, top-k 20 and presence penalty 1.5 over generated tokens. The original 30-question run was not changed. Limits remained 4,096 thought tokens and 32 final tokens per case, with a 720-second total bound for this six-case diagnostic.

Sampling result: **3/6 valid natural final answers, 3 correct**; 3 unresolved in the natural-output metric. Separately, 4/6 thoughts reached their natural closing marker. The constrained readout after completed thoughts scored 4/4 correctly. It used 21,017 thought tokens in 296.7 seconds. The same six cases had zero completed thoughts under the earlier greedy setting.

Thought completion, final-answer format and constrained semantic choice are distinct outcomes. A completed thought followed by an overlong explanation can have a usable constrained readout while failing the requested natural final-code format. A capped thought receives neither completion nor constrained-solution credit.

| Parent index | Primitive | Sampled termination | Natural answer | Gold | Correct natural answer | Conditional choice after completion |
|---|---|---|---|---|---|---|
| 18 | noul | final_token_cap | None | true | False | true |
| 9 | noul | eos_after_think | false | false | True | false |
| 7 | choice | scratchpad_cap | None | not_completed | False | None |
| 28 | choice | eos_after_think | unknown | unknown | True | unknown |
| 26 | score | scratchpad_cap | None | 2 | False | None |
| 17 | score | eos_after_think | 0 | 0 | True | 0 |

This is a post-hoc six-case decoding diagnostic on selected loop failures, not an accuracy estimate for the whole 30-question cohort. It changes a sampling recipe rather than isolating one parameter. Do not splice its successes into the greedy run and present the combination as a single-protocol benchmark.

Conditional answer-code probabilities after a generated rationale condition on that model-produced text. Their near-certainty is not evidence of calibrated correctness probabilities. Natural answers, conditional readouts, validity, caps and coverage are stored separately.

## Interpretation

The batching defect has a working cached inference fix on the tested runtime, with modest measured overhead and preserved observed task accuracy. It explains some close-call instability, not the larger semantic generalization gap.

The native probe provides positive evidence that the original model can derive some subtle answers with extra computation, including six direct-mode errors on the completed subset. It also exposes stopping/decoding limitations. These observations support investigating how to teach reliable direct judgments or verified intermediate distinctions; they do not establish that a bigger head, more LoRA rank, or full fine-tuning is necessary. No further training study has been launched.

## Verification and artifacts

The original datasets, H0 checkpoint, frozen model and historical implementation files remain unchanged. The full-prompt harness initially mislabeled derived response identities; original records remain intact and a metadata-only correction is documented in consistency-validation-bound/metadata-correction.json. Numerical payloads and timings were unchanged. The cached harness records its correct policy identity directly.

The final CPU suite passed 1,070 tests plus 48 subtests, with 15 existing runtime/deprecation warnings. Verification logs: cpu-suite.log, cached-focused-tests.log, sampling-focused-tests.log and cpu-suite-final.log. Independent implementation/result reviews are stored alongside the reports. Native journals preserve every generated token and complete response; the sampled probe certifies the greedy parent archive remains unchanged. No paid API calls, training, promotion or monitor activation occurred.

Main artifacts: cached-consistency-validation/report.json, probability-invariance.json, NATIVE_ANALYSIS.json, native/completed-v1/result.json, native/sampled-six-v1/result.json, and the immutable input/source manifests in each native archive.
