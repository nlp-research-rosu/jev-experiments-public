# Jev, our judge and LLM judges

This page collects what we observed about Jev while trying to reproduce it, how it compared with LLM judges in a separate audit, and what a small model can do when it is allowed to reason.

## What is public about Jev

Jev is TypeSafe's "System One" decision model. It reads a state once and answers typed questions in parallel: Noul, Choice and Score, each with a probability distribution. It does not generate text. All our comparisons used `jev-1.13.0`, published at $0.042 per million input tokens. In our calls, a request with several questions took about 0.35 seconds including the network.

TypeSafe describes a new architecture, a parallel sampler and a training method it calls RLCD, aimed at calibrated decisions. The architecture, parameter count, tokenizer and training recipe are not public, and our experiments do not reveal them ([source review](../research-log/reports/JEV_API_MODEL_BOUNDARY.md)).

## Jev on our tests

Jev was an external reference in every comparison: it never saw our training data, and our labels, not Jev's answers, defined correctness.

| Test | Judgments | Our best model | Jev 1.13 |
|---|---:|---:|---:|
| Paired-language test, 35 untouched families | 700 | 88.1% | 96.9% |
| Paired-language test, all 40 families | 800 | 87.0% | 95.5% |
| Contrast pilot, fresh domains and layout | 700 | 95.3% | 99.4% |
| Revised contrast test (from our generator) | 1,185 | 98.7% | 95.2% |
| Original semantic-contrast suite | 764 | 83.5% | 99.5% |
| Live check, eight selected requests | 51 | 30 (base model) | 51 |
| Teacher audit, 80 development cases | 400 | 58.0% (base model) | 96.5% |
| TypeSafe's published example workflows, reference agreement | 336 | 81.0% (base model) | 91.1% (stored answers) |

The same rows, with sources, are in [results/jev-comparisons.csv](../results/jev-comparisons.csv). The revised test is the one place our model led: its families came from the same generator as our training data, and most of Jev's misses there followed a "latest snapshot still holds" reading that the test's contract ruled out.

### Where Jev was strong

- **Claims versus evidence.** For an assistant's unsupported claim that a job was done, Jev put 6% on confirmation; our base model put 96.5%. For a timed-out call, Jev put 5% on completion; our base model put 95.3% ([live check](../research-log/reports/JEV_LIVE_SMOKE.md)).
- **Unknown, when offered.** For an unverified record and for two equally authoritative records that conflict, Jev put 100% on Unknown. Our base model put 27.5% and 22.2%.
- **Probability quality.** On the paired-language test, Jev gave one or two wrong answers with at least 90% probability out of 700–800, against 55–65 for our model. Its mean negative log-likelihood was 0.09–0.12, against 0.55–0.57.

### Where Jev erred

From our error analyses ([paired language](../research-log/reports/paired-language-v1/jev-error-notes.md), [revised test](../research-log/reports/REVISED_INITIALIZATION.md), [contrast pilot](../research-log/reports/CONTRAST_PILOT.md)):

- **Absent evidence read as failure.** Where no attempt was made or no record existed, Jev sometimes chose a definite failure over Unknown (0.56–0.73).
- **Stale snapshots read as current.** It failed 32 of 36 temporal-scope judgments whose contract required evidence from the requested time window.
- **Approvals missed in some layouts.** It missed 18 of 84 recorded policy approvals, 17 of them in two of the three layouts.
- **Definitions in sibling questions.** Where a threshold was defined only in another question, Jev scored 86 of 100 judgments in the affected families; with the definitions copied into the shared state, 99. The questions really are independent.
- **Drift under rewording.** In one test, 9 of 76 meaning-preserving pairs moved by more than 0.05 in probability, although every answer stayed correct.

None of this contradicts TypeSafe's own list of known weaknesses, which includes incorrect decisions and no guaranteed consistency between related questions.

## Jev and LLM judges in a formal-verification audit

In a separate experiment, each judge received an intended behavior and a formal semantics written for it, and decided whether the semantics truly represents the intent. There were 45 cases: 27 flawed and 18 faithful.

![One pass versus deliberation in an audit](../results/figures/06-jev-vs-llm-judges.png)

| Judge | Accuracy | Flawed accepted | Faithful rejected | Latency | Cost per case (OpenRouter) |
|---|---:|---:|---:|---:|---:|
| Jev | 80.0% | 0 / 27 | 9 / 18 | 1.1 s | ~$0.0006 |
| Sonnet | 100.0% | 0 / 27 | 0 / 18 | 45.7 s | ~$0.17 |
| Luna | 95.6% | 2 / 27 | 0 / 18 | 69.6 s | ~$0.002 |
| Sonnet, one turn without tools | 100.0% | 0 / 27 | 0 / 18 | 42.6 s | ~$0.12 |
| Luna, one turn without tools | 97.8% | 1 / 27 | 0 / 18 | 24.6 s | ~$0.002 |

Jev was about 40 times faster than Sonnet and 200–280 times cheaper, and its errors ran one way. Every one of its 9 acceptances was right, while 27 of its 36 rejections were right. Forty-five cases is a small sample: zero false acceptances out of 27 flawed cases still allows a true rate of about 11% (one-sided 95% bound).

For an audit where accepting a flawed semantics is the expensive error, that profile suits a first stage: accept what Jev accepts with a high threshold, and send everything else to a deliberate judge. On this sample such a cascade would have sent 36 of 45 cases to the slower judge, so its main benefit is safety rather than savings. The rows are also in [results/llm-judge-audit.csv](../results/llm-judge-audit.csv).

## Deliberation and a single pass

Can a 2B model make these distinctions at all? We let the original Qwen3.5-2B reason before answering 30 held-out judgments, one case per semantic category and one question per primitive ([results](../research-log/reports/consistency-native-v1/RESULTS.md)).

![The distinctions are reachable with deliberation](../results/figures/05-deliberation-vs-single-pass.png)

- Fifteen answers finished within 4,096 thinking tokens, and all 15 were right.
- On those 15, the same model scoring directly without reasoning was right on 9, and our trained judge on 13.
- The other 15 hit the cap, often repeating themselves. Over all 30, reasoning answered 15 correctly, our judge 23 and direct scoring 13.
- Reasoning used 90,289 thinking tokens and 20 minutes for the 30 questions. Our judge answers a five-question request in about 0.18 seconds.

The finished traces are selected by termination, so 15 of 15 is conditional. Still, the capability is present in the weights. In a six-case follow-up, the sampling settings recommended on the model card let four of six previously capped traces finish. The open problem is to make these distinctions without chain of thought, at single-pass cost.

## What we take from this

- **Different tools for different judgments.** A fast judge handles what the evidence shows directly; a deliberate model is still needed for what has to be worked out. The formal-verification audit and the reasoning probe point the same way.
- **Error direction matters as much as accuracy.** Decide which error is expensive in your application and place thresholds, or a cascade, accordingly.
- **Accuracy is the easier target to match.** Our 2B model came within 9 points of Jev's accuracy in about 70 minutes of training. Matching its probability quality, particularly the rarity of confident mistakes, is what we have not achieved.
