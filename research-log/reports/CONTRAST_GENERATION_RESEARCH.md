# Research for generating controlled semantic contrasts

Primary sources checked 2026-09-20. This follows the local
[contrast inventory](../design/contrast-taxonomy.md), compiled first from existing
experiments and supplied discussions. This is a research shortlist and proposed
adaptation, not a benchmark of these generators on OpenJev. No new model weights,
training data, or paid Jev responses were generated for this review.

Subsequent work: the [first generation pilot](GENERATION_PILOT.md) compared
Luna/Terra on 96 candidates and retained 48 Terra requests as development data.
It reports construction failures separately from blind semantic-label agreement;
no student training has yet tested those generated records.

Further follow-up: the [actual tool and effort probe](TOOL_EFFORT_PROBE.md) now runs
Polyjuice/Tailor generator cores and compares Terra/Sol low/medium/high effort. The
research-only “not run” statuses below describe the original literature review;
the follow-up records precisely what ran and what parts of the old wrappers were
not reproduced.

## Main finding

The most relevant literature calls this **controlled text perturbation**,
**counterfactual data augmentation**, **contrast sets**, and **semantic control**.
Paraphrasing is one component. A useful generator must make either a specified
meaning change or a meaning-preserving rewrite, and tell us which it attempted.
The acceptance decision still requires checking what the resulting text says.

My recommendation is to use modern instruction models as writers, with explicit
semantic controls inspired by Tailor and CoBA, plus independent review and the
evidence-necessity checks already found in Nimble. Treat specialist generators
as comparison baselines or sources of additional edit strategies. This is an
engineering recommendation for our setting, not a demonstrated ranking of models.

## Papers and tools mapped to our needs

IDs below refer to the [inventory](../design/contrast-taxonomy.md).

| Reference | What it contributes | Useful local contrasts | Limitation / available artifact |
|---|---|---|---|
| **Tailor — ACL 2022** | Conditions generation on semantic-role representations and composable controls, including role/content and tense/voice changes. This directly separates what should change from how text is realized. | S01/S04 tense and completed-versus-planned wording; S12/S13 roles and arguments; I01/I02 equivalent expression | Linguistic controls do not themselves define our evidence or permission rules. Public code, contrast sets, augmentation data and automatic generator-loading API; not run here. [Paper](https://aclanthology.org/2022.acl-long.228/), [code](https://github.com/allenai/tailor). |
| **Polyjuice — ACL 2021** | Generates sentence perturbations with selectable spans and edit controls: negation, quantifier, lexical, restructure, shuffle, insertion/deletion and semantic changes. It supports both prediction-changing and prediction-preserving exploration. | S04 negation/plans; S12 argument changes; S21 quantifier scope; I01/I02 wording diversity | An edit category is not a guaranteed label transformation. Even its negation examples can negate a different scope. Public generator `uw-hai/polyjuice`. [Paper](https://aclanthology.org/2021.acl-long.523/), [code and controls](https://github.com/tongshuangwu/polyjuice), [weights](https://huggingface.co/uw-hai/polyjuice). |
| **MiCE — Findings ACL 2021** | Uses a trained editor and search to find small fluent changes that produce a contrastive model prediction. | Find candidate hard cases near our current model's decision boundary | Flipping the model prediction does not prove the correct answer changed; using our flawed model as judge can reproduce its shortcuts. Released editors/predictors target IMDB, Newsgroups and RACE, not general agent records. [Paper](https://aclanthology.org/2021.findings-acl.336/), [code/checkpoint instructions](https://github.com/allenai/mice). |
| **CoBA — EMNLP 2025** | Decomposes text into subject–predicate–object triples, edits selected triples, then reconstructs varied text. Targets spurious correlations and broader augmentation rather than only minimal surface edits. | S12/S13 argument binding; S19/S20 changes to structured meaning; independent realizations for I02 | A triple alone can lose time, attribution, uncertainty and negation scope. Our adaptation needs richer event/evidence records. Paper includes generation prompts; no project implementation was verified in this review. [Paper](https://aclanthology.org/2025.emnlp-main.520/), [full text and prompts](https://aclanthology.org/2025.emnlp-main.520.pdf). |
| **DIPPER — 2023** | A T5-XXL paraphraser with surrounding context, lexical-diversity and content-order controls. Useful for producing alternative prose rather than repeating one template. | I01/I02 wording diversity; paragraph-level realizations | Primarily an invariance writer, not a factual counterfactual generator. Reordering event descriptions must preserve temporal meaning. Released 11B weights and sample code; semantic preservation still needs review. [Paper](https://arxiv.org/abs/2303.13408), [author code](https://github.com/martiansideofthemoon/ai-detection-paraphrases), [weights](https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl). |
| **AdaTest — ACL 2022** | Uses a generative model and human feedback to grow organized behavioral tests around discovered failures. | Expand S03 predicate confusion into independently worded subfamilies; explore interactions rather than repeat fixtures | Adaptive examples used to improve the system become development data; reserve an independent outer test. The official repository is archived/read-only as of this review, so borrow the process without assuming a current maintained stack. [Paper](https://aclanthology.org/2022.acl-long.230/), [official code](https://github.com/microsoft/adaptive-testing). |
| **CEval — INLG 2024** | Compares counterfactual generators using both counterfactual and text-quality measures; releases benchmark code/data/results. Finds tradeoffs between plausible prose and satisfying counterfactual criteria. | Design our generator comparison: edit minimality, intended change, untouched facts, fluency and accepted-family yield | It evaluates tasks such as sentiment and NLI, not our full evidence contract. Use its methodology, not its target-model flip rate as a truth oracle. [Paper](https://aclanthology.org/2024.inlg-main.6.pdf), [official code](https://github.com/aix-group/CEval-Counterfactual-Generation-Benchmark). |
| **Dually Self-Improved Counterfactual Data Augmentation — ACL 2025** | For NLI, combines task-specific token selection, preference-based improvement of generated counterfacts, and balanced original/augmented training. | Later generator-quality optimization after we have reliable accepted/rejected edit pairs | More machinery than the next small pilot; evidence is from NLI. Does not establish that attention-selected words are causal evidence in our tasks. No runnable release was verified here. [Paper](https://aclanthology.org/2025.acl-long.260/). |

## Existing datasets and evaluation foundations

**PAWS is especially relevant to the user's “similar words” requirement.** It
deliberately contains high-overlap pairs that either do or do not preserve meaning,
using controlled swapping/back-translation followed by human judgments. We already
train on a PAWS subset as an equivalence task. We can borrow its construction
principle, but a negative paraphrase label does not imply every downstream
predicate must flip. [Primary paper page](https://research.google/pubs/paws-paraphrase-adversaries-from-word-scrambling/).

**Contrast Sets** argues for small, meaningful changes with reviewed gold labels
to expose local decision boundaries. **CheckList** organizes capabilities across
test types rather than relying only on aggregate test accuracy. These were already
references for our original semantic suite; the gap is breadth and independence
of the generated language, not a missing name for the evaluation method.
[Contrast Sets](https://aclanthology.org/2020.findings-emnlp.117/),
[CheckList](https://aclanthology.org/2020.acl-main.442/).

Our existing reference folder also has useful generation/audit machinery:

| Already local | Reusable idea | Important boundary |
|---|---|---|
| Nimble | Change one evidence sentence, audit changed and unchanged facts, remove necessary evidence to test sufficiency | Model audits are not independent truth proofs; upstream certificates can use the same model as generator/verifier. |
| Kev | Generate varied logical structures and hold out canonicalized rule shapes | Useful for rules, not sufficient for natural prose or attribution. |
| Simple Jev / RFDT | Cache teacher annotations, retain supplied labels, group related records before splitting | Teacher-authored probability estimates are not automatically calibrated targets. |

These findings are documented in the earlier [pinned-code review](REFERENCE_STRATEGIES.md).

## Concrete generator options

| Candidate | Practical role | What is verified / not verified |
|---|---|---|
| **Luna/Terra agents already discussed with the user** | First small batch of independently written cases and rewrites using a strict family specification | Available in this workspace; not compared on generation acceptance rate. Separate author/reviewer instructions, preferably different model families for some audits, can reduce correlated mistakes without guaranteeing correctness. |
| **Qwen3.5-9B** | Candidate local instruction-model writer or additional reviewer | Official post-trained weights are available. The full checkpoint files total about 19.3 GB, so an entirely GPU-resident unquantized load plus runtime overhead is unsuitable for our 16 GB 4080. Quantization/offload would need a separate smoke test; no download or fit/throughput measurement was done. It is a candidate, not a known best generator. [Official model](https://huggingface.co/Qwen/Qwen3.5-9B), [files](https://huggingface.co/Qwen/Qwen3.5-9B/tree/main). |
| **`uw-hai/polyjuice`** | Small specialist baseline for controlled, localized sentence edits | Public GPT-2-family checkpoint and span/edit APIs verified. Its released config has 12 layers and width 768. Actual installation, memory and generation quality remain untested. [Model](https://huggingface.co/uw-hai/polyjuice), [config](https://huggingface.co/uw-hai/polyjuice/resolve/main/config.json). |
| **Tailor generator through `tailor_nlp`** | Baseline for explicit role/tense/voice manipulation | Official API downloads its generator; dependency list includes old AllenNLP/Transformers versions. Investigate in an isolated environment if selected, rather than changing the working training environment. [Repository](https://github.com/allenai/tailor), [requirements](https://github.com/allenai/tailor/blob/main/requirements.txt). |
| **`kalpeshk2011/dipper-paraphraser-xxl`** | Optional context-aware paraphrase baseline | Published 11B model. At two bytes per parameter, weights alone are roughly 22 GB, above 16 GB before runtime memory: plain BF16/FP16 GPU-only use is not a fit. Quantized/offloaded behavior is untested here. Its implementation converts requested diversity to similarity codes (`100 − diversity`); use the author's wrapper. [Model card](https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl), [minimal script](https://github.com/martiansideofthemoon/ai-detection-paraphrases/blob/main/dipper_paraphrases/paraphrase_minimal.py). |

The specialist methods are older because they expose particularly useful control
interfaces, not because their generators are known to outperform current general
models. None of the reviewed papers establishes the best generator for our exact
claim/confirmation/Unknown/rubric task. Compare accepted semantic families per
unit of generation and review effort, not just raw generated row counts.

## Proposed adaptation for our next small batch

This is our synthesis, not an implemented pipeline or a claim made by any one paper.

1. **Specify meaning before text.** Define predicates, actor/target, action,
   timestamps, modality, quotation scope, source authority, result and permission.
   Record whether the task asks about a claim, a world fact or what evidence
   establishes. Keep exact complements distinct from Unknown.
2. **Choose the intended relation.** Generate one atomic evidence change, one
   predicate change on the same evidence, and one invariance sibling where useful.
   State which labels should change and which should stay fixed. Keep positive
   and negative controls, not only difficult negatives.
3. **Use two writing strategies.** For a minimal contrast, edit the allowed span
   while preserving the rest. For independent realization, ask another writing
   pass to express the same semantic record in a different discourse style without
   seeing the canonical wording. Include claims, tool excerpts and narrative
   descriptions without using the gold label as text. Apply comparable style
   variation to both positive and negative members.
4. **Review the actual text, blind to the proposed label first.** Recover the
   facts and predicate meaning, then compare with the intended record and gold.
   Check the changed fact, untouched facts, missing qualifiers, fluency and
   sufficiency. A round-trip extraction or NLI model is a screen, not a proof;
   disagreements need adjudication. Review a sample of accepted cases as well
   as rejected cases to estimate missed errors.
5. **Audit dependencies.** Remove evidence claimed to be necessary; the judgment
   should become unresolved only if no redundant support remains. Add an
   irrelevant failed tool call; it should not erase a matching successful record.
   Evaluate all predicates on each record, because one changed fact may affect
   several outputs and leave the others invariant.
6. **Split by semantic family before expansion.** Keep edits, paraphrases,
   renderings and reused source text together. Hold out language patterns,
   predicate formulations, rule structures and domain combinations. Use current
   known failures as development examples; write a new outer test independently.
7. **Keep quality and uncertainty separate.** Reject unintended semantic drift;
   do not disguise a flawed example with a softer label. Legitimate ambiguity
   can be separately annotated for a later distribution-target experiment.
   Preserve all paid Jev observations and provenance, as already implemented.

### Small worked family (illustrative, not added to training)

Predicate A: **Does the assistant explicitly assert that sending invoice 17 is
already complete?** Predicate B: **Does the supplied authenticated execution
record confirm that invoice 17 was sent?** A missing result means B=false for this
evidence predicate; it does not prove that sending physically failed.

| Assistant statement | Execution evidence | A: claimed completion | B: completion confirmed |
|---|---|---|---|
| “I've sent invoice 17.” | No sending event/result is supplied | True | False |
| “I've sent invoice 17.” | Matching authenticated real sending result: success | True | True |
| “I'll send invoice 17.” | Later matching authenticated real sending result: success | False | True |
| “I'll send invoice 17.” | No sending event/result is supplied | False | False |

A writer can vary realization (e.g. “Invoice 17 has gone out” versus “Invoice 17
is next on my list”) while a reviewer checks that completed-versus-future scope
survives. For a categorical execution-status question, missing evidence should
be Unknown, explicit no-effect rejection should be confirmed failure, and a
matching successful event should establish success under its contract.

### What to measure before scaling

| Dimension | Measurement |
|---|---|
| Semantic validity | Intended label relation and all unchanged relevant facts survive independent review |
| Contrast strength | Both endpoints correct; question-only and evidence-only pairs reported separately |
| Surface control | Token edit distance / overlap for minimal pairs; syntactic and wording diversity for independent realizations |
| Unintended cue dependence | Labels predicted from questions alone, formatting alone or source markers; only interpret as an artifact where those views should be uninformative |
| Generation efficiency | Accepted complete families, rejection reasons, review effort and duplicate-family rate |
| Student effect | Per-predicate positive/negative accuracy, Unknown recall, NLL/Brier, confident errors, Score accuracy/MAE and broad retention |

A rise in embedding diversity, fluency, model disagreement or model-flip rate
alone does not establish better supervision. The most useful immediate comparison
is our old template approach versus independently written, semantically audited
families at a matched training budget; distribution targets can then be a separate
controlled factor.
