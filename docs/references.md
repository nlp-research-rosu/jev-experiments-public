# References

What we drew on, and how. "Ran" means we executed released code; "tested" means we implemented the idea and measured it; "adopted" means it shaped the design; "consulted" means it informed a decision without a dedicated experiment.

## Starting point

| Source | How we used it |
|---|---|
| [Harsha Gundala, Parallel Constrained Decoding](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) (revision `2af86848`) | Adopted and extended. Shared prefill, cache broadcast to every field, scoring only permitted answers, and JSON assembled in code. We added Qwen3.5's recurrent-state forking, verified single-token answer codes and full accounting of forward calls, and removed a heuristic confidence floor. Its four example schemas are in `data/upstream/` under Apache-2.0. |

## Community reproductions of Jev

Reviewed read-only at the pinned commits below ([full review](../research-log/reports/REFERENCE_STRATEGIES.md)). None was executed by us.

| Repository | Commit | What we took | What to watch for |
|---|---|---|---|
| [SemIf](https://github.com/TheoLeeCJ/SemIf) | `ca3ba65f` | Frozen direct-logit baselines; perturbations of option order, wording and distractors | No new training method; different readout and hardware |
| [Simple Jev](https://github.com/featherless-ai/simple-jev) | `7cba7d12` | Distribution targets keyed by answer meaning; cached teacher annotations; grouped splits | Teacher probabilities are authored estimates, not calibrated truth |
| [Laya](https://github.com/NandhaKishorM/laya) | `42626c34` | Bidirectional candidate markers, an action head, per-type temperatures | Its own sweep records zero accuracy on Khmer text at about 0.95 mean confidence |
| [Von](https://github.com/wfzyx/von) | `bed7e733` | Candidate markers, dynamic distractors, "other" examples with the gold answer omitted, cross-entropy plus Brier | 48 of 78 benchmark inputs appear verbatim in its training generator's pools; this shows overlap in the recipe, not which data any released weights saw |
| [Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev) | `30f15564` | Candidate scoring with ModernBERT, cross-entropy plus Brier, abstention experiments, scalar calibration | Training is supervised probability fitting despite the RLCD name; inference can truncate context |
| [Kev](https://github.com/jaredpalmer/kev) | `b339f446` | Mixed public, policy and compositional data; rule trees over AND/OR/NOT/UNLESS; KL anchoring; ordinal loss; its lower learning rate transferred better (75.9% against 70.4%) | Its results are on a different model and task mix |
| [Nimble](https://github.com/bespokelabsai/nimble) | `f136b3f7` | Counterfactual pairs that change one evidence sentence, with audits that the intended fact changed and nothing else did; balanced Choice, Noul and Score quotas | Its audits are model judgments, not human review |
| [OpenJev](https://github.com/razorback16/openjev) | `cddbd962` | A different runtime: diffusion canvases with masked answer slots and averaged repeated reads | Token entropy is not a calibration guarantee |

## Papers

### Contrast data and behavioral testing

| Paper | How we used it |
|---|---|
| [Evaluating Models' Local Decision Boundaries via Contrast Sets](https://aclanthology.org/2020.findings-emnlp.117/) (Findings of EMNLP 2020) | Adopted: minimal, meaning-changing edits with reviewed labels |
| [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://aclanthology.org/2020.acl-main.442/) (ACL 2020) | Adopted: capability-organized tests, invariance and directional checks |
| [Polyjuice](https://aclanthology.org/2021.acl-long.523/) (ACL 2021) | Ran the released generator: 3 usable edits of 32 on our tasks |
| [Tailor](https://aclanthology.org/2022.acl-long.228/) (ACL 2022) | Ran the released generator: 6 usable edits of 32, then 3 after a context fix |
| [MiCE: Minimal Contrastive Editing](https://aclanthology.org/2021.findings-acl.336/) (Findings of ACL 2021) | Consulted: a flipped prediction is not a changed label |
| [CoBA](https://aclanthology.org/2025.emnlp-main.520/) (EMNLP 2025) | Consulted: edit structured meaning, then realize varied text |
| [DIPPER paraphraser](https://arxiv.org/abs/2303.13408) (2023) | Consulted: controlled paraphrase for invariance siblings |
| [AdaTest](https://aclanthology.org/2022.acl-long.230/) (ACL 2022) | Consulted: grow tests around discovered failures, keep an untouched outer test |
| [CEval](https://aclanthology.org/2024.inlg-main.6.pdf) (INLG 2024) | Consulted: evaluate generators on validity and minimality, not flip rate |
| [Dually Self-Improved Counterfactual Data Augmentation](https://aclanthology.org/2025.acl-long.260/) (ACL 2025) | Consulted |
| [PAWS](https://research.google/pubs/paws-paraphrase-adversaries-from-word-scrambling/) (NAACL 2019) | Training data, and the high-overlap construction principle |

### Probabilities, calibration and confidence

| Paper | How we used it |
|---|---|
| [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) (ICML 2017) | Tested: temperature scaling cut log loss from 1.55 to 0.51 without changing any decision |
| [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437) (JASA 2007) | Adopted: log and Brier losses as proper objectives |
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015) | Tested: teacher distributions from Qwen3.5-4B and Jev, no gain over a calibrated baseline |
| [When Does Label Smoothing Help?](https://research.google/pubs/when-does-label-smoothing-help/) (NeurIPS 2019) | Consulted: smoothing as a control for teacher targets |
| [Beyond Binary Rewards (RLCR)](https://arxiv.org/abs/2507.16806) (2025) | Consulted: a reward that combines correctness with calibrated confidence |
| [Rewarding Doubt](https://arxiv.org/abs/2503.02623) (2025) | Consulted |
| [ConfTuner](https://arxiv.org/abs/2508.18847) (2025) | Consulted: its tokenized Brier is not categorical Brier over outcomes |
| [An analysis of confidence reward functions, arXiv 2607.04332](https://arxiv.org/abs/2607.04332) (2026 preprint) | Consulted: confidence rewards can be gamed |
| [Just Ask for Calibration](https://arxiv.org/abs/2305.14975) (2023) | Consulted: verbal confidence and token probabilities differ |
| [Adaptive Temperature Scaling](https://arxiv.org/abs/2409.19817) (2024) | Consulted |

### Adaptation and reasoning transfer

| Paper | How we used it |
|---|---|
| [LoRA](https://arxiv.org/abs/2106.09685) (2021) | Adopted: rank-8 adapters on a frozen backbone |
| [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (2024) | Consulted: why retention favored continuing from our base model |
| [rsLoRA](https://arxiv.org/abs/2312.03732) (2023) and [DoRA](https://arxiv.org/abs/2402.09353) (2024) | Consulted: options for a rank study |
| [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301) (2023) | Basis for the running intermediate-supervision experiment |
| [From Explicit CoT to Implicit CoT](https://arxiv.org/abs/2405.14838) (2024) | Consulted: a curriculum for removing reasoning tokens |
| [Coconut](https://arxiv.org/abs/2412.06769) (2024) | Consulted: latent reasoning adds sequential compute |

## Datasets

BoolQ (NAACL 2019), PAWS (NAACL 2019), CLINC150 (EMNLP-IJCNLP 2019), BANKING77 (2020) through Tasksource (2023), and UCI Wine Quality (2009). Licenses and pinned revisions are in [THIRD_PARTY.md](../THIRD_PARTY.md) and the [data preparation report](../research-log/reports/DATA_PREPARATION.md).

## TypeSafe documentation

[API](https://docs.typesafe.ai/api) · [Choice](https://docs.typesafe.ai/primitives/choice) · [Score](https://docs.typesafe.ai/primitives/score) · [Noul](https://docs.typesafe.ai/primitives/noul) · [Confidence](https://docs.typesafe.ai/confidence) · [Known weaknesses of Jev 1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13) · [Structured extraction cascade](https://docs.typesafe.ai/cookbooks/sde_cascade) · [Launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
