# Remaining options after the calibrated screen

2026-09-22. Originally a discussion/research note. The subsequently authorized [bounded architecture diagnostics](../reports/architecture-diagnostics-v1/RESULTS.md) are now complete; no follow-on rank/unfreezing study is launched. The [six-run results](../reports/calibrated-screen-v1/RESULTS.md) remain complete and their monitor is paused. These options extend the [staged roadmap](calibrated-training-roadmap-v1.md), not the frozen completed protocol.

## What the existing experiment held fixed

The student uses Qwen3.5-2B with 1,890,238,785 parameters in the recorded adapted model inventory. Rank-8, alpha-16 LoRA updates attention, gated-recurrence projections and feed-forward projections across the body. Trainable adapters contain 8,409,600 parameters; two linear readouts contain 4,097. Total trainable parameters are 8,413,697, about 0.45%. The original backbone matrices remain frozen, but LoRA changes their effective transformations throughout the network; this is not merely head-only training.

The binary head has a bias; the compatibility head is bias-free and shared between Choice and Score. Both read the final prompt-token representation. Heads were initialized from the original vocabulary A-minus-B projection, then trained. Original embeddings, normalization parameters, recurrence parameters outside the named projections, and convolution weights are not directly updated. Sources: [model implementation](../../src/openjev/judgment_model.py) and the checkpoint's recorded LoRA policy.

The six latest arms varied objective and teacher distribution, not rank, placement, readout architecture, base initialization, student size, or input representation. Repeated errors under these fixed choices do not isolate which is limiting. The earlier [component swap diagnostic](../reports/CONTRAST_DIAGNOSIS.md) found that trained heads could amplify certainty, while resetting heads did not restore completion semantics; that was an older checkpoint and a coadapted-component probe, not unique causal attribution.

## A verified information-flow limitation

The [compiler](../../src/openjev/judgments.py) sends each Choice/Score candidate to a separate scoring branch. Each branch gets state, question and its own criterion. Choice/Score probabilities are normalized together, so the loss couples candidates, but there is no automatic full-candidate-set input or cross-candidate feature interaction in the model. Some experiment states already contain the full rubric, which can supply the missing information indirectly.

A read-only compiler check (no model/GPU) confirmed:

- Fixed question: select the second-largest number among supplied candidates.
- Candidates A=2, B=3: answer A.
- Add C=4 without changing state/question/A/B: answer B.
- The rendered A and B inputs are byte-for-byte identical in both requests.

With `z_i=f(state, question, criterion_i)` and plain softmax, `p_A/p_B=exp(z_A-z_B)` cannot change when only C is added. Consequently this interface cannot represent every set-dependent choice rule. More LoRA capacity or a nonlinear per-candidate head alone cannot supply absent set information.

This demonstrates a generic API expressiveness boundary, not the established cause of the current benchmark gap: many of its states explicitly include all levels/definitions, and binary failures do not require comparing candidates.

Candidate-context experiment: hold facts/labels/backbone/loss fixed and give every candidate the complete rubric plus its selected criterion. Separately test a whole-question selector or a small permutation-aware candidate interaction layer. Preserve dynamic answer spaces; do not replace the generic API with a classifier whose output classes are fixed business labels. Test option permutation, additions/removals, meaning-changing rubric contrasts and latency. Comparing isolated versus full-context inputs without retraining is an interface diagnostic, not a fair trained-model architecture comparison.

## Staged diagnostic and adaptation options

| ID | Intervention | Question answered | Main controls / costs | Status |
| --- | --- | --- | --- | --- |
| DIAG-FIT | Inspect accuracy on a small audited training subset and held-out contrast families; use a bounded fitting check if needed | Can this interface/optimizer learn the training distinctions at all? | Separate fitting, generalization and label/input defects; low training loss alone is insufficient | D3 complete: 80 updates, fit 183→199/200, development 151→151/200; perfect-fit criterion inconclusive at cap |
| DIAG-NATIVE | Same 2B in its original answer interface with full rubric; compare no-thinking and bounded reasoning | Is the capability accessible in the original model/interface or only with extra computation? | Exact same visible evidence; original post-trained release versus OpenJev adapters explicit; diagnostic latency is not production latency | Follow-up complete: 30 questions at 4,096 thought tokens; 15 naturally completed, all correct; 15 capped. Matched direct 9/15 vs reasoning 15/15. Separate six-loop-case sampling probe: 4 completed thoughts/correct constrained answers, 3 valid natural answers. Capability demonstrated on some cases; full coverage unresolved |
| NUMERIC-BATCH | Deterministic per-unit cache prefix and fixed physical suffix layouts | Can request grouping change numerical results? | Keep weights/kernels fixed; preserve sharing; measure time, memory, probabilities and label flips | Fixed on tested runtime: exact zero differences across 43 cases / 408 units, about 6.4% median cached latency overhead; see consistency-native-v1 |
| INPUT-SET | Complete candidate set in every scoring unit | Does missing comparative context impede decisions? | Same facts, training labels and readout; measure added tokens/cache reuse and unseen-set transfer | D0/D1 complete: 152→155/200, raw NLL 1.002→.627; real states already contain banks; all 6 set-flip sentinel pairs unsolved in both modes |
| HEAD-SPLIT | Separate Choice and Score compatibility heads | Does sharing one readout cause interference? | Same scalar candidate interface and body; one changed factor | D4 complete: 200/200 fit; no development correctness improvement over refitted shared linear head |
| HEAD-MLP | Linear readout versus small nonlinear candidate readout | Are useful features present but not linearly readable? | First fit heads with the same frozen representations; retain variable candidate counts; then separately test body coadaptation | D4 complete: current linear head already fits 200/200; MLP has same development correctness as linear, sharper errors; joint body/MLP training remains untested |
| HEAD-SCALE | Lower head LR or normalized readout with a controlled positive scale | Is logit-scale growth dominating learning? | Compare calibrated CE; this may soften confidence without fixing semantic ranking | Unrun as a controlled training comparison |
| ADAPT-RANK | Rank8 versus32/64 | Is rank8 constraining adaptation? | Match starting predictions, module placement, exposure; control scaling/LR; report memory and retention | Unrun |
| ADAPT-PLACEMENT | Attention/recurrent projections versus feed-forward versus both | Where do useful updates occur? | Current setup already covers both; removing a placement is an ablation, not fixing a missing MLP target | Unrun |
| ADAPT-UPPER | Full updates to a final complete block group and final normalization, with lower layers fixed or identically adapted | Do unrestricted updates improve over low-rank changes? | Prespecify exact trainable set; same readout/data; profile local memory and forgetting | Unrun |
| ADAPT-FULL | Full 2B fine-tuning as a capacity control | Is limited adaptation capacity the bottleneck? | Conventional optimizer memory can exceed16GB; profile/offload or use larger-memory hardware; match data/selection and report different compute | Unrun |
| ADAPT-VARIANT | rsLoRA or DoRA | Does a different update parameterization help? | Compare against well-scaled LoRA at the same rank before attributing gains to rank | Unrun |
| INIT | Fresh original instruction model versus pretraining-only Base | Does previous task/upstream post-training matter? | Earlier fresh-adapter pilot existed, but was not a matched Base-model test of this recipe; equal task-interface preparation matters | Still unresolved |
| CAPACITY | Larger student with selected recipe | Does additional backbone capacity help? | A larger teacher is not this test; measure student latency/memory/compute and retention | Unrun |
| STRUCTURE | Auxiliary evidence/source/role/time supervision or reasoning distillation | Does supervision teach intermediate semantic distinctions? | Verify intermediate targets; retain direct-judgment training so inference does not depend on missing rationales | Unrun |
| CONSISTENCY | Certified invariance/complement loss | Does relation supervision improve correct consistency? | Roadmap S3; no equality constraints across valid rubric changes; keep both-correct metrics | Unrun in the latest contrast screens |
| TARGET-CONTROLS | Uniform smoothing and teacher-top-answer controls | Were teacher gains semantic information or generic softening? | Roadmap S2; same samples and target-loss allocation | Unrun |
| DISTRIBUTIONS | Known conditional probabilities with visible priors/noise | Does honest probability supervision transfer? | Different data experiment; do not soften deterministic Unknown merely because it represents missing evidence | Unrun |
| RL | Sampled reasoning/verifiable reward, optionally a teacher later distilled to the fast student | Does learning a decision procedure help? | New interface/verifier/cost; not an automatic confidence fix | Unrun |
| BACKBONE | Different attention/encoder architecture | Is the body architecture imposing the relevant limitation? | Different pretrained models also change data/training history; not a clean architecture-only comparison | Later option |

## How to compare heads and trainable parts fairly

The [consistency and completed-reasoning follow-up](../reports/consistency-native-v1/RESULTS.md) provides a tested stable cached inference policy and positive evidence of original-model reasoning on some semantic judgments. Use the [new inference guide](consistent-inference.md) for subsequent inference experiments. Preserve the distinction between completed-subset accuracy and all-question coverage: the small native run is not a uniformly superior replacement for the trained fast judge, and the six-case sampling follow-up is not a full-cohort benchmark.

Use the existing hard-CE recipe as a reference, retaining raw and separately calibrated metrics. Start with one axis at a time. If a new head improves a frozen-feature probe, it is evidence that useful information is more easily accessible to that readout, not proof that the full model has learned a general reasoning rule. If it fails, that does not prove the information is absent.

A compact model comparison could be: current rank8 + linear heads; rank32/64 + the same heads; rank8 + an MLP head; then a final-block full-update control. Add the head/capacity interaction only if individual results warrant it. Candidate-context changes form a separate first comparison, not a hidden change in every arm.

Preserve a common initial function when changing rank or unfreezing. For example, merge the fixed original v0.2 update into an isolated copy, then attach fresh zero-output adapters to every rank arm; this requires a matching rank8 control and verification of initial logits. Simply comparing a newly initialized high-rank adapter against a continued trained rank8 adapter would mix initialization and capacity. The originals stay immutable.

Rank scaling is not a cosmetic parameter: changing rank with fixed alpha changes `alpha/r`; rsLoRA uses `alpha/sqrt(r)`. Tune/declare scaling and learning rates on development, not the final comparison set, and report gradient/update norms and clipping. A more flexible head, larger rank or full fine-tuning can also increase overconfidence. They must earn gains in semantic decisions and calibrated probability quality rather than just raw confidence.

LoRA rank and which matrices are trained are training choices, not necessarily extra inference cost. Compatible merged adapters and full fine-tuning retain the base model's inference shape. Actual merge/output/kernel equivalence and latency still need verification on this hybrid model. A nonlinear head adds some work; fuller rubrics/extra reasoning can add substantially more.

A final-block experiment is a practical intermediate step before full2B training. Keep embeddings frozen initially; test normalization/recurrent-specific parameters only as named hypotheses. Full fine-tuning with ordinary optimizer states is memory-heavy on the16GB4080; a larger-memory GPU would enable that controlled test, not guarantee an accuracy improvement.

## Suggested order for discussion

1. Establish training fit versus transfer, and compare the original answer interface/full candidate context on a small frozen diagnostic.
2. Probe linear versus split/nonlinear heads using fixed representations; check retained capabilities and probability scale separately.
3. Compare adapter rank, then an explicitly selected full-update block group if needed, keeping data/loss/head fixed.
4. Explore intermediate supervision/consistency, better teacher target construction, initialization or larger student based on those outcomes.

This order investigates information, readout and adaptation before assuming a fundamental small-model ceiling or paying for a large full-model study. It does not assert that any single change will close the Jev gap. The remaining options and future execution require discussion; the completed study's monitor stays paused.

## Primary references checked

- [LoRA](https://arxiv.org/abs/2106.09685): low-rank adaptation, parameter/memory efficiency and mergeable updates.
- [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673): full fine-tuning outperformed low-rank LoRA in studied programming/math settings, while LoRA retained more outside-domain performance; not proof for this task.
- [Rank-Stabilized LoRA](https://arxiv.org/abs/2312.03732): rank-dependent scaling and higher-rank training behavior.
- [DoRA](https://arxiv.org/abs/2402.09353): magnitude/direction decomposition; different adaptation parameterization with reported gains on its tasks.
- [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301): rationale supervision as a separate training signal for smaller models; no guarantee of transfer to our evidence contracts.
- [TypeSafe primer](https://docs.typesafe.ai/introduction/machine-learning-primer): public calibrated-decision goal, not a disclosed Jev backbone/freezing/head recipe.
