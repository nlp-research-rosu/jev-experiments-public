# Fresh initialization and strategies from the community references

Read-only review of the eight pinned community repositories, 2026-09-20.
Sources below are the checked-in implementations and artifacts, not newly
reproduced upstream benchmarks. No reference was updated, installed or executed;
no model weights, local test suites or paid Jev observations were changed.

## What our experiment does and does not say about restarting

The pilot's **starting checkpoint was our already fine-tuned v0.2**, not the
original Qwen checkpoint. Both the original-data control and the contrast arm
continued from that same checkpoint with fresh optimizers. We have not compared
fresh adapters against continued adapters on the revised training mixture.

Starting over should mean loading the original **Qwen3.5-2B checkpoint before our
fine-tuning**, initializing new LoRA adapters and resetting our numerical readouts.
It does not mean random-weight pretraining, or undoing Qwen's own post-training.
This is easy to separate because the original backbone weights remain frozen.

A broad-plus-contrast mixture from the outset is a reasonable hypothesis: it
could avoid teaching one objective and then shifting heavily toward another.
However, the first run also raised our broad numerical-interface accuracy from
65.5% to 87.3%. Treating that entire run as noise would discard useful learning.
The later losses could reflect mixture weights, learning rate, narrow templates
or representation shift, rather than damage that requires a restart.

The recent pilot gives two particularly direct clues:

- 46 of its 53 net broad-test losses came from Score tasks. New contrast examples
  contained no Score judgments, so the effective primitive mixture changed.
- The contrast model scores 112/112 on a small probe with structured task fields,
  including unfamiliar question wording, but 106/112 with a nested record and
  prose task brief. Resetting weights alone does not create missing format diversity.

Keep the existing frozen suites as regression tests. Add contrastive examples
to the **training partition** of the original task collection, and create a new
versioned test extension with separate families. Do not turn current inspected
test cases or their siblings into training records and then call the same scores
held-out performance.

## Most useful references for the next training comparison

### Nimble: audit the evidence change itself

Nimble constructs a base/counterfactual pair by changing exactly one evidence
sentence while retaining the policy. It checks that the intended atomic fact
changes, other facts do not, and removing either of two supporting sentences
makes the focus unresolved. Its training loader reconstructs the evidence
certificate before accepting the record. This is stronger supervision hygiene
than simply asking a generator for one positive and one negative example.

Relevant code: pair construction and audits,
certificate checks.

The published corpus contains **2,676 training examples / 1,338 pairs**: 856
Choice, 888 Noul and 932 Score. Its 324-example evaluation has six source families
disjoint from the 34 training families. These quotas are directly relevant to
our Score omission. Manifest.

Its trainer builds fresh LoRA on pinned Qwen3.5-9B and uses ordinary hard-label
cross-entropy over permitted answer tokens. Inner family splits select settings
before a fresh refit. Its saved aggregate reports 292/324 versus 215/324 for the
base, but this is a synthetic-label result, not broad retention evidence or a
restart-versus-continuation comparison. Its “mixed” corpus mixes domains and
generation sources within contrastive data; it is not our proposed mixture of
broad public tasks plus contrasts.
Trainer,
selection/refit,
saved results.

The audits are model judgments, not independent truth proofs. The stored
certificate explicitly marks `human_reviewed=False` and
`verifier_independent_model=False`. We can borrow the audit questions while
retaining deterministic checks and independent review where available.

### Kev: joint mixture, structural variety and restrained updates

Kev constructs fresh Qwen3-base LoRA plus a learned pointer readout. It combines
public task records with programmatic policy and compositional cases. Its v7
manifest records 10,000 public, 896 policy and 1,680 compositional records.
Rule trees vary AND/OR/NOT/UNLESS/IF structure, with held-out structures filtered
after canonicalization and De Morgan normalization. This is a useful expansion
beyond changing entity names in our 25 fixed templates.
Construction and sampling,
rule structures and invariant siblings,
mixture manifest.

Its stored comparable 4B, seed-zero runs provide a practical caution:

| Recipe | In-domain accuracy | Transfer accuracy |
|---|---:|---:|
| Learning rate 2e-4 | 84.9% | 70.4% |
| Learning rate 5e-5 | 84.3% | 75.9% |
| 2e-4 with triple synthetic repetition | See source ledger | 68.6% |

Baseline ledger,
lower-rate ledger,
mixture ablation.
Other lower-rate seeds reach approximately 75.8–75.9% transfer. These are upstream
recorded results on a different model/task mix, not our measurements; historical
code hashes also differ. They support testing optimization strength alongside
data balance rather than assuming that more synthetic data or a fresh start is
the sole answer.

Kev implements optional KL anchoring to frozen-base option distributions,
aligning probabilities by option keys and skipping changed candidate sets.
For our continuation arm, a separately tested analogue could anchor broad replay
to the frozen v0.2 model, especially Score. That is our proposed adaptation, not a
demonstrated guarantee: stronger anchoring hurts some Kev runs, and its selected
v7 4B configuration has `anchor_w=0`.
Anchor targets,
KL and ordinal losses,
anchor experiments.

It also implements an optional ranked probability score for ordered levels,
per-task retention checks and group-aware pair evaluation. Those are useful
separate loss/evaluation experiments; changing all of them during a restart
comparison would obscure what helped. Its architecture uses block-causal question
branches and option-end representations, unlike our isolated criterion scorer.
That packing is not a drop-in change for Qwen3.5's recurrent layers.
Model, gates.

Prefer stored results to stale model-card summaries. For example, the selected
run's temperature scaling improves transfer NLL .6977→.5645 and ECE .1018→.0621,
while its in-domain Score MAE worsens .4539→.4785. Calibration metrics and Score
value quality need separate evaluation.
Raw and calibrated artifact.

### Simple Jev / RFDT: distribution targets with explicit meaning

RFDT validates distributions keyed by public answer meanings, preserves supplied
labels, caches teacher annotations, and minimizes soft cross-entropy. It unions
records sharing a context or source group before splitting. These are useful
contracts for our later distribution-target stage, although our trainer already
supports hard/soft categorical and Bernoulli targets.
Target validation,
preparation/cache/splitting,
loss.

Its teacher-generated probabilities are authored estimates, not extracted
teacher-token probabilities. Interpolating a fractional Score between adjacent
levels is a chosen target rule, not recovered uncertainty. The checked-in RFDT
verification uses tiny random-model workflow tests; it does not establish a
trained quality gain or reveal TypeSafe's RLCD algorithm.
RFDT documentation.

## Other useful approaches and limitations

| Reference | What the code contributes | What we should not infer or copy blindly |
|---|---|---|
| **SemIf** | Frozen direct-logit baselines; shared-prefix paths; explicit option-order, wording and distractor perturbations; row-level evidence and checksums. | No new training algorithm or proven calibration. Its candidate-list readout differs from our isolated scorer; its precise speed measurements use different hardware/model/workloads. |
| **Verdict** | Bidirectional GLiClass/ModernBERT candidate scoring; supervised CE+Brier; separate backbone/head rates; explicit abstention experiments and scalar calibration. | The implemented training is supervised probability fitting, despite RLCD terminology. Proper-loss training does not guarantee calibrated deployment probabilities. The inference code can truncate context/candidates and inject abstention options, changing our contract. |
| **Laya** | Bidirectional candidate markers, type embeddings, extra decision layers, an action head, and per-type/option-count temperature lookup. | The repository's own research reports severe overconfidence under language shift. A single batched forward duplicates state per question; it is not necessarily one shared state encoding. Full training provenance is less complete than the inference/benchmark code. |
| **Von** | Packed ModernBERT candidate markers, dynamic distractors, gold-omission “other” examples, CE+Brier and a calibration routine in the older trainer. | Benchmark-generation overlap, row-level splitting of repeated pools, format mismatches and incomplete calibration weaken performance claims. It does not prove fresh-start superiority. |
| **razorback16/OpenJev** | A genuinely different runtime: seeded diffusion canvases, masked answer slots, per-slot token distributions and optional repeated reads averaged for uncertainty. | It targets DiffusionGemma/vLLM or MLX rather than our Qwen scorer. Native token probabilities and entropy concentration are not an empirical calibration certificate. Architectural exploration should be separate from fixing the training recipe. |

Primary code/evidence for this table:

- SemIf: direct scorer,
  perturbation generator,
  results and limitations.
- Verdict: training,
  CE+Brier,
  inference formatting and truncation.
- Laya: model and input formatting,
  runtime,
  research findings.
- Von: marker model,
  older trainer,
  marker trainer.
- OpenJev: read-only canvases, averaging and probability extraction,
  model/runtime settings.

Von deserves a specific reproducibility caution. Static comparison found 48 of
78 checked-in peer-benchmark state strings verbatim in its supplied training
generator pools. The generator resamples small pools and splits rows afterward.
For example, all nine frustration benchmark states occur in its training pool.
This establishes overlap in the supplied recipe, **not proof which data any
particular released weights consumed**. Its marker trainer also does not fit a
temperature, and the backend defaults to one. We should borrow useful mechanisms
without adopting its headline scores as independent evidence.
Training pools/split,
benchmark cases,
temperature default.

Laya provides a useful caution of its own: its recorded English-model language
sweep reports zero Khmer accuracy with approximately .952 mean confidence.
Confidence-based routing cannot rescue every distribution shift. We do not need
to add multilingual support to this pilot to use that lesson: state layout,
question wording, primitive and domain shifts also need explicit checks.
Saved language sweep.

## Proposed next comparison

First build one improved training mixture containing the original broad tasks,
audited positive/negative contrasts across Noul/Choice/**Score**, and equivalent
plain/nested/prose representations. Track effective loss weight and decision
counts per primitive, not just scenario-row counts. Preserve complete candidate
groups and family-disjoint splits. The first initialization comparison should
keep our model interface and probability loss fixed.

| Arm | Initialization | Subsequent data |
|---|---|---|
| Fresh | Original Qwen3.5-2B checkpoint; new LoRA and initial numerical heads | Revised broad-plus-contrast mixture from its first update |
| Continued | Our original v0.2 adapters and heads; fresh optimizer | The same revised mixture and schedule |

Use the same conservative optimization policy for the first practical comparison,
measure learning curves, and report the continuation arm's extra historical
training explicitly. Equal *new* update budgets answer “which starting point is
more useful now?”; they do not establish a fair end-to-end curriculum comparison.
To attribute a benefit specifically to mixing from the outset, compare fresh
mixed and fresh sequential recipes with the same total data exposure/optimization
budget, including the first-stage training cost.

Select settings using validation families and primitive-specific retention
slices. Keep the old tests unchanged as regressions and freeze new outer test
families before final scoring. Only after initialization/data effects are clear
should we separately add teacher-distribution supervision, probability anchoring
or ordinal losses. Existing Jev observations remain comparison evidence and are
not silently converted into training labels.

**Recommendation:** test a fresh mixed run, but retain a matched continuation
control. The evidence favors improved curation, primitive balance, representation
coverage and conservative updates; it does not yet identify restart as the fix.

## Revisions inspected

| Repository | Pinned commit |
|---|---|
| SemIf | `ca3ba65f142967030ecb453346e94d6f476a69df` |
| simple-jev | `7cba7d121980e6a230477afa808ea0da0a7719e8` |
| laya | `42626c348753fbb17572a813127df2278a1ec527` |
| von | `bed7e7337791c5124a557eb59ad918cb6747b276` |
| Verdict-open-jev | `30f15564821626ca5c1ad5b2638c4eb7078787dd` |
| kev | `b339f446a0ef0d691d9541daa42fb5416dfba63b` |
| nimble | `f136b3f75721fda4ea961f73993cc50b08488835` |
| openjev | `cddbd962c88a76d7344c09273ebc581f7658566e` |
