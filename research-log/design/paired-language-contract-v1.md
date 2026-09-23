# Paired language experiment: frozen design and data interface

User authorized the paired training experiment and a rich contrast/metric matrix.
This extends existing research runners; it does not change model architecture or
the public input/output interface. Historical datasets, checkpoints and reports
stay intact. No git staging, commits, resets or submodule changes are required.

## Controlled comparison

Both arms start from `checkpoints/judgment-full-v0.2/final`, with identical weights,
fresh optimizers, 1,000 updates, adapter/head learning rates 5e-5/2.5e-5,
rank-8/alpha-16 LoRA, frozen original body and FP32 numerical heads/loss. Reuse
the six-slot update: original/new × Noul/Choice/Score, each weight 1/6. Original
replay is the same canonical 800 examples per source, with identical schedule
and IDs in both arms. New data has the same IDs, latent facts, targets, candidate
meanings, ordering and relations in both arms. Only rendered evidence and question
wording differ. Original replay prompts must match exactly; contrast prompts may
differ. This is equal updates/examples, not equal token compute.

Within each new-data primitive slot, cycle the ten contrast categories equally;
within an original-data primitive slot, retain the existing source balance.

Selection uses shared independent validation plus 300 canonical broad validation
examples at updates 0, 250, 500 and 1000. Objective is 0.5 new + 0.5 broad NLL,
macro-averaged across the three primitives within each. Lowest validation objective
wins, ties prefer earlier step. Lock both selections before final testing. Run
the unchanged v0.2 reference under the same evaluation protocol. Keep hard targets;
teacher distributions, calibration fitting, RL and new consistency losses are
not experimental factors in this run.

Also report a fixed-1,000-update comparison of both arms, declared before training.
If that is the selected checkpoint, reuse its test observations; otherwise evaluate
the final checkpoint separately. This separates the practical validation-selected
recipe from effects measured at exactly equal optimizer updates. Neither set of
test scores may change checkpoint selection or training.

GPU smoke must verify both arms, finite adapter/head gradients, frozen weights,
save/reload and exact arm matching before full training. Do not truncate inputs;
use the existing 1536-token ceiling and stop on oversized paired examples. Training
time cap 3600 seconds per arm excluding evaluation; evaluation phase cap 1800
seconds. Partial outputs and completed-update count must survive interruption.

## Target matrix and population

| Specific contrast category | General category |
|---|---|
| claim_vs_completion | evidence |
| permission_vs_execution | policy |
| unknown_vs_failure | epistemic |
| attribution_and_endorsement | evidence |
| entity_binding | binding |
| action_binding | binding |
| temporal_scope | temporal |
| reversal_and_current_state | temporal |
| negation_and_quantifiers | logic |
| ordered_rubrics | rubric |

Target training population: 200 distinct situation families, four controlled
variants each, 800 requests / 4,000 judgments, with 20 situation families per
specific category. Natural realizations are independently written/reviewed;
IDs and semantic skeletons alone do not establish linguistic independence.
Generation failures are retained and repaired as new versions or excluded from
both arms together. Actual counts and every exclusion remain explicit.

Independent evaluation must be authored and reviewed **before training language
generation**: four test families and two validation families per category, each
with four variants. Target test: 160 requests / 800 judgments; validation: 80 /
400. Authors see this semantic contract, not training templates or examples.
Training writers must not inspect the evaluation records. Test output distributions
are not examined until checkpoint selections are locked. Author/reviewer model
provenance and shared high-level concepts limit claims of independence.

Every case contains three Noul, one Choice and one Score question. Every family
contains positive and negative binary controls and meaning-changing or invariant
relations; do not achieve high accuracy through an always-no label mix. Across
each category deliberately balance relevant binary outcomes and represent Unknown
where its contract applies. Choice uses an explicit `unknown` label when evidence
is insufficient. A binary question asking whether evidence confirms X may be
false while a world-outcome Choice is unknown. Do not invent probability 0.5 gold.
Score is an explicit ordered rubric, not confidence, with multiple levels covered.

## Case/suite JSON interface

```json
{
  "contract": "authored evidence contracts, not hidden world-state assumptions",
  "cases": [{
    "id": "test/claim_vs_completion/01/v0",
    "family_id": "test/claim_vs_completion/01",
    "domain": "evidence",
    "category": "claim_vs_completion",
    "variant": "v0",
    "layout": "prose",
    "predicate_tags": {"n1":"completed_claim","n2":"confirmed_completion","n3":"permission","c1":"evidence_status","s1":"evidence_stage"},
    "request": {
      "state": "A self-contained natural record with any necessary scope/authority/rule definitions.",
      "questions": {
        "n1": {"type":"noul","instructions":"Precise predicate","criteria":{"true":"Meaning of yes","false":"Meaning of no"}},
        "n2": {"type":"noul","instructions":"A different predicate","criteria":{"true":"Meaning of yes","false":"Meaning of no"}},
        "n3": {"type":"noul","instructions":"A third predicate","criteria":{"true":"Meaning of yes","false":"Meaning of no"}},
        "c1": {"type":"choice","instructions":"Evidence outcome","criteria":{"confirmed":"Defined supported outcome","contradicted":"Defined contrary evidence","unknown":"Neither is established"}},
        "s1": {"type":"score","instructions":"Stage or degree under this rubric","criteria":["Lowest defined level","Middle defined level","Highest defined level"]}
      }
    },
    "expected":{"n1":true,"n2":false,"n3":true,"c1":"unknown","s1":0},
    "rationale":{"n1":"...","n2":"...","n3":"...","c1":"...","s1":"..."}
  }],
  "relations":[{
    "id":"unique-relation-id",
    "kind":"question_contrast",
    "left":{"case_id":"test/claim_vs_completion/01/v0","question_id":"n1"},
    "right":{"case_id":"test/claim_vs_completion/01/v0","question_id":"n2"},
    "reason":"The same record supports different predicates differently."
  }]
}
```

Use meaningful natural instructions; `n1` etc are routing IDs, not model inputs.
Criteria and state must be self-contained. Choice labels and Score levels may
differ by family, but relation endpoints need equal answer spaces. Noul relations
may compare independent predicates without claiming they are complements.
Supported relation kinds: `flip`, `question_contrast`, `invariant`,
`layout_invariant`. A flip/question_contrast has different expected labels;
invariant kinds have equal labels. Relations stay within a semantic family.
Question contrasts must preserve case_id. State may be prose or nested JSON;
questions remain a flat five-leaf mapping for the research evaluator.

For every family: include at least two evidence-change relations, at least two
same-record differing-predicate relations where semantically justified, and at
least one invariant relation on a predicate unaffected by a relevant edit.
Do not force a wrong relation merely to meet a quota. Include reasoning about
the labels in evaluator-only rationale, not the model state. Quoted intent,
explicit denial, attribution, source authority, time boundaries and conflict
priority must be unambiguous. No added answer-explaining sentences in the record.
Do not call an absent result proof of physical failure.

## Artifact layout

- `data/paired-language-v1/test/suite.json` and `validation/suite.json`: frozen
  independently authored common evaluation suites and manifests.
- `data/paired-language-v1/template/train.jsonl` and `natural/train.jsonl`:
  matching bundle IDs/groups/targets, as used by `load_bundles` / `prepare_bundle`.
- Each bundle has `id`, `group_id`, `examples` (state/question/target),
  `relations: []`, and provenance `dataset`, `assigned_split: train`, `family`,
  `category`, `label_origin`. Keep all sibling groups together.
- `data/paired-language-v1/paired-manifest.json`: pair alignment, generation and
  review provenance, labels and source hashes; no labels/rationales enter prompts.
- `reports/paired-language-v1/`: study logs, all review observations and reporting.
- `checkpoints/paired-language-v1/`: both arms, curves, selection lock and final
  predictions. Keep the original checkpoint/default unchanged.

## Metrics and decision rules fixed before results

Report micro and macro-category accuracy, whole-case exact match, per-category and
per-predicate binary sensitivity/specificity/balanced accuracy, categorical confusion
matrices, Unknown precision/recall and each primitive's NLL/Brier. Report Score
argmax accuracy plus expected-level MAE and normalized MAE. Count incorrect answers
whose selected probability is >=.90 and >=.95. Probability loss is not by itself
a proof of calibration; retain raw distributions.

Report evidence-flip and same-record question pair both-correct scores, invariance
correctness and probability drift, by both general and specific category. Bootstrap
paired natural-minus-template differences over complete test families (not individual
correlated variants); label this exploratory uncertainty. Do not tune on test results.
Keep all 1,500 canonical broad tests and the original 114-case semantic regression
suite. Broad retention: overall and each Score-source accuracy within 2 pp of the
matched v0.2 reference; each Score-source MAE within .05 level units; broad NLL within
.05. Interpret new gains together with these fixed retention checks, not just a
single pass/fail score.

Jev is an external reference, not a label oracle or a teacher in this training run.
Use the existing durable archive, exact eligible-response reuse, pinned model and
current validated API interface. Preserve every raw response, probability vector,
usage record and error. Archive the frozen test now if useful, but inspect its
judgment outcomes only after local selection is locked. Do not leak labels into
requests or silently normalize values outside the existing metric policy.
