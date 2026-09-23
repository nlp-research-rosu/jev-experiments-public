# Factorial semantic factory and portable language cards

This interface is frozen before training-language authoring. Authors receive this document and the public contract, never assessment/calibration narratives, gold, or earlier data. Actual language authoring starts only after the evaluation freeze exists.

## Portable card schema (version 1)

Each author supplies a JSON list of 200 cards: 20 per category across all ten contract categories. Across the two independent author sources there are 400 cards. A card is exactly:

```json
{
  "schema_version": 1,
  "id": "author/category/000",
  "author": "author-source",
  "category": "claim_vs_completion",
  "heading": "Recorded observations",
  "claim_template": "At {time}, {source} wrote: {actor} has carried out {operation} on {target}.",
  "intent_template": "At {time}, {source} recorded this intention: {actor} plans to carry out {operation} on {target}.",
  "denial_template": "At {time}, {source} denied that {actor} had carried out {operation} on {target}.",
  "endorsement_template": "At {time}, {source} explicitly endorsed the assertion that {actor} had carried out {operation} on {target}.",
  "execution_template": "For run {run_id}, {source} logged {actor} / {operation} / {target} at {time}: mode={mode}, outcome={outcome}, verified={verified}.",
  "observation_templates": {
    "approval": "At {time}, {source} records a permission decision for {actor} / {operation} / {target}: decision={decision}, verified={verified}.",
    "corroboration": "For {run_id}, {source} at {time} records {actor} / {operation} / {target}: mode={mode}, outcome={outcome}, verified={verified}."
  },
  "construction_note": "A short private description of the sentence/clause construction."
}
```

`claim_template`, `intent_template`, `denial_template`, and `endorsement_template` must each use exactly once: `actor`, `operation`, `target`, `source`, `time`. The renderer privately selects the semantic template. Tense, polarity, endorsement flags, and template names never appear as public fields. A claim positively asserts past completion; an intention concerns future action; a denial negates past completion; endorsement explicitly adopts a positive completion assertion (and therefore also constitutes a claim). A report that somebody else endorsed an assertion is not an endorsement by the reporting source. `execution_template` must use each exactly once: `actor`, `operation`, `target`, `run_id`, `source`, `time`, `verified`, `mode`, `outcome`. Placeholder attribute/index access, conversions, format specifications, duplicate slots, escaped braces, and extra slots are rejected. Values are literal data; templates must not presuppose truth, permission, success, trust, verification, or endorsement. In particular an execution record may be unverified, simulated, unsuccessful, or conflicting. A statement records an assertion, intention, or denial; its content is not execution evidence. No template may infer the meaning of a status value. No instructions to the model or answer hints belong in any card. Headings must be neutral and at most 80 characters. Every individual template must be at most 360 characters.

The example above is a synthetic interface fixture, not an authoring pattern to expand mechanically. Cards must vary actual linguistic construction: source attribution, clause ordering, grammatical focus, and the relationship between run identity and observations. Renamed actors, wrappers, IDs, punctuation-only edits, and synonym combinations do not establish independent constructions. `construction_note` explains the difference and stays private. Independent review evaluates both semantic preservation and meaningful structural variation. Preserve author inputs, drafts, and rejected cards; do not manufacture the pool with a synonym combinator.

The factory alone chooses raw facts, domain/query entities, time windows, trusted-source policy, statement count, execution count, and all other records. It fills every supplied template slot. Primary observations are rendered using the typed observation templates below; only scope, inventory completeness, and query policy remain structured. Authors cannot select or omit facts or alter questions/rubrics. No language-author identity, authorship-source/card ID, construction note, oracle program, target label, rationale, or feature vector enters request state. Genuine operational sources and query targets remain public raw facts. Public rules, Choice definitions, and every Score level are visible in state to all candidate branches.

The reviewed bank is a JSON object with `schema_version: 1`, `blueprints: [...]`, and `review: {approved: true, bank_sha256: <SHA256 of canonical blueprints JSON>, review_path: <repo-relative retained review artifact>}`. Canonical JSON uses UTF-8, sorted keys, separators `(',', ':')`, `ensure_ascii=False`. The separate retained review artifact has this required schema:

```json
{
  "kind": "independent_language_review",
  "approved": true,
  "bank_sha256": "<exact canonical blueprints hash>",
  "reviewer_identity": "<nonempty independent reviewer identity, distinct from authors>",
  "semantic_preservation": "pass",
  "construction_diversity": "pass",
  "reviewed_blueprint_ids": ["<every one of the 400 exact blueprint IDs, once each>"],
  "unresolved_flags": []
}
```

The review must resolve to a different file from the bank; symlink and hard-link self-review aliases are rejected. A twenty-card pilot approval cannot approve the complete bank. Its file hash is recorded in the output manifest. Machine validation is a necessary formatting/fact-preservation gate; the independent review certifies language semantics and diversity.

## Primary observation templates (required extension)

Every card also has `observation_templates`, an object containing exactly the keys required by its category. Each typed template must use every listed placeholder exactly once with the same safe-format and neutral-framing restrictions as the other templates. Render the individual fields as a meaningful sentence, never a JSON-block placeholder. This is essential: the language treatment must change the category’s main evidence, not just its heading or an auxiliary statement.

| Type | Required placeholders |
| --- | --- |
| `approval` | `actor`, `operation`, `target`, `source`, `time`, `verified`, `decision` |
| `corroboration` | `actor`, `operation`, `target`, `run_id`, `source`, `time`, `verified`, `mode`, `outcome` |
| `reversal` | `actor`, `operation`, `target`, `run_id`, `source`, `time`, `verified`, `mode`, `outcome`, `reverses` |
| `state_event` | `target`, `source`, `time`, `verified`, `active` |
| `item` | `id`, `verified`, `status` |
| `measurement` | `value` |

| Category | Required `observation_templates` keys |
| --- | --- |
| claim_vs_completion | approval, corroboration |
| permission_vs_execution | approval, corroboration |
| unknown_vs_failure | approval, corroboration |
| attribution_and_endorsement | approval, corroboration |
| entity_binding | none (empty object) |
| action_binding | none (empty object) |
| temporal_scope | state_event |
| reversal_and_current_state | approval, corroboration, reversal |
| negation_and_quantifiers | item |
| ordered_rubrics | measurement |

For `measurement`, `value` is the observed numeric value or the literal `missing`; do not assume it is numeric/present. For `state_event`, `active` is the raw reported boolean, not an inferred current state. For `item`, `status` is an individual reported `success` or `no_effect`; conflicts appear as separate observations. Reversal `run_id` identifies the recorded operation context and `reverses` identifies the operation whose effect was targeted. Permission `decision` is the raw allow/deny value. Template text must not turn those reported values into an inferred result.

## Semantic and emission API

The stable entry points are `make_plan(category, index, narrow_k, broad_k)`, `build_matched_suites(blueprints, seed=42)`, `validate_matched_suites(suites)`, `training_bundles(suite, arm)`, and `emit_study(repo_root, bank_path, evaluation_freeze_path, base_checkpoint, output_dir, seed=42)`. Emission is fail closed on missing freeze or unreviewed bank. Synthetic unit fixtures never constitute an approved bank.


`make_plan` enumerates legal raw worlds independently of labels, then evaluates explicit programs and selects two worlds that differ in one observation group. Selection covers legal outcomes; it never writes target bits or feature vectors into raw facts. The plan retains both raw records, private rules, original world indices, canonical fact hashes, and a legal raw witness for every level of every rubric. Queried and distractor actors/actions/targets/runs receive symmetric role-exchangeable aliases from the same neutral vocabulary. Source names rotate across trusted and untrusted membership. These deterministic permutations depend only on category/index, not outcomes; all arms share them. A positive affine clock transforms every observation time, window endpoint, query cutoff and retained witness consistently. `surface_context` retains the bijection, inverse and clock parameters. `normalize_record` reverses the transformation; aliases and clock coordinates do not count as new semantic worlds. A conflicting observation group may contain two competing reports.

The three boolean predicates are category-specific. Choice partitions established outcomes with explicit Unknown handling; temporal, inventory, and numeric categories use their own partitions. Score programs use category-specific requirement counts, ordered evidence tiers, successful-versus-resolved inventory counts, and explicit numeric thresholds. A checklist score counts the listed requirements, not calibrated confidence; logical dependencies between requirements are permitted and stated. Temporal scores count separately specified current/historical predicates. Unknown is never an implicit numeric midpoint. Every program uses the highest applicable explicitly defined level, with a reachable fallback 0.

For a complete bank, `build_matched_suites` consumes the root-owned `assign_blueprints` and returns `A/B/C/D` in memory. Each suite contains 400 ordered families, 1600 cases, 8000 judgments, and canonical relations. The factory checks all levels separately for each arm/category/K/rubric variant, independently crossed claim/intent/approval/completion, exact compiler inputs, 40 distinct raw-world/program pairs per category, valid same-answer-space relations, canonical fact hashes, and evidence/question equality under the appropriate factors. `validate_plan` normalizes actual raw records, binds any retained world indices to the legal enumeration, checks each private program against its exact canonical executable/public definition, verifies all legal witnesses and selected levels, and recomputes actual program/plan fingerprints. Diversity is hashed from normalized raw records and programs, not from world IDs. The suite validator reconstructs and compares every complete public request, including scope, questions, rules, candidate definitions, ordinality and selection policy; a shared corruption across all arms is rejected. Private provenance contains the plan and card only outside requests. `training_bundles` returns exactly 20 hard-target examples per family with `truth`, `choice`, or `level_index` targets and no training consistency relations.

`validate_matched_suites(suites, require_language_variation=True)` additionally requires primary evidence, excluding headings, to change in exactly the 200 language-changed families. Small synthetic fixtures may exercise all metadata assignments with a tiny repeated vocabulary; their ordinary validation does not certify real language diversity. Production emission always enables the stricter gate.

`language_quality_report(blueprints)` strips IDs, headings, case, and punctuation from the primary templates before counting constructions. It reports per-category authorship, distinct primary constructions, trigram-overlap summaries, and highest-overlap pairs. Emission requires 40 distinct primary constructions per category. These machine checks cannot certify meaningful natural-language variation: the retained independent bank review is also mandatory.

`validate_emission_inputs(repo_root, bank_path, evaluation_freeze_path, base_checkpoint)` performs non-emitting authorization and metadata preflight. Its repository root must resolve to the checkout containing the factory; a caller cannot substitute another directory with its own approval file. Its approval path is fixed at root-owned `reports/contrast-factorial-v1/STUDY_APPROVAL.json`; a caller cannot select its own approval document. The approval must identify this approved A/B/C/D, 400-update pilot and pin the contract path/hash, evaluation-freeze path/hash, original `checkpoints/judgment-full-v0.2/final` path/tree hash, checkpoint ID, model ID and revision.

The freeze metadata must identify study `contrast-factorial-v1`, status `frozen`, 480 reviewed requests, 2400 reviewed judgments, zero unresolved review flags, no model predictions used and no training language authored before freezing. Assessment/calibration metadata must contain exactly 80/40 families, 320/160 cases, 1600/800 judgments, and 8/4 families per category. The gate verifies the actual suite, author, review, frozen-source and legacy-manifest byte hashes. Evaluation artifacts are never decoded or parsed; only the metadata-only freeze is parsed.

The base must contain nonempty `checkpoint.json`, `training.pt`, `readouts.safetensors`, `adapter/adapter_config.json`, and `adapter/adapter_model.safetensors`. Its v0.2 identity is recomputed using the existing checkpoint-identity helper, its format/model/revision/rank8/alpha16 checked, and its whole tree hash matched to root approval. This is an identity/component gate, not a model load or weights-deserialization test.

`emit_study` invokes this preflight and additionally requires a real balanced bank with a complete separate independent review, normalized construction diversity and actual changed primary language evidence. It does not open evaluation/calibration examples or parse their narratives. It refuses synthetic-author fixtures and preexisting output directories. It writes `A/B/C/D/train.jsonl`, `A/B/C/D/train-suite.json`, `private-audit.json`, and `manifest.json` only after semantic and language gates pass. The suites expose `ordered_family_ids` for the runner. The audit retains raw records, all legal level witnesses, rules, assignments, and quality/coverage results. All manifest paths are repository-relative. Source/data mappings contain actual SHA256 values; every arm has training and suite hashes. Base hashing uses the runner's `tree_sha256` helper. The manifest includes the actual study-approval hash, pinned base checkpoint ID and verified frozen-artifact mappings. The runner module itself is not frozen by this factory while its independent review is pending.

Example coordinator invocation after freeze and language approval:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.runtime-training:. .venv/bin/python -m experiments.contrast_factorial_data \
  --bank data/contrast-factorial-v1/language/reviewed-bank.json \
  --evaluation-freeze data/contrast-factorial-v1/evaluation-freeze.json \
  --base-checkpoint checkpoints/judgment-full-v0.2/final \
  --output data/contrast-factorial-v1/train/v1 \
  --seed 42
```

Preserve author prompts, draft banks, review decisions, rejected configurations, and CLI logs under the study paths. A rejected configuration must be corrected without overwriting its source artifacts. Full real-tokenizer preflight follows the reviewed-bank rendering and must reject every branch above 1536 tokens without truncation. The CPU serializer probe in the tests verifies compiler/training interface equivalence, not the tokenizer limit. No training, GPU work, or API calls occur in this module.


The selected Score level remains the highest applicable explicitly listed index. The independent review found no demonstrated label error in this convention; this corrective batch preserves the existing public criterion wording and labels. Real-tokenizer preflight remains mandatory after approved-bank rendering. Role permutations and affine clocks improve representation support but do not add independently authored semantic mechanisms or establish unrestricted generalization.
