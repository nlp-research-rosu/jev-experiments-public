# Contrast data scaling v1: authorized experiment contract

User approved measuring data scale before introducing techniques from the calibration/RL papers. This extends the existing local judgment experiment. Preserve historical datasets/checkpoints/reports, the user-owned staged reference submodules, and the original default. Do not train with soft targets, Brier additions, RL, temperature calibration or an explicit pair loss in this study.

## Fixed model and supervised objective

Use `checkpoints/judgment-full-v0.2/final` as the common starting model, rank8/alpha16 LoRA and FP32 binary/compatibility heads; the original BF16 body stays frozen. AdamW and adapter/head learning rates remain 5e-5/2.5e-5; clip gradient norm1, seed42, TF32off, FLA backend, input limit1536 without truncation. This holds initialization fixed while changing data exposure. New Base models/capacity comparisons remain subsequent experiments.

Every training update includes ALL20 judgments from one four-case family (three Noul, one Choice, one Score per case), plus one old canonical replay question per primitive. Each of new/old × three primitives contributes1/6 loss. Average within each primitive, not across the unequal question counts. Backpropagate bounded microbatches of complete candidate groups; never split Choice/Score normalization. Loss is BCE/CE only. Relations are metadata for coverage/evaluation. Old replay comes from the existing canonical800-per-source training pool and its source balance; same old IDs at equal update positions across streams.

## Nested data and complete exposure

Build 5000 varied scenario families, balanced across10 categories. Each has four variants and20 labeled questions: full population20000cases/100000judgments. The prefix datasets contain200/1000/5000families, with20/100/500families per category. Their populations are strictly nested and the ordered common prefixes match. Report unique latent programs, fact combinations, authored language forms and domain settings; these are scenario families, not automatically5000 independently invented reasoning templates. Increasing size must vary semantic facts/rules and composition, not only names. Keep rendering/quality criteria common across sizes.

Families should combine evidence contrasts and same-evidence/rubric-or-question contrasts. Use varied explicit rubric mappings, thresholds, rule priorities, action/evidence/source/entity/time binding, negation/quantifiers, unknown evidence and reversals. Hard labels must follow a reproducible latent-fact oracle, not a language generator's confidence. Keep labels/rationales/IDs/provenance outside model inputs. Complete-family boundaries remain intact throughout splitting/sampling.

Ten categories: claim_vs_completion; permission_vs_execution; unknown_vs_failure; attribution_and_endorsement; entity_binding; action_binding; temporal_scope; reversal_and_current_state; negation_and_quantifiers; ordered_rubrics. The last includes rubric-changing counterfactuals; other categories must also vary their score rubric rather than all sharing one action-stage ladder.

## Primary and compute-control streams

Primary: one deterministic one-pass stream of5000families; save/evaluate validation at0,200,400,600,800,1000,2000,3000,4000,5000. The200/1000/5000 endpoints are mathematically the same as separate runs from identical initial weights/optimizer/RNG over these identical ordered prefixes, because optimizer/LR/schedule do not depend on eventual dataset size. This avoids redundant training; verify shared-prefix/restart equivalence in smoke.

Fixed-compute control: branch from the200-family checkpoint with optimizer and RNG restored; present only the same200families for four more complete epochs (1000totalfamily updates). Use the same old replay IDs at equal update positions. Compare that endpoint with the primary1000-family endpoint: repeated smaller data versus1000distinctfamilies at equal updates/primitive weights. The5000-family endpoint measures the practical gain of more unique data plus additional training; there is no matched5000-update small-data control in this bounded run. State this limit explicitly.

All primary dataset endpoints see every declared judgment and both endpoints of every declared training relation. Validate coverage as a hard gate. Also choose validation-selected checkpoints within each dataset's permitted prefix budget and within the repeat-control run; lock all choices before final testing. Headline scaling uses complete-pass endpoints; do not hide partial exposure behind selection. Starting checkpoint and unchanged legacy results are references.

## Independent evaluation authored first

Before emitting training language/data, independent authors construct80testfamilies and40validationfamilies (8test/4validation per category), four cases each,1600test/800validationjudgments. They see only this contract, not training sources/blueprints or previous test text/outcomes. Different authors own disjoint categories and exchange review afterward. Freeze hashes and corrected labels before training generation. These remain model-authored/reviewed, not human-certified ground truth.

Canonical case interface matches revised_metrics.validate_suite:
`{id,family_id,domain,category,variant,layout,predicate_tags,request:{state,questions},expected,rationale}`; `questions` ordered n1,n2,n3,c1,s1. Three Noul boolean targets, one Choice label, one Score integer index. Each family needs meaningful evidence flips, question contrasts and invariants where valid; rubric changes are `flip` relations with `contrast_axis:'rubric'`, without extending historical metric definitions. Same-record question_contrast must use identical case ID and different questions. All relation endpoints share type/answer space and the intended equal/different labels.

State must expose all public definitions needed by ANY leaf or candidate. Use state `{evidence: ..., rules: ..., choice_definitions: <exact c1.criteria>, score_levels: <exact s1.criteria>}`. This copies public schema definitions, never labels/rationales. Rules provide all time windows, authority, closed/open-world assumptions and predicate referents. Each candidate definition should also be self-contained. Do not rely on another question's answer. Noul instructions explicitly identify the subject/action/proposition. Review the exact rendered units and assert shared definitions survive compilation; whole-request label agreement alone is insufficient.

Question counts/classes must not be achieved by inventing invalid contrasts. Across categories include both binary outcomes and explicit unknown Choice labels where epistemically justified. Unknown is not automatic50/50 supervision. Score is a supplied ordinal rubric, never generic confidence. Rule changes must actually change the supplied definition, not secretly alter gold.

## Gates and evaluation

Run complete CPU tests, independently review code/data, and smoke GPU with finite adapter/head gradients, frozen-weight hashes, save/reload/resume equivalence and full family coverage. Determine safe microbatch size from smoke and freeze it for all streams. No dropped or truncated new examples. No real-model test outcomes before final selection lock. Preserve partial artifacts and accurate status if interrupted; no automatic promotion.

Evaluate the new frozen test for the common baseline, complete-pass endpoints, repeat control and any distinct selected checkpoints. Keep1500original broad tests,764original semantic judgments and the prior clarified paired suite as inspected regressions; reuse exact matches when checkpoint/request identities permit. Report accuracy, pair correctness by axis, whole-case correctness, NLL/Brier, Unknown P/R, selected-probability errors≥.90/.95, ScoreMAE and retention. Bootstrap complete test families within category; one training seed is not training-variance evidence. Report training exposure/tokens/time, diversity statistics and fixed-compute limitations.

Jev reference is authorized by existing session instructions; use the durable archive and preserve all paid responses. No teacher-label generation or training target distillation in this experiment. Query only the final frozen test; keep its outcomes unread until local checkpoint selections lock. Respect throttling and do not discard errors or duplicate responses.
