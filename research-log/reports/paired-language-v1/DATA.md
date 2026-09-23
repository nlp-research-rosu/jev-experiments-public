# Paired training data inventory

Each arm has the same 200 workflow/category families, 800 case variants and 4,000 judgments. Each case contributes three Noul questions, one Choice and one Score. These are 20 workflow settings crossed with ten common semantic patterns; the population does not represent 200 independent reasoning templates.

| General category | Specific category | Families | Cases | Judgments | Noul true/false | Choice Unknown | Score level counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binding | action_binding | 20 | 80 | 400 | 100/140 | 60 | 0: 40, 1: 20, 3: 20 |
| evidence | attribution_and_endorsement | 20 | 80 | 400 | 140/100 | 40 | 0: 20, 1: 20, 2: 20, 3: 20 |
| evidence | claim_vs_completion | 20 | 80 | 400 | 120/120 | 40 | 0: 20, 1: 20, 2: 20, 3: 20 |
| binding | entity_binding | 20 | 80 | 400 | 140/100 | 40 | 0: 40, 1: 20, 3: 20 |
| logic | negation_and_quantifiers | 20 | 80 | 400 | 140/100 | 20 | 0: 20, 1: 20, 2: 20, 3: 20 |
| rubric | ordered_rubrics | 20 | 80 | 400 | 100/140 | 20 | 0: 20, 1: 20, 2: 20, 3: 20 |
| policy | permission_vs_execution | 20 | 80 | 400 | 120/120 | 40 | 0: 20, 1: 20, 2: 20, 3: 20 |
| temporal | reversal_and_current_state | 20 | 80 | 400 | 80/160 | 40 | 0: 20, 1: 20, 2: 20, 3: 20 |
| temporal | temporal_scope | 20 | 80 | 400 | 140/100 | 40 | 0: 20, 1: 20, 2: 20, 3: 20 |
| epistemic | unknown_vs_failure | 20 | 80 | 400 | 80/160 | 40 | 0: 20, 1: 40, 3: 20 |

Natural authors only replace record text and question instructions. Context, policy, rubric definitions, labels, candidate order and paired identifiers are fixed. Language generation and review artifacts are retained, including rejected wording and repairs.

Independent validation contains 20 families / 80 cases / 400 judgments. Independent testing contains 40 families / 160 cases / 800 judgments. Both were independently authored and frozen before training language generation. Four variants from the same family remain correlated.

Training uses one original and one new example of each primitive per update, with 1,000 updates per arm. This provides 3,000 replay-question presentations and 3,000 new-question presentations per arm. Each primitive receives 1,000 presentations; new categories cycle equally. The pool contains 2,400 new Noul questions, 800 Choice and 800 Score, so this budget does not exhaust every Noul example. Both arms use exactly the same sampled identities and order.

The original replay pool contains 800 canonical examples from each of six original sources (4,800 in total); its source balance and prompt text are shared. This is equal examples/updates, not equal token compute.

Metrics include per-question and whole-case accuracy; binary sensitivity/specificity; Choice confusion and Unknown precision/recall; Score expected-level MAE; NLL/Brier; confidently wrong counts; both-correct evidence and question pairs; invariant probability drift; family bootstrap and broad-task retention. The complete matrix is written after checkpoint selections lock and testing completes.

Actual fixed-schedule coverage: 2,600 distinct new judgments, with all 800 cases represented through Choice/Score; only 71 cases have all five questions presented. New Noul presentations contain 475 true and 525 false targets. Exact counts are in `actual-training-exposure.json`. Relations are evaluation metadata here, not additional training losses.
