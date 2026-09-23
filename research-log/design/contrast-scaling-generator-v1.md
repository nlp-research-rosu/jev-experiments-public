# Scaling corpus generator interface

Implements docs/contrast-scaling-contract-v1.md. A semantic family is four cases in a2×2cross: evidenceA/rubricA, evidenceA/rubricB, evidenceB/rubricA, evidenceB/rubricB. Private latent facts produce labels; language only renders facts and public rules. Source question/candidate IDs, expected values, feature truth vectors and oracle programs never enter request.state.

Use experiments.contrast_scaling_logic.py derive_features(raw), eval_rule(expr,features), select_level(level_rules,features), select_choice(option_rules,features). Grammar supports booleanfeature,not/all/any,at_least and numericcomparison. Family metadata may store private rawfacts/programs for audit; only state/question enter model preparation.

## Language blueprint bank schema

Exactly20blueprints/category eventually, independently authored afterevalfreeze (200total). Each JSON record:

- id, category, scenario_domain: nonemptystrings.
- operation and other_operation: objects {verb,past,noun}; distinctverbs. Example: dispatch/dispatched/dispatch.
- target_template: string with only `{identifier}` placeholder; other-target distinguished byidentifier.
- sources: {primary,secondary,untrusted}: three distinctrealistic sourcenames.
- evidence_heading: contextualheading, no claim about outcomes or completeness.
- question_prefix: neutral framing phrase, no outcome assertion.
- score_title: neutral supplied-policy/rubric description, not a fixed stage mapping.
- claim_template: uses `{speaker}`,`{actor}`,`{past}`,`{target}`; affirmative completed assertion about actor; speaker may differ.
- intent_template: same with `{verb}`; future intention, not completed claim.
- denial_template: same with `{verb}`; explicit denial of completion.
- endorsement_template: uses `{authority}`,`{speaker}`,`{actor}`,`{past}`,`{target}`; authority endorses the claim, without asserting external execution evidence.

Validate format fields with string.Formatter; no arbitrary field access/conversions. No blueprint supplies gold labels or hidden completion hints in frame text. Two authors eachprovide10blueprints/category. Corpus generation waits for data/contrast-scaling-v1/evaluation-freeze.json.

## Corpus construction

`build_family(blueprint, instance, seed=42) -> dict` with id,category,domain,cases[4],relations,privateprovenance. Fourvariantsv0..v3 asabove. `build_training_suite(blueprints, families_per_category=500, seed=42) -> suite` emits a canonical ordered set: balanced categories interleaved, cycleall20blueprints before their nextinstance. Thus first200 containsonefamily/blueprint, first1000five, first5000twentyfive. These are5000semantic scenarios, not5000independentlywritten language templates. `training_bundles(suite) -> list` gives one20-example bundle/family orderedcasevariantthen n1,n2,n3,c1,s1; targethard types only; relations:[]; provenance assigned_split=train/category/dataset.

Vary actual latent conditions: source/authenticity, modes/outcomes, conflicting reports, exact actor/action/target/run bindings, inclusive windows, event order/reversals, inventories/unknown status, numericobservations. Distractors are raw irrelevant observations, not answer explanations. Distinctness audit normalizes names/IDs/source aliases so mere renamings do not count as new semantic configurations.

Within each family evidenceBchanges a meaningful fact or event relative toA. Never manufacture labels first then paste expected status into the state. Claims are rendered as textualmessages only; latent tense/endorsement/polarity tags stayprivate. Execution observations may be structured rawtoolrecords (mode/outcome/identity/source/time) because those are actual observations, not derived conclusion fields. Approval is a permission grant to the requested actor, not automatically execution of the requested action.

Rules specify exact queryactor/operation/target/runID, inclusivewindow, trustworthysources and semantics for conflicts/timeouts. Modelstate is `{evidence:...,rules:...,choice_definitions:<exact c1criteria>,score_levels:<exact s1criteria>}`. Evidence can be structured or prose, but allfacts and needed rule definitions must remain available.

## Questions and rules

n1/n2 probe the category's core distinction; n3 probes another facet or exact negation. Core candidates:
claim: claimed vscompleted; permission: approved vscompleted; unknown: unknown vsfailed; attribution: claim vsendorsement; binding: any/unrelated recorded success vs exactcompletion; temporal: active_earlier vsactive_now; reversal: completed vseffective/reversed; quantifiers:all_success/any_failure/item_unknown; rubric:thresholds undercurrent suppliedscale.

Choice criteria must form a complete exclusive partition under explicitdefinitions and include an `unknown` label when appropriate. For binding distinguish exactsuccess, exactfailure, unrelatedsuccess, unknown. Literal unknown doesnotmeanfalseworldstate. Requestedattempt scope and conflictresolutionmustbe explicit. Parentcanextendoracle with independentlytested rawfeatures if required.

Score has2–5levels (sameKwithin a family). Use multiple suppliedrule systems acrossallcategories, not a fixedaction-stage ladder. Generic priority rules are allowed when the task explicitly says choose highest applicablelevel. Level0isfallback; non-defaultlevels have substantive conditions. Vary featureconditions,negation,conjunction/disjunction/count thresholds and ordering; at leastone same-evidence rubricpair mustchangeScoretarget. Publiccriteria describeeachrule in natural language; allotherlevels visibleinstate. Ordered_rubrics category additionally uses variable numeric cutoffs/count scales with exactoracleoutcomes. Do not relyon hidden rubric indices or expectedvalues. Allprivate programs must useknownfeatures/operators.

Relations: standardflip/question_contrast/invariant. Add contrast_axis metadata (evidence/rubric/question). Rubricpairs v0/v1,v2/v3 mustkeep evidencebyte-identical. Evidencepairs v0/v2,v1/v3 keepquestion/rubricconstant. Create only valid endpoints with equalanswer spaces andcorrect equal/different labels. Require at leastone rubricScoreflip, oneevidenceflip, two same-record differingquestionpairs wherepossible, andoneinvariant per family; reject/resamplelatentconfigurationifnotcoherent ratherthanforcewronggold.

## Outputs and audit

CLI emits only afterevaluationfreeze exists. Write train.jsonl, train-suite.json, manifest.json with ordered_family_ids/nested_prefixes200/1000/5000 and allfilehashes; include originalsourcebank hashes, semanticprogram/fact signaturecounts, percategorylabelcounts, property/grammar/domaincounts, duplicatecontrols, nominalquestionunitcounts. Include private oraclefacts/programs in auditsuiteonly, never trainingmodelstate. Strictsuitevalidator and independentlabel fixtures mustpass. NoGPUorAPIcalls. KeepdeclaredfirstNmembershipstableforresumable experiments.
