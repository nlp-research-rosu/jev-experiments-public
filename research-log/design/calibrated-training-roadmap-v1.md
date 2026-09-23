# Calibrated training: staged experiment roadmap and tracker

Created 2026-09-22. Status: user authorized starting S0/S1 on 2026-09-22; all six S1 arms and evaluation are complete. Execution details and live state are in the screen protocol and STATUS.json. The later branches remain conditional. The user requested this staged record after reviewing the completed four-way pilot. It is a research roadmap, not an executable launch manifest or an instruction to run every combination.

## Purpose and starting evidence

Separate three questions: can we improve semantic decisions, can we improve probability estimates, and can either improvement retain useful high-confidence coverage and the original tasks?

The [completed four-way pilot](../reports/contrast-factorial-v1/RESULTS.md) found a promising rubric effect (B minus A: +4.55 percentage points on category-balanced contrast-pair accuracy), no clear benefit from doubling that particular language bank, and substantial probability sharpness that separate calibration could reduce. B achieved 73.63% individual accuracy and 48.92% paired accuracy. Its raw versus per-primitive-calibrated NLL was 0.9960 versus 0.5925; wrong answers at probability >=0.90 fell from 202 to 18 while coverage fell from 74.56% to 25.88%. The four-way study used one training seed and does not establish a loss, model-size, or initialization ceiling.

Use [the literature recap](../reports/CALIBRATED_TRAINING_RESEARCH.md) and [the checked methods review](../reports/post-pause-audit-v1/methods-review.md) for method definitions and source details. BCE/CE already learns probabilities; adding Brier or using teacher targets is an empirical intervention, not a calibration guarantee.

## Stage map and progress

| Stage | Work / decision | Depends on | Current status | Result / next decision |
| --- | --- | --- | --- | --- |
| P0 | Completed rubric/language factorial pilot | Earlier approved experiment | Complete | [Results](../reports/contrast-factorial-v1/RESULTS.md); monitor paused |
| S0 | Fix shared protocol, teacher audit, target capture, and numerical integration checks | This roadmap and selection of the next experiment | Complete | Preparation and fixed protocol |
| S1 | Six-run loss/teacher screen: H0, H1, Q0, Q1, J0, J1 | S0 and archived teacher targets | Complete: six × 400 updates and evaluation | [Verified results](../reports/calibrated-screen-v1/RESULTS.md); no follow-on run selected |
| S2 | Smoothing and teacher-top-answer controls | An interpretable S1 teacher signal | Conditional | Does target-specific distribution information help? |
| S3 | Six consistency-on counterparts: H2, H3, Q2, Q3, J2, J3 | Valid relation certificates and comparisons selected from S1 | Conditional | Does explicit consistency add correct reasoning rather than just agreement? |
| S4 | Matched additional seeds and independent confirmation | Candidate recipe and matched control | Conditional | Does the result repeat and transfer? |
| S5 | Exact-distribution data, adaptive calibration, initialization/capacity, or generative/RL branches | A specific unresolved hypothesis after earlier results | Deferred alternatives | Select one question; do not launch their Cartesian product |

Stages are decision points. They do not all need to run, and an unhelpful teacher or method can be reported and left unexpanded. Skipped combinations must be visible; a partial matrix cannot support all factorial interaction claims.

## S0: common protocol and teacher preparation

Proposed shared student setup:

- Start each arm independently from `checkpoints/judgment-full-v0.2/final`, with matched fresh optimizer and RNG state for each seed. The trained factorial B endpoint is a historical reference, not the starting weights for some arms.
- Reuse the immutable broad-rubric / 200-rendering B training inputs: 400 families, 1,600 cases, 8,000 new judgments, and 1,200 matched replay judgments over 400 updates. Freeze exact paths and hashes in the eventual launch manifest. Keep family ordering, replay, trainable parameters, learning rates, batching, and runtime matched to the [factorial contract](contrast-factorial-contract-v1.md).
- Keep authored labels intact. Teacher outputs are separately versioned opinions attached by exact request and answer-space identity.
- Preserve original-task replay supervision. Apply the proposed Brier/teacher interventions to new judgments in this first screen, keeping replay BCE/CE unchanged. Any broader application is a separately named comparison.
- Predeclare loss coefficients, teacher targets, target normalization, repeats, endpoint choice, metrics, and tolerances before assessment. This roadmap does not select numerical coefficients or a larger-model checkpoint.
- An H0 implementation check must establish equivalence to the old recipe. The existing B result can be reused as the same-seed H0 only after exact source, exposure, initialization, runtime, and numerical-equivalence checks; otherwise make a clearly named fresh matched control. Never relabel a historical result as a new run.
- Test complete candidate-group normalization, finite gradients, full checkpoint/optimizer/RNG resume, and frozen-body preservation before a long run. No new broad authoring campaign is implied.

Evaluation separation:

- The completed factorial assessment has now been inspected. It can be reused as a development/regression suite, but must not be described as a fresh untouched confirmation set for decisions informed by these findings.
- Use separate teacher-audit/development, calibration, and confirmatory populations; keep entire related families together. The exact new partition plan must be frozen before the next study, and teachers must not see gold answers or private latent facts in their inputs.
- Teacher-training targets never come from calibration or assessment requests. Audit predictions and external-reference predictions remain outside training, even when stored in the same durable archive.
- Preserve the original quarantined TEST/Jev outcomes. Releasing that quarantine needs a separately documented evaluation decision; it is not a side effect of this roadmap.

### Two teacher sources

| Teacher ID | Proposed source | Target representation | Before using it |
| --- | --- | --- | --- |
| Q | A larger Qwen or other explicitly selected model | Probabilities aligned to the supplied Noul/Choice/Score outcomes | Pin weights/version, prompt, candidate mapping, inference settings, and probability-extraction method; audit on independent labeled development cases |
| J | Jev, proposed pinned version `jev-1.13.0` | Noul `p(true)`; complete Choice and Score probability vectors | Verify returned version, exact semantics, schema, normalization policy, and independent development accuracy/probability quality |

Use the same visible evidence, predicates, and criteria. A larger teacher may reason internally, but should not receive privileged gold or latent facts. Token likelihoods, verbal confidence, and sampled-answer frequencies are different estimators; select and document Q's estimator before treating it as a comparable target source. Jev's derived `confidence` field is not a substitute for its full distribution. For Score, retain the distribution and level mapping, not only its mean.

Collect once per teacher and dataset version; all arms using that teacher reuse identical archived targets. Preselect any repeated-call subset to measure nondeterminism, preserve every draw, and fix a selection/aggregation rule before outcomes. Do not choose the best-looking draw. Keep teacher/gold disagreements and audit them; do not silently relabel or remove difficult cases. A teacher that fails a predeclared quality/semantic gate should not be presumed suitable merely because it is bigger or branded calibrated.

## S1: six initial training configurations

Hard-label BCE/CE remains in every run. Brier means categorical probability-vector squared error, not ConfTuner's confidence-token objective.

| ID | Brier | Consistency | Teacher | Question | Execution status | Result artifact |
| --- | --- | --- | --- | --- | --- | --- |
| H0 | Off | Off | None | Matched ordinary CE baseline | Complete: 400 updates | [Results](../reports/calibrated-screen-v1/RESULTS.md) |
| H1 | On | Off | None | Does altered probability-loss emphasis help? | Complete: 400 updates | [Results](../reports/calibrated-screen-v1/RESULTS.md) |
| Q0 | Off | Off | Larger model | Does Q's distribution supervision help? | Complete: 400 updates | [Results](../reports/calibrated-screen-v1/RESULTS.md) |
| Q1 | On | Off | Larger model | Does Brier add value with Q? | Complete: 400 updates | [Results](../reports/calibrated-screen-v1/RESULTS.md) |
| J0 | Off | Off | Jev | Does Jev's distribution supervision help? | Complete: 400 updates | [Results](../reports/calibrated-screen-v1/RESULTS.md) |
| J1 | On | Off | Jev | Does Brier add value with Jev? | Complete: 400 updates | [Results](../reports/calibrated-screen-v1/RESULTS.md) |

Interpret the first contrasts as H1-H0, Q0-H0, J0-H0, Q1-Q0, and J1-J0. Compare Q0/J0 and Q1/J1 with matched inputs and loss weights; teacher source includes its extraction protocol, so this is not a controlled model-size experiment.

Proposed loss convention to remove ambiguity before implementation:

For new judgments, use `(1-alpha) CE(p,y) + alpha CE(p,q_teacher) + lambda Brier(p,y)`, with `alpha=0` when teacher supervision is off and `lambda=0` when Brier is off. With a teacher, keep `0 < alpha < 1`, the same across Q and J. This normalized CE mixture keeps hard supervision and avoids an accidental doubling of the CE scale. Brier is against the authored target in every arm; using soft Brier targets would be another factor. For Noul use its equivalent two-outcome vector and stable binary CE.

Retain equal new/old by primitive balancing; add an explicitly normalized consistency term only in S3. Numerical coefficient choices and component/gradient reporting are part of S0, not values to tune against assessment. Match the baseline clipping settings and record any change in clipping frequency or training cost caused by the objective.

Every run gets the same evaluation variants:

| Suffix | Probability processing | Extra student training? |
| --- | --- | --- |
| RAW | None | No |
| GT | One positive temperature fitted on separate calibration | No |
| PT | One positive temperature per primitive, fitted on separate calibration | No |

Report all three; do not select the most flattering variant on assessment. Temperature scaling preserves argmax decisions but can change Score means and confidence coverage. Distillation temperatures, if used, are different from these post-hoc temperatures and need separate provenance. Six student configurations produce 18 metric views per seed, not 18 training jobs.

## S2: controls that explain teacher gains

Run the relevant controls on the same inputs, weights, updates, and teacher-loss allocation as their parent arm. Their results distinguish an informative distribution from generic softening or a changed preferred label.

| Control ID | Replaces | Matched comparison | Question | Status |
| --- | --- | --- | --- | --- |
| SM-Q-B0 / SM-Q-B1 | Q's distribution with label-centered uniform smoothing | Q0 / Q1 | Is Q's benefit just softening? | Conditional |
| SM-J-B0 / SM-J-B1 | J's distribution with label-centered uniform smoothing | J0 / J1 | Is Jev's benefit just softening? | Conditional |
| QH-B0 / QH-B1 | Q's distribution with a one-hot vector at Q's top answer | Q0 / Q1 | Does the full Q vector help beyond its selected answer? | Conditional |
| JH-B0 / JH-B1 | J's distribution with a one-hot vector at J's top answer | J0 / J1 | Does the full Jev vector help beyond its selected answer? | Conditional |

Fix smoothing strength from training/development only and report effective target entropy against the teacher arm. Matched average entropy, when feasible, gives a stronger control than arbitrarily weak smoothing, but does not match case-specific uncertainty. Retain the same mixture with authored hard labels for the teacher-top-answer controls. A tie-breaking policy must be fixed before deriving teacher hard targets. Identical smoothing controls can be reused only when their complete targets and configurations match.

These controls reuse existing teacher captures; they do not trigger another full API pass.

## S3: explicit consistency and interactions

| ID | Brier | Consistency | Teacher | Matched consistency-off row | Execution status | Result artifact |
| --- | --- | --- | --- | --- | --- | --- |
| H2 | Off | On | None | H0 | Conditional | None |
| H3 | On | On | None | H1 | Conditional | None |
| Q2 | Off | On | Larger model | Q0 | Conditional | None |
| Q3 | On | On | Larger model | Q1 | Conditional | None |
| J2 | Off | On | Jev | J0 | Conditional | None |
| J3 | On | On | Jev | J1 | Conditional | None |

Together S1 and S3 form the 12 training configurations originally discussed: Brier off/on x consistency off/on x teacher none/Q/J. This count excludes S2 controls and additional seeds. All 12 with RAW/GT/PT produce 36 metric views per seed. Full interactions require the corresponding complete matched cells; selecting only some rows makes the follow-up conditional/exploratory.

Only penalize disagreement on certified invariances or true logical complements under the same evidence policy and explicit answer mapping. A change of label, a metadata `flip`, or two different rubrics does not establish complementary probabilities. Missing evidence and Unknown must retain their declared meanings. Report relation distance and both-endpoints-correct accuracy together; a constant or consistently wrong prediction must not pass as successful reasoning.

Use existing eligible relation endpoints. If additional inputs or forwards are needed, expose the same inputs to the consistency-off control and record the compute difference. The streaming training implementation may need graph scheduling/recomputation; preserve the original experimental artifacts and verify the new path separately.

## S4: confirmation and comparison rules

Repeat each selected recipe together with its matched control on at least one additional training seed before claiming repeatable benefit. A family-bootstrap interval from one trained model does not measure seed variability. Select candidates using development results and a prespecified policy, then lock endpoint identities/calibration before fresh confirmation. Do not extend a promising arm opportunistically or select a different best epoch after inspecting assessment.

The reporting matrix for every executed configuration, seed, and RAW/GT/PT view includes:

| Metric group | Required outputs |
| --- | --- |
| Semantic decisions | Individual accuracy; category-balanced both-correct contrast-pair accuracy; complete-case accuracy; evidence/question/rubric axes; invariants separately |
| Probability quality | NLL and full-vector Brier; reliability; wrong >=0.90 and >=0.95 with counts, coverage, and conditional error |
| Useful uncertainty | Risk versus coverage; Unknown precision/recall and confusion matrix; disagreement strata |
| Structured outputs | Noul/Choice/Score separately; Score by number of levels; Score mean MAE alongside distribution metrics |
| Semantic categories | Claim/completion, permission/execution, unknown/failure, attribution, entity/action binding, time, reversals, negation/quantifiers, ordered rubrics |
| Retention and cost | Fixed old-task accuracy/NLL/Brier; actual candidate units/tokens/forwards, time, memory, clipping statistics, teacher requests and usage |
| Uncertainty of comparisons | Paired whole-family intervals and per-seed differences; clearly mark exploratory categories and multiple comparisons |

Success is improvement beyond calibrated H0 without an unacceptable loss in semantic accuracy, Score behavior, or retention. Numeric acceptance tolerances and coverage comparisons must be locked before the next assessment. Fewer high-confidence mistakes achieved by becoming uncertain on everything is insufficient. On small categories, show counts rather than only percentages.

## S5: later branches, tracked separately

These address different hypotheses and should not be multiplied into all 12 rows automatically.

| Branch ID | Method or factor | Controlled question | Preconditions / limitations | Status |
| --- | --- | --- | --- | --- |
| DIST-EXACT | Mathematically known conditional-distribution targets | Does explicit evidence-dependent uncertainty transfer better than sampled hard outcomes? | Match visible priors/noise/evidence and input distribution; hard control uses a seeded draw from the same distribution, not its argmax; retain deterministic regressions | Deferred |
| CAL-ADAPT | Input-dependent positive temperature | Can a learned calibration function beat GT/PT at useful coverage? | Extra calibration model/data; same positive scale across candidates of a question; independent fit/evaluation | Deferred |
| INIT | Fresh post-trained student versus pretraining-only Base, with v0.2 as reference | Does starting-model history matter? | Matched task-interface preparation and exposure; separate from loss/data changes | Deferred |
| CAPACITY | Selected 2B recipe versus a larger student | Is remaining error capacity-limited? | Actual memory, training/inference latency, and compute reporting; larger teacher does not test larger student capacity | Deferred |
| RLCR | Generated reasoning/answer plus calibration-aware reward | Does sampled reasoning improve verified decisions and uncertainty? | New generative interface, verifier/reward audit, cost-matched non-RL baseline; potentially use as a teacher then distill | Deferred |
| CONFTUNER | Tokenized Brier for verbal confidence | Is a verbal confidence interface useful for a generative teacher? | Different target space from categorical Brier; no implication that the fast student should generate confidence text | Deferred |
| DOUBT | Log-score RL for confidence about fixed answers | Can a separate confidence estimator improve routing? | Keep answer selection fixed during rewarded confidence action; does not directly fix answer selection | Deferred |
| DATA-SEM | Targeted semantic/rule coverage with controlled edits | Do new semantic distinctions fix residual time, quantifier, or source errors? | Preserve known labels and hold out rule/scenario families; CoBA/Tailor/contrast-set ideas are generation tools, not gold oracles | Deferred |
| DATA-LANG | More independent language/domain realizations | Does genuinely different evidence language transfer? | Keep semantic/rubric exposure matched; the 200-to-400 result is limited to its reviewed synthetic bank | Deferred |

Relevant papers: [proper scoring rules](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), [temperature scaling](https://proceedings.mlr.press/v70/guo17a.html), [distillation](https://arxiv.org/abs/1503.02531), [label smoothing](https://research.google/pubs/when-does-label-smoothing-help/), [adaptive temperature](https://arxiv.org/abs/2409.19817), [RLCR](https://arxiv.org/abs/2507.16806), [ConfTuner](https://arxiv.org/abs/2508.18847), and [Rewarding Doubt](https://arxiv.org/abs/2503.02623). Data-generation alternatives and actual probe limitations are recorded in [generation research](../reports/CONTRAST_GENERATION_RESEARCH.md) and [tool/effort results](../reports/TOOL_EFFORT_PROBE.md).

## Jev collection: courteous throughput and durable records

The user's latest 2026-09-22 clarification: use **5-10 requests/second, with at most 10 in flight**, and overlap collection with other useful work. This supersedes the earlier conservative 1 request/second suggestion and the intermediate 2 request/second draft. Proposed defaults for a future collector are **5 requests/second initially, rising to a ceiling of 10 requests/second after a healthy initial batch, at most 10 in flight, with smooth pacing and no accumulated bursts**. Rate and concurrency are separate ceilings, not throughput guarantees. Slower fallback follows actual throttling or token pressure. Retries consume the same shared rate/concurrency budget as first attempts.

TypeSafe's [models page](https://docs.typesafe.ai/models), checked 2026-09-22, currently lists 1,200 requests/minute, 250,000 input tokens/second, and $0.042 per million input tokens with free outputs. It explicitly says limits can change. At 5-10 requests/second the request rate is 300-600/minute; still pace by estimated total input-token demand, reconcile estimates with returned usage, and honor any lower service/account limits. Ten maximum-size requests each second could exceed the token limit despite staying below the request limit. Large payloads, aggregate traffic from other clients, and overload can require slowing down. Never increase traffic just to discover the maximum throttle threshold.

Collection policy to implement before bulk requests:

1. Pin the model version, verify returned identities, and group independent questions over the same state within the documented context budgets. Run one coordinated collector so multiple workers do not each apply a separate rate allowance.
2. Honor `Retry-After` for 429/529; otherwise use bounded exponential backoff with jitter. Proposed fallback delays start at 2 seconds and double up to 60 seconds, with at most 5 retries per request. Never shorten a server-specified wait to the local cap. On persistent overload, save progress and pause; reduce the global rate rather than repeatedly hammering the service. These behaviors follow [TypeSafe's API guidance](https://docs.typesafe.ai/api#handling-rate-limits).
3. Treat authentication and schema errors as concrete failures, not retry loops. Preserve ambiguous network timeouts/interrupted reads as separate attempts; a retry may incur another charge, and must never be reported as exactly-once execution.
4. Extend the existing [durable archive](jev-comparison.md), which already preserves requests, raw received bodies, safe metadata, usage, validation, version, and distinct attempts. Its current client stops on throttle responses; pacing/concurrent scheduling/bounded retries are proposed additions, not implemented features. Exclude credentials from saved artifacts.
5. Reuse validated identical requests only with matching fixed model and input identity, within permitted train/audit/evaluation partitions. Preserve all errors and repeated draws; never overwrite an old result or resample until a desired answer appears. Resume without discarding partial artifacts. Where a service supplies no bytes or usage, record that information as unknown.
6. Share cached target data across all student arms. For a 1,600-case corpus with five questions per case, same-state batching suggests 1,600 fresh requests per teacher before audit/repeat requests, not 8,000 separate field requests or another collection per arm. Actual packaging and cache hits must be counted.
7. Track reported input tokens, attempts/retries, and estimated cost separately from verified billing or account balance. At the current list price, 20 million input tokens cost approximately $0.84. The user reports a $5 trial credit and is comfortable with this scale; remaining balance has not been verified here. An estimate is not evidence that a particular number of requests is affordable without measuring their billed input size.
8. Run the future collector as a durable background task with one owner/lock, resumable request ledger, and visible progress (queued, in flight, captured, validated, failed, retrying, usage, and elapsed time). Work on independent local integration, CPU validation, or report preparation while Jev is generating. Never launch a duplicate collector. Coordinate local larger-model inference and student GPU work exclusively when they share the 4080. Freeze the complete teacher-target manifest before any student arm consumes it; do not start an arm against targets still changing in the background.

No API call was made to create this roadmap. All historical paid results and quarantines remain intact.

## Execution record and decision log

When a run is selected, update its status above and add a row here. Keep the scientific configuration immutable once launched; repairs and repetitions get new attempt identifiers and retain earlier artifacts. Status vocabulary: proposed, selected, prepared, running, paused, complete, rejected, or skipped. Conditional/deferred describe the roadmap branch until selected.

| Date | Experiment / attempt | Decision or status | Configuration / data / target hashes | Training / evaluation artifacts | Finding / next decision |
| --- | --- | --- | --- | --- | --- |
| 2026-09-22 | ROADMAP-v1 | Documented | Existing data and model artifacts unchanged | This document only | Six-run first screen proposed; additional controls and branches remain conditional |
| 2026-09-22 | JEV-PACING-v1 | Superseded draft | No client mutation | [Existing client behavior](jev-comparison.md) | Earlier 2 requests/second, 2 in flight proposal; replaced by the user's higher-rate preference below |
| 2026-09-22 | JEV-PACING-v2 | User preference documented; implementation pending | No client mutation | This document's collection section | Start 5 requests/second, ceiling 10 requests/second and 10 in flight; shared token-aware pacing/backoff; background collection with durable progress |

For each actual attempt, record seed, starting checkpoint, source/data/target fingerprints, coefficients, exposure schedule, teacher protocol, output path, process identity, checkpoint state, endpoint/calibration locks, metrics, uncertainty, cost, and reason for continuing or stopping. A design cell without an artifact is not a completed experiment.

The completed factorial run and its monitor stay complete/paused; the old contrast-scaling run stays intentionally paused at update 1,692. The new S0/S1 authorization does not resume either earlier job, alter their frozen artifacts, or promote a checkpoint.

Execution started 2026-09-22: live status, queue progress, launch, preparation and teacher results. The queue completed all six 400-update arms and locked evaluation. Results are verified and the monitor is paused. All later roadmap stages remain conditional.

Discussion extension, 2026-09-22: [architecture, readout, and adaptation options](architecture-options-v1.md) records the actual rank8/two-head configuration, a static candidate-context limitation, and remaining diagnostic/training comparisons. This is an options note; no follow-on experiment or monitor was started.

Subsequent user-authorized execution, 2026-09-22: the [small architecture diagnostics](../reports/architecture-diagnostics-v1/RESULTS.md) are complete. The fixed protocol and machine-readable summary record candidate presentation, native direct/capped reasoning, 80-update tiny fitting, and frozen-feature head comparisons. Existing linear readouts fit 200/200 with frozen features; larger heads do not improve development correctness. Tiny LoRA fitting reaches 199/200, with development 151/200 unchanged and worse raw NLL. Repeated full criteria improve raw NLL and 3/200 answers, but the criteria already existed in STATE. All 200 reasoning traces truncate at 128 tokens, so full native reasoning remains unresolved. A 16.7-second same-instance control reproduces 4 batch-packing decision flips with exactly repeatable outputs within each packing. The next decision should account for this numerical sensitivity and the limited native reasoning test. No larger-rank, unfreezing, new model, RL or paid collection study starts automatically; the old monitor stays paused.

Further user-authorized follow-up, 2026-09-22: [consistency and native reasoning](../reports/consistency-native-v1/RESULTS.md) is complete. A new cached inference policy passes exact layout invariance on 43 requests / 215 judgments / 408 units, retains 152/200 development and 55/60 retention accuracy, and costs about 6.4% more median request time than legacy caching in the small warmed sample. Original Qwen naturally completes 15/30 questions at a 4,096-token cap and gets all 15 right; on that same subset direct native scoring gets 9 and trained H0 gets 13. A separately declared six-case sampling diagnostic recovers four completed thoughts with correct constrained decisions (three valid natural final answers); two still cap. These results identify both numerical execution sensitivity and decoding/termination limitations, while demonstrating some useful reasoning ability. No next learning study or checkpoint promotion is automatic.
