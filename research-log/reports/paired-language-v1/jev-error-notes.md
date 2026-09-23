# Jev error notes after checkpoint selection

This is descriptive analysis of the frozen test and archived Jev predictions, performed after the selection lock recorded template update 0 and natural update 500. It changes no labels, examples, checkpoint choices, or metric policy. No new API calls were made.

Jev has **36 incorrect judgments out of 800** (764/800 correct, 95.5%). Unknown recall is **36/41** (87.8%). Two incorrect judgments have selected-answer probability at least .90; one reaches .95. These probabilities come from the answer distribution, not the separately returned Jev `confidence` field.

## 1. No attempt or no record can become a definite failed outcome

These are clear disagreements with the supplied evidence definitions, rather than errors that require a hidden world-state assumption:

| Case ID / question | Frozen target → Jev answer | Evidence and interpretation |
| --- | --- | --- |
| `test/permission_vs_execution/02/v3` / `c1` | `unknown` → `not_transmitted` (.73; Unknown .27) | The audit says “no E77 run entry.” The `not_transmitted` criterion requires the audit to establish that no rows were transmitted; an absent entry supplies no such affirmative result. |
| `test/unknown_vs_failure/02/v3` / `c1` | `unknown` → `failed` (.59; Unknown .41) | Inoculation was canceled before starting, and the state explicitly defines that as no fermentation attempt. `failed` requires an affirmative failed disposition, which is absent. The paired `n1` also incorrectly predicts false (.60) for whether the result is unknown. |
| `test/unknown_vs_failure/04/v3` / `c1` | `unknown` → `failed` (.56; Unknown .44) | Sampling was deferred before drill deployment; the state explicitly defines that as no attempt. No failed extraction is logged. The paired `n1` also incorrectly predicts false (.64) for whether the outcome is unknown. |

The other two missed Unknown targets are `test/entity_binding/01/v3` / `c1` (`both`, .61, despite explicitly blank working-key tests) and `test/reversal_and_current_state/02/v2` / `c1` (`released`, .63, despite no issued hold instruction and no established current state). None of the five missed Unknown targets is a ≥.90 error. This is a recurring tendency in a small set of examples, not evidence that all Unknown judgments are weak.

## 2. Many rubric errors have an input-context limitation

Ordered rubrics account for 12 of the 36 incorrect judgments. Ten are threshold questions `n1` or `n2` whose named bands are defined only in sibling Choice/Score criteria, not in the state or the binary question's own criteria. The binary criteria merely ask whether the predicate is supported.

| Case ID / question | Frozen target → Jev answer | Missing definition in that binary judgment |
| --- | --- | --- |
| `test/ordered_rubrics/04/v0` / `n1` | `true` → `false` (.92) | The state lists three passing checks, but the mapping “Release Ready = 3 checks” appears only in `c1`/`s1`. Jev correctly selects `release_ready` in `c1` and level 2 in `s1`, both at 1.00. |
| `test/ordered_rubrics/01/v2` / `n2` | `true` → `false` (.75) | The state gives 1,400 affected customers, but the Critical trigger of at least 1,000 appears only in `c1`/`s1`. Both explicitly defined categorical judgments are correct at 1.00. |
| `test/ordered_rubrics/03/v2` / `n2` | `true` → `false` (.75) | The state gives 10 points, but “Expedited Award = 9–10” appears only in `c1`/`s1`. Both explicitly defined categorical judgments are correct at 1.00. |

The ten affected error endpoints are:

- `test/ordered_rubrics/01/v0` / `n1`; `test/ordered_rubrics/01/v2` / `n1`, `n2`.
- `test/ordered_rubrics/02/v0` / `n1`; `test/ordered_rubrics/02/v2` / `n2`.
- `test/ordered_rubrics/03/v0` / `n1`; `test/ordered_rubrics/03/v2` / `n1`, `n2`.
- `test/ordered_rubrics/04/v0` / `n1`; `test/ordered_rubrics/04/v2` / `n2`.

Local `compile_request` inspection confirms that each binary unit contains only state, its own question, and its own binary criteria. Sibling criteria are absent. Jev's internal treatment of sibling questions is not established by these observations, so its internal cause cannot be asserted. The frozen labels follow the authors' complete intended rubric, but these binary inputs are not independently self-contained. Consequently, these errors—including the .92 error—should not be presented as unqualified evidence of failed threshold arithmetic. The original scores remain unchanged; this is an interpretation caveat, not a post-hoc exclusion.

## 3. The .95 error confuses clearance with crew staging, with a wording caveat

`test/permission_vs_execution/03/v0` / `s1` has frozen target **1, “Clearance issued,”** but Jev selects **2, “Crew staged without ignition,” at .95** (level 1 receives .05). The record says Pike cleared the burn, but high wind kept the crew at base and no ignition occurred. It does not affirmatively record staging. Jev separately gets the no-ignition Choice correct at 1.00, so the error is specifically about intermediate progress, not whether a fire started.

The family supplies a useful contrasting example: `test/permission_vs_execution/03/v2` explicitly records staging engines and drip torches, and Jev correctly selects level 2 at .96. The v0 gold is defensible from the absence of a staging event. However, “crew staged” does not specify a location or operational definition; someone could interpret a crew waiting at base as staged. Treat this as a high-confidence disagreement with the intended progress rubric, with some wording ambiguity, rather than an indisputable physical-state error.

## Input identity

- Frozen suite: `data/paired-language-v1/test/suite.json`; SHA-256 `e6c0b4f5148538427ffce282f2ac475681551ee774eb2970a2e1430372d8ab89`.
- Jev predictions: `reports/jev-paired-language-test-v1/predictions.json`; SHA-256 `54278b08bd2c4e90cf0782ab0bc347d25b71c2d16def12a666466a3ca64b3d60`.
- Selection lock: `checkpoints/paired-language-v1/selection-locked.json`; SHA-256 `8f90ace6dff56e45c58fdb80328a0d18dfbfa59df2a6c901584daa20de2892a9`.

Only the frozen suite, archived predictions, selection lock, and local request compiler were inspected for these notes. No labels or inputs were repaired after observing outcomes.
