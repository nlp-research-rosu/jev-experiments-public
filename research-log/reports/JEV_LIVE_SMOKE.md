# First live Jev comparison

Live access succeeded on 2026-09-20. The API returned **jev-1.13.0** for all eight
requests. All **51/51** judgments match the existing authored diagnostic labels,
versus **25/51** for our saved base interface and **30/51** for our saved trained
checkpoint. These eight scenarios were selected to exercise known behavior gaps;
they are not representative production samples or an independent benchmark.

The inputs were the exact full states and question definitions from the frozen
semantic-contrast suite, with no source filtering or prompt edits. Jev received
only state, questions and the requested model. Expected labels and evaluator
rationales were not transmitted. Local references use the same full inputs.

| Question / case | Expected | Our trained model | Live Jev |
|---|---|---:|---:|
| Assistant claimed completion; claim-only case | Yes | P(yes) 96.7% | P(yes) 98% |
| Operational result confirms completion; same claim-only case | No | P(yes) 96.5% | P(yes) 6% |
| Completion confirmed; matching live success | Yes | P(yes) 99.1% | P(yes) 98% |
| Completion confirmed; uncertain timeout | No | P(yes) 95.3% | P(yes) 5% |
| Registry confirms change; sender claims change but registry is unchanged | No | P(yes) 73.7% | P(yes) 8% |
| Current record status; unverified record | Unknown | P(Unknown) 27.5%; selects Changed | P(Unknown) 100% |
| Current record status; conflicting equally authoritative records | Unknown | P(Unknown) 22.2%; selects Unchanged | P(Unknown) 100% |

The Unknown results are the API's returned probabilities, not guarantees about
calibration. Under these fixtures' explicit contract, an unverified record or a
conflict gives Unknown; we have not changed the contract to force a particular
Jev answer. Missing completion evidence likewise means the binary proposition
“the record confirms completion” is false, not that physical failure is proven.

All six included same-record question contrasts have both answers correct for
Jev. This selected subset contains no complete declared evidence-flip or invariance
relations, so there is no new measurement of those pair metrics yet.

| Measurement | Result |
|---|---:|
| Requests completed | 8/8 |
| Judgments correct | 51/51 |
| Mean NLL from returned probabilities | 0.04035 |
| Mean categorical Brier score | 0.00504 |
| Median client-observed API time | 352 ms |
| Minimum / maximum API time | 290 / 387 ms |
| Reported input / output tokens | 11,015 / 1,189 |

Timing includes network and HTTPS connection setup and is not a local GPU kernel
measurement. NLL uses a declared probability floor of 1e-12 because the API does
not expose raw logits. No expected answer received zero probability in this run.
Brier sums across categories, so its binary form is twice the scalar convention.
The API confidence statistic is retained separately and is not treated as
probability of correctness. No post-hoc calibration was fitted.

The result establishes access and provides concrete contrasting behaviors. It
does not reveal Jev's architecture, parameter count, training data or RLCD loss.
The remaining frozen cases and future fresh pilot cases can use the same runner.
No Jev observations were added to training or substituted for gold labels.

## Artifacts and checks

- Live report and local disagreements
- Exact requests
- Returned responses, probabilities, confidence and timing
- Question-level predictions
- Included question-contrast results
- [Setup and replay guide](../design/jev-comparison.md)

The integration passed nine focused offline tests and the full repository suite:
**263 tests and 45 subtests**. The suite emitted the existing harmless pytest
assert-rewrite warning for an already imported `anyio` module. Lint passed for the
new runner and tests. The credential is kept in the private ignored key file or
environment and is not stored in these reports.
