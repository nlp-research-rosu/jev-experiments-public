# Can Jev judge a proof? (KleverBench judge eval, v1)

A prover can tell you a K claim holds. It can't tell you the claim states what
the task asked for. A spec that bounds the input to one point, or asserts
nothing about the result, still proves. Catching that is the judge's job.

This experiment asks whether **jev-1.13.0** (TypeSafe System One) can do that
job as well as the agent judges we use today: **claude-sonnet-5** (the
kit-eval judge) and **gpt-5.6-luna** (via Codex CLI).

**Short answer:** Jev never accepted a bad spec (0 false positives out of 27),
but it rejected half the correct ones (9 of 18) when asked a single question.
Asking it three narrower questions and requiring all of them to pass lowers
that to 5 of 18. Claude and Luna were at or near 100% either way. Jev is about
1 second and well under a tenth of a cent per case, so it works as a cheap
first filter, not as a replacement for the agent judge.

Date of runs: 2026-09-22. Full write-up with charts: [`report.html`](report.html)
(GitHub shows HTML as source; download it or open it from a local checkout).

## The suite

45 cases = 3 problems × 5 variants × 3 lanes. 18 GOOD, 27 BAD.

Every case is built from a KleverBench gold `reference-spec.k`
(`nlp-research-rosu/KleverBench` @ `ae3a9fea`), so the labels are exact by
construction and no one had to hand-label anything.

| variant | label | how it is built |
|---|---|---|
| `ref` | GOOD | the reference spec, unchanged |
| `equiv` | GOOD | renamed variables, reordered cells, a trivially true `requires` |
| `vacuous` | BAD | input collapsed to a single point (`requires A ==Int 1 ...`) |
| `narrowed` | BAD | input bounded to a window (`requires A >Int 0 andBool A <Int 100`) |
| `weakened` | BAD | result asserts nothing (`res \|-> (_ => _)`) |

Problems: `abs-times`, `array-sum`, `gcd-euclid`.

Lanes are the same problems in three languages:
- `imp`: operators mean what they look like.
- `imp-obf`: same meanings, operators written as unfamiliar glyphs.
- `imp-swap`: operators lie. `+` is subtraction, `/` is multiplication, and so
  on. Only `semantics.k` says what they do. The gap between `imp` and
  `imp-swap` shows whether a judge reads the semantics or guesses from symbols.

Each case directory (`suite/<lane>__<problem>__<variant>/`) holds only
`TASK.md` (the intent), `program.imp`, `semantics.k`, `verification.k` and the
candidate `spec.k`. No judge saw the reference spec or the label.

All specs were checked with the Prover (`validate.json`): the `vacuous` and
`narrowed` mutants compile and prove, and the `weakened` ones fail to kompile.
So 18 of the 27 BAD specs are defects only a judge can catch. Scores on that
subset are in the report and do not change the conclusions.

## Results

One call per case for every judge, all 45 cases.

| judge | N | accuracy | false positive (bad called good) | false negative (good called bad) | latency | cost / case |
|---|---|---|---|---|---|---|
| jev | 45 | 80.0% | 0/27 (0%) | 9/18 (50%) | 1.1 s | ~$0.0006 |
| sonnet | 45 | 100% | 0/27 (0%) | 0/18 (0%) | 45.7 s | ~$0.17 |
| luna | 45 | 95.6% | 2/27 (7.4%) | 0/18 (0%) | 69.6 s | ~$0.0018 |
| sonnet (1 turn, no tools) | 45 | 100% | 0/27 (0%) | 0/18 (0%) | 42.6 s | ~$0.12 |
| luna (1 turn, no tools) | 45 | 97.8% | 1/27 (3.7%) | 0/18 (0%) | 24.6 s | ~$0.0016 |

What the rows mean:
- **jev**: one API request, the single question `faithful_overall`
  ("does the spec state exactly the intended property over the full domain?").
  Jev returns P(yes); GOOD if > 0.5.
- **sonnet / luna**: the production-style agent judge. The CLI runs in the
  case directory with tools, reads the files it wants (several turns), and
  ends with `VERDICT: YES|NO`. Numbers are from the first of three runs.
- **(1 turn, no tools)**: same models, tools disabled, all five files inlined
  into one prompt, asked Jev's exact `faithful_overall` wording. This is the
  same shape of call Jev gets.

Other configurations:

| judge / config | accuracy | false positive | false negative |
|---|---|---|---|
| jev, 3 questions (all must pass) | 88.9% | 0/27 | 5/18 |
| claude, 3 runs (must agree) | 100% | 0/27 | 0/18 |
| codex, 3 runs (must agree) | 100% | 0/27 | 0/18 |

"3 questions" = `domain_unrestricted`, `results_constrained` and
`relation_matches_intent`, asked in the same request; GOOD only if all three
are > 0.5. "3 runs" = three independent agent runs; GOOD only if all three
say YES (kit-eval's `all_pass`).

**Cost** is what the calls would cost at OpenRouter list prices
(`typesafe/jev-1.13`: $0.042/M input; `anthropic/claude-sonnet-5`: $2/M in,
$10/M out, $0.20/M cache read; `openai/gpt-5.6-luna`: $0.20/M in, $1.20/M out,
$0.02/M cached). Jev's cost is exact from its token counts. For the CLIs we
computed a cache-aware blended rate from the runs where the full usage
breakdown was recoverable (5 to 30 runs per row) and applied it to the mean
tokens over all 45 cases, so treat those as rough. The Claude CLI adds about
41.6k tokens of fixed framing to every call; Codex adds about 640.

## Findings

1. **Jev's errors are all false negatives, and they cluster in `imp-swap`.**
   With 3 questions, every one of its 5 errors is a correct `imp-swap` spec
   (both `abs-times` specs, both `array-sum` specs, and `gcd-euclid/ref`)
   where `relation_matches_intent` scored 0.24 to 0.45. The labels were rechecked by hand under the swapped rules and
   are correct.
2. **It is not the size of the semantics file.** Rerunning all 45 cases with
   `semantics.k` cut from 26 KB to only the rule lines (about 5 KB), and again
   to only the rules for operators the program uses, did not help: accuracy
   went 89% → 87% → 84%. Jev struggles to simulate arithmetic whose meaning has
   been swapped, even when given exactly the rules it needs.
3. **Tools are not why the agent judges win.** With tools removed and the same
   inlined input as Jev, Claude stays at 100% and Luna at 97.8%.
4. **A retracted claim.** `imp/gcd-euclid` and `imp-swap/gcd-euclid` are
   byte-identical apart from `semantics.k`, and Jev accepted one and rejected
   the other. We first read this as the irrelevant semantics file
   contaminating the answer. The trim ablation and the scores across all lanes
   (all near 0.5) show it is a hard problem for Jev in every lane, so that
   claim is withdrawn.
5. **The suite is too easy for the agent judges.** Both hit the ceiling, so it
   can't rank them or show where they start to fail. Harder or real agent
   failures are needed for that.

## Caveats

- 45 cases, 3 problems. Small.
- The BAD specs are systematic mutants. Real agent mistakes are messier.
- Jev returns only a probability, no explanation. The agent judges explain
  their verdict.

## Files

| path | contents |
|---|---|
| `suite/` | the 45 case directories exactly as the judges saw them |
| `manifest.json` | case → lane, problem, variant, label |
| `validate.json`, `logs/validate.log`, `logs/prove.log` | Prover checks of every spec |
| `jev_results/` | raw Jev responses, full semantics (all 4 questions per case) |
| `jev_results_rulesonly/`, `jev_results_relevant/` | Jev under the two semantics trims |
| `results/` | agent judges with tools, `<case>__<claude\|codex>__s<1-3>.json` |
| `notools_results/` | agent judges, 1 turn, no tools |
| `notools_overhead.json` | fixed CLI token overhead on a trivial prompt |
| `codex_split.json` | Codex input/output token split on an 8-case resample |
| `scored.json`, `scored2.json` | per-case predictions from the scoring scripts |
| `report.html` | the full report |

Code: [`experiments/kleverbench_judge/`](../../experiments/kleverbench_judge/).
