# KleverBench judge eval: code

Scripts for the Jev vs agent-judge comparison. Results and the write-up are in
[`data/kleverbench-judge-v1/`](../../data/kleverbench-judge-v1/README.md).
All scripts read and write that data directory, so the committed results are
reused and only missing cases are run.

Plain Python 3, standard library only.

## Rescore without any API calls

```bash
python3 experiments/kleverbench_judge/build_table.py    # main tables
python3 experiments/kleverbench_judge/build_table2.py   # no-tools and semantics-trim tables
python3 experiments/kleverbench_judge/build_report.py   # rebuilds data/.../report.html
```

## Full pipeline

| step | script | needs |
|---|---|---|
| build the 45 cases | `build_suite.py` | a KleverBench checkout (`KLEVERBENCH_ROOT`, default `../KleverBench`) |
| check every spec with the Prover | `validate_specs.py`, `prove_check.py` | `kprover` on `PATH` |
| Jev | `run_jev.py` | `TYPESAFE_API_KEY` |
| Jev, trimmed semantics | `run_jev_ablation.py` (uses `semantics_trim.py`) | `TYPESAFE_API_KEY` |
| agent judges with tools, 3 runs each | `run_agent_judge.py` (prompt in `criterion.md`) | `claude` and `codex` CLIs, logged in |
| agent judges, 1 turn, no tools | `run_notools.py` | same |
| Codex token split | `codex_split_probe.py` | same |

Versions used: `jev-1.13.0`, `claude-sonnet-5` via Claude Code 2.1.267,
`gpt-5.6-luna` via codex-cli 0.147.0.

The API key is read from the environment only. Never commit it.
