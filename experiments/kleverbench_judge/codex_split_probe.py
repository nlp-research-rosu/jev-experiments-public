#!/usr/bin/env python3
"""Small resample to recover codex's real input/output token split.

Plain `codex exec` stdout only ever printed one blended "tokens used" total
(confirmed: the CLI's human-readable output has no input/output breakdown).
`codex exec --json` does expose it, in the `turn.completed` event's `usage`
object. This resamples a handful of cases per condition purely to measure
that split ratio; the totals themselves stay anchored to the full 45-case
runs already on disk.
"""
import json
import pathlib
import random
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "kleverbench-judge-v1"
SUITE = DATA / "suite"
EMPTY_WS = DATA / "empty_ws"
CRITERION = (HERE / "criterion.md").read_text()
NOTOOLS_CRITERION_MOD = HERE / "run_notools.py"

random.seed(7)
manifest = json.loads((DATA / "manifest.json").read_text())
sample = random.sample(manifest, 8)


def usage_of(jsonl_text):
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "turn.completed" and "usage" in d:
            return d["usage"]
    return None


def with_tools(case):
    p = subprocess.run(
        ["codex", "exec", "--model", "gpt-5.6-luna", "--skip-git-repo-check",
         "--json", CRITERION],
        cwd=SUITE / case, capture_output=True, text=True, timeout=300)
    return usage_of(p.stdout)


def no_tools(case):
    import importlib.util
    s = importlib.util.spec_from_file_location("n", NOTOOLS_CRITERION_MOD)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    prompt = f"{m.SYSTEM_PROMPT}\n\n{m.build_prompt(case)}"
    p = subprocess.run(
        ["codex", "exec", "--model", "gpt-5.6-luna", "--skip-git-repo-check",
         "--sandbox", "read-only", "--json", prompt],
        cwd=EMPTY_WS, capture_output=True, text=True, timeout=300)
    return usage_of(p.stdout)


def main():
    results = {"with_tools": [], "no_tools": []}
    for m in sample:
        c = m["case"]
        u = with_tools(c)
        if u:
            results["with_tools"].append(u)
        print("with_tools", c, u, flush=True)
    for m in sample:
        c = m["case"]
        u = no_tools(c)
        if u:
            results["no_tools"].append(u)
        print("no_tools", c, u, flush=True)
    (DATA / "codex_split.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
