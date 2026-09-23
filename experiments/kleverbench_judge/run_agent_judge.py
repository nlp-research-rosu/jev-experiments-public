#!/usr/bin/env python3
"""Run the agent judges (claude-code, codex) over the suite.

Production replica of kit-eval's judge: an agent with filesystem access, the
judge-template.md untrusted-data preamble, one binary criterion, three
independent samples per case aggregated `all_pass`.

One JSON file per (case, judge, sample) under results/, so the run is
resumable: a 429 wave costs the cases in flight, not the run.

Usage:  python3 run_agent_judge.py [--judges claude,codex] [--samples 3]
                                   [--concurrency 6]
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "kleverbench-judge-v1"
SUITE = DATA / "suite"
RESULTS = DATA / "results"
CRITERION = (HERE / "criterion.md").read_text()

VERDICT_RE = re.compile(r"VERDICT:\s*(YES|NO)", re.I)
CODEX_TOKENS_RE = re.compile(r"tokens used\s+([\d,]+)")

JUDGES = {
    "claude": ["claude", "-p", CRITERION, "--model", "claude-sonnet-5",
               "--allowedTools", "Read,Grep,Glob", "--output-format", "json"],
    "codex": ["codex", "exec", "--model", "gpt-5.6-luna",
              "--skip-git-repo-check", CRITERION],
}


def parse(judge: str, stdout: str):
    """Return (verdict, tokens, cost). Verdict is None if unparseable."""
    tokens = cost = None
    text = stdout
    if judge == "claude":
        try:
            d = json.loads(stdout)
            text = d.get("result", "") or ""
            cost = d.get("total_cost_usd")
            u = d.get("usage") or {}
            tokens = sum(v for k, v in u.items()
                         if isinstance(v, int) and "token" in k)
        except (json.JSONDecodeError, AttributeError):
            pass
    else:
        m = CODEX_TOKENS_RE.search(stdout)
        if m:
            tokens = int(m.group(1).replace(",", ""))
    hits = VERDICT_RE.findall(text)
    return (hits[-1].upper() if hits else None), tokens, cost


def run_one(case: str, judge: str, sample: int):
    dest = RESULTS / f"{case}__{judge}__s{sample}.json"
    if dest.exists():
        return "cached"
    t0 = time.time()
    try:
        p = subprocess.run(JUDGES[judge], cwd=SUITE / case, capture_output=True,
                           text=True, timeout=600)
        out, err, rc = p.stdout, p.stderr[-2000:], p.returncode
    except subprocess.TimeoutExpired:
        out, err, rc = "", "TIMEOUT after 600s", -1
    verdict, tokens, cost = parse(judge, out)
    dest.write_text(json.dumps({
        "case": case, "judge": judge, "sample": sample, "verdict": verdict,
        "tokens": tokens, "cost_usd": cost, "seconds": round(time.time() - t0, 1),
        "returncode": rc, "stderr": err,
        "raw_tail": out[-3000:],
    }, indent=2))
    return verdict or "UNPARSED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="claude,codex")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text())
    jobs = [(m["case"], j, s)
            for m in manifest
            for j in a.judges.split(",")
            for s in range(1, a.samples + 1)]

    todo = [j for j in jobs
            if not (RESULTS / f"{j[0]}__{j[1]}__s{j[2]}.json").exists()]
    print(f"{len(jobs)} invocations, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run at concurrency {a.concurrency}", flush=True)

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(run_one, *j): j for j in todo}
        for f in as_completed(futs):
            done += 1
            case, judge, s = futs[f]
            try:
                r = f.result()
            except Exception as e:                      # noqa: BLE001
                r = f"ERROR {e}"
            if done % 10 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1)
                eta = (len(todo) - done) / rate / 60 if rate else 0
                print(f"  {done}/{len(todo)}  last={judge}:{r}  "
                      f"eta {eta:.1f} min", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
