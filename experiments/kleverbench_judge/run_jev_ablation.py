#!/usr/bin/env python3
"""Rerun Jev with two trims of `semantics` (see semantics_trim.py), holding
everything else identical, over ALL 45 cases -- not just the ones it got
wrong, so the trim can't be accused of being picked to flatter it.

Usage: TYPESAFE_API_KEY=... python3 run_jev_ablation.py [--concurrency 4]
"""
import argparse
import importlib.util
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "kleverbench-judge-v1"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


jv = _load("jv", HERE / "run_jev.py")
tr = _load("tr", HERE / "semantics_trim.py")

VARIANTS = {
    "rulesonly": lambda sem, prog: tr.rules_only(sem),
    "relevant": lambda sem, prog: tr.operator_relevant(sem, prog),
}


def ask_variant(case: str, variant: str, key: str) -> dict:
    state = jv.state_for(case)
    state["semantics"] = VARIANTS[variant](state["semantics"], state["program"])
    body = json.dumps({"state": state, "model": jv.MODEL,
                       "questions": jv.QUESTIONS}).encode()
    import urllib.request, urllib.error, time
    req = urllib.request.Request(
        jv.ENDPOINT, data=body, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        out = {"error": f"HTTP {e.code}"}
    out["_seconds"] = round(time.time() - t0, 2)
    out["_semantics_bytes"] = len(state["semantics"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    key = os.environ["TYPESAFE_API_KEY"]

    manifest = json.loads((DATA / "manifest.json").read_text())
    for variant in VARIANTS:
        outdir = DATA / f"jev_results_{variant}"
        outdir.mkdir(exist_ok=True)
        todo = [m["case"] for m in manifest
                if not (outdir / f"{m['case']}.json").exists()]
        print(f"[{variant}] {len(todo)} to run", flush=True)
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futs = {ex.submit(ask_variant, c, variant, key): c for c in todo}
            for f in as_completed(futs):
                c = futs[f]
                out = f.result()
                (outdir / f"{c}.json").write_text(json.dumps(out, indent=2))
        print(f"[{variant}] done -> {outdir}")


if __name__ == "__main__":
    main()
