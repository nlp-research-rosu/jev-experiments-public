#!/usr/bin/env python3
"""Run Jev over the suite -- one request per case.

Jev gets exactly the state the agent judges can read (same five files, no
reference spec, no label), but asked the way TypeSafe's docs say to ask:
independent narrow judgements batched into one request, composed in code.

`compose_all` mirrors the agent judges' `all_pass` aggregation: GOOD iff every
narrow question clears its threshold. `faithful_overall` rides along in the
same request and is scored separately, as the apples-to-apples row against the
agent judges' single broad criterion.

Usage:  TYPESAFE_API_KEY=... python3 run_jev.py [--concurrency 4]
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "kleverbench-judge-v1"
SUITE = DATA / "suite"
OUT = DATA / "jev_results"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = os.environ.get("JEV_MODEL", "jev-latest")

# The four narrow questions are the decomposition of the single criterion the
# agent judges are given. Each names one way a claim can fail to be the
# intended theorem. NARROW is the set composed into a verdict; `faithful_overall`
# is scored on its own.
NARROW = ["domain_unrestricted", "results_constrained", "relation_matches_intent"]

QUESTIONS = {
    "domain_unrestricted": {
        "type": "noul",
        "instructions": (
            "`task_intent` describes the domain of inputs the property is meant to "
            "hold over. Look at the precondition of the claim in `candidate_spec` — "
            "its `requires` clause, and any input value it pins to a literal. Does "
            "the claim hold over the WHOLE domain `task_intent` describes?"
        ),
        "criteria": {
            "true": "The claim's precondition admits the full range of inputs the "
                    "intended property is stated over.",
            "false": "The precondition restricts the claim to a strict subset — a "
                     "bounded window, a single point, a sign restriction, or an "
                     "unsatisfiable condition — so most of the intended domain is "
                     "never covered.",
        },
    },
    "results_constrained": {
        "type": "noul",
        "instructions": (
            "In `candidate_spec`, the claim's `<state>` cells written as "
            "`var |-> (before => after)` say what each variable becomes. Does the "
            "claim actually pin down the values of the program's result variables, "
            "rather than leaving them unconstrained (for example rewriting to an "
            "anonymous `_`, which asserts nothing)?"
        ),
        "criteria": {
            "true": "Every result variable the intended property talks about is "
                    "given a definite value or expression.",
            "false": "At least one result variable the intended property talks about "
                     "is left unconstrained, so the claim asserts nothing about it.",
        },
    },
    "relation_matches_intent": {
        "type": "noul",
        "instructions": (
            "Compare what the claim in `candidate_spec` asserts the result variables "
            "become against the relation `task_intent` describes. `semantics` is the "
            "formal definition of the language `program` is written in and defines "
            "what each operator in the program actually means — an operator does not "
            "necessarily mean what its symbol conventionally means, so read the "
            "program through `semantics`. Does the claim assert the same relation "
            "`task_intent` describes?"
        ),
        "criteria": {
            "true": "The asserted final values are the relation the intended "
                    "property describes, read under the given semantics.",
            "false": "The asserted final values differ from the intended relation, "
                     "or describe a weaker one.",
        },
    },
    "faithful_overall": {
        "type": "noul",
        "instructions": (
            "Does the claim in `candidate_spec` state exactly the property described "
            "in `task_intent`, over the full domain `task_intent` describes, rather "
            "than a narrowed, bounded, single-point, vacuous or otherwise weaker "
            "property? Read `program` through `semantics`, which defines what the "
            "program's operators actually mean."
        ),
        "criteria": {
            "true": "The claim faithfully captures the intended property over its "
                    "full intended domain.",
            "false": "The claim is restricted to a smaller domain than intended, or "
                     "fails to constrain the results as intended.",
        },
    },
}


def state_for(case: str) -> dict:
    d = SUITE / case
    return {
        "task_intent": (d / "TASK.md").read_text(),
        "program": (d / "program.imp").read_text(),
        "semantics": (d / "semantics.k").read_text(),
        "verification": (d / "verification.k").read_text(),
        "candidate_spec": (d / "spec.k").read_text(),
    }


def ask(case: str, key: str) -> dict:
    body = json.dumps({"state": state_for(case), "model": MODEL,
                       "questions": QUESTIONS}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, method="POST",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        out = {"error": f"HTTP {e.code}", "detail": e.read()[:600].decode("replace")}
    out["_seconds"] = round(time.time() - t0, 2)
    out["_request_bytes"] = len(body)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()

    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        print("TYPESAFE_API_KEY not set", file=sys.stderr)
        return 2

    OUT.mkdir(exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text())
    todo = [m["case"] for m in manifest if not (OUT / f"{m['case']}.json").exists()]
    print(f"{len(manifest)} cases, {len(todo)} to run", flush=True)

    errors = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(ask, c, key): c for c in todo}
        for f in as_completed(futs):
            case = futs[f]
            out = f.result()
            (OUT / f"{case}.json").write_text(json.dumps(out, indent=2))
            if "error" in out:
                errors += 1
                print(f"  ERROR {case}: {out['error']} {out.get('detail','')[:200]}",
                      flush=True)
    print(f"done in {time.time() - t0:.1f}s, {errors} errors -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
