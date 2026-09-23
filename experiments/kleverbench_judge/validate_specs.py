#!/usr/bin/env python3
"""Check every case's spec.k compiles against the real Prover.

The suite's premise is that the BAD variants are things a prover ACCEPTS --
that only a judge can separate them from the reference. `kprover validate`
runs the server-side kompile and type check without paying for a proof, which
is where a malformed mutation would show up. Results land in validate.json.
"""
import json
import pathlib
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "kleverbench-judge-v1"
SUITE = DATA / "suite"
OUT = DATA / "validate.json"


def spec_module(spec_text: str) -> str:
    return re.search(r"^module\s+(\S+)", spec_text, re.M).group(1)


def validate(case: str) -> dict:
    """Validate in a scratch project holding ONLY spec.k + verification.k.

    The case directories deliberately carry a plain `semantics.k` for the
    judges to read, but kprover checks a project's semantics *identity*
    against the server registry, and a hand-copied file has no commit
    metadata -- so validating inside a case dir fails with "semantics identity
    is missing or mismatched". The scratch project sidesteps that and leaves
    the judged workspaces untouched.
    """
    src = SUITE / case
    lane = case.split("__")[0]
    d = DATA / "validate_ws" / case
    d.mkdir(parents=True, exist_ok=True)
    for f in ("spec.k", "verification.k"):
        shutil.copy(src / f, d / f)
    try:
        s = subprocess.run(
            ["kprover", "session", "start", "--project", ".",
             "--semantics", lane],
            cwd=d, capture_output=True, text=True, timeout=120)
        sess = json.loads(s.stdout)
        sid, ws = sess["sessionId"], pathlib.Path(sess["workspaceDir"])
        for f in ("spec.k", "verification.k"):
            shutil.copy(d / f, ws / f)
        v = subprocess.run(
            ["kprover", "validate", "--session", sid,
             "--spec", "spec.k",
             "--spec-module", spec_module((src / "spec.k").read_text()),
             "--verification", "verification.k",
             "--verification-module", "VERIFICATION"],
            cwd=d, capture_output=True, text=True, timeout=1800)
        out = json.loads(v.stdout) if v.stdout.strip() else {"status": "no-output"}
        # The server's own verdict is authoritative. kprover additionally
        # post-checks that the PROJECT carries a matching seeded semantics,
        # and reports failure when it does not -- which is always true for
        # this scratch project. Read the task record the server wrote.
        tasks = sorted((d / ".kprover" / "sessions" / sid).glob(
            "validation-*/task.json"), key=lambda p: p.stat().st_mtime)
        if tasks:
            t = json.loads(tasks[-1].read_text())
            valid = (t.get("result") or {}).get("valid")
            if valid is not None:
                return {"case": case,
                        "status": "success" if valid else "invalid",
                        "message": str(t.get("error") or "")[:300],
                        "ms": (t.get("timing") or {}).get("totalMilliseconds")}
        return {"case": case, "status": out.get("status"),
                "message": (out.get("message") or "")[:300]}
    except Exception as e:                                  # noqa: BLE001
        return {"case": case, "status": "harness-error", "message": str(e)[:300]}


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    cases = [m["case"] for m in manifest]
    results = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(validate, c): c for c in cases}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results[r["case"]] = r
            print(f"  {i}/{len(cases)} {r['case']}: {r['status']}", flush=True)
    OUT.write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results.values() if r["status"] == "success")
    print(f"\n{ok}/{len(results)} validated successfully -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
