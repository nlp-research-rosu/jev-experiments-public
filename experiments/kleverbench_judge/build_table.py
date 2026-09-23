#!/usr/bin/env python3
"""Score every judge against the exact labels and emit the comparison table.

Judge configurations compared (row-label constants live in build_configs()):

  jev (3-question check)        GOOD iff every narrow noul clears 0.5
                                 (mirrors the agent judges' must-agree rule)
  jev (1-question check)        GOOD iff the single broad noul clears 0.5
  <judge> (3 runs, must agree)  GOOD iff all 3 samples say YES (the deployed
                                 config: an agentic, multi-turn run per sample)
  <judge> (1 run)                first sample only, still multi-turn

Error names are chosen from the consequence, not the confusion matrix:
  MISS  = a BAD spec called GOOD -- the judge waved through a claim that
          proves but is not the intended theorem. The dangerous error.
  ALARM = a GOOD spec called BAD -- a false accusation against correct work.
"""
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "kleverbench-judge-v1"
MANIFEST = json.loads((DATA / "manifest.json").read_text())
NARROW = ["domain_unrestricted", "results_constrained", "relation_matches_intent"]
LANES = ["imp", "imp-obf", "imp-swap"]


def load_jev():
    out = {}
    for m in MANIFEST:
        p = DATA / "jev_results" / f"{m['case']}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        a = d.get("answers") or {}
        if not a:
            continue
        out[m["case"]] = {
            "narrow": {k: a[k]["noul"] for k in NARROW if k in a},
            "broad": a.get("faithful_overall", {}).get("noul"),
            "seconds": d.get("_seconds"),
            "tokens": (d.get("usage") or {}).get("input_tokens", 0)
                      + (d.get("usage") or {}).get("output_tokens", 0),
        }
    return out


CODEX_TOKENS = __import__("re").compile(r"tokens used\s+([\d,]+)")


def load_agent(judge):
    out = {}
    for m in MANIFEST:
        samples = []
        for s in (1, 2, 3):
            p = DATA / "results" / f"{m['case']}__{judge}__s{s}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            if not d.get("tokens"):
                # codex prints its usage line on stderr, not stdout
                hit = CODEX_TOKENS.search(d.get("stderr") or "")
                if hit:
                    d["tokens"] = int(hit.group(1).replace(",", ""))
            samples.append(d)
        if samples:
            out[m["case"]] = samples
    return out


def load_validate():
    """case -> True if the Prover's own kompile/type check ACCEPTS this spec.

    Only cases the prover accepts are ones a judge is needed for: if kompile
    rejects a spec, the deterministic layer already caught it.
    """
    p = DATA / "validate.json"
    if not p.exists():
        return {}
    return {c: v["status"] == "success" for c, v in json.loads(p.read_text()).items()}


VALID = load_validate()


def score(preds, subset=None, prover_invisible_only=False):
    """preds: case -> 'GOOD'/'BAD'/None. Returns metric dict.

    prover_invisible_only restricts the BAD side to specs the Prover accepts,
    i.e. the defects the deterministic layer cannot see. GOOD cases are kept
    in either way -- a false alarm costs the same regardless.
    """
    rows = [m for m in MANIFEST
            if m["case"] in preds and preds[m["case"]] is not None
            and (subset is None or m["lane"] == subset)
            and not (prover_invisible_only and m["label"] == "BAD"
                     and not VALID.get(m["case"], True))]
    n = len(rows)
    if not n:
        return None
    correct = sum(1 for m in rows if preds[m["case"]] == m["label"])
    bad = [m for m in rows if m["label"] == "BAD"]
    good = [m for m in rows if m["label"] == "GOOD"]
    miss = sum(1 for m in bad if preds[m["case"]] == "GOOD")
    alarm = sum(1 for m in good if preds[m["case"]] == "BAD")
    return {"n": n, "acc": correct / n,
            "miss": miss / len(bad) if bad else None,
            "alarm": alarm / len(good) if good else None,
            "miss_n": miss, "bad_n": len(bad),
            "alarm_n": alarm, "good_n": len(good)}


def build_configs(jev, agents):
    """Return (configs, perf, order) -- shared by the terminal table and the
    HTML report so the two can never disagree."""
    configs = {}

    # Row labels are written for a table cell read in isolation -- on a
    # narrow screen only this column may be visible, so each label states
    # both WHAT varies (questions asked vs. runs of one question) and HOW
    # the verdict is decided (must all pass / must all agree), rather than
    # leaving either to be inferred or carried over from internal jargon
    # ("battery", "broad", "all_pass").
    JEV_3Q = "jev, 3 questions (all must pass)"
    JEV_1Q = "jev, 1 question"

    def agree3(j):
        return f"{j} (3 runs, must agree)"

    def run1(j):
        return f"{j} (1 run)"

    configs[JEV_3Q] = {
        c: ("GOOD" if all(v > 0.5 for v in d["narrow"].values()) else "BAD")
        for c, d in jev.items() if len(d["narrow"]) == len(NARROW)}
    configs[JEV_1Q] = {
        c: ("GOOD" if d["broad"] > 0.5 else "BAD")
        for c, d in jev.items() if d["broad"] is not None}

    for j, data in agents.items():
        full = {c: s for c, s in data.items()
                if len(s) == 3 and all(x["verdict"] for x in s)}
        configs[agree3(j)] = {
            c: ("GOOD" if all(x["verdict"] == "YES" for x in s) else "BAD")
            for c, s in full.items()}
        configs[run1(j)] = {
            c: ("GOOD" if s[0]["verdict"] == "YES" else "BAD")
            for c, s in data.items() if s and s[0]["verdict"]}

    # ---- cost / latency -------------------------------------------------
    perf = {}
    if jev:
        perf[JEV_3Q] = perf[JEV_1Q] = {
            "secs": statistics.mean(d["seconds"] for d in jev.values()),
            "tokens": statistics.mean(d["tokens"] for d in jev.values()),
            "calls": 1.0}
    for j, data in agents.items():
        flat = [x for s in data.values() for x in s]
        if not flat:
            continue
        toks = [x["tokens"] for x in flat if x["tokens"]]
        secs = statistics.mean(x["seconds"] for x in flat)
        tk = statistics.mean(toks) if toks else 0
        perf[agree3(j)] = {"secs": secs * 3, "tokens": tk * 3, "calls": 3.0}
        perf[run1(j)] = {"secs": secs, "tokens": tk, "calls": 1.0}

    order = [JEV_3Q, JEV_1Q, agree3("claude"), run1("claude"),
             agree3("codex"), run1("codex")]
    order = [o for o in order if configs.get(o)]
    return configs, perf, order


def main():
    jev = load_jev()
    agents = {j: load_agent(j) for j in ("claude", "codex")}
    configs, perf, order = build_configs(jev, agents)

    w = max(len(o) for o in order) + 1
    print(f"\n{'judge / config':<{w}} {'n':>4} {'acc':>7} {'MISS':>13} "
          f"{'ALARM':>13} | " + " ".join(f"{l:>9}" for l in LANES))
    print("-" * (w + 52 + 10 * len(LANES)))
    for o in order:
        s = score(configs[o])
        if not s:
            continue
        lanes = []
        for l in LANES:
            ls = score(configs[o], l)
            lanes.append(f"{ls['acc']:>8.0%} " if ls else f"{'-':>9}")
        print(f"{o:<{w}} {s['n']:>4} {s['acc']:>6.0%} "
              f"{s['miss']:>7.0%} ({s['miss_n']}/{s['bad_n']}) "
              f"{s['alarm']:>6.0%} ({s['alarm_n']}/{s['good_n']}) | "
              + " ".join(lanes))

    if VALID:
        accepted = sum(1 for m in MANIFEST
                       if m["label"] == "BAD" and VALID.get(m["case"]))
        total_bad = sum(1 for m in MANIFEST if m["label"] == "BAD")
        print(f"\nRestricted to the {accepted}/{total_bad} BAD specs the Prover "
              f"ACCEPTS (kompile+typecheck pass) -- the defects no\n"
              f"deterministic check can see, which is the only place a judge "
              f"earns its keep:")
        print(f"\n{'judge / config':<{w}} {'n':>4} {'acc':>7} {'MISS':>13} "
              f"{'ALARM':>13} | " + " ".join(f"{l:>9}" for l in LANES))
        print("-" * (w + 52 + 10 * len(LANES)))
        for o in order:
            s = score(configs[o], prover_invisible_only=True)
            if not s:
                continue
            lanes = []
            for l in LANES:
                ls = score(configs[o], l, prover_invisible_only=True)
                lanes.append(f"{ls['acc']:>8.0%} " if ls else f"{'-':>9}")
            print(f"{o:<{w}} {s['n']:>4} {s['acc']:>6.0%} "
                  f"{s['miss']:>7.0%} ({s['miss_n']}/{s['bad_n']}) "
                  f"{s['alarm']:>6.0%} ({s['alarm_n']}/{s['good_n']}) | "
                  + " ".join(lanes))

    print(f"\n{'judge / config':<{w}} {'calls':>6} {'sec/case':>9} "
          f"{'tokens/case':>12}")
    print("-" * (w + 30))
    for o in order:
        p = perf.get(o)
        if p:
            print(f"{o:<{w}} {p['calls']:>6.0f} {p['secs']:>9.1f} "
                  f"{p['tokens']:>12,.0f}")

    # ---- per-variant breakdown -----------------------------------------
    variants = ["ref", "equiv", "vacuous", "weakened", "narrowed"]
    print(f"\n{'judge / config':<{w}} " + " ".join(f"{v:>9}" for v in variants))
    print("-" * (w + 10 * len(variants)))
    for o in order:
        cells = []
        for v in variants:
            rows = [m for m in MANIFEST
                    if m["variant"] == v and configs[o].get(m["case"])]
            if not rows:
                cells.append(f"{'-':>9}")
                continue
            ok = sum(1 for m in rows if configs[o][m["case"]] == m["label"])
            cells.append(f"{ok/len(rows):>8.0%} ")
        print(f"{o:<{w}} " + " ".join(cells))

    json.dump({"configs": configs, "perf": perf},
              (DATA / "scored.json").open("w"), indent=2)
    print(f"\nscored -> {DATA / 'scored.json'}")


if __name__ == "__main__":
    main()
