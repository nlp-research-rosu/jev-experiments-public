#!/usr/bin/env python3
"""Score the fairness-audit ablations: no-tools single-turn Claude/Codex, and
Jev under two semantics trims. Imports build_table.py for MANIFEST/score/VALID
so every script agrees on labels and the prover-invisible subset."""
import importlib.util
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "kleverbench-judge-v1"
spec = importlib.util.spec_from_file_location("bt", HERE / "build_table.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

NARROW = bt.NARROW


def load_notools(judge):
    out = {}
    for m in bt.MANIFEST:
        p = DATA / "notools_results" / f"{m['case']}__{judge}.json"
        if p.exists():
            out[m["case"]] = json.loads(p.read_text())
    return out


def load_jev_variant(variant):
    """variant is None for the original full-semantics run."""
    dirname = "jev_results" if variant is None else f"jev_results_{variant}"
    out = {}
    for m in bt.MANIFEST:
        p = DATA / dirname / f"{m['case']}.json"
        if p.exists():
            d = json.loads(p.read_text())
            a = d.get("answers")
            if a:
                out[m["case"]] = {
                    "narrow": {k: a[k]["noul"] for k in NARROW if k in a},
                    "broad": a.get("faithful_overall", {}).get("noul"),
                }
    return out


def preds_notools(judge):
    data = load_notools(judge)
    return {c: ("GOOD" if d["verdict"] == "YES" else "BAD")
            for c, d in data.items() if d.get("verdict")}


def preds_battery(variant):
    data = load_jev_variant(variant)
    return {c: ("GOOD" if all(v > 0.5 for v in d["narrow"].values()) else "BAD")
            for c, d in data.items() if len(d["narrow"]) == len(NARROW)}


def preds_broad(variant):
    data = load_jev_variant(variant)
    return {c: ("GOOD" if d["broad"] > 0.5 else "BAD")
            for c, d in data.items() if d["broad"] is not None}


def print_score_row(label, preds, w):
    s = bt.score(preds)
    if not s:
        print(f"  {label:<{w}} (no data)")
        return
    lanes = "".join(f"{(bt.score(preds, l) or {}).get('acc', float('nan')):>8.0%} "
                    for l in bt.LANES)
    print(f"  {label:<{w}} n={s['n']:<3} acc={s['acc']:.0%}  "
          f"miss={s['miss_n']}/{s['bad_n']}  alarm={s['alarm_n']}/{s['good_n']}  "
          f"| {lanes}")


def main():
    overhead_path = DATA / "notools_overhead.json"
    overhead = json.loads(overhead_path.read_text()) if overhead_path.exists() else {}

    w = 30
    print("=== No-tools single-turn (inline state, faithful_overall criterion) ===")
    print_score_row("claude / no-tools", preds_notools("claude"), w)
    print_score_row("codex / no-tools", preds_notools("codex"), w)
    print_score_row("jev / broad (for reference)", preds_broad(None), w)

    print("\n=== Jev / battery under semantics trims (all 45 cases, uniform) ===")
    print_score_row("full semantics (26 KB)", preds_battery(None), w)
    print_score_row("rules only (~5 KB)", preds_battery("rulesonly"), w)
    print_score_row("operator-relevant only", preds_battery("relevant"), w)

    print("\n=== Per-problem margin on relation_matches_intent, full semantics ===")
    jev_full = load_jev_variant(None)
    for prob in ("abs-times", "array-sum", "gcd-euclid"):
        for lane in bt.LANES:
            for v in ("ref", "equiv"):
                c = f"{lane}__{prob}__{v}"
                d = jev_full.get(c)
                if d:
                    print(f"  {c:<30} relation={d['narrow'].get('relation_matches_intent', float('nan')):.2f}")
        print()

    print("=== Fixed CLI tax (trivial PONG prompt, no case content) ===")
    for j, tok in overhead.items():
        print(f"  {j:<8} {tok:,} tokens before any case is even read")

    json.dump({
        "notools_claude": preds_notools("claude"),
        "notools_codex": preds_notools("codex"),
        "jev_rulesonly_battery": preds_battery("rulesonly"),
        "jev_relevant_battery": preds_battery("relevant"),
        "overhead": overhead,
    }, (DATA / "scored2.json").open("w"), indent=2)
    print(f"\nscored -> {DATA / 'scored2.json'}")


if __name__ == "__main__":
    main()
