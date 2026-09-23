#!/usr/bin/env python3
"""Build the labeled judge-evaluation suite from KleverBench gold specs.

Each case is a self-contained workspace holding exactly what a judge may see:

    program.imp      the program
    semantics.k      the lane's semantics -- says what the operators MEAN
    verification.k   the proof vocabulary the claim imports
    spec.k           the CANDIDATE claim under judgement
    TASK.md          the intended property, in prose

No reference-spec.k, no problem.toml, no label. The label lives only in
manifest.json, outside every case directory.

Variants per problem (2 GOOD / 3 BAD). All three BAD operators are chosen to
PRESERVE provability -- each still reaches #Top -- which is the whole point:
a prover cannot separate them from the reference, only a judge can.

    ref       GOOD  reference verbatim
    equiv     GOOD  alpha-renamed metavariables + reflow; same theorem
    vacuous   BAD   domain collapsed to a single point
    weakened  BAD   every output cell asserts nothing
    narrowed  BAD   domain bounded to a finite window
"""
import json
import os
import pathlib
import re
import shutil
import sys
import tomllib

ROOT = pathlib.Path(os.environ.get("KLEVERBENCH_ROOT", "../KleverBench"))
OUT = pathlib.Path(__file__).resolve().parents[2] / "data" / "kleverbench-judge-v1" / "suite"

LANES = ["imp", "imp-obf", "imp-swap"]
PROBLEMS = ["abs-times", "array-sum", "gcd-euclid"]

# Alpha-renaming map for the `equiv` GOOD variant. Chosen to avoid colliding
# with K builtins or with any metavariable name already in these specs.
RENAME = {"A": "P", "B": "Q", "I": "J", "S": "T", "N": "M",
          "R": "W", "X": "U", "Y": "V", "L": "G", "St": "Sv"}


def intent_of(problem_dir: pathlib.Path) -> str:
    """The property statement only -- drop provenance/translation appendices.

    gcd-euclid's intent carries a long dataset-provenance and translation-notes
    section that would both bloat the state and hand the judge reasoning it
    should be doing itself.
    """
    d = tomllib.load((problem_dir / "problem.toml").open("rb"))
    text = (d.get("intent") or "").strip()
    for marker in ("Provenance:", "Translation notes", "Provenance "):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def claims(spec: str):
    """Split a spec module into (head, [claim blocks], tail).

    A claim runs from `claim` to just before the next `claim` or `endmodule`.
    """
    starts = [m.start() for m in re.finditer(r"^\s*claim\b", spec, re.M)]
    if not starts:
        raise ValueError("no claim found")
    end = spec.rindex("endmodule")
    bounds = starts + [end]
    blocks = [spec[bounds[i]:bounds[i + 1]] for i in range(len(starts))]
    return spec[:starts[0]], blocks, spec[end:]


def int_metavars(block: str):
    seen = []
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9]*):Int\b", block):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def int_metavar(block: str):
    vs = int_metavars(block)
    return vs[0] if vs else None


def list_metavar(block: str):
    m = re.search(r"\b([A-Z][A-Za-z0-9]*):List\b", block)
    return m.group(1) if m else None


def add_requires(block: str, condition: str) -> str:
    """AND a condition onto the claim's precondition, or introduce one."""
    m = re.search(r"^(\s*)requires\s", block, re.M)
    if m:
        # Append to the existing requires clause (it runs to end of block).
        return block.rstrip() + f"\n     andBool {condition}\n"
    return block.rstrip() + f"\n    requires {condition}\n"


def op_vacuous(block: str) -> str:
    """Collapse the domain to ONE point -- a point that is still inside any
    existing precondition, so the claim stays satisfiable.

    Pinning to 1 rather than 0 matters: several reference claims already
    require their inputs positive, and AND-ing `== 0` onto those would make
    the precondition unsatisfiable. That proves too, but it is empty-domain
    vacuity -- a different and far more obvious defect than the narrow-but-real
    domain we want to test a judge on.
    """
    vs = int_metavars(block)
    if vs:
        return add_requires(block, " andBool ".join(f"{v} ==Int 1" for v in vs))
    v = list_metavar(block)
    if v:
        return add_requires(block, f"size({v}) ==Int 1")
    raise ValueError("no metavariable to collapse")


def op_narrowed(block: str) -> str:
    """Bound the domain to a finite window. Still proves; not the stated domain."""
    v = int_metavar(block)
    if v:
        return add_requires(block, f"{v} >Int 0 andBool {v} <Int 100")
    v = list_metavar(block)
    if v:
        return add_requires(block, f"size({v}) <Int 100")
    raise ValueError("no metavariable to bound")


def op_weakened(block: str) -> str:
    """Every output cell asserts nothing about its result. Still proves.

    Hand-scanned rather than regexed: postconditions nest parentheses to
    arbitrary depth (`sumRange(A, 0, size(A), 0)`), and a fixed-depth pattern
    silently leaves the deeper ones intact -- which would mislabel the case.
    """
    out, i, n = [], 0, 0
    while i < len(block):
        m = re.compile(r"\|->\s*\(").search(block, i)
        if not m:
            out.append(block[i:])
            break
        depth, j, arrow = 1, m.end(), None
        while j < len(block) and depth:
            if block[j] == "(":
                depth += 1
            elif block[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1 and block.startswith("=>", j):
                arrow = j
            j += 1
        if arrow is None:          # an input binding, not an output cell
            out.append(block[i:j + 1])
        else:
            out.append(block[i:arrow] + "=> _")
            n += 1
        i = j
    if n == 0:
        raise ValueError("no output cell to weaken")
    return "".join(out)


def op_equiv(spec: str) -> str:
    """Alpha-rename metavariables and reflow. Identical theorem."""
    out = spec
    for old, new in RENAME.items():
        out = re.sub(rf"(?<![A-Za-z0-9_]){old}(?![A-Za-z0-9_])", new, out)
    return ("// Claims module (metavariables renamed; formatting reflowed).\n"
            + out.replace("  claim", "\n  claim"))


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = []
    problems = 0

    for lane in LANES:
        semantics = (ROOT / "pl" / lane / "semantics" / "semantics.k").read_text()
        for pname in PROBLEMS:
            pdir = ROOT / "problems" / lane / pname
            ref = (pdir / "reference-spec.k").read_text()
            head, blocks, tail = claims(ref)
            problems += 1

            def rebuilt(fn):
                """Apply fn to the FIRST claim; leave any others untouched."""
                return head + fn(blocks[0]) + "".join(blocks[1:]) + tail

            variants = {
                "ref": (ref, "GOOD"),
                "equiv": (op_equiv(ref), "GOOD"),
                "vacuous": (rebuilt(op_vacuous), "BAD"),
                "weakened": (rebuilt(op_weakened), "BAD"),
                "narrowed": (rebuilt(op_narrowed), "BAD"),
            }

            for vname, (spec, label) in variants.items():
                case = f"{lane}__{pname}__{vname}"
                d = OUT / case
                d.mkdir(parents=True)
                (d / "program.imp").write_text((pdir / "program.imp").read_text())
                (d / "semantics.k").write_text(semantics)
                (d / "verification.k").write_text(
                    (pdir / "verification.k").read_text())
                (d / "spec.k").write_text(spec)
                (d / "TASK.md").write_text(
                    "# Intended property\n\n" + intent_of(pdir) + "\n")
                manifest.append({"case": case, "lane": lane, "problem": pname,
                                 "variant": vname, "label": label})

    (OUT.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))

    good = sum(1 for m in manifest if m["label"] == "GOOD")
    print(f"{len(manifest)} cases from {problems} (lane, problem) pairs "
          f"-> {OUT}")
    print(f"  {good} GOOD / {len(manifest) - good} BAD")
    print(f"  manifest -> {OUT.parent / 'manifest.json'} (labels live ONLY here)")
    return manifest


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
