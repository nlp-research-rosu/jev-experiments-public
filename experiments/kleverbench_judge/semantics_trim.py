#!/usr/bin/env python3
"""Two principled trims of semantics.k, to separate "Jev can't read swapped
semantics" from "Jev drowns in 26KB of syntax declarations and LaTeX prose
it never needed."

Both trims are applied uniformly to all 45 cases, not picked per-case to
flatter Jev on the ones it got wrong -- that would just be a different way
of rigging the test.

  rules_only         Strip comments and the SYNTAX/SYMBOLIC modules (pure
                      grammar declarations, no behavior). Keep every `rule`
                      block from the semantic module -- including the
                      swapped operators. Tests whether bulk/noise alone
                      explains the errors.

  operator_relevant   From rules_only, additionally drop the one-line atomic
                      rules for +,-,*,/,%,<,<=,==,!,&&,|| whose surface
                      symbol does not appear in THIS program. Keeps every
                      structural rule (assignment, arrays, functions,
                      control flow) untouched. Approximates what a
                      tool-using judge would grep for.
"""
import os
import pathlib
import re

ATOMIC_OPS = ["<=", "==", "&&", "||", "+", "-", "*", "/", "%", "<", "!"]


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def extract_rules(text: str) -> list[str]:
    """Pull `rule ...` blocks (which may continue for a few lines) out of a
    comment-stripped K file, skipping the grammar-only SYNTAX/SYMBOLIC
    modules by construction: no `rule` keyword ever appears inside them."""
    lines = text.splitlines()
    blocks, cur = [], []
    for line in lines:
        s = line.strip()
        starts_new = s.startswith("rule ") or s.startswith("rule<") or s == "rule"
        if starts_new:
            if cur:
                blocks.append("\n".join(cur))
            cur = [line]
        elif cur:
            if not s or s.startswith(("syntax", "endmodule", "module", "configuration")):
                blocks.append("\n".join(cur))
                cur = []
            else:
                cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def rules_only(semantics_text: str) -> str:
    blocks = extract_rules(strip_comments(semantics_text))
    return "\n\n".join(blocks)


def operator_relevant(semantics_text: str, program_text: str) -> str:
    blocks = extract_rules(strip_comments(semantics_text))
    used = {op for op in ATOMIC_OPS if op in program_text}
    kept = []
    for b in blocks:
        head = b.split("=>", 1)[0]
        # An atomic-operator one-liner has the shape "rule I1 <op> I2 => ...":
        # its head has no "<k>" cell and no "requires" of its own, just the
        # bare pattern.
        is_atomic = "<k>" not in head and len(head.split()) <= 5
        if is_atomic:
            hit_ops = [op for op in ATOMIC_OPS if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(op)}(?![A-Za-z0-9_=])", head)]
            if hit_ops and not any(op in used for op in hit_ops):
                continue  # this operator never appears in the program
        kept.append(b)
    return "\n\n".join(kept)


if __name__ == "__main__":
    import sys
    root = pathlib.Path(os.environ.get("KLEVERBENCH_ROOT", "../KleverBench"))
    for lane in ("imp", "imp-obf", "imp-swap"):
        full = (root / "pl" / lane / "semantics" / "semantics.k").read_text()
        ro = rules_only(full)
        prog = (root / "problems" / lane / "abs-times" / "program.imp").read_text()
        rel = operator_relevant(full, prog)
        print(f"{lane:<10} full={len(full):>6}  rules_only={len(ro):>6} "
              f"({len(ro)/len(full):.0%})  operator_relevant={len(rel):>6} "
              f"({len(rel)/len(full):.0%})  [{len(extract_rules(strip_comments(full)))} rule blocks]")
    if "--show" in sys.argv:
        full = (root / "pl" / "imp-swap" / "semantics" / "semantics.k").read_text()
        prog = (root / "problems" / "imp-swap" / "abs-times" / "program.imp").read_text()
        print("\n--- operator_relevant(imp-swap, abs-times) ---")
        print(operator_relevant(full, prog))
