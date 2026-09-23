"""Recompute the headline comparison from the saved predictions.

Run from anywhere:  python evaluation/paired-language-v1/score_predictions.py
Uses only the standard library.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEMS = [
    ("Public-data base (v0.2)", "public-data-base-v0.2.json"),
    ("Templated language, 1,000 updates", "template-step-1000.json"),
    ("Natural language, 1,000 updates", "natural-step-1000.json"),
    ("Natural language, update 500 (selected)", "natural-step-500.json"),
    ("Jev 1.13.0", "jev-1.13.0.json"),
]
CONFIDENT = 0.90


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score(rows, families=None):
    """Return (correct, total, wrong answers whose selected probability is at least CONFIDENT)."""
    if families is not None:
        rows = [row for row in rows if row["family_id"] in families]
    correct = sum(1 for row in rows if row["prediction"]["correct"])
    confident_wrong = sum(
        1
        for row in rows
        if not row["prediction"]["correct"] and row["prediction"]["max_probability"] >= CONFIDENT
    )
    return correct, len(rows), confident_wrong


def main():
    audit = load(HERE / "test-contract-audit.json")
    untouched = set(audit["conservative_unaffected_family_ids"])
    header = f"{'System':42} {'All 800':>16} {'wrong>=0.90':>11}   {'35 untouched':>16} {'wrong>=0.90':>11}"
    print(header)
    print("-" * len(header))
    for name, filename in SYSTEMS:
        rows = load(HERE / "predictions" / filename)
        correct, total, wrong = score(rows)
        u_correct, u_total, u_wrong = score(rows, untouched)
        print(
            f"{name:42} {correct:>4}/{total} ({correct / total:7.2%}) {wrong:>11}   "
            f"{u_correct:>4}/{u_total} ({u_correct / u_total:7.2%}) {u_wrong:>11}"
        )


if __name__ == "__main__":
    main()
