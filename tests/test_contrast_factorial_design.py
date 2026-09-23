"""Counterexamples to confounded language assignment in the factorial pilot."""

from collections import Counter

import pytest


def bank():
    return [
        dict(id=f"{category}/{author}/{i:02}", category=category, author=author)
        for category in [f"category-{i}" for i in range(10)]
        for author in ("sol", "terra")
        for i in range(20)
    ]


def test_same_latents_but_exact200_twice_versus400_once():
    from experiments.contrast_factorial_design import assign_blueprints

    rows = assign_blueprints(bank(), seed=42)
    assert len(rows) == len({r["family_id"] for r in rows}) == 400
    assert len(Counter(r["low_blueprint"] for r in rows)) == 200
    assert set(Counter(r["low_blueprint"] for r in rows).values()) == {2}
    assert len(Counter(r["high_blueprint"] for r in rows)) == 400
    assert sum(r["low_blueprint"] == r["high_blueprint"] for r in rows) == 200
    assert Counter(r["narrow_k"] for r in rows) == {2: 200, 3: 200}
    assert Counter(r["broad_k"] for r in rows) == {2: 100, 3: 100, 4: 100, 5: 100}
    assert rows == assign_blueprints(list(reversed(bank())), seed=42)


def test_changed_language_is_not_exclusively_assigned_to_large_rubrics_or_one_author():
    from experiments.contrast_factorial_design import assign_blueprints

    rows = assign_blueprints(bank(), seed=42)
    lookup = {x["id"]: x for x in bank()}
    changed = [r for r in rows if r["low_blueprint"] != r["high_blueprint"]]
    assert Counter(r["broad_k"] for r in changed) == {2: 50, 3: 50, 4: 50, 5: 50}
    for k in (2, 3, 4, 5):
        selected = [r for r in changed if r["broad_k"] == k]
        assert Counter(lookup[r["high_blueprint"]]["author"] for r in selected) == {"sol": 25, "terra": 25}
    for category in {r["category"] for r in rows}:
        selected = [r for r in rows if r["category"] == category]
        for k in (2, 3, 4, 5):
            assert Counter(lookup[r["high_blueprint"]]["author"] for r in selected if r["broad_k"] == k) == {
                "sol": 5,
                "terra": 5,
            }


def test_duplicate_and_incomplete_banks_fail_before_assignment():
    from experiments.contrast_factorial_design import assign_blueprints

    with pytest.raises(ValueError):
        assign_blueprints(bank()[:-1])
    bad = bank()
    bad[-1] = bad[0]
    with pytest.raises(ValueError):
        assign_blueprints(bad)
