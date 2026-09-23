"""Freeze must fail closed when a blind review is incomplete or disagrees."""

import copy


def cases():
    return [
        {"review_id": "x", "expected": {"n1": True, "n2": False, "n3": True, "c1": "unknown", "s1": 2}},
        {"review_id": "y", "expected": {"n1": False, "n2": True, "n3": False, "c1": "yes", "s1": 1}},
    ]


def reviews():
    return [{**copy.deepcopy(row), "flags": [], "reason": "Independent supplied-rule judgment."} for row in cases()]


def test_missing_extra_duplicate_or_incorrect_blind_answers_cannot_freeze():
    from experiments.contrast_factorial_freeze import compare_blind_review

    assert compare_blind_review(cases(), reviews())["approved"]
    for rows in [
        reviews()[:1],
        reviews() + reviews()[:1],
        reviews() + [{"review_id": "z", "expected": {}, "flags": []}],
    ]:
        assert not compare_blind_review(cases(), rows)["approved"]
    rows = reviews()
    rows[0]["expected"]["n2"] = True
    result = compare_blind_review(cases(), rows)
    assert not result["approved"]
    assert result["disagreements"] == [{"review_id": "x", "question_id": "n2", "author": False, "reviewer": True}]


def test_flags_and_missing_predicates_block_even_when_other_labels_match():
    from experiments.contrast_factorial_freeze import compare_blind_review

    rows = reviews()
    rows[0]["flags"] = ["overlapping rubric levels"]
    assert not compare_blind_review(cases(), rows)["approved"]
    rows = reviews()
    del rows[1]["expected"]["c1"]
    assert not compare_blind_review(cases(), rows)["approved"]


def test_bool_int_aliases_are_not_valid_agreement():
    from experiments.contrast_factorial_freeze import compare_blind_review

    rows = reviews()
    rows[0]["expected"]["n1"] = 1
    assert not compare_blind_review(cases(), rows)["approved"]
    rows = reviews()
    rows[1]["expected"]["s1"] = True
    assert not compare_blind_review(cases(), rows)["approved"]
