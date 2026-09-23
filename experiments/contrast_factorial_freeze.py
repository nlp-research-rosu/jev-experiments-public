"""Fail-closed comparison of independently reviewed evaluation annotations."""

from collections import Counter

QUESTION_IDS = {"n1", "n2", "n3", "c1", "s1"}


def _valid_labels(expected):
    return (
        isinstance(expected, dict)
        and set(expected) == QUESTION_IDS
        and all(type(expected[q]) is bool for q in ("n1", "n2", "n3"))
        and isinstance(expected["c1"], str)
        and bool(expected["c1"])
        and type(expected["s1"]) is int
        and expected["s1"] >= 0
    )


def compare_blind_review(author_rows, review_rows):
    """Report all coverage, type, ambiguity and label failures before a freeze.

    This is a necessary annotation gate, not an automatic certification of
    natural-language validity or independent authorship. A separate reviewed
    diversity/semantics disposition must also approve an evaluation freeze.
    """
    if not author_rows or any(
        not isinstance(r.get("review_id"), str) or not _valid_labels(r.get("expected")) for r in author_rows
    ):
        raise ValueError("author mapping must contain nonempty canonical annotations")
    authors = {r["review_id"]: r for r in author_rows}
    if len(authors) != len(author_rows):
        raise ValueError("duplicate author review ID")
    counts = Counter(row.get("review_id") for row in review_rows)
    expected_ids = set(authors)
    result = {
        "author_requests": len(authors),
        "reviewed_rows": len(review_rows),
        "missing_ids": sorted(expected_ids - set(counts)),
        "extra_ids": sorted(set(counts) - expected_ids, key=str),
        "duplicate_ids": sorted((key for key, count in counts.items() if count > 1), key=str),
        "invalid_annotations": [],
        "flagged": [],
        "disagreements": [],
    }
    for row in review_rows:
        identity = row.get("review_id")
        labels = row.get("expected")
        if not _valid_labels(labels) or not isinstance(row.get("flags"), list) or not row.get("reason"):
            result["invalid_annotations"].append(identity)
        if row.get("flags"):
            result["flagged"].append({"review_id": identity, "flags": row["flags"]})
        if identity not in authors or not isinstance(labels, dict):
            continue
        for question, gold in authors[identity]["expected"].items():
            answer = labels.get(question)
            if type(answer) is not type(gold) or answer != gold:
                result["disagreements"].append(
                    {"review_id": identity, "question_id": question, "author": gold, "reviewer": answer}
                )
    failure_keys = ("missing_ids", "extra_ids", "duplicate_ids", "invalid_annotations", "flagged", "disagreements")
    result["approved"] = not any(result[key] for key in failure_keys)
    return result
