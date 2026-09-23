"""Matched language/rubric assignment independent of any authored text or labels."""

import hashlib
from collections import Counter, defaultdict


def assign_blueprints(blueprints, seed=42):
    """Assign400latent families to200reused or400unique reviewed blueprints.

    The shared/changed language half is balanced over Score cardinality and
    authors. This prevents the extra language from appearing only at K=4/5.
    The function only consumes provenance, never evidence, gold or outcomes.
    """
    if type(seed) is not int or not isinstance(blueprints, list) or len(blueprints) != 400:
        raise ValueError("expected400blueprints and an integer seed")
    if any(
        not isinstance(b, dict) or any(not isinstance(b.get(k), str) or not b[k] for k in ("id", "category", "author"))
        for b in blueprints
    ):
        raise ValueError("blueprint provenance must identify id/category/author")
    if len({b["id"] for b in blueprints}) != 400:
        raise ValueError("duplicate blueprint ID")
    strata = defaultdict(lambda: defaultdict(list))
    for blueprint in blueprints:
        strata[blueprint["category"]][blueprint["author"]].append(blueprint["id"])
    if len(strata) != 10:
        raise ValueError("expectedten categories")
    authors = sorted({b["author"] for b in blueprints})
    if len(authors) != 2:
        raise ValueError("expectedtwo balanced author sources")
    if any(set(group) != set(authors) or any(len(group[a]) != 20 for a in authors) for group in strata.values()):
        raise ValueError("expected20blueprints per category/author source")

    def order(value):
        return hashlib.sha256(f"factorial-language/{seed}/{value}".encode()).hexdigest()

    per_category = {}
    # Ten first/second shared positions; five changed examples at every K.
    shared_offsets = [0, 0, 0, 0, 1, 1, 1, 1] * 2 + [1, 1, 0, 0]
    for category_index, category in enumerate(sorted(strata)):
        pools = {a: sorted(strata[category][a], key=order) for a in authors}
        consumed = Counter()
        rows = []
        for pair in range(20):
            author = authors[(pair // 2 + category_index) % 2]
            local = consumed[author]
            consumed[author] += 1
            shared, additional = pools[author][local], pools[author][10 + local]
            for offset in range(2):
                index = 2 * pair + offset
                rows.append(
                    {
                        "family_id": f"factorial-v1/{category}/{index:03d}",
                        "category": category,
                        "category_index": index,
                        "low_blueprint": shared,
                        "high_blueprint": shared if offset == shared_offsets[pair] else additional,
                        "narrow_k": 2 + index % 2,
                        "broad_k": 2 + index % 4,
                    }
                )
        per_category[category] = rows
    return [per_category[c][index] for index in range(40) for c in sorted(per_category)]
