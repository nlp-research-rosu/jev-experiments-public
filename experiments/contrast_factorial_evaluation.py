"""Disjoint calibration-population helpers for the four-arm study."""

import hashlib
import json
from collections import Counter


def _context_key(record):
    return hashlib.sha256(
        json.dumps(record["examples"][0]["state"], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def select_legacy_calibration(records, retention, *, per_source=25):
    """Choose canonical VALIDATION contexts without looking at labels/outcomes."""
    if type(per_source) is not int or per_source < 1:
        raise ValueError("positive per-source calibration count required")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate legacy record IDs")
    for r in records:
        p = r.get("provenance", {})
        if p.get("assigned_split") != "validation" or not p.get("dataset") or not r.get("group_id"):
            raise ValueError("only identified validation source/groups can be used")
        if not r.get("examples") or "state" not in r["examples"][0]:
            raise ValueError("legacy record has no canonical state")
    blocked_ids = {r["id"] for r in retention}
    blocked_groups = {r["group_id"] for r in retention}
    blocked_contexts = {_context_key(r) for r in retention}
    selected, counts = [], Counter()
    sources = {r["provenance"]["dataset"] for r in records}
    def order(record):
        return hashlib.sha256(f"factorial-calibration-v1/{record['id']}".encode()).hexdigest()
    for r in sorted(records, key=order):
        source, context = r["provenance"]["dataset"], _context_key(r)
        if (
            counts[source] >= per_source
            or r["id"] in blocked_ids
            or r["group_id"] in blocked_groups
            or context in blocked_contexts
        ):
            continue
        selected.append(r)
        counts[source] += 1
        blocked_ids.add(r["id"])
        blocked_groups.add(r["group_id"])
        blocked_contexts.add(context)
    if not sources or any(counts[source] != per_source for source in sources):
        raise ValueError("insufficient disjoint legacy calibration contexts")
    return selected
