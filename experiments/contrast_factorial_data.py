"""Matched factorial data from raw observations and independently authored prose.

This module never reads evaluation examples. All public rendering is a projection
of private raw records; the original raw-fact evaluator remains unmodified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import re
import string
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from experiments.contrast_factorial_design import assign_blueprints
from experiments.contrast_scaling_logic import derive_features, eval_rule, select_choice, select_level
from experiments.revised_metrics import validate_suite
from openjev.judgments import compile_request

CATEGORIES = (
    "claim_vs_completion", "permission_vs_execution", "unknown_vs_failure", "attribution_and_endorsement",
    "entity_binding", "action_binding", "temporal_scope", "reversal_and_current_state",
    "negation_and_quantifiers", "ordered_rubrics",
)
QUESTION_IDS = ("n1", "n2", "n3", "c1", "s1")
STATEMENT_SLOTS = {"actor", "operation", "target", "source", "time"}
EXECUTION_SLOTS = STATEMENT_SLOTS | {"run_id", "verified", "mode", "outcome"}
STATEMENT_TEMPLATES = ("claim_template", "intent_template", "denial_template", "endorsement_template")
CARD_FIELDS = {"schema_version", "id", "author", "category", "heading", "construction_note",
               "execution_template", "observation_templates", *STATEMENT_TEMPLATES}
OBSERVATION_SLOTS = {
    "approval": STATEMENT_SLOTS | {"verified", "decision"},
    "corroboration": EXECUTION_SLOTS,
    "reversal": EXECUTION_SLOTS | {"reverses"},
    "state_event": {"target", "source", "time", "verified", "active"},
    "item": {"id", "verified", "status"},
    "measurement": {"value"},
}
OBSERVATION_TYPES = {
    "entity_binding": (), "action_binding": (), "temporal_scope": ("state_event",),
    "reversal_and_current_state": ("approval", "corroboration", "reversal"),
    "negation_and_quantifiers": ("item",), "ordered_rubrics": ("measurement",),
    **{c: ("approval", "corroboration") for c in CATEGORIES[:4]},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def validate_blueprint(card):
    if not isinstance(card, dict) or set(card) != CARD_FIELDS or card["schema_version"] != 1:
        raise ValueError("blueprint schema_version/fields invalid")
    if card["category"] not in CATEGORIES or any(not isinstance(card[k], str) or not card[k].strip()
                                                for k in CARD_FIELDS - {"schema_version", "observation_templates"}):
        raise ValueError("blueprint metadata must be nonempty strings in a known category")
    if len(card["heading"]) > 80 or "{" in card["heading"] or "}" in card["heading"]:
        raise ValueError("neutral heading must be short literal text")
    forbidden = re.compile(r"\b(gold|correct answer|expected label|score level|ignore instructions|ground truth)\b", re.I)
    observations = card["observation_templates"]
    if not isinstance(observations, dict) or set(observations) != set(OBSERVATION_TYPES[card["category"]]):
        raise ValueError("required primary observation templates must match category")
    templates = [(card[key], STATEMENT_SLOTS) for key in STATEMENT_TEMPLATES]
    templates += [(card["execution_template"], EXECUTION_SLOTS)]
    templates += [(template, OBSERVATION_SLOTS[key]) for key, template in observations.items()]
    for template, expected in templates:
        if not isinstance(template, str) or not template.strip() or len(template) > 360 or forbidden.search(template) or "{{" in template or "}}" in template:
            raise ValueError("template is too long or contains interpretation hints/braces")
        try:
            parts = list(string.Formatter().parse(template))
        except ValueError as exc:
            raise ValueError("invalid template") from exc
        slots = []
        for _, name, spec, conversion in parts:
            if name is not None:
                if name not in expected or spec or conversion:
                    raise ValueError("only plain named semantic slots are permitted")
                slots.append(name)
        if Counter(slots) != Counter(expected):
            raise ValueError("every raw-fact placeholder must occur exactly once")
    if forbidden.search(card["heading"]):
        raise ValueError("heading contains interpretation hints")
    return card


def F(name):
    return {"feature": name}


def _primary_fields(category):
    return {
        "claim_vs_completion": ("statements", "executions"),
        "attribution_and_endorsement": ("statements", "executions"),
        "permission_vs_execution": ("approvals", "executions"),
        "unknown_vs_failure": ("executions",),
        "entity_binding": ("executions",), "action_binding": ("executions",),
        "temporal_scope": ("state_events",), "reversal_and_current_state": ("executions", "reversals"),
        "negation_and_quantifiers": ("items",), "ordered_rubrics": ("measurement",),
    }[category]


def language_quality_report(blueprints):
    """Report structural overlap, never equate surface count with mechanisms."""
    groups = defaultdict(list)
    singular = {"approvals": "approval", "reversals": "reversal", "state_events": "state_event",
                "items": "item", "measurement": "measurement"}
    for card in blueprints:
        validate_blueprint(card)
        templates = []
        for field in _primary_fields(card["category"]):
            if field == "statements":
                templates.extend(card[k] for k in STATEMENT_TEMPLATES)
            elif field == "executions":
                templates.append(card["execution_template"])
            else:
                templates.append(card["observation_templates"][singular[field]])
        normalized = tuple(" ".join(re.findall(r"\w+", text.lower())) for text in templates)
        words = " | ".join(normalized).split()
        shingles = {tuple(words[i:i + 3]) for i in range(max(1, len(words) - 2))}
        groups[card["category"]].append((card, normalized, shingles))
    result = {"blueprint_ids": len({b["id"] for b in blueprints}), "categories": {},
              "limitation": "Distinct authored constructions do not imply statistically independent reasoning mechanisms."}
    for category, rows in sorted(groups.items()):
        overlaps = []
        for a, b in itertools.combinations(rows, 2):
            union = a[2] | b[2]
            similarity = len(a[2] & b[2]) / len(union) if union else 1.0
            overlaps.append((similarity, a[0]["id"], b[0]["id"]))
        result["categories"][category] = {
            "cards": len(rows), "authors": dict(Counter(row[0]["author"] for row in rows)),
            "distinct_primary_constructions": len({row[1] for row in rows}),
            "max_trigram_jaccard": max((r[0] for r in overlaps), default=0),
            "mean_trigram_jaccard": sum(r[0] for r in overlaps) / max(1, len(overlaps)),
            "highest_overlap_pairs": [list(r) for r in sorted(overlaps, reverse=True)[:10]],
        }
    return result


def AND(*rules):
    return {"all": list(rules)}


def OR(*rules):
    return {"any": list(rules)}


def NOT(rule):
    return {"not": rule}


def CMP(name, operator, number):
    return {"compare": {"left": F(name), "operator": operator, "right": number}}


FEATURE_TEXT = {
    "claim": "a positive assertion of past completion for the queried actor/action/target is present",
    "intent": "an intention for future action by the queried actor/action/target is present",
    "denied_claim": "a denial of past completion for the queried actor/action/target is present",
    "endorsement": "a source explicitly adopts a positive completion assertion for the queried actor/action/target",
    "approved": "the latest authentic in-window permission records unanimously allow the queried actor/action/target",
    "forbidden": "the latest authentic in-window permission records unanimously deny the queried actor/action/target",
    "invocation": "an in-window execution record matches all four query fields, regardless of trust or mode",
    "completed": "matching authentic live execution reports contain success and no no_effect",
    "failed": "matching authentic live execution reports contain no_effect and no success",
    "unknown": "neither completion nor failure is established (including conflicting success/no_effect reports)",
    "success_report": "an in-window execution record matching all four query fields reports success, regardless of trust/mode",
    "failure_report": "an in-window execution record matching all four query fields reports no_effect, regardless of trust/mode",
    "any_live_success": "any authentic in-window live execution record reports success, for any query binding",
    "other_live_success": "an authentic in-window live success record differs on at least one of the four query fields",
    "actor_seen": "any execution record names the queried actor, regardless of time/trust/other fields",
    "operation_seen": "any execution record names the queried operation, regardless of time/trust/other fields",
    "reversed": "completion is established and an authentic in-window live successful reversal matches actor/action/target, names the successful run in reverses, and is no earlier than its success",
    "effective": "completion is established and no qualifying successful reversal exists",
    "corroborated": "completion has an authentic in-window success corroboration matching all four query fields, from a different source at or after a success",
    "earlier_known": "latest authentic target-state reports at or before query_time agree on a boolean active value",
    "current_known": "latest authentic target-state reports at or before window end agree on a boolean active value",
    "active_earlier": "the latest authentic target-state reports at or before query_time unanimously say active=true",
    "active_now": "the latest authentic target-state reports at or before window end unanimously say active=true",
    "inactive_now": "the latest authentic target-state reports at or before window end unanimously say active=false",
    "ever_active": "some authentic target-state record at or before window end reports active=true, even if contradicted",
    "all_success": "inventory_complete=true and all scoped items have success status",
    "any_success": "at least one scoped item has success status",
    "any_failure": "at least one scoped item has no_effect status",
    "has_measurement": "a numeric measurement is present",
}


def describe(rule):
    if rule is True:
        return "always applicable"
    if rule is False:
        return "never applicable"
    op, value = next(iter(rule.items()))
    if op == "feature":
        return FEATURE_TEXT[value]
    if op == "not":
        return "NOT (" + describe(value) + ")"
    if op in ("all", "any"):
        return (" AND " if op == "all" else " OR ").join("(" + describe(v) + ")" for v in value)
    if op == "compare":
        labels = {"measurement": "measurement", "success_count": "number of scoped success items",
                  "unknown_count": "number of scoped unknown items"}
        return f"{labels[value['left']['feature']]} {value['operator']} {value['right']}"
    if op == "at_least":
        return f"at least {value['count']} of the listed requirements hold"
    raise ValueError("unknown program operator")


def _base():
    return {"query": {"actor": "operator", "operation": "seal", "target": "parcel", "run_id": "run"},
            "window": [0, 10], "query_time": 3, "trusted_sources": ["ledger", "inspector"],
            "statements": [], "executions": [], "approvals": [], "corroborations": [], "reversals": [],
            "state_events": [], "item_ids": ["one", "two", "three", "four"], "items": [],
            "inventory_complete": True, "measurement": None}


def _row(record, **changes):
    return {**record["query"], "source": "ledger", "time": 5, "verified": True, "mode": "live",
            "outcome": "success", **changes}


def _statement(record, **changes):
    return {k: v for k, v in {**record["query"], "source": "speaker", "time": 4,
                              "tense": "completed", "polarity": "positive", "endorsed": False,
                              **changes}.items() if k != "run_id"}


def _approval(record, decision):
    raw = _row(record, decision=decision, time=2)
    return {key: raw[key] for key in OBSERVATION_SLOTS["approval"]}


def _execution_status(record, state):
    if state == 0:
        return []
    if state == 1:
        return [_row(record, outcome="timeout")]
    if state == 2:
        return [_row(record, outcome="no_effect")]
    if state == 3:
        return [_row(record)]
    if state == 4:
        return [_row(record), _row(record, outcome="no_effect", source="inspector")]
    if state == 5:
        return [_row(record, verified=False)]
    return [_row(record, mode="simulation")]


@lru_cache(maxsize=10)
def _worlds(category):
    """Enumerate legal raw worlds BEFORE selecting any labels or contrasts.

    Adjacent worlds change one independently controlled observation. A conflicting
    event/status is one observation group containing its competing raw reports.
    """
    records, keys = [], []
    if category in {"claim_vs_completion", "permission_vs_execution", "attribution_and_endorsement",
                    "unknown_vs_failure"}:
        dimensions = (3, 2, 3, 7, 2, 2)
    elif category in {"entity_binding", "action_binding"}:
        dimensions = (12, 2, 2)
    elif category == "temporal_scope":
        dimensions = (4, 4)
    elif category == "reversal_and_current_state":
        dimensions = (4, 5, 2, 2, 2)
    elif category == "negation_and_quantifiers":
        dimensions = (4, 4, 4, 4, 2)
    elif category == "ordered_rubrics":
        dimensions = (22,)
    else:
        raise ValueError("unknown category")
    for key in itertools.product(*(range(n) for n in dimensions)):
        record = _base()
        if len(dimensions) == 6:
            claim, intent, permission, status, corroboration, denial = key
            if claim:
                record["statements"].append(_statement(record, endorsed=claim == 2))
            if intent:
                record["statements"].append(_statement(record, tense="future"))
            if denial:
                record["statements"].append(_statement(record, polarity="negative"))
            if permission:
                record["approvals"] = [_approval(record, "allow" if permission == 1 else "deny")]
            record["executions"] = _execution_status(record, status)
            if corroboration:
                record["corroborations"] = [_row(record, source="inspector", time=7)]
        elif category in {"entity_binding", "action_binding"}:
            status, decoy, claim = key
            variants = [None, {"actor": "other operator"}, {"operation": "open"}, {"target": "crate"},
                        {"run_id": "other run"}, {"verified": False}, {"mode": "simulation"},
                        {"outcome": "timeout"}, {"outcome": "no_effect"}, {},
                        {"outcome": "no_effect", "verified": False}, {"source": "untrusted"}]
            if variants[status] is not None:
                record["executions"] = [_row(record, **variants[status])]
            if decoy:
                record["executions"].append(_row(record, actor="other operator", operation="open",
                                                  target="crate", run_id="other run", outcome="no_effect"))
            if claim:
                record["statements"] = [_statement(record)]
        elif category == "temporal_scope":
            for state, time in zip(key, (2, 8)):
                if state:
                    values = [False, True] if state == 3 else [state == 2]
                    record["state_events"].extend({"target": "parcel", "time": time, "source": "ledger",
                                                    "verified": True, "active": value} for value in values)
        elif category == "reversal_and_current_state":
            status, reversal, corroboration, permission, claim = key
            record["executions"] = _execution_status(record, status)
            if reversal:
                record["reversals"] = [_row(record, time=8, reverses="other run" if reversal == 4 else "run",
                                             target="crate" if reversal == 3 else "parcel",
                                             outcome="no_effect" if reversal == 2 else "success")]
            if corroboration:
                record["corroborations"] = [_row(record, source="inspector", time=7)]
            if permission:
                record["approvals"] = [_approval(record, "allow")]
            if claim:
                record["statements"] = [_statement(record)]
        elif category == "negation_and_quantifiers":
            for identity, status in zip(record["item_ids"], key[:4]):
                outcomes = [] if status == 0 else ["no_effect", "success"] if status == 3 else [
                    "success" if status == 2 else "no_effect"]
                record["items"].extend({"id": identity, "verified": True, "status": outcome} for outcome in outcomes)
            record["inventory_complete"] = bool(key[-1])
        else:
            record["measurement"] = None if key[0] == 0 else 5 * (key[0] - 1)
        records.append(record)
        keys.append(key)
    lookup = {key: i for i, key in enumerate(keys)}
    edges = []
    for i, key in enumerate(keys):
        for axis, n in enumerate(dimensions):
            # Any replacement of one observation is a meaningful controlled change.
            for value in range(key[axis] + 1, n):
                neighbor = (*key[:axis], value, *key[axis + 1:])
                edges.append((i, lookup[neighbor]))
    return records, [derive_features(r) for r in records], edges


def _noul_rules(category):
    features = {
        "claim_vs_completion": ("claim", "completed", "intent"),
        "permission_vs_execution": ("approved", "completed", "forbidden"),
        "unknown_vs_failure": ("failed", "unknown", "invocation"),
        "attribution_and_endorsement": ("endorsement", "claim", "completed"),
        "entity_binding": ("completed", "other_live_success", "actor_seen"),
        "action_binding": ("completed", "other_live_success", "operation_seen"),
        "temporal_scope": ("active_now", "active_earlier", "ever_active"),
        "reversal_and_current_state": ("effective", "completed", "reversed"),
        "negation_and_quantifiers": ("all_success", "any_success", "any_failure"),
    }
    if category == "ordered_rubrics":
        return [AND(F("has_measurement"), CMP("measurement", ">=", 40)),
                AND(F("has_measurement"), CMP("measurement", ">=", 70)), F("has_measurement")]
    return [F(x) for x in features[category]]


def _choice_rules(category):
    if category == "temporal_scope":
        return {"Active": F("active_now"), "Inactive": F("inactive_now"), "Unknown": NOT(F("current_known"))}
    if category == "negation_and_quantifiers":
        return {"All succeeded": F("all_success"), "Failure present": F("any_failure"),
                "Unknown": AND(NOT(F("all_success")), NOT(F("any_failure")))}
    if category == "ordered_rubrics":
        return {"Below 40": AND(F("has_measurement"), CMP("measurement", "<", 40)),
                "40 or above": AND(F("has_measurement"), CMP("measurement", ">=", 40)),
                "Unknown": NOT(F("has_measurement"))}
    return {"Completed": F("completed"), "Failed": F("failed"), "Unknown": F("unknown")}


def _thresholds(k, index):
    options = {2: ((1,), (2,), (3,), (4,)), 3: ((1, 3), (2, 4), (1, 4), (2, 3)),
               4: ((1, 2, 4), (1, 3, 4), (2, 3, 4)), 5: ((1, 2, 3, 4),)}
    return options[k][(index // 4) % len(options[k])]


def _program(category, k, index, variant):
    thresholds = _thresholds(k, index)
    if category == "unknown_vs_failure":
        thresholds = {2: (3,), 3: ((1, 3), (2, 3), (3, 4))[(index // 4) % 3],
                      4: (1, 3, 4), 5: (1, 2, 3, 4)}[k]
    elif category in {"entity_binding", "action_binding"} and k == 2:
        thresholds = (3 + (index // 4) % 2,)
    requirements = {
        "claim_vs_completion": (("claim", "endorsement", "completed", "corroborated"),
                                 ("intent", "approved", "completed", "corroborated")),
        "permission_vs_execution": (("approved", "invocation", "completed", "corroborated"),
                                     ("claim", "invocation", "completed", "corroborated")),
        "attribution_and_endorsement": (("claim", "endorsement", "completed", "corroborated"),
                                        ("intent", "endorsement", "completed", "corroborated")),
        "temporal_scope": (("earlier_known", "current_known", "active_earlier", "active_now"),
                            ("earlier_known", "current_known", "ever_active", "active_now")),
        "reversal_and_current_state": (("completed", "effective", "corroborated", "approved"),
                                        ("completed", "effective", "corroborated", "claim")),
    }
    if category in requirements:
        names = requirements[category][variant]
        predicates = [F(name) for name in names]
        rules = [True] + [{"at_least": {"count": threshold, "of": predicates}} for threshold in thresholds]
        instruction = "Score checklist fulfillment: count these four listed requirements (one point each): " + "; ".join(
            f"({i + 1}) {FEATURE_TEXT[name]}" for i, name in enumerate(names)) + "."
        ordinal = "Higher levels mean more of the explicitly listed requirements are met, not physical confidence."
    elif category in {"entity_binding", "action_binding", "unknown_vs_failure"}:
        first = F("operation_seen" if category == "action_binding" else "actor_seen")
        if category == "unknown_vs_failure":
            first = F("invocation")
            second = OR(F("success_report"), F("failure_report"))
            third = OR(F("completed"), F("failed")) if not variant else F("completed")
            fourth = F("corroborated")
        else:
            second = F("invocation")
            third = F("success_report" if not variant else "failure_report")
            fourth = F("completed" if not variant else "failed")
        tiers = [True, first, second, third, fourth]
        rules = [True] + [tiers[t] for t in thresholds]
        instruction = "Score the highest qualifying evidence tier for " + (
            "a resolved outcome" if category == "unknown_vs_failure" and not variant else
            "successful completion" if not variant or category == "unknown_vs_failure" else "a no-effect outcome") + "."
        ordinal = "Higher tiers require increasingly specific reported or authenticated outcome evidence."
    elif category == "negation_and_quantifiers":
        rules = [True] + [CMP("success_count", ">=", t) if not variant else CMP("unknown_count", "<=", 4 - t)
                          for t in thresholds]
        instruction = "Score inventory " + ("successful-item coverage." if not variant else
                                               "resolved-item coverage; both success and no_effect count as resolved.")
        ordinal = "Higher levels mean more scoped items meet the stated coverage condition; inventory_complete does not affect this count."
    else:
        offset = 2 * (index // 4 % 4) + 7 * variant
        rules = [True] + [AND(F("has_measurement"), CMP("measurement", ">=", 20 * t + offset)) for t in thresholds]
        instruction = "Score observed measurement against the supplied numeric thresholds; missing measurement stays at level 0."
        ordinal = "Higher levels mean a larger directly observed measurement."
    levels = [f"Level {i}: {describe(rule)}." for i, rule in enumerate(rules)]
    return {"rules": rules, "instruction": instruction, "ordinal": ordinal, "levels": levels,
            "selection": "Use the highest numbered applicable level; level 0 is the fallback."}


def _outputs(features, nouls, choices, programs):
    return [{"n": tuple(eval_rule(r, f) for r in nouls), "c": select_choice(choices, f),
             "s": {arm: tuple(select_level(p["rules"], f) for p in variants)
                   for arm, variants in programs.items()}} for f in features]


def _surface_context(category, index):
    """Outcome-independent balanced role permutations and a positive affine clock."""
    aliases = {}
    pairs = (("actor", ("operator", "other operator"), ("Mira", "Tobin")),
             ("operation", ("seal", "open"), ("latch", "release")),
             ("target", ("parcel", "crate"), ("case A", "case B")),
             ("run", ("run", "other run"), ("attempt A", "attempt B")))
    for kind, original, vocabulary in pairs:
        offset = int(digest(["aliases", category, kind, index // 2])[:8], 16) % 2
        offset = (offset + index % 2) % 2
        aliases.update({name: vocabulary[(i + offset) % 2] for i, name in enumerate(original)})
    sources = ("ledger", "inspector", "untrusted", "speaker")
    vocabulary = sorted(("source A", "source B", "source C", "source D"),
                        key=lambda name: digest(["sources", category, index // 4, name]))
    aliases.update({name: vocabulary[(i + index % 4) % 4] for i, name in enumerate(sources)})
    aliases.update({name: f"item {letter}" for name, letter in zip(("one", "two", "three", "four"), "ABCD")})
    clock = int(digest(["clock", category, index])[:12], 16)
    return {"aliases": aliases, "inverse_aliases": {value: key for key, value in aliases.items()},
            "time_scale": 1 + clock % 7, "time_offset": (clock // 7) % 101 - 40}


def _transform_record(record, context, *, inverse=False):
    aliases = context["inverse_aliases" if inverse else "aliases"]
    scale, offset = context["time_scale"], context["time_offset"]

    def clock(value):
        if type(value) not in (int, float):
            raise ValueError("raw time coordinates must be numeric, never boolean")
        value = (value - offset) / scale if inverse else scale * value + offset
        return int(value) if isinstance(value, float) and value.is_integer() else value

    def visit(value, key=None):
        if isinstance(value, str):
            return aliases.get(value, value)
        if key in {"time", "query_time"}:
            return clock(value)
        if key == "window":
            return [clock(v) for v in value]
        if isinstance(value, list):
            return [visit(v) for v in value]
        if isinstance(value, dict):
            return {k: visit(v, k) for k, v in value.items()}
        return value

    return visit(record)


def transform_record(record, context):
    return _transform_record(record, context)


def normalize_record(record, context):
    return _transform_record(record, context, inverse=True)


def _program_core(program):
    return {key: value for key, value in program.items() if key != "witnesses"}


def _plan_hash(plan):
    return digest({key: value for key, value in plan.items() if key != "plan_sha256"})


def bind_plan_fingerprints(plan):
    """Bind data declarations; validation separately proves their semantic legality."""
    plan["canonical_fact_hashes"] = [digest(record) for record in plan["records"]]
    plan["semantic_fact_hashes"] = [digest(normalize_record(record, plan["surface_context"])) for record in plan["records"]]
    plan["program_hashes"] = {arm: [digest(_program_core(p)) for p in programs] for arm, programs in plan["programs"].items()}
    plan["plan_sha256"] = _plan_hash(plan)
    return plan


@lru_cache(maxsize=10)
def _legal_world_hashes(category):
    return frozenset(digest(record) for record in _worlds(category)[0])


def validate_plan(plan):
    """Validate actual normalized raw worlds, exact programs, and every witness."""
    try:
        category, index = plan["category"], plan["index"]
        if category not in CATEGORIES or type(index) is not int or not 0 <= index < 40:
            raise ValueError("invalid plan identity")
        if plan["family_id"] != f"factorial-v1/{category}/{index:03d}":
            raise ValueError("plan family identity differs from its category/index")
        context = plan["surface_context"]
        if canonical(context) != canonical(_surface_context(category, index)):
            raise ValueError("surface context must use the authorized reversible role/time mapping")
        if len(plan["records"]) != 2 or len(plan["world_indices"]) != 2:
            raise ValueError("plan must contain two raw worlds")
        worlds, _, edges = _worlds(category)
        normalized = [normalize_record(record, context) for record in plan["records"]]
        for record, world_index in zip(normalized, plan["world_indices"]):
            if type(world_index) is not int or not 0 <= world_index < len(worlds) or canonical(record) != canonical(worlds[world_index]):
                raise ValueError("actual normalized raw facts do not match a legal world index")
        if tuple(plan["world_indices"]) not in edges:
            raise ValueError("raw world pair is not a legal single-observation contrast")
        if canonical(plan["noul_rules"]) != canonical(_noul_rules(category)) or canonical(plan["choice_rules"]) != canonical(_choice_rules(category)):
            raise ValueError("private Noul/Choice program changed")
        if set(plan["programs"]) != {"narrow", "broad"}:
            raise ValueError("missing rubric programs")
        program_hashes = {}
        witness_count = 0
        for arm, allowed in (("narrow", (2, 3)), ("broad", (2, 3, 4, 5))):
            k = plan[f"{arm}_k"]
            variants = plan["programs"][arm]
            if type(k) is not int or k not in allowed or len(variants) != 2:
                raise ValueError("invalid program cardinality")
            program_hashes[arm] = []
            for variant, program in enumerate(variants):
                if canonical(_program_core(program)) != canonical(_program(category, k, index, variant)):
                    raise ValueError("private executable program/public description differs from canonical program")
                program_hashes[arm].append(digest(_program_core(program)))
                if set(program["witnesses"]) != {str(level) for level in range(k)}:
                    raise ValueError("witness levels incomplete")
                for level, witness in program["witnesses"].items():
                    raw = normalize_record(witness, context)
                    if digest(raw) not in _legal_world_hashes(category):
                        raise ValueError("reachability witness is not a legal raw world")
                    if select_level(program["rules"], derive_features(witness)) != int(level):
                        raise ValueError("reachability witness does not establish its declared level")
                    witness_count += 1
        if plan["canonical_fact_hashes"] != [digest(record) for record in plan["records"]]:
            raise ValueError("actual raw fact fingerprints changed")
        if plan["semantic_fact_hashes"] != [digest(record) for record in normalized]:
            raise ValueError("normalized semantic fact fingerprints changed")
        if plan["program_hashes"] != program_hashes or plan["plan_sha256"] != _plan_hash(plan):
            raise ValueError("actual plan/program fingerprints changed")
        return {"semantic_signature": digest({"records": normalized, "programs": program_hashes,
                                              "noul_rules": plan["noul_rules"], "choice_rules": plan["choice_rules"]}),
                "witness_count": witness_count}
    except (KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
        raise ValueError("malformed raw plan/program/witness") from exc


def make_plan(category, index, narrow_k, broad_k, *, coverage=None):
    if category not in CATEGORIES or type(index) is not int or index < 0 or narrow_k not in (2, 3) or broad_k not in (2, 3, 4, 5):
        raise ValueError("invalid latent plan configuration")
    records, features, edges = _worlds(category)
    programs = {"narrow": [_program(category, narrow_k, index, v) for v in (0, 1)],
                "broad": [_program(category, broad_k, index, v) for v in (0, 1)]}
    nouls, choices = _noul_rules(category), _choice_rules(category)
    outputs = _outputs(features, nouls, choices, programs)
    program_signature = digest({arm: [p["rules"] for p in variants] for arm, variants in programs.items()})
    # Reachability is checked over legal raw witnesses before selecting a pair.
    for arm, variants in programs.items():
        for v, program in enumerate(variants):
            witnesses = {}
            for r, y in zip(records, outputs):
                level = str(y["s"][arm][v])
                if level not in witnesses:
                    witnesses[level] = copy.deepcopy(r)
            if set(witnesses) != {str(i) for i in range(len(program["rules"]))}:
                raise ValueError(f"unreachable level: {category}/{arm}/{index}/{v}")
            program["witnesses"] = witnesses
    viable = []
    for a, b in edges:
        left, right = outputs[a], outputs[b]
        if not any(len(set(y["n"])) > 1 for y in (left, right)):
            continue
        if not all(any(y["s"][arm][0] != y["s"][arm][1] for y in (left, right)) for arm in programs):
            continue
        if not all(left["n"] != right["n"] or left["c"] != right["c"] or left["s"][arm] != right["s"][arm]
                   for arm in programs):
            continue
        gain = 0
        previous = (coverage or {}).get((category, "semantic_pairs"), set())
        gain += 1000 * int((program_signature, a, b) not in previous)
        for arm, k in (("narrow", narrow_k), ("broad", broad_k)):
            levels = set(left["s"][arm] + right["s"][arm])
            seen = (coverage or {}).get((category, arm, k), set())
            gain += 10 * len(levels - seen)
            for v in (0, 1):
                seen_variant = (coverage or {}).get((category, arm, k, v), set())
                gain += 6 * len({left["s"][arm][v], right["s"][arm][v]} - seen_variant)
            gain += int((index // 4) % k in levels)
        # Stable tie-break varies worlds without adding target-driven proof bits.
        tie = hashlib.sha256(f"{category}/{index}/{a}/{b}".encode()).hexdigest()
        viable.append((gain, tie, a, b))
    if not viable:
        raise ValueError(f"no legal matched contrast for {category}/{index}/{narrow_k}/{broad_k}")
    _, _, a, b = max(viable)
    selected = [copy.deepcopy(records[a]), copy.deepcopy(records[b])]
    context = _surface_context(category, index)
    selected = [transform_record(record, context) for record in selected]
    for variants in programs.values():
        for program in variants:
            program["witnesses"] = {level: transform_record(record, context) for level, record in program["witnesses"].items()}
    if coverage is not None:
        coverage.setdefault((category, "semantic_pairs"), set()).add((program_signature, a, b))
        for arm, k in (("narrow", narrow_k), ("broad", broad_k)):
            coverage.setdefault((category, arm, k), set()).update(outputs[a]["s"][arm] + outputs[b]["s"][arm])
            for v in (0, 1):
                coverage.setdefault((category, arm, k, v), set()).update((outputs[a]["s"][arm][v], outputs[b]["s"][arm][v]))
    return bind_plan_fingerprints({"family_id": f"factorial-v1/{category}/{index:03d}", "category": category, "index": index,
            "records": selected, "programs": programs, "noul_rules": nouls, "choice_rules": choices,
            "world_indices": [a, b], "surface_context": context, "narrow_k": narrow_k, "broad_k": broad_k})


def render_evidence(record, card):
    validate_blueprint(card)
    statements = []
    for statement in record["statements"]:
        if statement["tense"] == "future":
            template = "intent_template"
        elif statement["polarity"] == "negative":
            template = "denial_template"
        elif statement["endorsed"]:
            template = "endorsement_template"
        else:
            template = "claim_template"
        statements.append(card[template].format(**{name: statement[name] for name in STATEMENT_SLOTS}))
    executions = [card["execution_template"].format(**{name: str(row[name]).lower() if type(row[name]) is bool else row[name]
                                                      for name in EXECUTION_SLOTS}) for row in record["executions"]]
    evidence = {"heading": card["heading"], "statements": statements, "executions": executions}
    # These are raw operational observations, not private derived truth values.
    def observation(kind, row):
        values = {name: str(row[name]).lower() if type(row[name]) is bool else row[name]
                  for name in OBSERVATION_SLOTS[kind]}
        return card["observation_templates"][kind].format(**values)

    for key, kind in (("approvals", "approval"), ("corroborations", "corroboration"),
                      ("reversals", "reversal"), ("state_events", "state_event")):
        if record[key]:
            evidence[key] = [observation(kind, row) for row in record[key]]
    if card["category"] == "negation_and_quantifiers":
        evidence.update({k: copy.deepcopy(record[k]) for k in ("item_ids", "inventory_complete")})
        evidence["items"] = [observation("item", row) for row in record["items"]]
    if card["category"] == "ordered_rubrics":
        evidence["measurement"] = observation("measurement", {"value": record["measurement"] if record["measurement"] is not None else "missing"})
    return evidence


def _public_rules(category):
    common = ("Judge only the supplied record. Authentic means verified=true and source in trusted_sources. "
              "Execution outcomes require all four query fields, inclusive window, authentic source and live mode. "
              "Success and no_effect together conflict; timeout, absence, simulation or unverified data alone establish neither. "
              "Statements, intentions, endorsements and permissions alone do not establish execution. "
              "Statement predicates match actor/operation/target, without a time or source-trust restriction. "
              "Endorsement means the source adopts the assertion, not merely quotes another person's endorsement.")
    if category == "temporal_scope":
        common += " State predicates match target only; inspect all authentic events up to the stated cutoff, including before window start. Ties disagreeing on active yield unknown."
    if category == "negation_and_quantifiers":
        common += " Scope is item_ids. An item's status is its sole verified success/no_effect value; absent or conflicting values mean unknown. No item source/time test applies. All-success also requires inventory_complete=true."
    if category == "reversal_and_current_state":
        common += " Reversed means " + FEATURE_TEXT["reversed"] + ". Such a reversal preserves historical completion but removes effectiveness."
    return common


def render_family(plan, card, rubric):
    if card["category"] != plan["category"] or rubric not in ("narrow", "broad"):
        raise ValueError("language category or rubric mismatch")
    cases, relations = [], []
    for e, record in enumerate(plan["records"]):
        features = derive_features(record)
        evidence = render_evidence(record, card)
        choice_defs = {label: describe(rule) for label, rule in plan["choice_rules"].items()}
        for r, program in enumerate(plan["programs"][rubric]):
            questions = {f"n{i + 1}": {"type": "noul", "instructions": "Is this supported: " + describe(rule) + "?"}
                         for i, rule in enumerate(plan["noul_rules"])}
            questions["c1"] = {"type": "choice", "instructions": "Which supplied outcome definition applies?",
                               "criteria": choice_defs}
            questions["s1"] = {"type": "score", "instructions": "Apply score_rule and select the highest applicable score_levels index.",
                               "criteria": program["levels"]}
            expected = {f"n{i + 1}": eval_rule(rule, features) for i, rule in enumerate(plan["noul_rules"])}
            expected.update(c1=select_choice(plan["choice_rules"], features), s1=select_level(program["rules"], features))
            state = {"scope": {k: copy.deepcopy(record[k]) for k in ("query", "window", "query_time", "trusted_sources")},
                     "evidence": copy.deepcopy(evidence), "rules": _public_rules(plan["category"]),
                     "choice_definitions": choice_defs, "score_rule": program["instruction"],
                     "score_ordinality": program["ordinal"], "score_selection": program["selection"],
                     "score_levels": program["levels"]}
            cases.append({"id": f"{plan['family_id']}/e{e}r{r}", "family_id": plan["family_id"],
                          "domain": "operational observations", "category": plan["category"],
                          "variant": f"e{e}r{r}", "layout": "authored observations and public definitions",
                          "predicate_tags": [plan["category"]], "request": {"state": state, "questions": questions},
                          "expected": expected, "rationale": {q: "Evaluated from raw observations under the explicit supplied definition."
                                                                  for q in QUESTION_IDS}})

    def relation(axis, a, qa, b, qb, kind="flip"):
        relations.append({"id": f"{plan['family_id']}/{axis}/{len(relations)}", "family_id": plan["family_id"],
                          "kind": kind, "contrast_axis": axis, "expected_equal": False,
                          "left": {"case_id": cases[a]["id"], "question_id": qa},
                          "right": {"case_id": cases[b]["id"], "question_id": qb}})

    for a, b in ((0, 2), (1, 3)):
        for q in QUESTION_IDS:
            if cases[a]["expected"][q] != cases[b]["expected"][q]:
                relation("evidence", a, q, b, q)
    for a, b in ((0, 1), (2, 3)):
        if cases[a]["expected"]["s1"] != cases[b]["expected"]["s1"]:
            relation("rubric", a, "s1", b, "s1")
    for a in range(4):
        for qa, qb in itertools.combinations(QUESTION_IDS[:3], 2):
            if cases[a]["expected"][qa] != cases[a]["expected"][qb]:
                relation("question", a, qa, a, qb, "question_contrast")
    if {r["contrast_axis"] for r in relations} != {"evidence", "rubric", "question"}:
        raise ValueError("family lacks a meaningful contrast axis")
    family = {"id": plan["family_id"], "category": plan["category"], "cases": cases, "relations": relations,
              "private_provenance": {"blueprint_id": card["id"], "author": card["author"], "blueprint": card, "rubric": rubric,
                                     "canonical_fact_hashes": plan["canonical_fact_hashes"], "plan": plan}}
    validate_suite(family)
    return family


def build_matched_suites(blueprints, seed=42):
    for card in blueprints:
        validate_blueprint(card)
    assignment = assign_blueprints(blueprints, seed=seed)
    by_id = {b["id"]: b for b in blueprints}
    suites = {arm: {"name": f"contrast-factorial-v1/{arm}", "cases": [], "relations": [], "families": [],
                    "metadata": {"seed": seed, "arm": arm}} for arm in "ABCD"}
    coverage = {}
    for row in assignment:
        plan = make_plan(row["category"], row["category_index"], row["narrow_k"], row["broad_k"], coverage=coverage)
        for arm, rubric, language in (("A", "narrow", "low"), ("B", "broad", "low"),
                                      ("C", "narrow", "high"), ("D", "broad", "high")):
            family = render_family(plan, by_id[row[f"{language}_blueprint"]], rubric)
            suites[arm]["families"].append(family)
            suites[arm]["cases"].extend(family["cases"])
            suites[arm]["relations"].extend(family["relations"])
    validate_matched_suites(suites)
    return suites


def validate_matched_suites(suites, *, require_language_variation=False):
    if set(suites) != set("ABCD"):
        raise ValueError("exactly four arms required")
    coverage = defaultdict(Counter)
    variant_coverage = defaultdict(Counter)
    widths = defaultdict(Counter)
    fact_crosses = defaultdict(Counter)
    semantic_pairs = defaultdict(set)
    validated_plans = {}
    role_coverage = defaultdict(lambda: defaultdict(Counter))
    time_coordinates = set()
    ids = None
    for arm, suite in suites.items():
        validate_suite(suite)
        if suite["cases"] != [case for family in suite["families"] for case in family["cases"]] or suite["relations"] != [
            relation for family in suite["families"] for relation in family["relations"]
        ]:
            raise ValueError("flattened suite differs from the canonical family view")
        current_ids = [f["id"] for f in suite["families"]]
        if len(current_ids) != 400 or len(set(current_ids)) != 400 or len(suite["cases"]) != 1600:
            raise ValueError("400 complete distinct latent families required")
        if ids is None:
            ids = current_ids
        elif ids != current_ids:
            raise ValueError("family identity/order mismatch")
        if Counter(f["category"] for f in suite["families"]) != Counter({c: 40 for c in CATEGORIES}):
            raise ValueError("category population mismatch")
        for family in suite["families"]:
            if len(family["cases"]) != 4:
                raise ValueError("incomplete family")
            k = len(family["cases"][0]["request"]["questions"]["s1"]["criteria"])
            widths[arm][k] += 1
            provenance = family["private_provenance"]
            plan = provenance["plan"]
            if [digest(r) for r in plan["records"]] != plan["canonical_fact_hashes"] or plan["canonical_fact_hashes"] != provenance["canonical_fact_hashes"]:
                raise ValueError("canonical fact hash does not match actual raw records")
            actual_plan_hash = _plan_hash(plan)
            if plan.get("plan_sha256") != actual_plan_hash:
                raise ValueError("actual private plan/program/witness fingerprint differs from its declaration")
            if actual_plan_hash not in validated_plans:
                validated_plans[actual_plan_hash] = validate_plan(plan)
            if plan["narrow_k"] != 2 + plan["index"] % 2 or plan["broad_k"] != 2 + plan["index"] % 4:
                raise ValueError("plan cardinalities differ from the matched assignment")
            if provenance["rubric"] != ("narrow" if arm in "AC" else "broad"):
                raise ValueError("arm uses the wrong rubric program")
            canonical_family = render_family(plan, provenance["blueprint"], provenance["rubric"])
            if family["id"] != plan["family_id"] or family["category"] != plan["category"]:
                raise ValueError("family metadata differs from the canonical raw plan")
            if family["relations"] != canonical_family["relations"]:
                raise ValueError("relations differ from canonical raw-fact/program relations")
            if arm == "A":
                semantic_pairs[family["category"]].add(validated_plans[actual_plan_hash]["semantic_signature"])
                context = plan["surface_context"]
                time_coordinates.add((context["time_scale"], context["time_offset"]))
                for kind, originals in {"actor": ("operator", "other operator"), "operation": ("seal", "open"),
                                        "target": ("parcel", "crate"), "run": ("run", "other run")}.items():
                    for role, original in zip(("queried", "distractor"), originals):
                        role_coverage[kind][context["aliases"][original]][role] += 1
                for original in ("ledger", "inspector", "untrusted", "speaker"):
                    role = "trusted" if original in {"ledger", "inspector"} else "not_trusted"
                    role_coverage["source"][context["aliases"][original]][role] += 1
                for record in plan["records"]:
                    values = derive_features(record)
                    for a, b in (("claim", "completed"), ("intent", "completed"), ("approved", "completed"), ("claim", "approved")):
                        fact_crosses[f"{a}/{b}"][f"{int(values[a])}/{int(values[b])}"] += 1
            for position, case in enumerate(family["cases"]):
                record = plan["records"][position // 2]
                features = derive_features(record)
                program = plan["programs"][provenance["rubric"]][position % 2]
                recomputed = {f"n{i + 1}": eval_rule(rule, features) for i, rule in enumerate(plan["noul_rules"])}
                recomputed.update(c1=select_choice(plan["choice_rules"], features), s1=select_level(program["rules"], features))
                if recomputed != case["expected"]:
                    raise ValueError("stored targets do not match raw-fact oracle")
                state = case["request"]["state"]
                if state["evidence"] != render_evidence(record, provenance["blueprint"]):
                    raise ValueError("rendered evidence does not match exact raw record and blueprint")
                if canonical(case["request"]) != canonical(canonical_family["cases"][position]["request"]):
                    raise ValueError("public request differs from complete canonical raw-fact/program request")
                coverage[f"{arm}/{family['category']}/K{k}"][str(case["expected"]["s1"])] += 1
                variant_coverage[f"{arm}/{family['category']}/r{position % 2}/K{k}"][str(case["expected"]["s1"])] += 1
                if list(case["request"]["questions"]) != list(QUESTION_IDS):
                    raise ValueError("canonical question order mismatch")
                if set(state) != {"scope", "evidence", "rules", "choice_definitions", "score_rule", "score_ordinality",
                                  "score_selection", "score_levels"}:
                    raise ValueError("private or unknown field in request state")
                serialized = canonical(state)
                if any(f'"{secret}"' in serialized for secret in ("expected", "rationale", "tense", "polarity", "endorsed",
                                                                  "programs", "features", "blueprint_id", "author", "family_id")):
                    raise ValueError("private annotation leaked into state")
                for unit in compile_request(case["request"]).units:
                    if unit.payload["state"] != state:
                        raise ValueError("compiler candidate lost public definitions")
            for a, b in ((0, 1), (2, 3)):
                if family["cases"][a]["request"]["state"]["evidence"] != family["cases"][b]["request"]["state"]["evidence"]:
                    raise ValueError("rubric contrast changed evidence")
            for a, b in ((0, 2), (1, 3)):
                if family["cases"][a]["request"]["questions"] != family["cases"][b]["request"]["questions"]:
                    raise ValueError("evidence contrast changed questions")
        counts = Counter(f["private_provenance"]["blueprint_id"] for f in suite["families"])
        if len(counts) != (200 if arm in "AB" else 400) or set(counts.values()) != ({2} if arm in "AB" else {1}):
            raise ValueError("language cardinality/reuse mismatch")
        expected_widths = {2: 200, 3: 200} if arm in "AC" else {2: 100, 3: 100, 4: 100, 5: 100}
        if widths[arm] != Counter(expected_widths):
            raise ValueError("Score cardinality population mismatch")
    for key, counts in {**coverage, **variant_coverage}.items():
        if set(counts) != {str(i) for i in range(int(key.rsplit("K", 1)[1]))}:
            raise ValueError(f"missing positive Score level: {key}")
    for left, right in (("A", "C"), ("B", "D")):
        for a, b in zip(suites[left]["cases"], suites[right]["cases"]):
            if a["expected"] != b["expected"] or a["request"]["questions"] != b["request"]["questions"]:
                raise ValueError("language factor changed labels/questions")
            sa, sb = a["request"]["state"], b["request"]["state"]
            if {k: v for k, v in sa.items() if k != "evidence"} != {k: v for k, v in sb.items() if k != "evidence"}:
                raise ValueError("language factor changed public definitions")
        for a, b in zip(suites[left]["families"], suites[right]["families"]):
            if a["private_provenance"]["canonical_fact_hashes"] != b["private_provenance"]["canonical_fact_hashes"]:
                raise ValueError("language factor changed canonical facts")
    for index in range(400):
        plans = [suites[arm]["families"][index]["private_provenance"]["plan"] for arm in "ABCD"]
        if any(plan != plans[0] for plan in plans[1:]):
            raise ValueError("arms differ in actual raw facts, programs, witnesses or reversible context")
    for left, right in (("A", "B"), ("C", "D")):
        for a, b in zip(suites[left]["cases"], suites[right]["cases"]):
            if canonical(a["request"]["state"]["evidence"]) != canonical(b["request"]["state"]["evidence"]):
                raise ValueError("rubric factor changed evidence bytes")
            for q in QUESTION_IDS[:-1]:
                if a["expected"][q] != b["expected"][q] or a["request"]["questions"][q] != b["request"]["questions"][q]:
                    raise ValueError("rubric factor changed non-Score question")
    shared = sum(a["private_provenance"]["blueprint_id"] == c["private_provenance"]["blueprint_id"]
                 for a, c in zip(suites["A"]["families"], suites["C"]["families"]))
    if shared != 200:
        raise ValueError("shared language half mismatch")
    changed_primary = 0
    for a, c in zip(suites["A"]["families"], suites["C"]["families"]):
        fields = _primary_fields(a["category"])
        changed_primary += any(
            {key: ca["request"]["state"]["evidence"].get(key) for key in fields} !=
            {key: cc["request"]["state"]["evidence"].get(key) for key in fields}
            for ca, cc in zip(a["cases"], c["cases"]))
    if require_language_variation and changed_primary != 200:
        raise ValueError("language treatment must change primary evidence in exactly the 200 changed families")
    if any(len(cells) != 4 for cells in fact_crosses.values()):
        raise ValueError("claim, intent, permission and completion must be independently crossed")
    if any(len(values) != 40 for values in semantic_pairs.values()):
        raise ValueError("duplicate within-category raw-fact/program pair")
    if any(len(roles) != 2 for group in role_coverage.values() for roles in group.values()):
        raise ValueError("lexical identities fail to cross queried/distractor or trusted/untrusted roles")
    return {"families": 400, "cases_per_arm": 1600, "shared_language_families": shared,
            "changed_primary_language_families": changed_primary,
            "validated_witnesses": sum(row["witness_count"] for row in validated_plans.values()),
            "distinct_affine_time_coordinates": len(time_coordinates),
            "lexical_role_coverage": {kind: {word: dict(roles) for word, roles in group.items()}
                                      for kind, group in role_coverage.items()},
            "distinct_fact_program_pairs_by_category": {k: len(v) for k, v in sorted(semantic_pairs.items())},
            "independent_fact_crosses": {k: dict(v) for k, v in sorted(fact_crosses.items())},
            "level_coverage": {k: dict(v) for k, v in sorted(coverage.items())},
            "variant_level_coverage": {k: dict(v) for k, v in sorted(variant_coverage.items())}}


def training_bundles(suite, arm):
    bundles = []
    for family in suite["families"]:
        examples = []
        for case in family["cases"]:
            for qid in QUESTION_IDS:
                question, expected = case["request"]["questions"][qid], case["expected"][qid]
                key = "truth" if question["type"] == "noul" else "choice" if question["type"] == "choice" else "level_index"
                examples.append({"state": copy.deepcopy(case["request"]["state"]), "question": copy.deepcopy(question),
                                 "target": {key: expected}, "case_id": case["id"], "question_id": qid})
        bundles.append({"id": family["id"], "group_id": family["id"], "examples": examples, "relations": [],
                        "provenance": {"dataset": "contrast-factorial-v1", "assigned_split": "train", "arm": arm,
                                       "category": family["category"]}})
    return bundles


def _repo_path(root, relative):
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("all manifest paths must be repository-relative")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("path escapes repository")
    return resolved


STUDY_APPROVAL_PATH = "reports/contrast-factorial-v1/STUDY_APPROVAL.json"
APPROVED_BASE_PATH = "checkpoints/judgment-full-v0.2/final"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_metadata(path, label):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid or missing {label} metadata") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} metadata must be an object")
    return value


def _verify_byte_hashes(root, mapping, label):
    """Hash opaque artifacts without decoding assessment, source, or review text."""
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{label} must contain artifact hashes")
    for relative, expected in mapping.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid {label} artifact hash")
        path = _repo_path(root, relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"{label} artifact byte hash mismatch: {relative}")


def validate_emission_inputs(repo_root, bank_path, evaluation_freeze_path, base_checkpoint):
    """Non-emitting metadata-only authorization, freeze, base and full-review gate.

    STUDY_APPROVAL.json is fixed and root-owned; the caller cannot supply a new
    approval path or authorize an arbitrary base directory by hashing it.
    Evaluation artifacts are opaque byte streams, never parsed as examples.
    """
    from experiments.contrast_factorial_training import tree_sha256
    from openjev.judgment_model import checkpoint_identity

    root = Path(repo_root).resolve()
    if root != REPO_ROOT:
        raise ValueError("repository root must be this factory checkout; caller directories cannot supply study approval")
    freeze_file = _repo_path(root, evaluation_freeze_path)
    if not freeze_file.is_file() or not freeze_file.stat().st_size:
        raise ValueError("evaluation freeze must exist before emission")
    approval_file = _repo_path(root, STUDY_APPROVAL_PATH)
    approval = _read_metadata(approval_file, "root study approval")
    if (approval.get("study") != "contrast-factorial-v1" or approval.get("status") != "approved"
            or approval.get("arms") != list("ABCD") or approval.get("updates_per_arm") != 400):
        raise ValueError("root study approval does not authorize this pilot")
    if str(base_checkpoint) != APPROVED_BASE_PATH or approval.get("base_checkpoint") != str(base_checkpoint):
        raise ValueError("base checkpoint path is not the root-pinned v0.2 initialization")
    if approval.get("evaluation_freeze_path") != str(evaluation_freeze_path):
        raise ValueError("evaluation freeze path is not root-pinned")
    _verify_byte_hashes(root, {str(evaluation_freeze_path): approval.get("evaluation_freeze_sha256")}, "root-pinned freeze")
    _verify_byte_hashes(root, {approval.get("contract_path"): approval.get("contract_sha256")}, "root-pinned contract")
    freeze = _read_metadata(freeze_file, "evaluation freeze")
    if (freeze.get("study") != "contrast-factorial-v1" or freeze.get("status") != "frozen"
            or freeze.get("reviewed_requests") != 480 or freeze.get("reviewed_judgments") != 2400
            or type(freeze.get("unresolved_review_flags")) is not int or freeze["unresolved_review_flags"] != 0
            or freeze.get("no_model_predictions_used") is not True
            or freeze.get("training_language_authored_before_freeze") is not False):
        raise ValueError("evaluation freeze is incomplete, unresolved or not frozen")
    if not isinstance(freeze.get("suites"), dict) or set(freeze["suites"]) != {"assessment", "calibration"}:
        raise ValueError("freeze must identify exactly assessment and calibration")
    frozen_artifacts = {}
    for split, families, per_category in (("assessment", 80, 8), ("calibration", 40, 4)):
        meta = freeze["suites"][split]
        if (not isinstance(meta, dict) or meta.get("families") != families or meta.get("cases") != 4 * families
                or meta.get("judgments") != 20 * families
                or meta.get("category_counts") != {category: per_category for category in CATEGORIES}):
            raise ValueError(f"frozen {split} population is not complete")
        hashes = {meta.get("path"): meta.get("suite_sha256")}
        _verify_byte_hashes(root, hashes, f"frozen {split}")
        frozen_artifacts.update(hashes)
    for key in ("author_sources", "review_sources", "source_sha256"):
        _verify_byte_hashes(root, freeze.get(key), f"freeze {key}")
        frozen_artifacts.update(freeze[key])
    legacy_path = "data/contrast-factorial-v1/legacy/manifest.json"
    _verify_byte_hashes(root, {legacy_path: freeze.get("legacy_manifest_sha256")}, "frozen legacy manifest")
    frozen_artifacts[legacy_path] = freeze["legacy_manifest_sha256"]

    base = _repo_path(root, base_checkpoint)
    components = ("checkpoint.json", "training.pt", "readouts.safetensors", "adapter/adapter_config.json",
                  "adapter/adapter_model.safetensors")
    if any(not (base / part).is_file() or not (base / part).stat().st_size for part in components):
        raise ValueError("approved base checkpoint components are incomplete")
    info = _read_metadata(base / "checkpoint.json", "base checkpoint identity")
    identity = info.get("checkpoint_id")
    if (info.get("format") != "openjev-judgment-v0.2" or not isinstance(identity, str)
            or not re.fullmatch(r"openjev-judgment-v0\.2/sha256-[0-9a-f]{64}", identity)
            or identity != approval.get("base_checkpoint_id") or info.get("model_id") != approval.get("model_id")
            or info.get("revision") != approval.get("revision") or not isinstance(info.get("lora"), dict)
            or info["lora"].get("rank") != 8 or info["lora"].get("alpha") != 16
            or checkpoint_identity(base, info) != identity):
        raise ValueError("base checkpoint identity is not the approved v0.2 identity")
    base_hash = tree_sha256(base)
    if base_hash != approval.get("base_checkpoint_sha256"):
        raise ValueError("base checkpoint tree hash differs from root approval")

    bank_file = _repo_path(root, bank_path)
    bank = _read_metadata(bank_file, "language bank")
    if type(bank.get("schema_version")) is not int or bank["schema_version"] != 1 or not isinstance(bank.get("blueprints"), list):
        raise ValueError("language bank schema is invalid")
    cards = bank["blueprints"]
    for card in cards:
        validate_blueprint(card)
    assign_blueprints(cards)
    bank_hash = digest(cards)
    review = bank.get("review")
    if (not isinstance(review, dict) or review.get("approved") is not True or review.get("bank_sha256") != bank_hash
            or not isinstance(review.get("review_path"), str)):
        raise ValueError("reviewed language bank with matching content hash required")
    review_file = _repo_path(root, review["review_path"])
    if not review_file.is_file() or review_file == bank_file or review_file.samefile(bank_file):
        raise ValueError("language review must be a separate artifact, not the bank or an alias of it")
    reviewed = _read_metadata(review_file, "independent language review")
    identities = reviewed.get("reviewed_blueprint_ids")
    reviewer = reviewed.get("reviewer_identity")
    if (reviewed.get("kind") != "independent_language_review" or reviewed.get("approved") is not True
            or reviewed.get("bank_sha256") != bank_hash or reviewed.get("semantic_preservation") != "pass"
            or reviewed.get("construction_diversity") != "pass" or reviewed.get("unresolved_flags") != []
            or not isinstance(reviewer, str) or not reviewer.strip() or reviewer in {card["author"] for card in cards}
            or not isinstance(identities, list) or len(identities) != 400
            or any(not isinstance(identity, str) for identity in identities)
            or len(set(identities)) != 400 or set(identities) != {card["id"] for card in cards}):
        raise ValueError("separate independent review must approve semantics and diversity for exactly all 400 blueprint IDs")
    return {"root": root, "bank": bank, "bank_file": bank_file, "review_file": review_file,
            "freeze_file": freeze_file, "base_hash": base_hash, "base_checkpoint_id": identity,
            "approval_file": approval_file, "approval_sha256": file_sha256(approval_file),
            "frozen_artifacts": frozen_artifacts}


def emit_study(repo_root, bank_path, evaluation_freeze_path, base_checkpoint, output_dir, seed=42):
    gate = validate_emission_inputs(repo_root, bank_path, evaluation_freeze_path, base_checkpoint)
    root, bank, bank_file = gate["root"], gate["bank"], gate["bank_file"]
    freeze, review_file, base_hash = gate["freeze_file"], gate["review_file"], gate["base_hash"]
    review = bank["review"]
    if any("synthetic" in b.get("author", "").lower() for b in bank["blueprints"]):
        raise ValueError("synthetic fixtures cannot be emitted as training language")
    language_audit = language_quality_report(bank["blueprints"])
    if any(row["distinct_primary_constructions"] != 40 for row in language_audit["categories"].values()):
        raise ValueError("language bank contains duplicate primary constructions; IDs/headings do not count")
    destination = _repo_path(root, output_dir)
    if destination.exists():
        raise ValueError("output already exists; preserve prior artifacts and choose a versioned path")
    suites = build_matched_suites(bank["blueprints"], seed=seed)
    audit = validate_matched_suites(suites, require_language_variation=True)
    audit["language_quality"] = language_audit
    destination.mkdir(parents=True)
    manifest = {"study": "contrast-factorial-v1", "seed": seed, "arms": {},
                "ordered_latent_ids": [f["id"] for f in suites["A"]["families"]],
                "evaluation_freeze_path": str(evaluation_freeze_path), "evaluation_freeze_sha256": file_sha256(freeze),
                "base_checkpoint": str(base_checkpoint), "base_checkpoint_sha256": base_hash,
                "base_checkpoint_id": gate["base_checkpoint_id"], "study_approval_path": STUDY_APPROVAL_PATH,
                "study_approval_sha256": gate["approval_sha256"],
                "source_fingerprints": {}, "data_fingerprint": {str(bank_path): file_sha256(bank_file),
                    review["review_path"]: file_sha256(review_file), STUDY_APPROVAL_PATH: gate["approval_sha256"],
                    **gate["frozen_artifacts"]}}
    source_paths = ["experiments/contrast_factorial_data.py", "experiments/contrast_factorial_design.py",
                    "experiments/contrast_scaling_logic.py", "experiments/revised_metrics.py",
                    "src/openjev/judgments.py", "src/openjev/judgment_training.py", "docs/contrast-factorial-contract-v1.md",
                    "docs/contrast-factorial-generator-v1.md"]
    for path in source_paths:
        manifest["source_fingerprints"][path] = file_sha256(root / path)
    for arm, suite in suites.items():
        arm_dir = destination / arm
        arm_dir.mkdir()
        train_file, suite_file = arm_dir / "train.jsonl", arm_dir / "train-suite.json"
        train_file.write_text("".join(canonical(b) + "\n" for b in training_bundles(suite, arm)))
        public_suite = {key: suite[key] for key in ("name", "cases", "relations", "metadata")}
        public_suite["ordered_family_ids"] = manifest["ordered_latent_ids"]
        suite_file.write_text(canonical(public_suite) + "\n")
        train_rel, suite_rel = str(train_file.relative_to(root)), str(suite_file.relative_to(root))
        manifest["arms"][arm] = {"train_file": train_rel, "suite_file": suite_rel,
                                 "ordered_family_ids": manifest["ordered_latent_ids"],
                                 "sha256": file_sha256(train_file), "suite_sha256": file_sha256(suite_file)}
        manifest["data_fingerprint"].update({train_rel: file_sha256(train_file), suite_rel: file_sha256(suite_file)})
    audit_path = destination / "private-audit.json"
    audit_path.write_text(canonical({"checks": audit, "plans": [f["private_provenance"]["plan"] for f in suites["A"]["families"]],
                                    "assignment": assign_blueprints(bank["blueprints"], seed=seed)}) + "\n")
    manifest["data_fingerprint"][str(audit_path.relative_to(root))] = file_sha256(audit_path)
    (destination / "manifest.json").write_text(canonical(manifest) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--evaluation-freeze", required=True)
    parser.add_argument("--base-checkpoint", default="checkpoints/judgment-full-v0.2/final")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = emit_study(args.repo_root, args.bank, args.evaluation_freeze, args.base_checkpoint, args.output, args.seed)
    print(canonical({"study": result["study"], "arms": list(result["arms"]), "families": len(result["ordered_latent_ids"])}))


if __name__ == "__main__":
    main()
