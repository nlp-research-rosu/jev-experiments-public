"""Deterministic factory for the frozen-before-language scaling corpus.

This module constructs *private raw facts first*, then uses
``contrast_scaling_logic`` to derive every target.  It deliberately has no
bank loader or writer: rendering a real training corpus remains gated on the
separately authored language bank and evaluation freeze.
"""

from __future__ import annotations

import copy
import hashlib
import json
import string
from collections import Counter

from experiments.contrast_scaling_logic import derive_features, eval_rule, select_choice, select_level


CATEGORIES = (
    "claim_vs_completion", "permission_vs_execution", "unknown_vs_failure",
    "attribution_and_endorsement", "entity_binding", "action_binding",
    "temporal_scope", "reversal_and_current_state", "negation_and_quantifiers",
    "ordered_rubrics",
)
QUESTION_IDS = ("n1", "n2", "n3", "c1", "s1")
_FORMAT_FIELDS = {
    "claim_template": {"speaker", "actor", "past", "target"},
    "intent_template": {"speaker", "actor", "verb", "target"},
    "denial_template": {"speaker", "actor", "verb", "target"},
    "endorsement_template": {"authority", "speaker", "actor", "past", "target"},
}


def _fail(message):
    raise ValueError(message)


def _template_fields(value, allowed, name):
    if not isinstance(value, str) or not value:
        _fail(f"{name} must be a nonempty string")
    fields = set()
    for _, field, spec, conversion in string.Formatter().parse(value):
        if field is None:
            continue
        if not field or field not in allowed or spec or conversion:
            _fail(f"{name} has an invalid placeholder")
        fields.add(field)
    if not fields:
        _fail(f"{name} needs a placeholder")


def validate_blueprint(blueprint):
    """Reject placeholders and source descriptions that cannot safely render facts."""
    if not isinstance(blueprint, dict):
        _fail("blueprint must be an object")
    for name in ("id", "category", "scenario_domain", "evidence_heading", "question_prefix", "score_title"):
        if not isinstance(blueprint.get(name), str) or not blueprint[name].strip():
            _fail(f"blueprint {name} must be a nonempty string")
    if blueprint["category"] not in CATEGORIES:
        _fail("blueprint category is unknown")
    for name in ("operation", "other_operation"):
        operation = blueprint.get(name)
        if not isinstance(operation, dict) or set(operation) != {"verb", "past", "noun"} or any(
            not isinstance(operation[key], str) or not operation[key].strip() for key in operation
        ):
            _fail(f"blueprint {name} must define verb/past/noun")
    if blueprint["operation"]["verb"] == blueprint["other_operation"]["verb"]:
        _fail("blueprint operations need distinct verbs")
    target = blueprint.get("target_template")
    _template_fields(target, {"identifier"}, "target_template")
    if set(blueprint.get("sources", {})) != {"primary", "secondary", "untrusted"}:
        _fail("blueprint sources must define primary/secondary/untrusted")
    source_values = list(blueprint["sources"].values())
    if any(not isinstance(value, str) or not value.strip() for value in source_values) or len(set(source_values)) != 3:
        _fail("blueprint sources must be distinct names")
    for name, allowed in _FORMAT_FIELDS.items():
        _template_fields(blueprint.get(name), allowed, name)
    return blueprint


def _record(blueprint, instance, evidence, configuration):
    """Create source facts, never targets; category branches change observed facts."""
    target = blueprint["target_template"].format(identifier=f"{instance:03d}")
    query = {"actor": "Mira", "operation": blueprint["operation"]["verb"], "target": target, "run_id": f"run-{instance:03d}"}
    source = blueprint["sources"]
    raw = {
        "query": query, "window": [10 + (configuration % 3), 30 + (configuration % 3)], "query_time": 15 + (configuration % 3),
        "trusted_sources": [source["primary"], source["secondary"]], "statements": [], "approvals": [],
        "executions": [], "reversals": [], "corroborations": [], "items": [], "item_ids": [],
        "inventory_complete": True, "state_events": [], "measurement": _numeric_scale(configuration)[0 if evidence == "A" else 1],
    }
    # Evidence A always has a genuine completion claim but no verified completion.
    raw["statements"].append({**query, "source": source["untrusted"], "time": 14, "verified": False,
                              "tense": "completed", "polarity": "positive", "speaker": "Rin"})
    raw["statements"].append({**query, "source": source["untrusted"], "time": 13, "verified": False,
                              "tense": "future", "polarity": "positive", "speaker": "Rin"})
    raw["approvals"].append({**query, "source": source["primary"], "time": 15, "verified": True, "decision": "allow"})
    if evidence == "B":
        raw["executions"].append({**query, "source": source["primary"], "time": 20, "verified": True,
                                  "mode": "live", "outcome": "success"})

    category = blueprint["category"]
    if category == "claim_vs_completion" and evidence == "B":
        raw["statements"] = [{**query, "source": source["untrusted"], "time": 14, "verified": False,
                              "tense": "completed", "polarity": "negative", "speaker": "Rin"}]
    elif category == "unknown_vs_failure" and evidence == "B":
        raw["executions"][-1]["outcome"] = "no_effect"
    elif category == "permission_vs_execution" and evidence == "B":
        raw["approvals"][-1]["decision"] = "deny"
    elif category == "attribution_and_endorsement" and evidence == "B":
        raw["statements"][0]["endorsed"] = True
    elif category == "entity_binding" and evidence == "A":
        raw["executions"].append({**query, "actor": "Omar", "source": source["primary"], "time": 20,
                                  "verified": True, "mode": "live", "outcome": "success"})
    elif category == "action_binding" and evidence == "A":
        raw["executions"].append({**query, "operation": blueprint["other_operation"]["verb"], "source": source["primary"], "time": 20,
                                  "verified": True, "mode": "live", "outcome": "success"})
    elif category == "temporal_scope":
        raw["state_events"] = [{"target": query["target"], "source": source["primary"], "verified": True, "time": 12 if evidence == "A" else 20, "active": True}]
        if evidence == "A":
            raw["state_events"].append({"target": query["target"], "source": source["primary"], "verified": True, "time": 22, "active": False})
    elif category == "reversal_and_current_state" and evidence == "A":
        raw["executions"].append({**query, "source": source["primary"], "time": 20, "verified": True, "mode": "live", "outcome": "success"})
        raw["reversals"].append({**query, "source": source["secondary"], "time": 25, "verified": True, "mode": "live",
                                 "outcome": "success", "reverses": query["run_id"]})
    elif category == "negation_and_quantifiers":
        raw["item_ids"] = ["item-a", "item-b"]
        status_b = "success" if evidence == "A" else "no_effect"
        raw["items"] = [
            {"id": "item-a", "verified": True, "status": "success"},
            {"id": "item-b", "verified": True, "status": status_b},
        ]
    _add_semantic_observation(raw, blueprint, configuration // 25, evidence)
    _apply_surface_permutation(raw, blueprint, instance, configuration)
    return raw


def _numeric_scale(configuration):
    """Return low/high observations and a varying cutoff strictly between them."""
    pattern, shape = configuration % 25, configuration // 25
    low = 10 + ((7 * pattern + 3 * shape) % 40)
    high = 50 + ((11 * pattern + 5 * shape) % 39)
    cutoff = low + max(1, (high - low) // 2)
    return low, high, cutoff


def _add_semantic_observation(raw, blueprint, shape, evidence):
    """Add a scoped evidence shape that remains relevant to the public decision.

    These are not names or inert distractors: each changes the authority, scope,
    outcome, mode, or time relation a reader must apply under ``rules``.
    """
    q, source = raw["query"], blueprint["sources"]

    def execution(**changes):
        return {**q, "source": source["primary"], "time": 20, "verified": True,
                "mode": "live", "outcome": "success", **changes}

    variants = (
        lambda: None,
        lambda: raw["executions"].append(execution(verified=False)),
        lambda: raw["executions"].append(execution(verified=False, outcome="no_effect")),
        lambda: raw["executions"].append(execution(mode="dry_run")),
        lambda: raw["executions"].append(execution(source=source["untrusted"])),
        lambda: raw["executions"].append(execution(actor="Omar")),
        lambda: raw["executions"].append(execution(operation=blueprint["other_operation"]["verb"])),
        lambda: raw["executions"].append(execution(target="other-target")),
        lambda: raw["executions"].append(execution(run_id="other-run")),
        lambda: raw["executions"].append(execution(time=raw["window"][0] - 1)),
        lambda: raw["executions"].append(execution(time=raw["window"][1] + 1)),
        lambda: raw["executions"].append(execution(mode="simulation")),
        lambda: raw["executions"].append(execution(mode="queued")),
        lambda: raw["executions"].append(execution(outcome="timeout")),
        lambda: raw["executions"].append(execution(outcome="missing")),
        lambda: raw["executions"].append(execution(actor="Omar", operation=blueprint["other_operation"]["verb"])),
        lambda: raw["executions"].append(execution(actor="Omar", target="other-target")),
        lambda: raw["executions"].append(execution(operation=blueprint["other_operation"]["verb"], run_id="other-run")),
        lambda: raw["executions"].append(execution(target="other-target", run_id="other-run")),
        lambda: raw["executions"].append(execution(actor="Omar", operation=blueprint["other_operation"]["verb"], target="other-target", run_id="other-run")),
    )
    variants[shape]()

    if evidence != "B":
        return

    category = blueprint["category"]
    # These are outcome-bearing observations. They exercise real oracle
    # branches only on evidence B, preserving the A-side rubric contrast.
    if shape % 5 == 0 and category not in {"entity_binding", "action_binding", "reversal_and_current_state"}:
        exact = [row for row in raw["executions"] if all(row.get(key) == q[key] for key in ("actor", "operation", "target", "run_id"))]
        if exact:
            opposite = "success" if exact[0].get("outcome") == "no_effect" else "no_effect"
            raw["executions"].append(execution(source=source["secondary"], time=21, outcome=opposite))
    if shape % 5 == 1 and category != "unknown_vs_failure":
        raw["corroborations"].append({**q, "source": source["secondary"], "time": 21,
                                      "verified": True, "outcome": "success"})

    if category == "negation_and_quantifiers":
        branch = shape % 5
        if branch == 0:
            raw["items"] = [row for row in raw["items"] if row["id"] != "item-b"]
        elif branch == 1:
            raw["items"][1]["verified"] = False
        elif branch == 2:
            raw["items"].append({"id": "item-b", "verified": True, "status": "success"})
        elif branch == 3:
            raw["inventory_complete"] = False


def _apply_surface_permutation(raw, blueprint, instance, configuration):
    """Rename semantic roles with neutral family-stable surface identities."""
    digest = hashlib.sha256(f"surface-v1:{blueprint['id']}:{instance}:{configuration}".encode()).digest()
    people = ("Mira", "Omar", "Nia", "Ivo", "Zara", "Pavel", "Lena", "Tariq")
    requested_actor = people[digest[0] % len(people)]
    alternate_actor = people[(digest[0] + 1 + digest[1] % (len(people) - 1)) % len(people)]

    operation_values = (blueprint["operation"]["verb"], blueprint["other_operation"]["verb"])
    requested_operation = operation_values[digest[2] % 2]
    alternate_operation = operation_values[1 - digest[2] % 2]

    token = int.from_bytes(digest[3:7], "big") % 10000
    query_target = blueprint["target_template"].format(identifier=f"{token:04d}")
    alternate_target = blueprint["target_template"].format(identifier=f"{(token + 5003) % 10000:04d}")
    query_run, alternate_run = f"run-{token:04d}", f"run-{(token + 4001) % 10000:04d}"
    query_item, alternate_item = f"entry-{token:04d}", f"entry-{(token + 3001) % 10000:04d}"

    source_values = list(blueprint["sources"].values())
    untrusted_index = digest[7] % 3
    trusted_values = [value for index, value in enumerate(source_values) if index != untrusted_index]
    if digest[8] % 2:
        trusted_values.reverse()
    source_map = {
        blueprint["sources"]["primary"]: trusted_values[0],
        blueprint["sources"]["secondary"]: trusted_values[1],
        blueprint["sources"]["untrusted"]: source_values[untrusted_index],
    }
    role_maps = {
        "actor": {"Mira": requested_actor, "Omar": alternate_actor},
        "operation": {blueprint["operation"]["verb"]: requested_operation,
                      blueprint["other_operation"]["verb"]: alternate_operation},
        "target": {raw["query"]["target"]: query_target, "other-target": alternate_target},
        "run_id": {raw["query"]["run_id"]: query_run, "other-run": alternate_run},
        "reverses": {raw["query"]["run_id"]: query_run, "other-run": alternate_run},
        "id": {"item-a": query_item, "item-b": alternate_item},
        "source": source_map,
    }

    def rename(value, key=None):
        if isinstance(value, dict):
            return {name: rename(part, name) for name, part in value.items()}
        if isinstance(value, list):
            if key == "trusted_sources":
                return [source_map[part] for part in value]
            if key == "item_ids":
                return [role_maps["id"].get(part, part) for part in value]
            return [rename(part, key) for part in value]
        return role_maps.get(key, {}).get(value, value)

    renamed = rename(raw)
    raw.clear()
    raw.update(renamed)


def _noul_programs(category):
    programs = {
        "claim_vs_completion": ({"feature": "claim"}, {"feature": "completed"}, {"feature": "denied_claim"}),
        "permission_vs_execution": ({"feature": "approved"}, {"feature": "completed"}, {"feature": "forbidden"}),
        "unknown_vs_failure": ({"feature": "unknown"}, {"feature": "failed"}, {"feature": "invocation"}),
        "attribution_and_endorsement": ({"feature": "endorsement"}, {"feature": "claim"}, {"feature": "completed"}),
        "entity_binding": ({"feature": "completed"}, {"feature": "actor_seen"}, {"feature": "unknown"}),
        "action_binding": ({"feature": "completed"}, {"feature": "operation_seen"}, {"feature": "unknown"}),
        "temporal_scope": ({"feature": "active_earlier"}, {"feature": "active_now"}, {"feature": "current_known"}),
        "reversal_and_current_state": ({"feature": "completed"}, {"feature": "effective"}, {"feature": "reversed"}),
        "negation_and_quantifiers": ({"feature": "all_success"}, {"feature": "any_failure"}, {"feature": "item_unknown"}),
        "ordered_rubrics": ({"feature": "has_measurement"}, {"compare": {"left": {"feature": "measurement"}, "operator": ">=", "right": 50}}, {"feature": "claim"}),
    }
    return programs[category]


def _choice_program(category):
    # This explicit partition preserves epistemic unknown rather than treating it as false.
    labels = {"completed": {"feature": "completed"}, "failed": {"feature": "failed"},
              "unknown": {"feature": "unknown"}}
    if category in {"entity_binding", "action_binding"}:
        unresolved = {"all": [{"feature": "unknown"}, {"not": {"feature": "other_live_success"}}]}
        return {
            "exact_success": labels["completed"], "exact_failure": labels["failed"],
            "unrelated_success": {"all": [{"feature": "unknown"}, {"feature": "other_live_success"}]},
            "unknown": unresolved,
        }
    return labels


def _score_programs(configuration, category):
    """Five hundred public rubric programs with distinct evidence logic.

    A configuration combines one of twenty rule shapes with one of twenty-five
    feature/threshold forms.  The A record satisfies ``positive`` and not
    ``negative``; therefore its two supplied rubrics must produce a real score
    flip before any language is rendered.
    """
    pattern, shape = configuration % 25, configuration // 25
    low_measurement, _, cutoff = _numeric_scale(configuration)
    if category == "ordered_rubrics":
        boundary = {"compare": {"left": {"feature": "measurement"}, "operator": ">=", "right": low_measurement}}
        above_observation = {"compare": {"left": {"feature": "measurement"}, "operator": ">=", "right": cutoff}}
        if pattern % 2 == 0:
            return [True, boundary], [True, above_observation]
        high_observation = {"compare": {"left": {"feature": "measurement"}, "operator": ">=", "right": _numeric_scale(configuration)[1]}}
        return [True, boundary, high_observation], [True, above_observation, high_observation]
    left = ("claim", "approved", "intent", "has_measurement", "inventory_complete")[pattern % 5]
    right = ("completed", "failed", "reversed", "effective", "corroborated")[pattern // 5]
    if category == "reversal_and_current_state":
        left, right = "reversed", "effective"
    applies = {"compare": {"left": {"feature": "measurement"}, "operator": "<", "right": cutoff}}
    does_not_apply = {"compare": {"left": {"feature": "measurement"}, "operator": ">=", "right": cutoff}}
    yes, no = {"feature": left}, {"feature": right}
    positives = (
        yes, {"all": [yes, {"not": no}]}, {"any": [yes, no]}, {"not": no},
        {"at_least": {"count": 1, "of": [yes, {"not": no}]}},
        {"at_least": {"count": 2, "of": [yes, {"not": no}]}},
        {"all": [yes, applies]}, {"any": [{"all": [yes, applies]}, no]},
        {"not": {"all": [no, {"not": yes}]}}, {"any": [yes, {"not": no}]},
        {"all": [{"any": [yes, no]}, {"not": no}]}, {"at_least": {"count": 1, "of": [yes, applies]}},
        {"all": [yes, {"not": {"all": [no, applies]}}]}, {"any": [{"all": [yes, applies]}, {"not": no}]},
        {"not": {"at_least": {"count": 2, "of": [no, {"not": yes}]}}}, {"all": [{"not": no}, applies]},
        {"any": [yes, {"all": [no, applies]}]}, {"at_least": {"count": 2, "of": [yes, {"not": no}, applies]}},
        {"all": [yes, {"any": [{"not": no}, applies]}]}, {"not": {"all": [no, does_not_apply]}},
    )
    negatives = (
        no, {"all": [no, {"not": yes}]}, {"all": [no, does_not_apply]}, {"not": yes},
        {"at_least": {"count": 2, "of": [no, {"not": yes}]}},
        {"all": [no, {"not": yes}, does_not_apply]}, {"all": [no, does_not_apply]}, {"all": [no, {"not": yes}]},
        {"all": [no, {"not": yes}, does_not_apply]}, {"all": [no, {"not": yes}]},
        {"at_least": {"count": 1, "of": [no, does_not_apply]}}, {"all": [no, does_not_apply]},
        {"all": [no, {"not": yes}]}, {"all": [no, does_not_apply]},
        {"at_least": {"count": 2, "of": [no, does_not_apply]}}, {"all": [no, does_not_apply]},
        {"all": [no, {"not": yes}]}, {"all": [no, does_not_apply]},
        {"all": [no, {"not": yes}, does_not_apply]}, {"all": [no, does_not_apply]},
    )
    positive = positives[shape]
    negative = negatives[shape]
    mode = pattern % 4
    if mode == 0:
        return [True, positive], [True, negative]
    if mode == 1:
        return [True, positive, {"all": [positive, {"not": negative}]}], [True, negative, {"not": positive}]
    if mode == 2:
        return [True, {"any": [positive, negative]}], [True, {"all": [negative, {"not": positive}]}]
    return [True, positive, {"at_least": {"count": 2, "of": [positive, {"not": negative}, applies]}}], [True, negative, {"all": [negative, {"not": positive}]}]


def _public_evidence(raw, blueprint):
    q, source = raw["query"], blueprint["sources"]
    operations = {
        blueprint["operation"]["verb"]: blueprint["operation"],
        blueprint["other_operation"]["verb"]: blueprint["other_operation"],
    }
    messages = []
    for statement in raw["statements"]:
        operation = operations.get(statement["operation"], {"verb": statement["operation"], "past": statement["operation"]})
        values = {"speaker": statement.get("speaker", "Rin"), "actor": statement["actor"], "past": operation["past"], "verb": operation["verb"], "target": statement["target"],
                  "authority": "Review authority"}
        if statement.get("endorsed"):
            messages.append(blueprint["endorsement_template"].format(**values))
        elif statement["tense"] == "future":
            messages.append(blueprint["intent_template"].format(**values))
        elif statement["polarity"] == "negative":
            messages.append(blueprint["denial_template"].format(**values))
        else:
            messages.append(blueprint["claim_template"].format(**values))
    return {
        "heading": blueprint["evidence_heading"], "messages": messages,
        "approvals": [{key: row[key] for key in ("actor", "operation", "target", "run_id", "source", "time", "verified", "decision")} for row in raw["approvals"]],
        "tool_results": [{key: row[key] for key in ("actor", "operation", "target", "run_id", "source", "time", "verified", "mode", "outcome")} for row in raw["executions"]],
        "reversal_records": [{key: row[key] for key in ("actor", "operation", "target", "run_id", "source", "time", "verified", "mode", "outcome", "reverses")} for row in raw["reversals"]],
        "corroboration_records": copy.deepcopy(raw["corroborations"]),
        "inventory": copy.deepcopy(raw["items"]),
        "inventory_scope": {"item_ids": copy.deepcopy(raw["item_ids"]), "complete": raw["inventory_complete"]},
        "state_events": copy.deepcopy(raw["state_events"]), "query_time": raw["query_time"],
        "measurement": raw["measurement"],
    }


def _rules(raw, blueprint):
    q = raw["query"]
    return {
        "scope": {"actor": q["actor"], "operation": q["operation"], "target": q["target"], "run_id": q["run_id"], "inclusive_window": raw["window"]},
        "authority": {"trusted_sources": raw["trusted_sources"], "requires_verified": True},
        "interpretation": {
            "statements": "Claim, intent, denial, and endorsement predicates concern recorded messages matching the requested actor, operation, and target regardless of message trust or timing; they do not establish execution.",
            "invocation": "A matching invocation is any tool result with the requested actor, operation, target, and run ID inside the inclusive window, regardless of whether it proves an outcome.",
            "completion": "Only a trusted, verified, live matching tool result inside the inclusive window establishes completion; success and no-effect conflict leaves the outcome unknown.",
            "approval": "Use the latest trusted, verified decision inside the inclusive window that matches actor, operation, and target. Approval authorizes but does not execute; conflicting latest decisions are unknown.",
            "reversal": "A reversal must be trusted, verified, live, successful, match actor, operation, and target, occur no earlier than the established success, and name that success run ID in reverses.",
            "corroboration": "Corroboration must be trusted, verified, successful, match all exact bindings inside the inclusive window, and come from a different source no earlier than the established success.",
            "inventory": "Evaluate only the declared item IDs. An item is known only from one unconflicted verified status, and all-item claims additionally require the inventory scope to be explicitly complete.",
            "state": "At the query time or window end, use the latest admissible state event at or before that time for the requested target; conflicting latest values remain unknown.",
            "binding": "Actor, operation, target, and run ID are exact bindings. An otherwise admissible success with any different binding is unrelated evidence.",
            "missing": "Missing, inadmissible, timeout, and unresolved observations do not establish a negative world fact; they leave the corresponding record-support predicate unestablished.",
        },
    }


def _choice_definitions(choice):
    result = {}
    for label in choice:
        if label in {"completed", "exact_success"}:
            result[label] = "A matching trusted live result establishes success without a conflicting matching failure."
        elif label in {"failed", "exact_failure"}:
            result[label] = "A matching trusted live result establishes no effect without a conflicting matching success."
        elif label == "unrelated_success":
            result[label] = "The requested outcome remains unresolved, but an admissible trusted live success report concerns a different actor, operation, target, or run."
        else:
            result[label] = "The supplied records do not establish one of the other alternatives; unrelated records do not satisfy the exact scope."
    return result


_FEATURE_TEXT = {
    "claim": "the record contains an affirmative completed claim for the scoped action",
    "intent": "the record contains a future-intention statement for the scoped action",
    "denied_claim": "the record contains an explicit denial of completed action for the scoped action",
    "endorsement": "the record contains an authority endorsement of an affirmative completed claim",
    "approved": "the latest admissible matching approval decision grants permission",
    "forbidden": "the latest admissible matching approval decision denies permission",
    "completed": "the record establishes historical completion for the exact scoped action from admissible evidence",
    "failed": "the record establishes a no-effect outcome for the exact scoped action from admissible evidence",
    "unknown": "the admissible record establishes neither exact success nor exact no-effect",
    "invocation": "the record contains a matching invocation inside the inclusive window",
    "reversed": "the record establishes a matching successful reversal of the completed run",
    "effective": "the record establishes that the completed effect remains established at the window end",
    "actor_seen": "an execution record names the requested actor",
    "operation_seen": "an execution record names the requested operation",
    "active_earlier": "the latest admissible state evidence establishes active status at the query time",
    "active_now": "the latest admissible state evidence establishes active status at the window end",
    "current_known": "the latest admissible state evidence establishes a nonconflicting status at the window end",
    "all_success": "the explicitly complete inventory record establishes verified success for every declared item",
    "any_failure": "the inventory record establishes verified no-effect for at least one declared item",
    "item_unknown": "the inventory record leaves at least one declared item unresolved or the declared scope incomplete",
    "has_measurement": "a finite measurement is recorded",
    "inventory_complete": "the inventory scope is explicitly complete",
    "corroborated": "the record contains admissible corroboration from a second trusted source for matching success",
}


def _rule_text(rule):
    if type(rule) is bool:
        return "the level is always applicable" if rule else "the level is never applicable"
    operator, value = next(iter(rule.items()))
    if operator == "feature":
        return _FEATURE_TEXT.get(value, value.replace("_", " ") + " is established")
    if operator == "not":
        return f"(NOT ({_rule_text(value)}))"
    if operator == "all":
        return "(" + " AND ".join(_rule_text(part) for part in value) + ")"
    if operator == "any":
        return "(" + " OR ".join(_rule_text(part) for part in value) + ")"
    if operator == "at_least":
        return f"(AT LEAST {value['count']} OF [" + "; ".join(f"({_rule_text(part)})" for part in value["of"]) + "])"
    if operator == "compare":
        left, right = value["left"], value["right"]
        left_text = "the recorded measurement" if left.get("feature") == "measurement" else left.get("feature", "numeric value").replace("_", " ")
        return f"{left_text} is {value['operator']} {right}"
    _fail("score program uses an unsupported public rule")


def _score_levels(program):
    return ["Fallback level: no higher supplied condition applies."] + [
        f"Level {index}: apply when {_rule_text(rule)}."
        for index, rule in enumerate(program[1:], 1)
    ]


def _question_set(noul, choice, score_levels, blueprint, query):
    scope = (
        f"actor {query['actor']!r}, operation {query['operation']!r}, target {query['target']!r}, "
        f"and run {query['run_id']!r}"
    )
    questions = {}
    for qid, program in zip(("n1", "n2", "n3"), noul):
        predicate = _rule_text(program)
        questions[qid] = {
            "type": "noul",
            "instructions": f"{blueprint['question_prefix']} for requested {scope}, is this record-support predicate true: {predicate}?",
            "criteria": {
                "true": f"Yes: the supplied evidence and rules establish ({predicate}).",
                "false": f"No: the supplied evidence and rules do not establish ({predicate}); this does not assert the opposite world fact.",
            },
        }
    questions["c1"] = {"type": "choice", "instructions": f"{blueprint['question_prefix']} choose the one exhaustive outcome for the exact scoped request.", "criteria": _choice_definitions(choice)}
    questions["s1"] = {"type": "score", "instructions": f"{blueprint['question_prefix']} select the highest applicable level in this supplied rubric.", "criteria": score_levels}
    return questions


def _case(blueprint, instance, variant, evidence, rubric, configuration):
    raw = _record(blueprint, instance, evidence, configuration)
    features = derive_features(raw)
    noul = _noul_programs(blueprint["category"])
    choice = _choice_program(blueprint["category"])
    score_a, score_b = _score_programs(configuration, blueprint["category"])
    score = score_a if rubric == "A" else score_b
    choice_definitions, levels = _choice_definitions(choice), _score_levels(score)
    questions = _question_set(noul, choice, levels, blueprint, raw["query"])
    state = {"evidence": _public_evidence(raw, blueprint), "rules": _rules(raw, blueprint),
             "choice_definitions": copy.deepcopy(choice_definitions), "score_levels": copy.deepcopy(levels)}
    expected = {"n1": eval_rule(noul[0], features), "n2": eval_rule(noul[1], features), "n3": eval_rule(noul[2], features),
                "c1": select_choice(choice, features), "s1": select_level(score, features)}
    rationale = {
        qid: f"The oracle evaluated the public record-support predicate ({_rule_text(program)}) under the supplied scope and interpretation rules."
        for qid, program in zip(("n1", "n2", "n3"), noul)
    }
    rationale["c1"] = "The oracle selected the sole exhaustive outcome whose public definition is satisfied by the scoped observations."
    rationale["s1"] = "The oracle selected the highest applicable level from this case's supplied ordered rubric."
    family_id = f"scaling-v1/{blueprint['category']}/{instance:03d}/{blueprint['id']}"
    return {"id": f"{family_id}/{variant}", "family_id": family_id, "domain": blueprint["scenario_domain"], "category": blueprint["category"],
            "variant": variant, "layout": "structured_observations", "predicate_tags": dict(zip(QUESTION_IDS, (*[blueprint["category"]] * 3, "outcome", "supplied_rubric"))),
            "request": {"state": state, "questions": questions}, "expected": expected,
            "rationale": rationale,
            "private_programs": {"noul": noul, "choice": choice, "score": score}, "_raw": raw}


def _relation(identity, kind, left, right, axis, cases):
    a, b = cases[left[0]]["expected"][left[1]], cases[right[0]]["expected"][right[1]]
    return {"id": f"{identity}/{kind}/{left[0]}.{left[1]}-{right[0]}.{right[1]}", "kind": kind,
            "left": {"case_id": cases[left[0]]["id"], "question_id": left[1]}, "right": {"case_id": cases[right[0]]["id"], "question_id": right[1]},
            "contrast_axis": axis, "expected_equal": a == b}


def _differing_question_relations(identity, cases, count=2):
    """Select real same-record Noul contrasts instead of assuming fixed pairs differ."""
    candidates = []
    for case_index, case in enumerate(cases):
        for left_index, left_qid in enumerate(("n1", "n2", "n3")):
            for right_qid in ("n1", "n2", "n3")[left_index + 1:]:
                if case["expected"][left_qid] != case["expected"][right_qid]:
                    candidates.append((case_index, left_qid, right_qid))
    if len(candidates) < count:
        _fail("raw configuration does not provide enough valid question contrasts")
    return [
        _relation(identity, "question_contrast", (case_index, left_qid), (case_index, right_qid), "question", cases)
        for case_index, left_qid, right_qid in candidates[:count]
    ]


def _evidence_flip_relation(identity, cases):
    """Choose a Noul predicate whose truth actually changes with evidence."""
    for qid in ("n1", "n2", "n3"):
        if cases[0]["expected"][qid] != cases[2]["expected"][qid]:
            return _relation(identity, "flip", (0, qid), (2, qid), "evidence", cases)
    _fail("raw configuration does not provide a valid Noul evidence flip")


def _signature(raw, programs):
    """Fingerprint facts after erasing names and aliases, preserving their roles and relations."""
    query = raw["query"]
    trusted_sources = set(raw["trusted_sources"])
    aliases = {
        "actor": {query["actor"]: "ACTOR_REQUESTED"},
        "operation": {query["operation"]: "OPERATION_REQUESTED"},
        "target": {query["target"]: "TARGET_REQUESTED"},
        "run": {query["run_id"]: "RUN_REQUESTED"},
        "item": {identity: f"ITEM_{index}" for index, identity in enumerate(raw.get("item_ids", []))},
        "speaker": {},
        "trusted_source": {},
        "untrusted_source": {},
    }

    def alias(group, value, prefix):
        mapping = aliases[group]
        if value not in mapping:
            mapping[value] = f"{prefix}_{len(mapping)}"
        return mapping[value]

    def normalize(value, key=None):
        if isinstance(value, dict):
            return {name: normalize(part, name) for name, part in value.items()}
        if isinstance(value, list):
            if key == "trusted_sources":
                return ["TRUSTED_SOURCE"] * len(set(value))
            if key == "item_ids":
                return [alias("item", part, "ITEM_OTHER") for part in value]
            return [normalize(part, key) for part in value]
        if key == "source":
            group = "trusted_source" if value in trusted_sources else "untrusted_source"
            prefix = "SOURCE_TRUSTED" if value in trusted_sources else "SOURCE_UNTRUSTED"
            return alias(group, value, prefix)
        if key == "actor":
            return alias("actor", value, "ACTOR_OTHER")
        if key == "speaker":
            if value in aliases["actor"]:
                return aliases["actor"][value]
            return alias("speaker", value, "SPEAKER")
        if key == "operation":
            return alias("operation", value, "OPERATION_OTHER")
        if key == "target":
            return alias("target", value, "TARGET_OTHER")
        if key in {"run_id", "reverses"}:
            return alias("run", value, "RUN_OTHER")
        if key == "id":
            return alias("item", value, "ITEM_OTHER")
        return value

    normalized = normalize(raw)
    return hashlib.sha256(json.dumps({"facts": normalized, "programs": programs}, sort_keys=True).encode()).hexdigest()


def _configuration_for_instance(category, instance, seed):
    """Return a seeded 500-item permutation stratified in blocks of twenty."""
    digest = hashlib.sha256(f"scaling-v1:{seed}:{category}".encode()).digest()
    shape_offset = int.from_bytes(digest[:2], "big") % 20
    pattern_offset = int.from_bytes(digest[2:4], "big") % 25
    position = instance % 500
    block, slot = divmod(position, 20)
    shape = (slot + block + shape_offset) % 20
    pattern = (7 * shape + block + pattern_offset) % 25
    return shape * 25 + pattern


def build_family(blueprint, instance, seed=42):
    """Build a four-case semantic family from raw facts and oracle programs."""
    validate_blueprint(blueprint)
    if type(instance) is not int or instance < 0 or type(seed) is not int:
        _fail("instance and seed must be nonnegative/integer")
    # IDs and order depend only on authored blueprint/instance. The seed changes
    # a frozen within-category permutation whose first twenty cover every shape.
    configuration = _configuration_for_instance(blueprint["category"], instance, seed)
    cases = [_case(blueprint, instance, variant, evidence, rubric, configuration) for variant, evidence, rubric in
             (("v0", "A", "A"), ("v1", "A", "B"), ("v2", "B", "A"), ("v3", "B", "B"))]
    identity = cases[0]["family_id"]
    relations = [
        _relation(identity, "flip", (0, "s1"), (1, "s1"), "rubric", cases),
        _evidence_flip_relation(identity, cases),
        *_differing_question_relations(identity, cases),
        _relation(identity, "invariant", (0, "c1"), (1, "c1"), "rubric", cases),
    ]
    # Required evidence flip is a hard construction criterion, never relabelled after rendering.
    if not relations[0]["expected_equal"] is False or not relations[1]["expected_equal"] is False:
        _fail("raw configuration did not yield required score/evidence flips")
    provenance = {"raw_facts": [case.pop("_raw") for case in cases],
                  "programs": [case.pop("private_programs") for case in cases], "seed": seed,
                  "semantic_configuration": configuration, "evidence_shape": configuration // 25,
                  "rule_form": configuration % 25}
    provenance["semantic_signature"] = _signature(provenance["raw_facts"][0], provenance["programs"][0])
    family = {"id": identity, "category": blueprint["category"], "domain": blueprint["scenario_domain"], "cases": cases,
              "relations": relations, "private_provenance": provenance}
    validate_family(family)
    return family


def validate_family(family):
    """Check complete cases and that every relation states an actual oracle result."""
    if not isinstance(family, dict) or not isinstance(family.get("cases"), list) or len(family["cases"]) != 4:
        _fail("family needs four cases")
    cases = family["cases"]
    lookup = {(case["id"], qid): (case["request"]["questions"][qid], case["expected"][qid]) for case in cases for qid in QUESTION_IDS}
    required_evidence = {
        "heading", "messages", "approvals", "tool_results", "reversal_records", "corroboration_records",
        "inventory", "inventory_scope", "state_events", "query_time", "measurement",
    }
    for case in cases:
        if tuple(case.get("request", {}).get("questions", {})) != QUESTION_IDS or set(case.get("expected", {})) != set(QUESTION_IDS):
            _fail("case needs ordered n1/n2/n3/c1/s1")
        state = case["request"].get("state", {})
        if set(state) != {"evidence", "rules", "choice_definitions", "score_levels"}:
            _fail("state must expose only public evidence and definitions")
        if set(state["evidence"]) != required_evidence or set(state["evidence"]["inventory_scope"]) != {"item_ids", "complete"}:
            _fail("public evidence omits an oracle-relevant raw premise")
        if set(state["rules"]) != {"scope", "authority", "interpretation"}:
            _fail("public rules must expose scope, authority, and exact interpretation")
        if case["request"]["questions"]["c1"]["criteria"] != state["choice_definitions"] or case["request"]["questions"]["s1"]["criteria"] != state["score_levels"]:
            _fail("candidate definitions must exactly survive compilation")
        if set(case.get("rationale", {})) != set(QUESTION_IDS):
            _fail("case needs one rationale per question")
        for qid in ("n1", "n2", "n3"):
            if type(case["expected"][qid]) is not bool:
                _fail("Noul target must be hard boolean")
        choice = case["request"]["questions"]["c1"]
        if case["expected"]["c1"] not in choice["criteria"]:
            _fail("Choice target must be a declared exclusive choice")
        score = case["request"]["questions"]["s1"]
        if type(case["expected"]["s1"]) is not int or not 0 <= case["expected"]["s1"] < len(score["criteria"]):
            _fail("Score target must be a hard level index")
    seen = set()
    for relation in family.get("relations", []):
        if relation.get("id") in seen:
            _fail("relation IDs must be unique")
        seen.add(relation.get("id"))
        left, right = relation.get("left", {}), relation.get("right", {})
        try:
            left_q, left_y = lookup[left["case_id"], left["question_id"]]
            right_q, right_y = lookup[right["case_id"], right["question_id"]]
        except (KeyError, TypeError):
            _fail("relation endpoint is not a case question")
        def answer_space(question):
            if question["type"] == "noul":
                return {"false", "true"}
            if question["type"] == "score":
                return set(range(len(question["criteria"])))
            return set(question["criteria"])

        left_space, right_space = answer_space(left_q), answer_space(right_q)
        if left_q["type"] != right_q["type"] or left_space != right_space or relation.get("expected_equal") is not (left_y == right_y):
            _fail("relation does not match its oracle targets")
        kind = relation.get("kind")
        if kind in {"flip", "question_contrast"} and left_y == right_y:
            _fail("contrast relation must compare different targets")
        if kind == "invariant" and left_y != right_y:
            _fail("invariant relation must compare equal targets")
        if kind == "question_contrast" and (left["case_id"] != right["case_id"] or left["question_id"] == right["question_id"]):
            _fail("question relation must compare different questions on one record")
    required = {("flip", "rubric"), ("flip", "evidence"), ("invariant", "rubric")}
    present = {(row.get("kind"), row.get("contrast_axis")) for row in family.get("relations", [])}
    if not required <= present or sum(row.get("kind") == "question_contrast" for row in family.get("relations", [])) < 2:
        _fail("family lacks required coherent contrasts")
    for left_index, right_index in ((0, 1), (2, 3)):
        left, right = cases[left_index], cases[right_index]
        left_state, right_state = left["request"]["state"], right["request"]["state"]
        if any(left_state[key] != right_state[key] for key in ("evidence", "rules", "choice_definitions")):
            _fail("rubric pairs must preserve evidence, scope, and choice definitions")
        if any(left["request"]["questions"][qid] != right["request"]["questions"][qid] for qid in ("n1", "n2", "n3", "c1")):
            _fail("rubric pairs may change only the score question")
    for left_index, right_index in ((0, 2), (1, 3)):
        left, right = cases[left_index], cases[right_index]
        if left["request"]["state"]["rules"] != right["request"]["state"]["rules"] or left["request"]["questions"] != right["request"]["questions"]:
            _fail("evidence pairs must preserve scope, questions, and rubric")
    if len({len(case["request"]["questions"]["s1"]["criteria"]) for case in cases}) != 1:
        _fail("all family variants need the same Score answer space")
    from experiments.revised_metrics import validate_suite as validate_canonical_suite
    validate_canonical_suite({"cases": cases, "relations": family["relations"]})
    return family


_COVERAGE_FEATURES = (
    "completed", "failed", "unknown", "conflict", "corroborated", "inventory_complete", "item_unknown",
)


def _comparisons(expression):
    """Yield numeric comparison nodes from a private rule expression."""
    if isinstance(expression, list) or isinstance(expression, tuple):
        for part in expression:
            yield from _comparisons(part)
    elif isinstance(expression, dict):
        if set(expression) == {"compare"}:
            yield expression
        else:
            for part in expression.values():
                yield from _comparisons(part)


def _contains_feature(expression, name):
    if isinstance(expression, (list, tuple)):
        return any(_contains_feature(part, name) for part in expression)
    if isinstance(expression, dict):
        return expression == {"feature": name} or any(_contains_feature(part, name) for part in expression.values())
    return False


def _semantic_coverage(families):
    feature_counts = {name: Counter() for name in _COVERAGE_FEATURES}
    threshold_counts = Counter()
    threshold_positions = Counter()
    numeric_cutoffs = set()
    measurements = set()
    inventory_branches = Counter()
    numeric_flips = 0
    conflicting_unknown = 0
    conflicting_choice_unknown = 0
    corroborated_score_operand_true = 0
    for family in families:
        cases = family["cases"]
        raw_facts = family["private_provenance"]["raw_facts"]
        programs = family["private_provenance"]["programs"]
        for case, raw, program in zip(cases, raw_facts, programs):
            features = derive_features(raw)
            measurements.add(features["measurement"])
            for name in _COVERAGE_FEATURES:
                feature_counts[name][str(features[name]).lower()] += 1
            if features["conflict"] and features["unknown"]:
                conflicting_unknown += 1
                if case["expected"]["c1"] == "unknown":
                    conflicting_choice_unknown += 1
            if features["corroborated"] and _contains_feature(program["score"], "corroborated"):
                corroborated_score_operand_true += 1
            for comparison in _comparisons(program["score"]):
                threshold_counts[str(eval_rule(comparison, features)).lower()] += 1
                spec = comparison["compare"]
                observed = features[spec["left"]["feature"]]
                threshold = spec["right"]
                numeric_cutoffs.add(threshold)
                position = "below" if observed < threshold else "above" if observed > threshold else "boundary"
                threshold_positions[position] += 1

            declared = set(raw["item_ids"])
            present = {row["id"] for row in raw["items"]}
            if declared - present:
                inventory_branches["missing_item"] += 1
            if any(row["id"] in declared and row.get("verified") is False for row in raw["items"]):
                inventory_branches["unverified_item"] += 1
            if any(
                len({row["status"] for row in raw["items"] if row["id"] == item_id and row.get("verified") is True}) > 1
                for item_id in declared
            ):
                inventory_branches["conflicting_item"] += 1
            if raw["inventory_complete"] is False:
                inventory_branches["incomplete_inventory"] += 1

        if (
            family["category"] == "ordered_rubrics"
            and cases[0]["expected"]["s1"] != cases[1]["expected"]["s1"]
            and all(set(rule) == {"compare"} for program in programs[:2] for rule in program["score"][1:])
        ):
            numeric_flips += 1

    fact_signatures = Counter(
        _signature(family["private_provenance"]["raw_facts"][0], {}) for family in families
    )
    program_signatures = Counter(
        hashlib.sha256(json.dumps(family["private_provenance"]["programs"][0], sort_keys=True).encode()).hexdigest()
        for family in families
    )
    joint_signatures = Counter(family["private_provenance"]["semantic_signature"] for family in families)
    return {
        "feature_truths": {name: {value: feature_counts[name][value] for value in ("true", "false")} for name in _COVERAGE_FEATURES},
        "numeric_threshold_truths": {value: threshold_counts[value] for value in ("true", "false")},
        "numeric_threshold_positions": {value: threshold_positions[value] for value in ("below", "boundary", "above")},
        "distinct_numeric_cutoffs": len(numeric_cutoffs),
        "distinct_measurements": len(measurements),
        "same_evidence_numeric_score_flips": numeric_flips,
        "conflicting_outcome_unknown": conflicting_unknown,
        "conflicting_choice_unknown": conflicting_choice_unknown,
        "corroborated_score_operand_true": corroborated_score_operand_true,
        "inventory_branches": {name: inventory_branches[name] for name in (
            "missing_item", "unverified_item", "conflicting_item", "incomplete_inventory",
        )},
        "normalized_semantic_fact_signatures": len(fact_signatures),
        "semantic_program_signatures": len(program_signatures),
        "normalized_joint_fact_program_signatures": len(joint_signatures),
    }


def _require_semantic_coverage(coverage, prefix_size):
    missing = []
    if not all(coverage["numeric_threshold_truths"].values()):
        missing.append("numeric threshold truth values")
    if not all(coverage["numeric_threshold_positions"].values()):
        missing.append("numeric threshold below/boundary/above positions")
    if coverage["distinct_numeric_cutoffs"] < 2 or coverage["distinct_measurements"] < 2:
        missing.append("variable numeric cutoffs and observations")
    if not coverage["same_evidence_numeric_score_flips"]:
        missing.append("same-evidence numeric Score flips")
    if not coverage["conflicting_outcome_unknown"]:
        missing.append("conflicting outcomes adjudicated unknown")
    if not coverage["conflicting_choice_unknown"]:
        missing.append("conflicting outcomes with unknown Choice target")
    if not coverage["corroborated_score_operand_true"]:
        missing.append("true corroboration used by a Score rule")
    for name, counts in coverage["feature_truths"].items():
        if not all(counts.values()):
            missing.append(f"{name} true/false")
    for name, count in coverage["inventory_branches"].items():
        if not count:
            missing.append(name)
    if missing:
        _fail(f"prefix {prefix_size} lacks semantic coverage: {', '.join(missing)}")


def build_training_suite(blueprints, families_per_category=500, seed=42):
    """Create the canonical interleaved ordering; 200/1000/5000 are nested prefixes."""
    if type(families_per_category) is not int or not 0 < families_per_category <= 500:
        _fail("families_per_category must be between 1 and 500")
    by_category = {category: [] for category in CATEGORIES}
    ids = set()
    for blueprint in blueprints:
        validate_blueprint(blueprint)
        if blueprint["id"] in ids:
            _fail("blueprint IDs must be unique")
        ids.add(blueprint["id"])
        by_category[blueprint["category"]].append(blueprint)
    if any(len(by_category[category]) != 20 for category in CATEGORIES):
        _fail("exactly 20 blueprints per category are required")
    families = []
    for round_index in range(families_per_category):
        for category in CATEGORIES:
            blueprint = by_category[category][round_index % 20]
            # Every twenty rounds advances a new semantic configuration cycle.
            instance = round_index
            families.append(build_family(blueprint, instance, seed))
    ordered = [family["id"] for family in families]
    if len(ordered) != len(set(ordered)):
        _fail("family IDs must be unique")
    configuration_counts = {category: len({family["private_provenance"]["semantic_configuration"] for family in families if family["category"] == category}) for category in CATEGORIES}
    signature_counts = Counter(family["private_provenance"]["semantic_signature"] for family in families)
    if families_per_category == 500 and len(signature_counts) != 5000:
        _fail("full scaling population has duplicate normalized fact/program signatures")
    prefix_coverage = {}
    semantic_coverage = {}
    for size in (200, 1000, 5000):
        if size > len(families):
            continue
        prefix = families[:size]
        prefix_coverage[str(size)] = {
            "evidence_shapes_by_category": {
                category: len({family["private_provenance"]["evidence_shape"] for family in prefix if family["category"] == category})
                for category in CATEGORIES
            },
            "rule_forms_by_category": {
                category: len({family["private_provenance"]["rule_form"] for family in prefix if family["category"] == category})
                for category in CATEGORIES
            },
        }
        semantic_coverage[str(size)] = _semantic_coverage(prefix)
        _require_semantic_coverage(semantic_coverage[str(size)], size)
    from experiments.revised_metrics import validate_suite as validate_canonical_suite
    validate_canonical_suite({
        "cases": [case for family in families for case in family["cases"]],
        "relations": [relation for family in families for relation in family["relations"]],
    })
    instances_per_blueprint = families_per_category // 20 if families_per_category % 20 == 0 else None
    limitation = "The full population is designed as 200 authored language blueprints paired with 25 semantic instances each, not 5,000 independent language templates."
    if instances_per_blueprint != 25:
        limitation += f" This in-memory prefix contains {instances_per_blueprint!r} complete semantic instances per blueprint."
    total_coverage = _semantic_coverage(families)
    surface_invariance = {
        "counted_as_semantic_diversity": False,
        "requested_actor_forms": len({family["private_provenance"]["raw_facts"][0]["query"]["actor"] for family in families}),
        "requested_operation_forms": len({family["private_provenance"]["raw_facts"][0]["query"]["operation"] for family in families}),
        "trusted_source_sets": len({tuple(sorted(family["private_provenance"]["raw_facts"][0]["trusted_sources"])) for family in families}),
        "description": "Family-stable neutral actor, operation, target, run, item, and source-role permutations are surface invariances excluded by normalized signatures.",
    }
    suite = {"contract": "scaling-data-factory-v1", "families": families, "ordered_family_ids": ordered,
             "nested_prefixes": {str(size): ordered[:size] for size in (200, 1000, 5000) if size <= len(ordered)},
             "diversity": {"semantic_configurations_per_category": max(configuration_counts.values()),
                           "configuration_counts_by_category": configuration_counts,
                           "evidence_shapes": len({family["private_provenance"]["evidence_shape"] for family in families}),
                           "normalized_semantic_fact_signatures": total_coverage["normalized_semantic_fact_signatures"],
                           "semantic_program_signatures": total_coverage["semantic_program_signatures"],
                           "normalized_joint_fact_program_signatures": len(signature_counts),
                           "normalized_semantic_fact_program_families": len(signature_counts),
                           "authored_language_blueprints": len(blueprints),
                           "semantic_instances_per_blueprint": instances_per_blueprint,
                           "surface_invariance": surface_invariance,
                           "prefix_configuration_coverage": prefix_coverage,
                           "semantic_coverage_by_prefix": semantic_coverage,
                           "normalized_signature_counts": dict(sorted(signature_counts.items()))},
             "metadata": {"seed": seed, "categories": list(CATEGORIES), "families_per_category": families_per_category,
                          "limitations": limitation}}
    return suite


def training_bundles(suite):
    """Return complete 20-example training groups without private audit metadata."""
    families = suite.get("families") if isinstance(suite, dict) else None
    if not isinstance(families, list):
        _fail("suite needs families")
    bundles = []
    for family in families:
        validate_family(family)
        examples = []
        for case in family["cases"]:
            for qid in QUESTION_IDS:
                question, expected = case["request"]["questions"][qid], case["expected"][qid]
                target = {"truth": expected} if question["type"] == "noul" else {"choice": expected} if question["type"] == "choice" else {"level_index": expected}
                examples.append({"state": copy.deepcopy(case["request"]["state"]), "question": copy.deepcopy(question), "target": target,
                                 "case_id": case["id"], "question_id": qid})
        bundles.append({"id": family["id"], "group_id": family["id"], "examples": examples, "relations": [],
                        "provenance": {"dataset": "contrast-scaling-v1", "assigned_split": "train", "category": family["category"]}})
    return bundles
