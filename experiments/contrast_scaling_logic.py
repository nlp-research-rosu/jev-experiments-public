"""Private raw-fact oracle and explicit supplied-rule evaluator for scaling data.

Feature values are annotation machinery. They must never be copied into model
state. Dataset renderers expose original observations and public rule definitions.
"""

import math
from collections import defaultdict


def _matches(row, query, *, run=True):
    keys = ("actor", "operation", "target", "run_id") if run else ("actor", "operation", "target")
    return all(key not in query or row.get(key) == query[key] for key in keys)


def derive_features(record):
    query = record["query"]
    start, end = record["window"]
    if start > end:
        raise ValueError("window endpoints reversed")
    trusted = set(record["trusted_sources"])

    def within(row):
        return type(row.get("time")) in (int, float) and start <= row["time"] <= end

    def authentic(row):
        return row.get("verified") is True and row.get("source") in trusted

    statements = [s for s in record.get("statements", []) if _matches(s, query, run=False)]
    claims = [s for s in statements if s.get("tense") == "completed" and s.get("polarity") == "positive"]
    approvals = [s for s in record.get("approvals", []) if _matches(s, query, run=False) and within(s) and authentic(s)]
    if approvals:
        latest = max(s["time"] for s in approvals)
        decisions = {s.get("decision") for s in approvals if s["time"] == latest}
    else:
        decisions = set()
    invocations = [s for s in record.get("executions", []) if _matches(s, query) and within(s)]
    live = [s for s in invocations if authentic(s) and s.get("mode") == "live"]
    # These are existence-of-report predicates, not a claim that conflicting
    # reports establish the physical success of some other action.
    all_live_success = [
        s
        for s in record.get("executions", [])
        if within(s) and authentic(s) and s.get("mode") == "live" and s.get("outcome") == "success"
    ]
    success = [s for s in live if s.get("outcome") == "success"]
    failure = [s for s in live if s.get("outcome") == "no_effect"]
    completed, failed = bool(success) and not failure, bool(failure) and not success
    reversals = [
        s
        for s in record.get("reversals", [])
        if completed
        and _matches(s, query, run=False)
        and within(s)
        and authentic(s)
        and s.get("mode") == "live"
        and s.get("outcome") == "success"
        and any(s.get("reverses") == e["run_id"] and s["time"] >= e["time"] for e in success)
    ]
    corroborations = [
        s
        for s in record.get("corroborations", [])
        if completed
        and _matches(s, query)
        and within(s)
        and authentic(s)
        and s.get("outcome") == "success"
        and any(s.get("source") != e["source"] and s["time"] >= e["time"] for e in success)
    ]
    item_rows = defaultdict(list)
    for row in record.get("items", []):
        item_rows[row["id"]].append(row)
    scope = list(dict.fromkeys(record.get("item_ids", list(item_rows))))
    statuses = []
    for identity in scope:
        known = {
            s["status"]
            for s in item_rows[identity]
            if s.get("verified") is True and s.get("status") in {"success", "no_effect"}
        }
        statuses.append(next(iter(known)) if len(known) == 1 else "unknown")
    total = len(scope)
    positive, negative = statuses.count("success"), statuses.count("no_effect")
    missing = statuses.count("unknown")
    complete_inventory = record.get("inventory_complete") is True
    events = [
        s
        for s in record.get("state_events", [])
        if s.get("target") == query.get("target") and authentic(s) and type(s.get("time")) in (int, float)
    ]

    def active_at(time):
        prior = [s for s in events if s["time"] <= time]
        if not prior:
            return None
        latest = max(s["time"] for s in prior)
        reported = [s.get("active") for s in prior if s["time"] == latest]
        if any(type(value) is not bool for value in reported):
            return None
        values = set(reported)
        return next(iter(values)) if len(values) == 1 else None

    now = active_at(end)
    earlier = active_at(record.get("query_time", start))
    measurement = record.get("measurement")
    if measurement is not None and (type(measurement) not in (int, float) or not math.isfinite(measurement)):
        raise ValueError("measurement must be a finite numeric observation")
    return {
        "intent": any(s.get("tense") == "future" and s.get("polarity") == "positive" for s in statements),
        "claim": bool(claims),
        "denied_claim": any(s.get("tense") == "completed" and s.get("polarity") == "negative" for s in statements),
        "endorsement": any(s.get("endorsed") is True for s in claims),
        "approved": decisions == {"allow"},
        "forbidden": decisions == {"deny"},
        "approval_unknown": decisions not in ({"allow"}, {"deny"}),
        "invocation": bool(invocations),
        "completed": bool(completed),
        "failed": bool(failed),
        "unknown": not completed and not failed,
        "conflict": bool(success) and bool(failure),
        "success_report": any(s.get("outcome") == "success" for s in invocations),
        "any_live_success": bool(all_live_success),
        "other_live_success": any(not _matches(s, query) for s in all_live_success),
        "failure_report": any(s.get("outcome") == "no_effect" for s in invocations),
        "reversed": bool(reversals),
        "effective": bool(completed) and not reversals,
        "corroborated": bool(corroborations),
        "actor_seen": any(s.get("actor") == query.get("actor") for s in record.get("executions", [])),
        "operation_seen": any(s.get("operation") == query.get("operation") for s in record.get("executions", [])),
        "target_seen": any(s.get("target") == query.get("target") for s in record.get("executions", [])),
        "run_seen": any(s.get("run_id") == query.get("run_id") for s in record.get("executions", [])),
        "inventory_complete": complete_inventory,
        "item_count": total,
        "success_count": positive,
        "failure_count": negative,
        "unknown_count": missing,
        "all_success": complete_inventory and positive == total,
        "all_failed": complete_inventory and negative == total,
        "any_success": positive > 0,
        "any_failure": negative > 0,
        "item_unknown": missing > 0 or not complete_inventory,
        "active_now": now is True,
        "inactive_now": now is False,
        "current_known": now is not None,
        "active_earlier": earlier is True,
        "inactive_earlier": earlier is False,
        "earlier_known": earlier is not None,
        "ever_active": any(s.get("active") is True and s["time"] <= end for s in events),
        "has_measurement": measurement is not None,
        "measurement": measurement,
    }


def _value(expression, features):
    if isinstance(expression, dict) and set(expression) == {"feature"}:
        name = expression["feature"]
        if name not in features:
            raise ValueError(f"unknown feature {name}")
        return features[name]
    if type(expression) in (int, float) and math.isfinite(expression):
        return expression
    raise ValueError("invalid numeric operand")


def eval_rule(expression, features):
    if type(expression) is bool:
        return expression
    if not isinstance(expression, dict) or len(expression) != 1:
        raise ValueError("rule requires one known operator")
    operator, value = next(iter(expression.items()))
    if operator == "feature":
        result = _value(expression, features)
        if type(result) is not bool:
            raise ValueError("numeric feature requires an explicit comparison")
        return result
    if operator == "not":
        return not eval_rule(value, features)
    if operator in {"all", "any"}:
        if not isinstance(value, list):
            raise ValueError("logical operator needs a list")
        return (all if operator == "all" else any)(eval_rule(v, features) for v in value)
    if operator == "at_least":
        if (
            not isinstance(value, dict)
            or set(value) != {"count", "of"}
            or type(value["count"]) is not int
            or value["count"] < 0
            or not isinstance(value["of"], list)
        ):
            raise ValueError("invalid count rule")
        return sum(eval_rule(v, features) for v in value["of"]) >= value["count"]
    if operator == "compare":
        if not isinstance(value, dict) or set(value) != {"left", "operator", "right"}:
            raise ValueError("comparison needs explicit operands")
        a, b = _value(value["left"], features), _value(value["right"], features)
        if type(a) not in (int, float) or type(b) not in (int, float):
            raise ValueError("missing/non-numeric comparison operand")
        if value["operator"] == ">=":
            return a >= b
        if value["operator"] == ">":
            return a > b
        if value["operator"] == "<=":
            return a <= b
        if value["operator"] == "<":
            return a < b
        if value["operator"] == "==":
            return a == b
        if value["operator"] == "!=":
            return a != b
        raise ValueError("unknown comparison operator")
    raise ValueError(f"unknown rule operator {operator}")


def select_level(level_rules, features):
    if not isinstance(level_rules, list) or not level_rules:
        raise ValueError("nonempty ordered rubric required")
    satisfied = [i for i, expression in enumerate(level_rules) if eval_rule(expression, features)]
    if not satisfied:
        raise ValueError("rubric supplies no applicable level")
    return max(satisfied)


def select_choice(option_rules, features):
    if not isinstance(option_rules, dict) or not option_rules:
        raise ValueError("nonempty option rules required")
    satisfied = [name for name, expression in option_rules.items() if eval_rule(expression, features)]
    if len(satisfied) != 1:
        raise ValueError("choice rules must establish exactly one alternative")
    return satisfied[0]
