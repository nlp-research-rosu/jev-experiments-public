"""Raw-fact seeds and a deterministic oracle for paired-language-v1 training.

The 20-workflow catalog is reused across ten contrast categories.  Each
workflow/category pair has its own scope identifier and four raw-evidence
variants.  Targets are computed only after the raw facts exist; no target or
adjudicated summary is rendered into model state.

This module performs no writes through ``build_training_seeds``.  Its CLI will
export seeds only when given an existing held-out evaluation freeze manifest.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from experiments.revised_metrics import validate_suite
from openjev.judgments import compile_request

STUDY = "paired-language-v1"
VARIANTS = ("v0", "v1", "v2", "v3")
QUESTION_IDS = ("n1", "n2", "n3", "c1", "s1")
CATEGORIES = (
    ("claim_vs_completion", "evidence"),
    ("permission_vs_execution", "policy"),
    ("unknown_vs_failure", "epistemic"),
    ("attribution_and_endorsement", "evidence"),
    ("entity_binding", "binding"),
    ("action_binding", "binding"),
    ("temporal_scope", "temporal"),
    ("reversal_and_current_state", "temporal"),
    ("negation_and_quantifiers", "logic"),
    ("ordered_rubrics", "rubric"),
)

WORKFLOWS = (
    ("Avery Chen", "publish", "the Northwind safety bulletin", "NW-SAFETY-041"),
    ("Beatriz Okafor", "release", "the Harbor payroll batch", "HB-PAY-112"),
    ("Carlos Ruiz", "rotate", "the Cedar signing key", "CD-KEY-309"),
    ("Dina Shah", "close", "the Meridian incident ticket", "MR-INC-077"),
    ("Elliot Park", "approve", "the Juniper supplier amendment", "JP-SUP-208"),
    ("Farah Ali", "migrate", "the Atlas customer index", "AT-INDEX-521"),
    ("Gavin Bell", "quarantine", "the Solstice inventory lot", "SO-LOT-184"),
    ("Hana Ito", "activate", "the Ember emergency roster", "EM-ROSTER-034"),
    ("Imani Cole", "reconcile", "the Willow settlement ledger", "WL-LEDGER-612"),
    ("Jonas Meyer", "issue", "the Plover access credential", "PL-CRED-459"),
    ("Kira Nwosu", "archive", "the Quartz retention record", "QZ-ARCH-253"),
    ("Luca Rossi", "route", "the Tidal support mailbox", "TD-MAIL-098"),
    ("Mina Patel", "restore", "the Birch backup snapshot", "BR-BACKUP-710"),
    ("Noah Kim", "escalate", "the Cobalt fraud review", "CB-REVIEW-166"),
    ("Omar Haddad", "revoke", "the Falcon deployment grant", "FC-GRANT-372"),
    ("Priya Das", "seal", "the Lumen audit package", "LM-AUDIT-287"),
    ("Quinn Walker", "transfer", "the Redwood equipment custody", "RW-CUSTODY-943"),
    ("Rosa Silva", "schedule", "the Nimbus maintenance window", "NB-MAINT-505"),
    ("Samira Khan", "verify", "the Orion identity proof", "OR-ID-128"),
    ("Theo Grant", "suspend", "the Vale partner connection", "VL-PARTNER-664"),
)

WINDOW = {
    "start": "2026-01-10T09:00:00Z",
    "end": "2026-01-10T17:00:00Z",
}
ACTION_CHOICE_CRITERIA = {
    "confirmed": "Authenticated real evidence establishes successful completion of the exact requested action in the inclusive window, with no conflicting result and no successful linked matching reversal.",
    "no_effect": "Authenticated real evidence explicitly establishes a guaranteed failure of the exact requested action in the inclusive window.",
    "reversed": "A confirmed exact completion occurred and a linked authenticated matching reversal succeeded before the window ended.",
    "unknown": "The exact requested result is missing, unverified, timed out, conflicting, or supported only by unrelated events.",
}
ACTION_SCORE_CRITERIA = [
    "0: no invocation matching actor, operation, target, and inclusive window is recorded.",
    "1: a matching invocation is recorded, but successful completion is not confirmed.",
    "2: authenticated real matching successful completion is confirmed, including a completion later reversed.",
    "3: authenticated real matching successful completion in the inclusive window is confirmed, and a separate authenticated source corroborates that exact invocation.",
]
QUANTIFIER_CHOICE_CRITERIA = {
    "all_success": "All three named items have authenticated success observations.",
    "all_failed": "All three named items have authenticated no-effect observations.",
    "mixed": "Every item is known and the set contains both success and no-effect observations.",
    "unknown": "At least one named item remains unknown.",
}
QUANTIFIER_SCORE_CRITERIA = [
    "0: zero named items have authenticated success observations.",
    "1: exactly one named item has an authenticated success observation.",
    "2: exactly two named items have authenticated success observations.",
    "3: all three named items have authenticated success observations.",
]
RUBRIC_CHOICE_CRITERIA = {
    "unknown": "No authenticated in-window impact observation establishes an issue.",
    "cosmetic": "The highest established issue is visual or cosmetic and affects no functionality.",
    "degraded": "The highest established issue degrades named functionality and has a working workaround.",
    "blocking": "The highest established issue blocks named functionality and has no workaround.",
}
RUBRIC_SCORE_CRITERIA = [
    "0: no issue is established by an authenticated in-window impact observation.",
    "1: the highest established issue is cosmetic and affects no functionality.",
    "2: the highest established issue degrades functionality and a workaround is available.",
    "3: the highest established issue blocks functionality and no workaround is available.",
]


def workflow_catalog():
    """Return the shared concrete workflow ontology."""
    return [
        {"actor": actor, "operation": operation, "target": target, "base_scope_id": scope_id}
        for actor, operation, target, scope_id in WORKFLOWS
    ]


def _context(category, workflow):
    actor, operation, target, base_scope_id = workflow
    category_scope = hashlib.sha256(category.encode()).hexdigest()[:8]
    return {
        "scope_id": f"{base_scope_id}-{category_scope}",
        "requested_action": {"actor": actor, "operation": operation, "target": target},
        "inclusive_window": dict(WINDOW),
    }


def _policy(category):
    if category == "negation_and_quantifiers":
        return {
            "item_rule": "Count only authenticated observations for the three named items.",
            "unknown_rule": "Any named item without authenticated success or no-effect evidence remains unknown.",
        }
    if category == "ordered_rubrics":
        return {
            "highest_applicable_rule": "Apply the highest level supported by an authenticated impact observation inside the inclusive window.",
            "response_rule": "A response is timely only when authenticated and recorded inside the inclusive window.",
        }
    return {
        "binding_rule": "An invocation matches only when actor, operation, target, and inclusive-window time all match exactly.",
        "completion_rule": "Only an authenticated real invocation with result success confirms completion.",
        "no_effect_rule": "Only an authenticated real matching invocation with result guaranteed_failure establishes no effect; timeout, missing, unverified, conflict, and unrelated events are unknown.",
        "reversal_rule": "A reversal changes current effect only when it is authenticated, real, successful, matching, and linked to a confirmed invocation.",
        "endorsement_rule": "The endorsed field records whether the reviewing authority explicitly adopts the attributed affirmative completion statement; it is independent of whether execution happened.",
        "stage_rule": "Select the highest established evidence stage; corroborated completion uses the top stage even though ordinary completion is also established.",
    }


def _statement(context, *, speaker, endorsed=False, polarity="affirmative", tense="completed"):
    requested = context["requested_action"]
    return {
        "speaker": speaker,
        "subject_actor": requested["actor"],
        "operation": requested["operation"],
        "target": requested["target"],
        "tense": tense,
        "polarity": polarity,
        "endorsed": endorsed,
    }


def _approval(context, decision):
    return {**context["requested_action"], "decision": decision, "authority": "Change Control"}


def _invocation(
    context,
    result,
    *,
    invocation_id="run-1",
    actor=None,
    operation=None,
    target=None,
    time="2026-01-10T12:00:00Z",
    authenticated=True,
    mode="real",
):
    requested = context["requested_action"]
    return {
        "id": invocation_id,
        "actor": requested["actor"] if actor is None else actor,
        "operation": requested["operation"] if operation is None else operation,
        "target": requested["target"] if target is None else target,
        "time": time,
        "mode": mode,
        "result": result,
        "authenticated": authenticated,
    }


def _corroboration(invocation_id="run-1"):
    return {"invocation_id": invocation_id, "source": "independent audit log", "authenticated": True}


def _action_record():
    return {
        "attributed_statements": [],
        "approval": None,
        "invocations": [],
        "reversals": [],
        "corroborations": [],
    }


def _build_action_record(category, context, variant):
    record = _action_record()
    actor = context["requested_action"]["actor"]

    if category == "claim_vs_completion":
        if variant in {"v0", "v1"}:
            record["attributed_statements"] = [_statement(context, speaker=actor)]
        elif variant == "v3":
            record["attributed_statements"] = [_statement(context, speaker=actor, endorsed=True)]
        if variant == "v1":
            record["invocations"] = [_invocation(context, "timeout")]
        elif variant in {"v2", "v3"}:
            record["invocations"] = [_invocation(context, "success")]
        if variant == "v3":
            record["corroborations"] = [_corroboration()]

    elif category == "permission_vs_execution":
        record["approval"] = _approval(context, "granted" if variant in {"v0", "v3"} else "denied")
        if variant == "v1":
            record["invocations"] = [_invocation(context, "timeout")]
        elif variant in {"v2", "v3"}:
            record["invocations"] = [_invocation(context, "success")]
        if variant == "v3":
            record["corroborations"] = [_corroboration()]

    elif category == "unknown_vs_failure":
        if variant == "v1":
            record["invocations"] = [_invocation(context, "conflict")]
        elif variant == "v2":
            record["invocations"] = [_invocation(context, "guaranteed_failure")]
        elif variant == "v3":
            record["invocations"] = [_invocation(context, "success")]
            record["corroborations"] = [_corroboration()]

    elif category == "attribution_and_endorsement":
        if variant in {"v0", "v1", "v3"}:
            record["attributed_statements"] = [
                _statement(context, speaker="Riley, outside contractor", endorsed=variant == "v1")
            ]
        else:
            record["attributed_statements"] = [_statement(context, speaker="Operations Authority", endorsed=True)]
        if variant == "v1":
            record["invocations"] = [_invocation(context, "timeout")]
        elif variant in {"v2", "v3"}:
            record["invocations"] = [_invocation(context, "success")]
        if variant == "v3":
            record["corroborations"] = [_corroboration()]

    elif category == "entity_binding":
        requested = context["requested_action"]
        if variant == "v0":
            record["invocations"] = [_invocation(context, "success", actor="Morgan Lee", target=requested["target"])]
        elif variant == "v1":
            record["invocations"] = [
                _invocation(context, "timeout", target=f"a different target from {requested['target']}")
            ]
        elif variant == "v2":
            record["invocations"] = [_invocation(context, "guaranteed_failure")]
        else:
            record["invocations"] = [_invocation(context, "success")]
            record["corroborations"] = [_corroboration()]

    elif category == "action_binding":
        wrong_operation = f"inspect instead of {context['requested_action']['operation']}"
        if variant == "v0":
            record["invocations"] = [_invocation(context, "success", operation=wrong_operation)]
        elif variant == "v1":
            record["invocations"] = [_invocation(context, "guaranteed_failure", operation=wrong_operation)]
        elif variant == "v2":
            record["invocations"] = [_invocation(context, "timeout")]
        else:
            record["invocations"] = [_invocation(context, "success")]
            record["corroborations"] = [_corroboration()]

    elif category == "temporal_scope":
        if variant == "v0":
            record["invocations"] = [_invocation(context, "success", time="2026-01-10T08:59:59Z")]
        elif variant == "v1":
            record["invocations"] = [_invocation(context, "timeout", time=WINDOW["start"])]
        elif variant == "v2":
            record["invocations"] = [_invocation(context, "success", time=WINDOW["end"])]
        else:
            record["invocations"] = [_invocation(context, "success")]
            record["corroborations"] = [_corroboration()]

    elif category == "reversal_and_current_state":
        if variant == "v0":
            record["reversals"] = [
                {
                    "linked_invocation_id": "unrelated-run",
                    **context["requested_action"],
                    "target": f"a different target from {context['requested_action']['target']}",
                    "time": "2026-01-10T15:00:00Z",
                    "mode": "real",
                    "result": "success",
                    "authenticated": True,
                }
            ]
        elif variant == "v1":
            record["invocations"] = [_invocation(context, "timeout")]
        elif variant in {"v2", "v3"}:
            record["invocations"] = [_invocation(context, "success")]
        if variant == "v3":
            record["corroborations"] = [_corroboration()]
            record["reversals"] = [
                {
                    "linked_invocation_id": "run-1",
                    **context["requested_action"],
                    "time": "2026-01-10T16:00:00Z",
                    "mode": "real",
                    "result": "success",
                    "authenticated": True,
                }
            ]
    else:
        raise ValueError(f"unsupported action category: {category}")
    return record


def _build_quantifier_record(variant):
    outcomes = {
        "v0": ("success", "unknown", "no_effect"),
        "v1": ("success", "success", "success"),
        "v2": ("no_effect", "no_effect", "no_effect"),
        "v3": ("success", "success", "no_effect"),
    }[variant]
    return {
        "item_observations": [
            {"item": item, "outcome": outcome, "authenticated": True, "source": "item execution log"}
            for item, outcome in zip(("alpha", "beta", "gamma"), outcomes, strict=True)
        ]
    }


def _build_rubric_record(variant):
    records = {
        "v0": {"impact_observations": [], "responses": []},
        "v1": {
            "impact_observations": [
                {
                    "time": "2026-01-10T10:00:00Z",
                    "authenticated": True,
                    "impact": "visual_only",
                    "affected_functionality": [],
                    "workaround_available": False,
                }
            ],
            "responses": [
                {
                    "responder": "Support Desk",
                    "action": "acknowledged the visual defect",
                    "time": "2026-01-10T10:15:00Z",
                    "authenticated": True,
                }
            ],
        },
        "v2": {
            "impact_observations": [
                {
                    "time": "2026-01-10T11:00:00Z",
                    "authenticated": True,
                    "impact": "functionality_degraded",
                    "affected_functionality": ["primary workflow"],
                    "workaround_available": True,
                }
            ],
            "responses": [],
        },
        "v3": {
            "impact_observations": [
                {
                    "time": "2026-01-10T12:00:00Z",
                    "authenticated": True,
                    "impact": "functionality_blocked",
                    "affected_functionality": ["primary workflow"],
                    "workaround_available": False,
                }
            ],
            "responses": [
                {
                    "responder": "Incident Commander",
                    "action": "opened the blocking-issue response",
                    "time": "2026-01-10T12:05:00Z",
                    "authenticated": True,
                }
            ],
        },
    }
    return copy.deepcopy(records[variant])


def _raw_facts(category, workflow, variant):
    context = _context(category, workflow)
    if category == "negation_and_quantifiers":
        record = _build_quantifier_record(variant)
    elif category == "ordered_rubrics":
        record = _build_rubric_record(variant)
    else:
        record = _build_action_record(category, context, variant)
    return {"context": context, "policy": _policy(category), "record": record}


def _in_window(time, context):
    window = context["inclusive_window"]
    return window["start"] <= time <= window["end"]


def _same_action(value, requested):
    return all(value.get(key) == requested[key] for key in ("actor", "operation", "target"))


def _matching_invocations(facts):
    context, record = facts["context"], facts["record"]
    requested = context["requested_action"]
    return [
        invocation
        for invocation in record.get("invocations", [])
        if _same_action(invocation, requested) and _in_window(invocation["time"], context)
    ]


def _confirmed_invocations(facts):
    matching = _matching_invocations(facts)
    confirmed = [
        row
        for row in matching
        if row.get("authenticated") is True and row.get("mode") == "real" and row.get("result") == "success"
    ]
    contradictory = any(
        row.get("result") in {"conflict", "guaranteed_failure"}
        and row.get("authenticated") is True
        and row.get("mode") == "real"
        for row in matching
    )
    return [] if contradictory else confirmed


def _successful_reversals(facts, confirmed):
    context, record = facts["context"], facts["record"]
    requested = context["requested_action"]
    ids = {row["id"] for row in confirmed}
    return [
        row
        for row in record.get("reversals", [])
        if row.get("linked_invocation_id") in ids
        and _same_action(row, requested)
        and _in_window(row["time"], context)
        and row.get("authenticated") is True
        and row.get("mode") == "real"
        and row.get("result") == "success"
    ]


def _action_summary(facts):
    matching = _matching_invocations(facts)
    confirmed = _confirmed_invocations(facts)
    reversals = _successful_reversals(facts, confirmed)
    failures = [
        row
        for row in matching
        if row.get("authenticated") is True and row.get("mode") == "real" and row.get("result") == "guaranteed_failure"
    ]
    authenticated_success = any(
        row.get("authenticated") is True and row.get("mode") == "real" and row.get("result") == "success"
        for row in matching
    )
    ambiguous = any(
        row.get("authenticated") is not True
        or row.get("mode") != "real"
        or row.get("result") in {"timeout", "missing_result", "conflict"}
        for row in matching
    ) or bool(authenticated_success and failures)
    if confirmed and reversals:
        status = "reversed"
    elif confirmed:
        status = "confirmed"
    elif failures and not ambiguous:
        status = "no_effect"
    else:
        status = "unknown"
    corroborated_ids = {
        row.get("invocation_id")
        for row in facts["record"].get("corroborations", [])
        if row.get("authenticated") is True
    }
    if not matching:
        score = 0
    elif not confirmed:
        score = 1
    elif any(row["id"] in corroborated_ids for row in confirmed):
        score = 3
    else:
        score = 2
    return {
        "matching": matching,
        "confirmed": confirmed,
        "reversals": reversals,
        "failures": failures,
        "status": status,
        "score": score,
    }


def _matching_statements(facts):
    requested = facts["context"]["requested_action"]
    return [
        row
        for row in facts["record"].get("attributed_statements", [])
        if row.get("subject_actor") == requested["actor"]
        and row.get("operation") == requested["operation"]
        and row.get("target") == requested["target"]
        and row.get("tense") == "completed"
        and row.get("polarity") == "affirmative"
    ]


def _matching_approval(facts, decision):
    approval = facts["record"].get("approval")
    return bool(
        approval
        and _same_action(approval, facts["context"]["requested_action"])
        and approval.get("decision") == decision
    )


def _adjudicate_action(category, facts):
    summary = _action_summary(facts)
    statements = _matching_statements(facts)
    record = facts["record"]
    requested = facts["context"]["requested_action"]

    if category == "claim_vs_completion":
        binary = (bool(statements), bool(summary["confirmed"]), any(row.get("endorsed") is True for row in statements))
    elif category == "permission_vs_execution":
        binary = (
            _matching_approval(facts, "granted"),
            bool(summary["confirmed"]),
            _matching_approval(facts, "denied"),
        )
    elif category == "unknown_vs_failure":
        binary = (
            summary["status"] == "unknown",
            summary["status"] in {"confirmed", "reversed"},
            summary["status"] == "no_effect",
        )
    elif category == "attribution_and_endorsement":
        binary = (
            any(row["speaker"] == "Riley, outside contractor" for row in statements),
            bool(summary["confirmed"]),
            any(row.get("endorsed") is True for row in statements),
        )
    elif category == "entity_binding":
        invocations = record.get("invocations", [])
        binary = (
            any(row.get("actor") == requested["actor"] for row in invocations),
            bool(summary["confirmed"]),
            any(row.get("target") == requested["target"] for row in invocations),
        )
    elif category == "action_binding":
        invocations = record.get("invocations", [])
        successful_same_entity = any(
            row.get("actor") == requested["actor"]
            and row.get("target") == requested["target"]
            and _in_window(row["time"], facts["context"])
            and row.get("authenticated") is True
            and row.get("mode") == "real"
            and row.get("result") == "success"
            for row in invocations
        )
        binary = (
            successful_same_entity,
            bool(summary["confirmed"]),
            bool(summary["matching"]),
        )
    elif category == "temporal_scope":
        boundary = {facts["context"]["inclusive_window"]["start"], facts["context"]["inclusive_window"]["end"]}
        binary = (
            bool(summary["matching"]),
            bool(summary["confirmed"]),
            any(row["time"] in boundary for row in summary["matching"]),
        )
    elif category == "reversal_and_current_state":
        binary = (
            bool(summary["confirmed"]),
            bool(summary["confirmed"]) and not summary["reversals"],
            bool(summary["reversals"]),
        )
    else:
        raise ValueError(category)
    return dict(zip(QUESTION_IDS, (*binary, summary["status"], summary["score"]), strict=True))


def _adjudicate_quantifiers(facts):
    observations = facts["record"].get("item_observations", [])
    by_item = {
        row["item"]: row["outcome"]
        for row in observations
        if row.get("authenticated") is True and row.get("item") in {"alpha", "beta", "gamma"}
    }
    outcomes = [by_item.get(item, "unknown") for item in ("alpha", "beta", "gamma")]
    successes = sum(outcome == "success" for outcome in outcomes)
    all_success = successes == 3
    some_success = successes > 0
    if "unknown" in outcomes:
        status = "unknown"
    elif all_success:
        status = "all_success"
    elif all(outcome == "no_effect" for outcome in outcomes):
        status = "all_failed"
    else:
        status = "mixed"
    return {"n1": all_success, "n2": some_success, "n3": not all_success, "c1": status, "s1": successes}


def _impact_level(observation):
    affected = bool(observation.get("affected_functionality"))
    workaround = observation.get("workaround_available") is True
    if observation.get("impact") == "functionality_blocked" and affected and not workaround:
        return 3
    if observation.get("impact") == "functionality_degraded" and affected and workaround:
        return 2
    if observation.get("impact") == "visual_only" and not affected:
        return 1
    return 0


def _adjudicate_rubric(facts):
    context, record = facts["context"], facts["record"]
    eligible = [
        row
        for row in record.get("impact_observations", [])
        if row.get("authenticated") is True and _in_window(row["time"], context)
    ]
    score = max((_impact_level(row) for row in eligible), default=0)
    status = ("unknown", "cosmetic", "degraded", "blocking")[score]
    workaround = any(_impact_level(row) == 2 for row in eligible)
    timely = any(
        row.get("authenticated") is True and _in_window(row["time"], context) for row in record.get("responses", [])
    )
    return {"n1": score >= 2, "n2": workaround, "n3": timely, "c1": status, "s1": score}


def adjudicate_case(category, facts):
    """Derive five hard targets from raw facts without consulting a variant label."""
    if category == "negation_and_quantifiers":
        return _adjudicate_quantifiers(facts)
    if category == "ordered_rubrics":
        return _adjudicate_rubric(facts)
    if category not in {name for name, _domain in CATEGORIES}:
        raise ValueError(f"unknown category: {category}")
    return _adjudicate_action(category, facts)


def _questions(category, workflow):
    actor, operation, target, _scope = workflow
    subject = f"{actor}'s {operation} operation on {target}"
    score = {
        "type": "score",
        "instructions": f"Select the highest established evidence stage for successful completion of {subject} in the inclusive window.",
        "criteria": ACTION_SCORE_CRITERIA,
    }
    choice = {
        "type": "choice",
        "instructions": f"What result is established for {subject} in the inclusive window?",
        "criteria": ACTION_CHOICE_CRITERIA,
    }

    def binary_criteria(yes, no):
        return {"true": yes, "false": no}

    if category == "claim_vs_completion":
        prompts = (
            (
                f"Does an attributed statement affirm that {subject} was completed?",
                "A matching affirmative completed-tense statement exists.",
                "No such statement exists.",
            ),
            (
                f"Does authenticated real evidence confirm successful completion of {subject}?",
                "A matching successful completion is confirmed.",
                "A matching successful completion is not confirmed.",
            ),
            (
                "Is a matching affirmative completion statement explicitly endorsed?",
                "A matching statement is endorsed.",
                "No matching statement is endorsed.",
            ),
        )
    elif category == "permission_vs_execution":
        prompts = (
            (
                f"Does the approval record grant permission for {subject}?",
                "An exact granted approval exists.",
                "No exact granted approval exists.",
            ),
            (
                f"Does authenticated real evidence confirm successful completion of {subject}?",
                "Successful completion is confirmed.",
                "Successful completion is not confirmed.",
            ),
            (
                f"Does the approval record explicitly deny {subject}?",
                "An exact denied approval exists.",
                "No exact denied approval exists.",
            ),
        )
    elif category == "unknown_vs_failure":
        prompts = (
            (
                f"Is the result of {subject} unknown under the supplied evidence?",
                "The result is missing, unverified, timed out, conflicting, or unrelated.",
                "The supplied evidence establishes confirmed completion, explicit no effect, or reversal.",
            ),
            (
                f"Does authenticated real evidence confirm successful completion of {subject}?",
                "Successful completion is confirmed.",
                "Successful completion is not confirmed.",
            ),
            (
                f"Does authenticated evidence explicitly establish guaranteed failure of {subject}?",
                "A matching guaranteed failure is established.",
                "No matching guaranteed failure is established.",
            ),
        )
    elif category == "attribution_and_endorsement":
        prompts = (
            (
                f"Is an affirmative completion statement about {subject} attributed to Riley, the outside contractor?",
                "Riley is the recorded speaker of a matching statement.",
                "Riley is not the recorded speaker of a matching statement.",
            ),
            (
                f"Does authenticated real evidence confirm successful completion of {subject}?",
                "Successful completion is confirmed.",
                "Successful completion is not confirmed.",
            ),
            (
                "Is a matching affirmative completion statement explicitly endorsed by authority?",
                "A matching statement is endorsed.",
                "No matching statement is endorsed.",
            ),
        )
    elif category == "entity_binding":
        prompts = (
            (
                f"Does any invocation name the requested actor {actor}?",
                "At least one invocation names the requested actor.",
                "No invocation names the requested actor.",
            ),
            (
                "Is successful completion confirmed for the exact actor, operation, target, and window?",
                "The exact completion is confirmed.",
                "The exact completion is not confirmed.",
            ),
            (
                f"Does any invocation name the requested target {target}?",
                "At least one invocation names the requested target.",
                "No invocation names the requested target.",
            ),
        )
    elif category == "action_binding":
        prompts = (
            (
                f"Is there an authenticated real in-window success for {actor} and {target}, regardless of operation name?",
                "Such a same-entity success exists.",
                "No such same-entity success exists.",
            ),
            (
                f"Is successful completion confirmed for the exact requested operation {operation}?",
                "The exact operation completed successfully.",
                "The exact operation is not confirmed complete.",
            ),
            (
                f"Is any invocation recorded for the exact requested operation {operation} with matching actor, target, and time?",
                "An exact matching invocation exists.",
                "No exact matching invocation exists.",
            ),
        )
    elif category == "temporal_scope":
        prompts = (
            (
                f"Is any exact invocation of {subject} inside the inclusive window?",
                "An exact in-window invocation exists.",
                "No exact in-window invocation exists.",
            ),
            (
                f"Does authenticated real in-window evidence confirm successful completion of {subject}?",
                "An in-window completion is confirmed.",
                "An in-window completion is not confirmed.",
            ),
            (
                "Does an exact invocation occur precisely at either inclusive boundary?",
                "An exact invocation occurs at the start or end timestamp.",
                "No exact invocation occurs at either boundary.",
            ),
        )
    elif category == "reversal_and_current_state":
        prompts = (
            (
                f"Was successful completion of {subject} historically confirmed, even if later reversed?",
                "A historical matching completion is confirmed.",
                "No historical matching completion is confirmed.",
            ),
            (
                f"Does the confirmed completion of {subject} still have current effect at the fixed window end?",
                "A confirmed completion remains effective at the window end.",
                "No confirmed completion remains effective at the window end.",
            ),
            (
                f"Did a linked authenticated matching reversal of {subject} succeed before the window ended?",
                "A successful linked matching reversal exists.",
                "No successful linked matching reversal exists.",
            ),
        )
    elif category == "negation_and_quantifiers":
        return {
            "n1": {
                "type": "noul",
                "instructions": "Are all three named items confirmed successful?",
                "criteria": binary_criteria(
                    "Every named item is confirmed successful.", "At least one named item is not confirmed successful."
                ),
            },
            "n2": {
                "type": "noul",
                "instructions": "Is at least one named item confirmed successful?",
                "criteria": binary_criteria(
                    "One or more named items are confirmed successful.", "No named item is confirmed successful."
                ),
            },
            "n3": {
                "type": "noul",
                "instructions": "Is it true that not all three named items are confirmed successful?",
                "criteria": binary_criteria(
                    "At least one named item is not confirmed successful.",
                    "All three named items are confirmed successful.",
                ),
            },
            "c1": {
                "type": "choice",
                "instructions": "Which set-level outcome is established for the three named items?",
                "criteria": QUANTIFIER_CHOICE_CRITERIA,
            },
            "s1": {
                "type": "score",
                "instructions": "How many of the three named items have authenticated success observations?",
                "criteria": QUANTIFIER_SCORE_CRITERIA,
            },
        }
    elif category == "ordered_rubrics":
        return {
            "n1": {
                "type": "noul",
                "instructions": "Does the highest authenticated in-window impact reach degraded or blocking severity?",
                "criteria": binary_criteria(
                    "Named functionality is degraded with a workaround or blocked without one.",
                    "No degraded-or-higher impact is established.",
                ),
            },
            "n2": {
                "type": "noul",
                "instructions": "Is affected functionality established together with an available workaround?",
                "criteria": binary_criteria(
                    "Affected functionality and a workaround are both established.",
                    "That combination is not established.",
                ),
            },
            "n3": {
                "type": "noul",
                "instructions": "Is an authenticated response recorded inside the inclusive window?",
                "criteria": binary_criteria(
                    "A timely authenticated response is recorded.", "No timely authenticated response is recorded."
                ),
            },
            "c1": {
                "type": "choice",
                "instructions": "What highest issue class is established by the raw impact observations?",
                "criteria": RUBRIC_CHOICE_CRITERIA,
            },
            "s1": {
                "type": "score",
                "instructions": "Select the highest applicable impact level under the stated rubric.",
                "criteria": RUBRIC_SCORE_CRITERIA,
            },
        }
    else:
        raise ValueError(category)

    questions = {}
    for qid, (instructions, yes, no) in zip(("n1", "n2", "n3"), prompts, strict=True):
        questions[qid] = {"type": "noul", "instructions": instructions, "criteria": binary_criteria(yes, no)}
    questions["c1"] = choice
    questions["s1"] = score
    return questions


def _predicate_tags(category):
    tags = {
        "claim_vs_completion": (
            "affirmative_completion_statement",
            "confirmed_completion",
            "endorsed_statement",
            "action_status",
            "completion_stage",
        ),
        "permission_vs_execution": (
            "permission_granted",
            "confirmed_completion",
            "permission_denied",
            "action_status",
            "completion_stage",
        ),
        "unknown_vs_failure": (
            "unknown_result",
            "confirmed_completion",
            "guaranteed_failure",
            "action_status",
            "completion_stage",
        ),
        "attribution_and_endorsement": (
            "contractor_attribution",
            "confirmed_completion",
            "authority_endorsement",
            "action_status",
            "completion_stage",
        ),
        "entity_binding": ("actor_named", "exact_completion", "target_named", "action_status", "completion_stage"),
        "action_binding": (
            "same_entity_success",
            "exact_operation_completion",
            "exact_operation_invocation",
            "action_status",
            "completion_stage",
        ),
        "temporal_scope": (
            "in_window_invocation",
            "in_window_completion",
            "boundary_invocation",
            "action_status",
            "completion_stage",
        ),
        "reversal_and_current_state": (
            "historical_completion",
            "current_effect",
            "matching_reversal",
            "action_status",
            "completion_stage",
        ),
        "negation_and_quantifiers": (
            "all_confirmed",
            "some_confirmed",
            "not_all_confirmed",
            "set_status",
            "confirmed_count",
        ),
        "ordered_rubrics": (
            "degraded_or_higher",
            "workaround_for_affected_functionality",
            "timely_response",
            "issue_class",
            "impact_level",
        ),
    }[category]
    return dict(zip(QUESTION_IDS, tags, strict=True))


def _rationale(category, expected):
    return {
        qid: f"The independent {category} oracle derives {qid}={value!r} from the raw record under the stated rules."
        for qid, value in expected.items()
    }


def _case(category, domain, family_number, workflow, variant):
    family_id = f"train/{category}/{family_number:02d}"
    facts = _raw_facts(category, workflow, variant)
    expected = adjudicate_case(category, facts)
    request = {"state": facts, "questions": _questions(category, workflow)}
    compile_request(request)
    requested = facts["context"]["requested_action"]
    return {
        "id": f"{family_id}/{variant}",
        "family_id": family_id,
        "domain": domain,
        "category": category,
        "variant": variant,
        "layout": "template_structured",
        "predicate_tags": _predicate_tags(category),
        "request": request,
        "expected": expected,
        "rationale": _rationale(category, expected),
        "facts": facts,
        "workflow": {**requested, "scope_id": facts["context"]["scope_id"]},
    }


def _changed_paths(left, right, prefix="record"):
    if type(left) is not type(right):
        return {prefix}
    if isinstance(left, dict):
        result = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}"
            if key not in left or key not in right:
                result.add(path)
            else:
                result |= _changed_paths(left[key], right[key], path)
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return {prefix}
        result = set()
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            result |= _changed_paths(a, b, f"{prefix}[{index}]")
        return result
    return set() if left == right else {prefix}


def _relations(cases):
    by_variant = {case["variant"]: case for case in cases}
    family = cases[0]["family_id"]
    pairs = [(a, b) for index, a in enumerate(VARIANTS) for b in VARIANTS[index + 1 :]]
    diffs = {
        pair: sorted(
            _changed_paths(
                by_variant[pair[0]]["facts"]["record"],
                by_variant[pair[1]]["facts"]["record"],
            )
        )
        for pair in pairs
    }
    flip_candidates = sorted(
        (
            len(diffs[(left, right)]),
            left,
            right,
            qid,
        )
        for left, right in pairs
        for qid in ("n1", "n2", "n3")
        if by_variant[left]["expected"][qid] != by_variant[right]["expected"][qid]
    )
    chosen_flips = [flip_candidates[0]]
    chosen_flips.append(next(item for item in flip_candidates[1:] if item[1:3] != chosen_flips[0][1:3]))
    invariant = min(
        (
            len(diffs[(left, right)]),
            left,
            right,
            qid,
        )
        for left, right in pairs
        for qid in ("n1", "n2", "n3")
        if by_variant[left]["expected"][qid] == by_variant[right]["expected"][qid]
    )
    contrasts = [
        (variant, left_qid, right_qid)
        for variant in VARIANTS
        for left_index, left_qid in enumerate(("n1", "n2", "n3"))
        for right_qid in ("n1", "n2", "n3")[left_index + 1 :]
        if by_variant[variant]["expected"][left_qid] != by_variant[variant]["expected"][right_qid]
    ]
    if len(contrasts) < 2:
        raise AssertionError(f"need two same-record contrasts for {family}")

    relations = []
    for index, (_count, left, right, qid) in enumerate(chosen_flips, 1):
        paths = diffs[(left, right)]
        relations.append(
            {
                "id": f"{family}/fact-flip-{index}",
                "kind": "flip",
                "left": {"case_id": by_variant[left]["id"], "question_id": qid},
                "right": {"case_id": by_variant[right]["id"], "question_id": qid},
                "changed_fact_paths": paths,
                "reason": f"Raw evidence differs at {', '.join(paths)}; these recorded changes alter {qid}.",
            }
        )
    for index, (variant, left_qid, right_qid) in enumerate(contrasts[:2], 1):
        relations.append(
            {
                "id": f"{family}/question-contrast-{index}",
                "kind": "question_contrast",
                "left": {"case_id": by_variant[variant]["id"], "question_id": left_qid},
                "right": {"case_id": by_variant[variant]["id"], "question_id": right_qid},
                "reason": f"The unchanged raw record supports {left_qid} and {right_qid} differently because they ask distinct predicates.",
            }
        )
    _count, left, right, qid = invariant
    paths = diffs[(left, right)]
    relations.append(
        {
            "id": f"{family}/invariant",
            "kind": "invariant",
            "left": {"case_id": by_variant[left]["id"], "question_id": qid},
            "right": {"case_id": by_variant[right]["id"], "question_id": qid},
            "changed_fact_paths": paths,
            "reason": f"Raw evidence differs at {', '.join(paths)}; {qid} is unaffected by those recorded changes.",
        }
    )
    return relations


def build_training_seeds():
    """Build 200 semantic families and 800 cases in memory without writing files."""
    cases, families, relations = [], [], []
    for category, domain in CATEGORIES:
        for number, workflow in enumerate(WORKFLOWS, 1):
            group = [_case(category, domain, number, workflow, variant) for variant in VARIANTS]
            cases.extend(group)
            relations.extend(_relations(group))
            families.append(
                {
                    "id": group[0]["family_id"],
                    "category": category,
                    "domain": domain,
                    "workflow": group[0]["workflow"],
                    "case_ids": [case["id"] for case in group],
                }
            )
    suite = {
        "contract": "Raw authored evidence is adjudicated by deterministic rules; targets and rationales never enter model state.",
        "cases": cases,
        "relations": relations,
    }
    validate_suite(suite)
    if len(families) != 200 or len(cases) != 800 or len(relations) != 1000:
        raise AssertionError("paired population must contain 200 families, 800 cases, and 1,000 relations")
    return {
        **suite,
        "families": families,
        "metadata": {
            "study": STUDY,
            "families": 200,
            "cases": 800,
            "judgments": 4000,
            "count_definition": "A family is one workflow/category situation with four raw-evidence variants and five questions per case.",
            "limitations": "The 200 situations reuse a shared 20-workflow ontology and category-specific reasoning patterns; they are not 200 independently authored reasoning templates.",
            "scope_rule": "Each workflow/category family has a unique scope ID and the stated inclusive window is the only temporal authority.",
            "label_origin": "Deterministic independent adjudication of raw evidence facts after variant construction.",
            "natural_authoring": "Natural writers replace only request.state.record and question.instructions; context, policy, IDs, question order, types, criteria, targets, rationales, facts, and relations remain immutable.",
        },
    }


def template_bundles(seeds):
    """Serialize seeds into sibling-preserving training bundles without writing them."""
    by_family = {}
    for case in seeds["cases"]:
        by_family.setdefault(case["family_id"], []).append(case)
    bundles = []
    for family_id, cases in sorted(by_family.items()):
        first = cases[0]
        examples = []
        for case in sorted(cases, key=lambda value: value["variant"]):
            for qid in QUESTION_IDS:
                question, target = case["request"]["questions"][qid], case["expected"][qid]
                if question["type"] == "noul":
                    encoded = {"truth": target}
                elif question["type"] == "choice":
                    encoded = {"choice": target}
                else:
                    encoded = {"level_index": target}
                examples.append(
                    {
                        "state": case["request"]["state"],
                        "question": question,
                        "target": encoded,
                        "case_id": case["id"],
                        "question_id": qid,
                    }
                )
        bundles.append(
            {
                "id": family_id,
                "group_id": family_id,
                "examples": examples,
                "relations": [],
                "provenance": {
                    "dataset": STUDY,
                    "assigned_split": "train",
                    "family": family_id,
                    "category": first["category"],
                    "label_origin": "deterministic raw-fact oracle",
                },
            }
        )
    if len(bundles) != 200 or any(len(bundle["examples"]) != 20 for bundle in bundles):
        raise AssertionError("every family must keep four case siblings and five questions")
    return bundles


def natural_replacement_contract(seeds):
    """Describe the two rendering fields natural writers may replace."""
    return {
        "allowed_changes": ["request.state.record", "request.questions.<qid>.instructions"],
        "immutable": [
            "case.id",
            "family_id",
            "category",
            "facts",
            "request.state.context",
            "request.state.policy",
            "question order",
            "question type",
            "criteria",
            "expected",
            "rationale",
            "relations",
        ],
        "cases": [
            {
                "id": case["id"],
                "family_id": case["family_id"],
                "facts": case["facts"],
                "question_ids": list(case["request"]["questions"]),
            }
            for case in seeds["cases"]
        ],
    }


def _write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _write_jsonl(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for deferred template seed exports.")
    parser.add_argument(
        "--evaluation-freeze-manifest",
        type=Path,
        required=True,
        help="Existing held-out evaluation manifest required before export.",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output directory")
    manifest = json.loads(args.evaluation_freeze_manifest.read_text())
    if not manifest.get("suite_sha256"):
        parser.error("evaluation freeze manifest must name its frozen suite hash")
    seeds = build_training_seeds()
    args.output.mkdir(parents=True)
    _write_json(args.output / "template-seeds.json", seeds)
    _write_json(args.output / "natural-replacement-contract.json", natural_replacement_contract(seeds))
    _write_jsonl(args.output / "template-train.jsonl", template_bundles(seeds))
    print(json.dumps({"families": 200, "cases": 800, "output": str(args.output)}))


if __name__ == "__main__":
    main()
