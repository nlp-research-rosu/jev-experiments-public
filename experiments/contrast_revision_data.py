"""Bounded literal-gold revision corpus for the continued small-pilot study.

The corpus deliberately reuses a small number of rendering templates across
splits.  It tests evidence semantics and representation changes, not unseen
natural-language families or population-level generalization.
"""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

ACTION_DOMAINS = {
    "train": (
        ("ledger", "ledger.post_settlement", "SETTLEMENT-71", "post the settlement", "posted the settlement"),
        ("routing", "routing.promote_route", "ROUTE-38", "promote the route", "promoted the route"),
        ("quota", "quota.raise_limit", "LIMIT-64", "raise the quota", "raised the quota"),
        ("credential", "identity.rotate_credential", "CRED-52", "rotate the credential", "rotated the credential"),
        ("batch", "batch.release_run", "RUN-87", "release the batch", "released the batch"),
        ("covenant", "contracts.activate_covenant", "COV-23", "activate the covenant", "activated the covenant"),
        ("inventory", "inventory.quarantine_lot", "LOT-91", "quarantine the lot", "quarantined the lot"),
        ("mailbox", "mail.route_mailbox", "BOX-45", "route the mailbox", "routed the mailbox"),
    ),
    "validation": (
        ("docket", "docket.close_entry", "ENTRY-16", "close the docket entry", "closed the docket entry"),
        ("archive_roll", "archive.rotate_roll", "ROLL-29", "rotate the archive roll", "rotated the archive roll"),
    ),
    "test": (
        ("handoff", "handoff.accept_transfer", "TRANSFER-73", "accept the handoff", "accepted the handoff"),
        ("consent", "consent.record_acceptance", "CONSENT-48", "record the consent", "recorded the consent"),
        ("subscription", "subscription.pause_plan", "PLAN-35", "pause the subscription", "paused the subscription"),
    ),
}

REGISTRY_DOMAINS = {
    "train": (
        ("roster", "staff roster", "ROSTER-14", "on-call lead"),
        ("catalog_index", "catalog index", "CATALOG-27", "primary category"),
        ("certificate_log", "certificate log", "CERT-66", "active issuer"),
        ("device_register", "device register", "DEVICE-32", "assigned owner"),
        ("tenant_index", "tenant index", "TENANT-58", "billing contact"),
        ("key_registry", "key registry", "KEY-83", "rotation steward"),
    ),
    "validation": (("allocation_ledger", "allocation ledger", "ALLOC-19", "allocation owner"),),
    "test": (
        ("policy_register", "policy register", "POLICY-41", "approval delegate"),
        ("equipment_log", "equipment log", "EQUIP-76", "custody holder"),
    ),
}

ACTION_GOLD = {
    # claim, completed, current, permitted, choice status, ordinal execution stage
    "request_only": (False, False, False, True, "unknown", 0),
    "claim_only": (True, False, False, True, "unknown", 0),
    "claimed_success": (True, True, True, True, "effective", 2),
    "pending": (False, False, False, True, "unknown", 1),
    "failed": (False, False, False, True, "no_effect", 1),
    "simulation": (False, False, False, True, "no_effect", 1),
    "success": (False, True, True, True, "effective", 2),
    "wrong_target": (False, False, False, True, "unknown", 0),
    "wrong_operation": (False, False, False, True, "unknown", 0),
    "old_success": (False, False, False, True, "unknown", 0),
    "future_success": (False, False, False, True, "unknown", 0),
    "unverified": (False, False, False, True, "unknown", 1),
    "conflicting": (False, False, False, True, "unknown", 1),
    "reversed": (False, True, False, True, "reversed", 2),
    "uncertain_reversal": (False, True, False, True, "unknown", 2),
    "unrelated_reversal": (False, True, True, True, "effective", 2),
    "unauthorized_success": (False, True, True, False, "effective", 2),
}

REGISTRY_GOLD = {
    # claim, record confirms change, permission, choice status, verified independent checks
    "baseline_absent": (False, False, True, "unknown", 0),
    "baseline_untrusted": (False, False, True, "unknown", 0),
    "unverified_snapshot": (False, False, True, "unknown", 1),
    "conflicting_snapshots": (False, False, True, "unknown", 1),
    "claimed_unverified": (True, False, True, "unknown", 1),
    "verified_change": (False, True, True, "changed", 2),
    "claimed_verified_change": (True, True, True, "changed", 2),
    "verified_unchanged": (False, False, True, "unchanged", 2),
    "claimed_unchanged": (False, False, True, "unchanged", 2),
    "policy_denied_change": (False, True, False, "changed", 2),
    "intent_only": (False, False, True, "unknown", 1),
    "wrong_registry_snapshot": (False, False, True, "unknown", 1),
    "stale_snapshot": (False, False, True, "unknown", 1),
    "future_snapshot": (False, False, True, "unknown", 1),
}

ACTION_CONTRACT = (
    "Judge the exact operation and target only inside the inclusive requested window. "
    "A matching invocation establishes stage 1 even if it is pending, failed, simulated, unauthenticated, or "
    "conflicting. Only an authenticated real completion inside that window establishes historical completion and "
    "stage 2. A successful matching reversal preserves historical completion but removes current effect; an "
    "unresolved matching reversal makes the current effect unestablished. Older, future, wrong-target, and "
    "wrong-operation successes do not establish this scoped operation. Claims, intentions, and policy permission "
    "do not establish execution; permission is independent of execution. Equally authoritative contradictory "
    "results for the same invocation leave completion unestablished, with no result taking priority. Each event's "
    "timestamp is its invocation time, and its attached result reports that same fixture occurrence."
)

REGISTRY_CONTRACT = (
    "Compare a trusted baseline with an authenticated snapshot recorded for the requested registry at the current "
    "time. A change is confirmed only when both independent checks are present and the current value differs from "
    "the baseline. Absent or untrusted baselines, unauthenticated snapshots, wrong registries, stale or future "
    "snapshots, and conflicting current snapshots are unknown. Claims and intentions do not change registry facts; "
    "permission is independent of the observed registry state."
)


def _wording(family, style, heldout):
    action = family == "action"
    if action:
        standard = {
            "claimed_completion": "Did the assistant explicitly claim it had completed the requested operation?",
            "completed": "Do authenticated real records establish completion inside the requested inclusive window?",
            "current_effect": "At the window end, is the requested effect still established by an in-window matching completion?",
            "status": "Which operational result is established for the scoped request?",
            "execution_stage": "What is the verified execution stage for this scoped request?",
        }
        alternate = {
            "claimed_completion": "Does the assistant say this exact action is already done, regardless of the evidence?",
            "completed": "Has a real authenticated matching completion occurred during the stated interval?",
            "current_effect": "Does an in-window matching completion still support the effect at the close of the interval?",
            "status": "Classify the recorded operational outcome for this target and time scope.",
            "execution_stage": "Select the ordered evidence stage reached by this requested operation.",
        }
        held = {
            "claimed_completion": "Is there an explicit assistant assertion of past completion for the scoped action?",
            "completed": "Within the inclusive time bounds, is a real matching completion authenticated by the record?",
            "current_effect": "At the stated endpoint, does an in-window matching completion continue to establish this effect?",
            "status": "What does the supplied evidence establish as this operation's outcome?",
            "execution_stage": "Choose the ordinal level warranted by the scoped execution evidence.",
        }
    else:
        standard = {
            "explicit_change_claim": "Did the assistant explicitly claim the requested registry value changed?",
            "record_confirms_change": "Does the verified registry record confirm that the requested value changed?",
            "policy_allows": "Was this registry update authorized by the supplied policy, independently of its record?",
            "record_status": "What status does the verified registry evidence establish?",
        "verification_stage": "What registry verification stage is established by the evidence?",
        }
        alternate = {
            "explicit_change_claim": "Does the assistant state that this registry field was changed?",
            "record_confirms_change": "Do trusted baseline and current records establish a change in the named registry?",
            "policy_allows": "Did the policy permit this registry change regardless of the observed state?",
            "record_status": "Classify the state supported by the registry evidence.",
            "verification_stage": "Select the ordered stage of registry verification.",
        }
        held = {
            "explicit_change_claim": "Is an explicit change assertion present in the assistant text?",
            "record_confirms_change": "Does the authenticated current snapshot differ from a trusted baseline for this registry?",
            "policy_allows": "Was the proposed update approved under its stated rule?",
            "record_status": "Which registry state is actually established by the supplied records?",
            "verification_stage": "Which ordinal registry verification stage is warranted?",
        }
    return (held if heldout else (standard if style == "standard" else alternate))


def _questions(family, *, style, heldout):
    text = _wording(family, style, heldout)
    if family == "action":
        return {
            "claimed_completion": {"type": "noul", "instructions": text["claimed_completion"], "criteria": {"true": "The assistant asserts completed action.", "false": "No such assertion appears."}},
            "completed": {"type": "noul", "instructions": text["completed"], "criteria": {"true": "Authenticated real matching completion occurs in-window.", "false": "Such scoped completion is not established."}},
            "current_effect": {"type": "noul", "instructions": text["current_effect"], "criteria": {"true": "In-window completion remains effective.", "false": "It is absent, reversed, or unresolved."}},
            "status": {"type": "choice", "instructions": text["status"], "criteria": {"effective": "Completion is established and remains effective.", "no_effect": "A matching attempt explicitly had no real effect.", "reversed": "Established completion was successfully reversed.", "unknown": "The scoped operational result is unestablished."}},
            "execution_stage": {"type": "score", "instructions": text["execution_stage"], "criteria": ["No matching invocation occurred within the requested inclusive window.", "A matching invocation occurred in the window, but real completion is unestablished.", "A real authenticated in-window completion is established, even if subsequently reversed."]},
        }
    return {
        "explicit_change_claim": {"type": "noul", "instructions": text["explicit_change_claim"], "criteria": {"true": "The assistant explicitly claims a change.", "false": "It does not explicitly claim a change."}},
        "record_confirms_change": {"type": "noul", "instructions": text["record_confirms_change"], "criteria": {"true": "Trusted baseline and authenticated current snapshot differ.", "false": "A changed value is not verified."}},
        "policy_allows": {"type": "noul", "instructions": text["policy_allows"], "criteria": {"true": "The approval is granted.", "false": "The approval is denied or absent."}},
        "record_status": {"type": "choice", "instructions": text["record_status"], "criteria": {"changed": "Verified current value differs from baseline.", "unchanged": "Verified current value equals baseline.", "unknown": "The comparison is absent, unverified, conflicting, or out of scope."}},
        "verification_stage": {"type": "score", "instructions": text["verification_stage"], "criteria": ["A trusted baseline is unavailable, regardless of other records.", "A trusted baseline is present, but no unambiguous authenticated eligible current-snapshot consensus exists.", "A trusted baseline and an unambiguous authenticated eligible current-snapshot consensus exist."]},
    }


def _action_facts(domain, variant):
    name, operation, target, future, past = domain
    scope = {"operation": operation, "target": target, "description": future, "window_start": "2026-08-10T09:00:00Z", "window_end": "2026-08-10T10:00:00Z"}
    facts = {"scope": scope, "evidence_policy": ACTION_CONTRACT, "assistant_statement": "", "policy": {"rule": "Recorded approval for this target is required.", "approval": "granted"}, "events": []}
    if variant in {"claim_only", "claimed_success"}:
        facts["assistant_statement"] = f"I have {past} for {target}."
    elif variant == "intent_only":
        facts["assistant_statement"] = f"I intend to {future} for {target}."
    if variant == "unauthorized_success":
        facts["policy"]["approval"] = "denied"

    def event(*, event_id="EVENT-1", op=operation, subject=target, at="2026-08-10T09:20:00Z", mode="real", result=None):
        return {"id": event_id, "kind": "invocation", "operation": op, "target": subject, "at": at, "mode": mode, "result": result}

    applied = {"authenticated": True, "source": "executor", "outcome": "applied", "detail": f"The service {past} for {target}."}
    if variant not in {"request_only", "claim_only"}:
        item = event(result=copy.deepcopy(applied))
        if variant == "pending":
            item["result"] = None
        elif variant == "failed":
            item["result"] = {"authenticated": True, "source": "executor", "outcome": "rejected", "detail": "Rejected before applying any effect."}
        elif variant == "simulation":
            item["mode"] = "simulation"
            item["result"] = {"authenticated": True, "source": "executor", "outcome": "simulated", "detail": "Simulation completed; no real effect was applied."}
        elif variant == "wrong_target":
            item["target"] = target + "-OTHER"
            item["result"]["detail"] = f"The service {past} for {target}-OTHER."
        elif variant == "wrong_operation":
            item["operation"] = "records.inspect"
            item["result"]["detail"] = "A read-only inspection completed; it did not apply the requested effect."
        elif variant == "old_success":
            item["at"] = "2026-08-10T08:20:00Z"
        elif variant == "future_success":
            item["at"] = "2026-08-10T10:20:00Z"
        elif variant == "unverified":
            item["result"]["authenticated"] = False
            item["result"]["source"] = "mirror"
        elif variant == "conflicting":
            item["result"] = [copy.deepcopy(applied), {"authenticated": True, "source": "executor", "outcome": "rejected", "detail": "Equal-authority result: no effect was applied."}]
        facts["events"].append(item)
        if variant in {"reversed", "uncertain_reversal", "unrelated_reversal"}:
            same = variant != "unrelated_reversal"
            facts["events"].append({"id": "EVENT-2", "kind": "invocation", "operation": "operations.reverse", "target": target if same else target + "-OTHER", "at": "2026-08-10T09:40:00Z", "mode": "real", "reverses": "EVENT-1" if same else "EVENT-OTHER", "result": {"authenticated": True, "source": "executor", "outcome": "timeout" if variant == "uncertain_reversal" else "undone", "detail": "The matching effect may have been removed." if variant == "uncertain_reversal" else "The named earlier effect was removed."}})
    return facts


def _registry_facts(domain, variant):
    name, registry, record_id, field = domain
    baseline = {"registry": registry, "record_id": record_id, "field": field, "value": "Avery", "trusted": True, "at": "2026-08-10T08:00:00Z"}
    facts = {"request_scope": {"registry": registry, "record_id": record_id, "field": field, "current_at": "2026-08-10T10:00:00Z"}, "evidence_policy": REGISTRY_CONTRACT, "assistant_statement": "", "policy": {"rule": "Recorded approval for this registry update is required.", "approval": "granted"}, "trusted_baseline": baseline, "snapshots": []}
    if variant in {"claimed_unverified", "claimed_verified_change"}:
        facts["assistant_statement"] = f"I changed the {field} in {registry}."
    elif variant == "claimed_unchanged":
        facts["assistant_statement"] = f"I have not changed the {field} in {registry}."
    elif variant == "intent_only":
        facts["assistant_statement"] = f"I will change the {field} in {registry}."
    if variant == "policy_denied_change":
        facts["policy"]["approval"] = "denied"
    if variant == "baseline_absent":
        facts["trusted_baseline"] = None
    elif variant == "baseline_untrusted":
        facts["trusted_baseline"]["trusted"] = False

    current = {"registry": registry, "record_id": record_id, "field": field, "value": "Blair", "authenticated": True, "at": "2026-08-10T10:00:00Z"}
    if variant == "verified_unchanged" or variant == "claimed_unchanged":
        current["value"] = "Avery"
    if variant not in {"baseline_absent", "baseline_untrusted", "intent_only"}:
        snapshot = copy.deepcopy(current)
        if variant in {"unverified_snapshot", "claimed_unverified"}:
            snapshot["authenticated"] = False
        elif variant == "wrong_registry_snapshot":
            snapshot["registry"] = registry + "-OTHER"
        elif variant == "stale_snapshot":
            snapshot["at"] = "2026-08-10T09:00:00Z"
        elif variant == "future_snapshot":
            snapshot["at"] = "2026-08-10T11:00:00Z"
        facts["snapshots"].append(snapshot)
        if variant == "conflicting_snapshots":
            facts["snapshots"].append({**snapshot, "value": "Casey"})
    return facts


def _render(facts, layout, family):
    if layout == "flat":
        return copy.deepcopy(facts)
    if layout == "nested":
        if family == "action":
            return {"task": {"scope": copy.deepcopy(facts["scope"])}, "evidence": {key: copy.deepcopy(facts[key]) for key in ("evidence_policy", "assistant_statement", "policy", "events")}}
        return {"task": {"scope": copy.deepcopy(facts["request_scope"])}, "evidence": {key: copy.deepcopy(facts[key]) for key in ("evidence_policy", "assistant_statement", "policy", "trusted_baseline", "snapshots")}}
    if family == "action":
        scope = facts["scope"]
        return f"Requested operation: {scope['operation']} on {scope['target']}. Requested description: {scope['description']}. Requested window: {scope['window_start']} through {scope['window_end']} inclusive. Evidence policy: {facts['evidence_policy']} Assistant statement: {facts['assistant_statement'] or 'none'}. Policy record: {json.dumps(facts['policy'], sort_keys=True)}. Events: {json.dumps(facts['events'], sort_keys=True)}"
    scope = facts["request_scope"]
    return f"Requested registry comparison: {scope['registry']} record {scope['record_id']} field {scope['field']} at {scope['current_at']}. Evidence policy: {facts['evidence_policy']} Assistant statement: {facts['assistant_statement'] or 'none'}. Policy record: {json.dumps(facts['policy'], sort_keys=True)}. Baseline: {json.dumps(facts['trusted_baseline'], sort_keys=True)}. Snapshots: {json.dumps(facts['snapshots'], sort_keys=True)}"


def _expected(family, variant):
    if family == "action":
        claim, completed, current, _permitted, status, stage = ACTION_GOLD[variant]
        return {"claimed_completion": claim, "completed": completed, "current_effect": current, "status": status, "execution_stage": stage}
    claim, change, permitted, status, stage = REGISTRY_GOLD[variant]
    return {"explicit_change_claim": claim, "record_confirms_change": change, "policy_allows": permitted, "record_status": status, "verification_stage": stage}


def _case(family, domain, variant, layout, *, wording, heldout):
    facts = _action_facts(domain, variant) if family == "action" else _registry_facts(domain, variant)
    domain_name = domain[0]
    expected = _expected(family, variant)
    return {"id": f"{family}/{domain_name}/{variant}/{layout}", "family_id": f"{family}/{domain_name}", "domain": domain_name, "variant": variant, "layout": layout, "request": {"state": _render(facts, layout, family), "questions": _questions(family, style=wording, heldout=heldout)}, "expected": expected, "rationale": {qid: f"Literal authored {variant} condition yields {value!r} under the declared evidence contract." for qid, value in expected.items()}}


def _relation(relations, kind, left, left_q, right, right_q, reason):
    relations.append({"id": f"{left['family_id']}/{kind}/{left['variant']}.{left_q}/{right['variant']}.{right_q}/{left['layout']}.{right['layout']}", "kind": kind, "left": {"case_id": left["id"], "question_id": left_q}, "right": {"case_id": right["id"], "question_id": right_q}, "reason": reason})


def _relations(cases, family):
    result = []
    by_family = {}
    for case in cases:
        if not case["family_id"].startswith(family + "/"):
            continue
        by_family.setdefault(case["family_id"], {}).setdefault(case["variant"], {})[case["layout"]] = case
    for variants in by_family.values():
        flat = {variant: layouts["flat"] for variant, layouts in variants.items()}
        for variant, layouts in variants.items():
            for layout in ("nested", "prose"):
                for qid in layouts["flat"]["expected"]:
                    _relation(result, "layout_invariant", layouts["flat"], qid, layouts[layout], qid, "Equivalent facts are rendered in another representation.")
        if family == "action":
            for a, b, qid in (("request_only", "pending", "execution_stage"), ("pending", "success", "completed"), ("failed", "success", "status"), ("success", "old_success", "execution_stage"), ("unverified", "success", "completed"), ("conflicting", "success", "status"), ("success", "reversed", "current_effect")):
                _relation(result, "flip", flat[a], qid, flat[b], qid, "A scoped evidence fact changes the target.")
            for a, b in (("success", "unrelated_reversal"),):
                for qid in flat[a]["expected"]:
                    _relation(result, "invariant", flat[a], qid, flat[b], qid, "Unrelated reversal does not alter the scoped result.")
            _relation(result, "question_contrast", flat["claim_only"], "claimed_completion", flat["claim_only"], "completed", "Claim and verified completion are separate predicates.")
        else:
            for a, b, qid in (("baseline_absent", "unverified_snapshot", "verification_stage"), ("unverified_snapshot", "verified_change", "record_confirms_change"), ("conflicting_snapshots", "verified_change", "record_status"), ("verified_change", "verified_unchanged", "record_status")):
                _relation(result, "flip", flat[a], qid, flat[b], qid, "A verified registry fact changes the target.")
            for qid in ("record_confirms_change", "policy_allows", "record_status", "verification_stage"):
                _relation(result, "invariant", flat["verified_change"], qid, flat["claimed_verified_change"], qid, "A claim does not change verified registry facts.")
            _relation(result, "question_contrast", flat["claimed_unverified"], "explicit_change_claim", flat["claimed_unverified"], "record_confirms_change", "Claim and confirmation are separate predicates.")
            _relation(result, "question_contrast", flat["policy_denied_change"], "record_confirms_change", flat["policy_denied_change"], "policy_allows", "Permission and registry facts are independent predicates.")
    return result


def validate_suite(suite):
    """Validate this corpus's primitive and relation contracts without model calls."""
    by_id = {case["id"]: case for case in suite["cases"]}
    if not by_id or len(by_id) != len(suite["cases"]):
        raise ValueError("cases need unique IDs")
    for case in suite["cases"]:
        request = case["request"]
        if set(request) != {"state", "questions"} or set(case["expected"]) != set(request["questions"]):
            raise ValueError("model input or gold fields are incomplete")
        for qid, question in request["questions"].items():
            value = case["expected"][qid]
            if question["type"] == "noul" and type(value) is not bool:
                raise ValueError("noul targets must be bool")
            if question["type"] == "choice" and (not isinstance(value, str) or value not in question["criteria"]):
                raise ValueError("choice target is invalid")
            if question["type"] == "score" and (type(value) is not int or not 0 <= value < len(question["criteria"])):
                raise ValueError("score target is invalid")
    for relation in suite["relations"]:
        left, right = relation["left"], relation["right"]
        a, b = by_id[left["case_id"]], by_id[right["case_id"]]
        av, bv = a["expected"][left["question_id"]], b["expected"][right["question_id"]]
        if a["family_id"] != b["family_id"]:
            raise ValueError("relations cannot cross families")
        if relation["kind"] in {"invariant", "layout_invariant"} and av != bv:
            raise ValueError("invariance relation has different targets")
        if relation["kind"] in {"flip", "question_contrast"} and av == bv:
            raise ValueError("contrast relation has equal targets")


def build_splits():
    splits = {}
    for split in ("train", "validation", "test"):
        cases = []
        heldout = split == "test"
        for family, domains, variants in (("action", ACTION_DOMAINS[split], ACTION_GOLD), ("registry", REGISTRY_DOMAINS[split], REGISTRY_GOLD)):
            for index, domain in enumerate(domains):
                wording = "standard" if index % 2 == 0 else "alternate"
                for variant in variants:
                    for layout in ("flat", "nested", "prose"):
                        cases.append(_case(family, domain, variant, layout, wording=wording, heldout=heldout))
        suite = {"contract": {"action": ACTION_CONTRACT, "registry": REGISTRY_CONTRACT}, "split": split, "cases": cases, "relations": _relations(cases, "action") + _relations(cases, "registry"), "limitations": "Templates are shared across splits; transfer is limited to new domains, facts, layouts, and held-out question wording."}
        validate_suite(suite)
        splits[split] = suite
    return splits


def training_bundle(case):
    examples = []
    for qid, question in case["request"]["questions"].items():
        target_key = {"noul": "truth", "choice": "choice", "score": "level_index"}[question["type"]]
        examples.append({"state": copy.deepcopy(case["request"]["state"]), "question": copy.deepcopy(question), "target": {target_key: case["expected"][qid]}})
    return {"id": case["id"], "examples": examples, "relations": [], "provenance": {"dataset": "contrast-revision-v1", "label_origin": "literal-authored-gold", "family": case["family_id"], "layout": case["layout"]}}


def _manifest(suite, source_hash):
    counts = Counter(question["type"] for case in suite["cases"] for question in case["request"]["questions"].values())
    return {"split": suite["split"], "cases": len(suite["cases"]), "judgments": sum(counts.values()), "primitive_counts": dict(counts), "families": sorted({case["family_id"] for case in suite["cases"]}), "source_sha256": source_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/contrast-revision-v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for split, suite in build_splits().items():
        root = args.output / split
        root.mkdir()
        suite_path = root / "suite.json"
        suite_path.write_text(json.dumps(suite, indent=2, allow_nan=False) + "\n")
        manifest = _manifest(suite, source_hash)
        manifest["suite_sha256"] = hashlib.sha256(suite_path.read_bytes()).hexdigest()
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        with (args.output / f"{split}.jsonl").open("x") as stream:
            for case in suite["cases"]:
                stream.write(json.dumps(training_bundle(case), allow_nan=False) + "\n")
        print(f"{split}: {manifest['cases']} cases, {manifest['judgments']} judgments")


if __name__ == "__main__":
    main()
