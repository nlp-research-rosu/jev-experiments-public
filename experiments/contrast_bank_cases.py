"""Frozen synthetic claim/fact/permission contrasts; no training or model calls."""

from copy import deepcopy

_QUESTION_IDS = (
    "explicit_change_claim",
    "different_destination",
    "record_confirms_change",
    "use_permitted",
    "record_status",
    "operational_change_signal",
)

_CONTRACT = """Frozen destination diagnostic, version 1. All variants in each domain
belong to one family and must stay together. This is a diagnostic, not a training
split. Only each case's request may enter a model; ids, expected answers, rationales
and relation metadata are evaluator-only.

Six distinct predicates are specified in each request. Explicit-change-claim
means that the current sender affirmatively asserts in their own voice that the
details already changed. Questions, denials, future-only plans, unendorsed third-
party quotations, and a different destination alone do not qualify. Adopting a
quotation as true does qualify. Different-destination compares the destination
requested for the current transaction with the trusted baseline, regardless of
sender assertions or verification. Operational-change-signal deliberately uses
the broader rule: an explicit already-changed claim OR a different destination
requested for the current transaction. Future-only destinations do not count.

The baseline is a trusted historical checkpoint. Current record status concerns
the net current destination versus that baseline, not whether any intermediate
change ever occurred. Use only authenticated snapshots from the stated registry
whose effective_at equals evaluated_at. An absent eligible snapshot or disagreeing
eligible snapshots gives Unknown; no snapshot has priority over another. Otherwise
their common destination determines Changed versus Unchanged. Sender assertions
do not establish registry state. Record-confirms-change is true only for Changed;
its false label includes both known Unchanged and Unknown, which the Choice
question distinguishes. Unknown is a categorical answer, never an invented 0.5
probability. These are evidence-relative labels, not assertions about inaccessible
world truth.

Permission is an explicit local policy: a unique established current registry
destination must exactly match the requested destination, and the transaction
must not be on hold. Verified unchanged details can be permitted; verified changed
details can be blocked by a hold. These predicates are not logical complements.
The two domains share a semantic template to test transfer, but constitute only
two independent families; the case count is not an independent-sample count.
"""

_DOMAINS = (
    {
        "domain": "payment",
        "family_id": "payment_destination_change",
        "details": "payment bank details",
        "transaction_kind": "invoice payment",
        "reference": "INV-104",
        "counterparty": "Alder Services",
        "sender": "Priya Shah",
        "sender_role": "supplier billing contact",
        "registry": "Finance master registry",
        "old": {"bank": "Cedar Demo Bank", "account": "804611", "routing": "110000001"},
        "new": {"bank": "Juniper Demo Bank", "account": "907522", "routing": "220000002"},
        "old_text": "Cedar Demo Bank, account 804611, routing 110000001",
        "new_text": "Juniper Demo Bank, account 907522, routing 220000002",
        "quoted_source": "our outside bookkeeper",
    },
    {
        "domain": "shipping",
        "family_id": "shipping_destination_change",
        "details": "delivery contact details",
        "transaction_kind": "parcel delivery",
        "reference": "SHIP-104",
        "counterparty": "Maple Field Office",
        "sender": "Jordan Reed",
        "sender_role": "recipient coordinator",
        "registry": "Dispatch contact registry",
        "old": {"contact": "Aline Teo", "email": "dispatch-old@example.test", "address": "18 Alder Way"},
        "new": {"contact": "Mina Cho", "email": "dispatch-new@example.test", "address": "42 Alder Way"},
        "old_text": "Aline Teo, dispatch-old@example.test, 18 Alder Way",
        "new_text": "Mina Cho, dispatch-new@example.test, 42 Alder Way",
        "quoted_source": "our outside delivery coordinator",
    },
)


def _questions(details: str) -> dict:
    current_record_rule = (
        "Use only authenticated current_record_snapshots from authority.registry_name "
        "with effective_at equal to evaluated_at. There is no precedence among eligible snapshots. "
        "If none qualify, or eligible snapshots disagree on destination, the current record "
        "status is Unknown. Otherwise compare their common destination with baseline.destination: "
        "different means Changed; identical means Unchanged. This is the net current value, "
        "not whether some intermediate change occurred. Ignore message claims as registry evidence. "
    )
    explicit_rule = (
        f"Does the message sender explicitly assert, in their own voice, that the {details} "
        "have already changed from the baseline by evaluated_at? An affirmative endorsement of "
        "a quoted claim counts as the sender's assertion. A denial, a question, a future-only "
        "plan, an unendorsed third-party quotation, or merely directing use of a different "
        "destination does not count. Judge what is asserted, even if registry evidence contradicts it."
    )
    return {
        "explicit_change_claim": {"type": "noul", "instructions": explicit_rule},
        "different_destination": {
            "type": "noul",
            "instructions": (
                "Does requested_destination for the current transaction differ from baseline.destination "
                "in at least one field? Compare the structured values exactly. This question asks "
                "about a proposed current destination, not whether the registry changed or use is permitted. "
                "A different destination mentioned only for the future does not count."
            ),
        },
        "record_confirms_change": {
            "type": "noul",
            "instructions": current_record_rule + "Is the current record status Changed? Return false for Unchanged or Unknown.",
        },
        "use_permitted": {
            "type": "noul",
            "instructions": (
                current_record_rule
                + "Does use_policy permit requested_destination for the current transaction now? "
                "Permission requires an established current registry destination that exactly matches "
                "requested_destination and transaction_on_hold=false. Unknown current records or a hold "
                "deny permission. Sender assertions alone cannot grant permission."
            ),
        },
        "record_status": {
            "type": "choice",
            "instructions": current_record_rule + "Select the status established by the supplied evidence.",
            "criteria": {
                "Changed": "The established current registry destination differs from the trusted baseline.",
                "Unchanged": "The established current registry destination equals the trusted baseline.",
                "Unknown": "No eligible current snapshot exists, or eligible current snapshots conflict.",
            },
        },
        "operational_change_signal": {
            "type": "noul",
            "instructions": (
                "Does this message trigger the following broad operational change rule? Return true "
                "if either the sender explicitly asserts an already-completed change in their own voice "
                "(including an endorsed quotation), OR requested_destination for the current transaction "
                "differs from baseline.destination. A differing current destination counts even with "
                "a denial of change, a question, or an unendorsed quotation. A future-only destination "
                "does not count. Do not require registry confirmation or permission."
            ),
        },
    }


def _variants(domain: dict) -> list[dict]:
    details, old, new = domain["details"], domain["old_text"], domain["new_text"]
    use_old = f"For this transaction today, please use {old}."
    use_new = f"For this transaction today, please use {new}."
    assertion = f"Our {details} have already changed to {new}. {use_new}"
    quote = f"A note from {domain['quoted_source']} says: 'Your {details} have already changed to {new}.' "
    # Gold is hand-assigned, not derived by a rule engine from the fixture state.
    # Order: explicit claim, different destination, confirmed change, permission,
    #        categorical record status, operational change signal.
    specs = [
        (
            "unchanged_repeat", f"Our {details} are unchanged. {use_old}", "old", "baseline", False,
            (False, False, False, True, "Unchanged", False),
            "The sender explicitly says the details are unchanged, and repeats the baseline destination.",
        ),
        (
            "explicit_unconfirmed", assertion, "new", "baseline", False,
            (True, True, False, False, "Unchanged", True),
            "The sender explicitly asserts an already-completed change, but the authoritative current record still has the baseline.",
        ),
        (
            "explicit_same_destination", f"Our {details} have already changed from the details in your baseline record. {use_old}",
            "old", "baseline", False,
            (True, False, False, True, "Unchanged", True),
            "The sender explicitly asserts a completed change but requests the baseline destination; the assertion still counts as a claim.",
        ),
        (
            "denied_redirect", f"Our {details} have not changed. {use_new}", "new", "baseline", False,
            (False, True, False, False, "Unchanged", True),
            "The sender denies a change while directing use of different details; that directive is not an explicit change assertion.",
        ),
        (
            "change_question", f"Have our {details} already changed to {new}? {use_new}", "new", "baseline", False,
            (False, True, False, False, "Unchanged", True),
            "The sender asks whether a change happened rather than asserting it, while separately requesting the different destination.",
        ),
        (
            "future_only",
            f"Starting tomorrow, our {details} will change to {new}; that change has not happened yet. {use_old}",
            "old", "baseline", False,
            (False, False, False, True, "Unchanged", False),
            "The only announced change is future-only; the sender explicitly says it has not happened and requests the baseline today.",
        ),
        (
            "same_destination", use_old, "old", "baseline", False,
            (False, False, False, True, "Unchanged", False),
            "The sender gives only a use instruction, with values identical to the baseline; there is no explicit change assertion.",
        ),
        (
            "different_destination", use_new, "new", "baseline", False,
            (False, True, False, False, "Unchanged", True),
            "The sender gives only a use instruction; its different values trigger the broad rule but are not an explicit assertion of change.",
        ),
        (
            "confirmed_without_claim", use_new, "new", "new", False,
            (False, True, True, True, "Changed", True),
            "The sender gives only a use instruction, while independent authoritative current evidence establishes changed details.",
        ),
        (
            "confirmed_with_claim", assertion, "new", "new", False,
            (True, True, True, True, "Changed", True),
            "The sender explicitly asserts a completed change, and independent authoritative current evidence confirms it.",
        ),
        (
            "unverified_record", assertion, "new", "unverified", False,
            (True, True, False, False, "Unknown", True),
            "The sender explicitly asserts a completed change, but the only current registry snapshot is unauthenticated.",
        ),
        (
            "conflicting_records", assertion, "new", "conflicting", False,
            (True, True, False, False, "Unknown", True),
            "The sender explicitly asserts a completed change, but equally authoritative current snapshots conflict without precedence.",
        ),
        (
            "attributed_quote", quote + f"I am forwarding their claim without endorsing it as true. {use_new}",
            "new", "baseline", False,
            (False, True, False, False, "Unchanged", True),
            "The completed-change assertion belongs to the quoted third party; the sender explicitly withholds endorsement.",
        ),
        (
            "adopted_quote", quote + f"I endorse that statement as true: our details have already changed. {use_new}",
            "new", "baseline", False,
            (True, True, False, False, "Unchanged", True),
            "The sender expressly adopts the quoted completed-change claim in their own voice, despite the unchanged registry.",
        ),
        (
            "explicit_paraphrase", f"We have already replaced our former {details} with {new}. {use_new}",
            "new", "baseline", False,
            (True, True, False, False, "Unchanged", True),
            "Already replacing the former details is an affirmative completed-change assertion, equivalent to saying they have changed.",
        ),
        (
            "explicit_distractor", assertion + " Separately, the office newsletter changed its font yesterday.",
            "new", "baseline", False,
            (True, True, False, False, "Unchanged", True),
            "The added newsletter sentence has no bearing on the sender's explicit destination-change claim, registry, or use policy.",
        ),
        (
            "verified_but_policy_hold", assertion, "new", "new", True,
            (True, True, True, False, "Changed", True),
            "The sender asserts a change and the registry confirms it, but a transaction hold independently blocks use.",
        ),
    ]
    return [
        dict(zip(("variant", "message", "destination", "record", "hold", "gold", "claim_reason"), spec, strict=True))
        for spec in specs
    ]


def _state(domain: dict, variant: dict) -> dict:
    now = "2026-02-10T12:00:00Z"
    record = {"source": domain["registry"], "authenticated": True, "effective_at": now, "destination": domain["old"]}
    snapshots = [record]
    if variant["record"] in {"new", "unverified"}:
        record["destination"] = domain["new"]
        record["authenticated"] = variant["record"] != "unverified"
    elif variant["record"] == "conflicting":
        snapshots.append({**record, "destination": domain["new"]})
    return deepcopy({
        "evaluated_at": now,
        "transaction": {"kind": domain["transaction_kind"], "reference": domain["reference"], "scheduled_at": now},
        "counterparty": {"name": domain["counterparty"]},
        "authority": {"registry_name": domain["registry"]},
        "baseline": {
            "source": domain["registry"], "authenticated": True,
            "captured_at": "2026-02-01T12:00:00Z", "destination": domain["old"],
        },
        "message": {"sender": domain["sender"], "sender_role": domain["sender_role"], "body": variant["message"]},
        "requested_destination": domain[variant["destination"]],
        "current_record_snapshots": snapshots,
        "use_policy": {
            "rule": (
                "Permit use only when authenticated snapshots from authority.registry_name effective at evaluated_at "
                "establish one nonconflicting current destination, requested_destination exactly matches it, "
                "and transaction_on_hold is false. No eligible snapshot or conflicting eligible snapshots deny "
                "permission. Sender assertions cannot override this rule."
            ),
            "transaction_on_hold": variant["hold"],
        },
    })


def _rationale(variant: dict) -> dict:
    evidence = {
        "baseline": "An authenticated current registry snapshot equals the trusted baseline: known Unchanged.",
        "new": "An authenticated current registry snapshot differs from the trusted baseline: established Changed.",
        "unverified": "The only current snapshot is unauthenticated, so no eligible current record establishes a value: Unknown, not known unchanged.",
        "conflicting": "Authenticated current snapshots disagree at the same effective time with no precedence: Unknown, not known unchanged.",
    }[variant["record"]]
    different = variant["destination"] == "new"
    if variant["hold"]:
        permission = "The transaction hold denies permission even though the requested destination is verified."
    elif variant["record"] in {"unverified", "conflicting"}:
        permission = "Unknown current registry state cannot satisfy the policy's verification requirement."
    elif variant["gold"][3]:
        permission = "The requested destination exactly matches the established current registry destination, and no hold applies."
    else:
        permission = "The requested destination differs from the established current registry destination; the sender cannot authorize an override."
    if different:
        operational = "The different destination requested now is sufficient for this broader rule, regardless of an explicit assertion."
    elif variant["gold"][0]:
        operational = "The explicit already-changed assertion is sufficient for the broad rule even though the requested destination equals the baseline."
    else:
        operational = "Neither an explicit already-changed assertion nor a different destination requested now is present."
    return {
        "explicit_change_claim": variant["claim_reason"],
        "different_destination": (
            "The destination requested for this transaction differs from the baseline."
            if different else "The destination requested for this transaction exactly equals the baseline."
        ),
        "record_confirms_change": evidence + " Confirmation is true only for established Changed.",
        "use_permitted": permission,
        "record_status": evidence,
        "operational_change_signal": operational,
    }


def _relations(domain: str) -> list[dict]:
    relations = []

    def add(kind: str, left: str, left_q: str, right: str, right_q: str, reason: str):
        relations.append({
            "id": f"{domain}.{kind}.{left}.{left_q}.{right}.{right_q}",
            "kind": kind,
            "left": {"case_id": f"{domain}.{left}", "question_id": left_q},
            "right": {"case_id": f"{domain}.{right}", "question_id": right_q},
            "reason": reason,
        })

    flips = [
        ("same_destination", "explicit_same_destination", "explicit_change_claim", "Adding an explicit completed-change assertion changes the narrow claim predicate while the requested destination stays at baseline."),
        ("same_destination", "explicit_same_destination", "operational_change_signal", "An explicit completed-change assertion independently satisfies the broad rule even without a different requested destination."),
        ("explicit_unconfirmed", "denied_redirect", "explicit_change_claim", "Only the assertion changes to a denial; the current redirect remains."),
        ("explicit_unconfirmed", "change_question", "explicit_change_claim", "Only the assertion changes to a question; the current redirect remains."),
        ("attributed_quote", "adopted_quote", "explicit_change_claim", "Changing withheld endorsement to explicit adoption changes the sender's own assertion."),
        ("same_destination", "different_destination", "different_destination", "The use request replaces baseline values with different values."),
        ("same_destination", "different_destination", "operational_change_signal", "Different current requested values trigger the broad rule even without an explicit change assertion."),
        ("same_destination", "different_destination", "use_permitted", "The differing requested destination no longer matches the unchanged verified registry."),
        ("explicit_unconfirmed", "confirmed_with_claim", "record_confirms_change", "Only the current registry destination changes; the sender message is identical."),
        ("explicit_unconfirmed", "confirmed_with_claim", "record_status", "Only the current registry destination changes from baseline to the new destination."),
        ("explicit_unconfirmed", "confirmed_with_claim", "use_permitted", "Only the registry update makes the unchanged request match verified current details."),
        ("different_destination", "confirmed_without_claim", "record_confirms_change", "Only independent registry evidence changes; neither sender message asserts a change."),
        ("confirmed_with_claim", "unverified_record", "record_confirms_change", "Removing snapshot authentication removes adequate confirmation without changing the supplied value."),
        ("confirmed_with_claim", "unverified_record", "record_status", "An unauthenticated current value gives Unknown, not established Changed."),
        ("confirmed_with_claim", "unverified_record", "use_permitted", "An unauthenticated snapshot cannot satisfy the policy."),
        ("confirmed_with_claim", "conflicting_records", "record_status", "Contradictory equally authoritative current evidence makes the net record status Unknown."),
        ("confirmed_with_claim", "verified_but_policy_hold", "use_permitted", "Only a transaction hold changes; the claim, requested values, and verified record stay the same."),
    ]
    for left, right, question_id, reason in flips:
        add("flip", left, question_id, right, question_id, reason)
    for variant, left_q, right_q, reason in (
        ("explicit_unconfirmed", "explicit_change_claim", "record_confirms_change", "A sender claim can be present while authoritative confirmation is absent."),
        ("explicit_unconfirmed", "explicit_change_claim", "use_permitted", "An explicit sender claim does not grant policy permission."),
        ("denied_redirect", "explicit_change_claim", "operational_change_signal", "The narrow assertion rule excludes a denied change; the broad rule includes the different current destination."),
        ("different_destination", "explicit_change_claim", "different_destination", "A proposed different destination is not itself an explicit assertion of completed change."),
        ("explicit_same_destination", "different_destination", "operational_change_signal", "The explicit assertion triggers the broad rule even though the requested destination is unchanged."),
        ("confirmed_without_claim", "explicit_change_claim", "record_confirms_change", "A registry can confirm changed details when the sender never asserts a change."),
        ("unchanged_repeat", "record_confirms_change", "use_permitted", "Verified unchanged current details are permitted without a change."),
        ("verified_but_policy_hold", "record_confirms_change", "use_permitted", "A confirmed change does not override a transaction hold."),
        ("attributed_quote", "explicit_change_claim", "operational_change_signal", "An unendorsed quotation is not the sender's assertion, but the current redirect triggers the broad rule."),
    ):
        add("question_contrast", variant, left_q, variant, right_q, reason)
    for variant in ("explicit_paraphrase", "explicit_distractor"):
        for question_id in _QUESTION_IDS:
            add("invariant", "explicit_unconfirmed", question_id, variant, question_id,
                "Equivalent assertion wording or an unrelated newsletter detail preserves the predicate's answer.")
    for question_id in ("different_destination", "record_confirms_change", "use_permitted", "record_status", "operational_change_signal"):
        add("invariant", "explicit_unconfirmed", question_id, "denied_redirect", question_id,
            "Changing assertion to denial preserves the requested destination, verified registry, and policy; the broad rule still triggers.")
    for question_id in ("explicit_change_claim", "different_destination", "record_confirms_change", "record_status", "operational_change_signal"):
        add("invariant", "confirmed_with_claim", question_id, "verified_but_policy_hold", question_id,
            "The transaction hold affects permission only.")
    return relations


def build_bank_suite() -> dict:
    """Return independent JSON-compatible diagnostic cases and relation metadata."""
    cases, relations = [], []
    for domain in _DOMAINS:
        for variant in _variants(domain):
            cases.append({
                "id": f"{domain['domain']}.{variant['variant']}",
                "family_id": domain["family_id"],
                "domain": domain["domain"],
                "variant": variant["variant"],
                "request": {"state": _state(domain, variant), "questions": _questions(domain["details"])},
                "expected": dict(zip(_QUESTION_IDS, variant["gold"], strict=True)),
                "rationale": _rationale(variant),
            })
        relations.extend(_relations(domain["domain"]))
    return {"cases": cases, "relations": relations, "contract": _CONTRACT}
