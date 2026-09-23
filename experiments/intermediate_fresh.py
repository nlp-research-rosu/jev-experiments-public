"""Blind, independently authored fresh evaluation material for supervision v1.

Only private raw observations and explicit supplied-rule programs determine the
targets.  Public STATE contains the observations and rule definitions, never
the derived oracle feature map or an answer label.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.contrast_scaling_logic import derive_features, eval_rule, select_choice, select_level
from openjev.judgments import compile_request


CATEGORIES = (
    "claim_vs_completion",
    "permission_vs_execution",
    "unknown_vs_failure",
    "attribution_and_endorsement",
    "entity_binding",
    "action_binding",
    "temporal_scope",
    "reversal_and_current_state",
    "negation_and_quantifiers",
    "ordered_rubrics",
)
QUESTION_IDS = ("n1", "n2", "n3", "c1", "s1")
SPLITS = ("validation", "calibration", "confirmation")
OUTPUT = Path("data/intermediate-supervision-v1/fresh-draft")
AUTHORING = Path("data/intermediate-supervision-v1/authoring")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _answer_space(question):
    if question["type"] == "noul":
        return {False, True}
    if question["type"] == "score":
        return set(range(len(question["criteria"])))
    return set(question["criteria"])


def validate_public_suite(suite):
    """CPU-only validation of the established public suite interface."""
    cases = suite["cases"]
    by_id = {case["id"]: case for case in cases}
    if not cases or len(by_id) != len(cases):
        raise ValueError("suite needs nonempty unique cases")
    for case in cases:
        request = case["request"]
        if set(request) != {"state", "questions"}:
            raise ValueError("only state and questions may enter the model")
        compile_request(request)
        if set(case["expected"]) != set(request["questions"]) or set(case["rationale"]) != set(case["expected"]):
            raise ValueError("targets and rationales must cover every question")
        for qid, question in request["questions"].items():
            if case["expected"][qid] not in _answer_space(question):
                raise ValueError("invalid target")
    seen = set()
    for relation in suite["relations"]:
        if relation["id"] in seen or relation["kind"] not in {"flip", "question_contrast", "invariant", "layout_invariant"}:
            raise ValueError("invalid relation")
        seen.add(relation["id"])
        left, right = relation["left"], relation["right"]
        a, b = by_id[left["case_id"]], by_id[right["case_id"]]
        qa, qb = a["request"]["questions"][left["question_id"]], b["request"]["questions"][right["question_id"]]
        ya, yb = a["expected"][left["question_id"]], b["expected"][right["question_id"]]
        if a["family_id"] != b["family_id"] or qa["type"] != qb["type"] or _answer_space(qa) != _answer_space(qb):
            raise ValueError("relation endpoint mismatch")
        if (ya == yb) != (relation["kind"] in {"invariant", "layout_invariant"}):
            raise ValueError("relation contradicts targets")
        if relation["kind"] == "question_contrast" and a["id"] != b["id"]:
            raise ValueError("question contrast must preserve its case")
    return True


def F(name):
    return {"feature": name}


def AND(*values):
    return {"all": list(values)}


def OR(*values):
    return {"any": list(values)}


def NOT(value):
    return {"not": value}


def CMP(name, operator, number):
    return {"compare": {"left": F(name), "operator": operator, "right": number}}


FEATURE_TEXT = {
    "intent": "a matching positive future statement is present",
    "claim": "a matching positive completed-action statement is present",
    "denied_claim": "a matching negative completed-action statement is present",
    "endorsement": "a matching completed-action statement is explicitly adopted by its source",
    "approved": "the latest authentic in-window approval decision is unambiguously allow",
    "forbidden": "the latest authentic in-window approval decision is unambiguously deny",
    "approval_unknown": "the latest qualifying approval does not establish exactly allow or exactly deny",
    "invocation": "an in-window execution entry matches actor, operation, target, and run",
    "completed": "matching authentic live evidence establishes success without a matching no_effect conflict",
    "failed": "matching authentic live evidence establishes no_effect without a matching success conflict",
    "unknown": "neither completion nor no_effect failure is established",
    "conflict": "both matching authentic live success and no_effect evidence are present",
    "success_report": "a matching in-window execution entry reports success, regardless of trust or mode",
    "failure_report": "a matching in-window execution entry reports no_effect, regardless of trust or mode",
    "any_live_success": "some authentic in-window live execution reports success",
    "other_live_success": "an authentic in-window live success differs from the query on at least one binding",
    "actor_seen": "some execution entry names the queried actor",
    "operation_seen": "some execution entry names the queried operation",
    "target_seen": "some execution entry names the queried target",
    "run_seen": "some execution entry names the queried run",
    "reversed": "a qualifying later authentic live reversal names an established successful run",
    "effective": "completion is established and no qualifying later reversal removes its effect",
    "corroborated": "a different trusted source later corroborates the matching successful run",
    "current_known": "the latest authentic target-state reports by window end agree on active or inactive",
    "earlier_known": "the latest authentic target-state reports by query_time agree on active or inactive",
    "active_earlier": "the agreed latest authentic target state at query_time is active",
    "inactive_earlier": "the agreed latest authentic target state at query_time is inactive",
    "active_now": "the agreed latest authentic target state at window end is active",
    "inactive_now": "the agreed latest authentic target state at window end is inactive",
    "ever_active": "an authentic target-state report at or before window end says active",
    "inventory_complete": "the inventory is explicitly marked complete",
    "all_success": "the complete scoped inventory contains only known success statuses",
    "any_success": "at least one scoped item has a known success status",
    "any_failure": "at least one scoped item has a known no_effect status",
    "item_unknown": "at least one scoped item is unresolved or the inventory is not complete",
    "has_measurement": "a finite numeric measurement is present",
}


def describe(rule):
    if rule is True:
        return "the fallback condition, which always applies"
    if rule is False:
        return "a condition that never applies"
    operator, value = next(iter(rule.items()))
    if operator == "feature":
        return FEATURE_TEXT[value]
    if operator == "not":
        return "NOT (" + describe(value) + ")"
    if operator in {"all", "any"}:
        joiner = " AND " if operator == "all" else " OR "
        return joiner.join("(" + describe(item) + ")" for item in value)
    if operator == "at_least":
        return f"at least {value['count']} of these hold: " + "; ".join(describe(item) for item in value["of"])
    if operator == "compare":
        left = value["left"]["feature"]
        labels = {
            "measurement": "the recorded measurement",
            "success_count": "the number of scoped items with known success",
            "failure_count": "the number of scoped items with known no_effect",
            "unknown_count": "the number of unresolved scoped items",
            "item_count": "the number of scoped item identifiers",
        }
        return f"{labels[left]} {value['operator']} {value['right']}"
    raise ValueError(f"unsupported rule operator {operator}")


# Each entry is deliberately authored as a scenario/language frame.  Controlled
# e/r variants share the entry by design; entries do not arise from name filling.
FRAME_BANK = {
    "claim_vs_completion": [
        ("museum-crate", "collections", "A registrar's closing note", "A museum registrar says the climate crate was sealed; the control-room trace is the separate execution record.", "Read the signed note and the machine trace as different kinds of evidence."),
        ("orchard-gate", "agriculture", "Orchard handoff ledger", "At the north orchard, a crew lead reports the irrigation gate closed while the pump console records commands independently.", "Do not turn the crew's report into a console event."),
        ("clinic-freezer", "clinical logistics", "Freezer transfer brief", "A pharmacy courier's transfer brief states that a cold-chain bin was locked; device telemetry appears below on its own terms.", "The brief can establish a claim even when telemetry establishes nothing."),
        ("stage-rig", "theater operations", "Fly-system cue sheet", "The stage manager marks a lighting truss as lowered, beside a separate rig-controller history.", "A cue-sheet assertion and an executed motor run remain distinct."),
        ("harbor-boom", "port operations", "Harbor watch memorandum", "A watch officer writes that the containment boom was deployed; launch-control observations are listed afterward.", "Treat narrative reporting and operational completion as separate predicates."),
        ("archive-vault", "archives", "Vault access folio", "An archivist records that the manuscript vault was latched, with access hardware observations kept in a second section.", "The folio records what was said as well as what the hardware showed."),
        ("lab-shutter", "laboratory safety", "Laser-room turnover card", "A technician checks a box saying the beam shutter was closed; an interlock log independently records actuation.", "The checked box is a statement, not an inferred actuator success."),
        ("ferry-ramp", "marine transit", "Ferry departure packet", "The deck chief reports the vehicle ramp raised while the hydraulic controller supplies a separate run history.", "Keep the human completion claim separate from the hydraulic evidence."),
    ],
    "permission_vs_execution": [
        ("roof-hatch", "facilities", "Storm-prep authorization", "A facilities desk decides whether the roof hatch may be locked; the maintenance controller records any actual lock cycle.", "Permission answers may, while controller evidence answers did."),
        ("quarry-blast", "quarry safety", "Quarry clearance board", "The safety board may authorize a warning siren before a scheduled blast, and the siren controller separately records activation.", "An allowed action is not thereby an executed action."),
        ("server-drain", "site reliability", "Maintenance change ticket", "A change manager rules on draining a service node; orchestration events record whether the drain command took effect.", "Read the decision and orchestration trail independently."),
        ("gallery-lights", "museum operations", "After-hours lighting permit", "A curator may permit the west gallery lights to dim, while the building system logs its own run.", "Authorization is a prerequisite in this story, not proof of execution."),
        ("reservoir-valve", "water operations", "Reservoir control docket", "A duty engineer records allow or deny for opening a bypass valve; the actuator ledger reports physical command outcomes.", "The docket exposes both decision status and actuator status."),
        ("rail-switch", "rail dispatch", "Siding movement release", "Dispatch can release a switch movement, and a wayside unit later reports whether the points moved.", "A release and a successful movement answer different questions."),
        ("cleanroom-cycle", "manufacturing", "Cleanroom cycle waiver", "A supervisor may approve a decontamination cycle; chamber controls preserve separate execution evidence.", "Do not promote a waiver decision into a completed cycle."),
        ("satellite-pass", "ground operations", "Antenna pass clearance", "A flight director clears a tracking maneuver, while the antenna controller records the actual slew attempt.", "Clearance and controller outcome remain separately observable."),
    ],
    "unknown_vs_failure": [
        ("bakery-oven", "food production", "Oven purge incident card", "A bakery controller attempts to purge an oven; a timeout and a verified no-effect report have different meanings.", "Unknown means unresolved by the qualifying evidence, not successful or failed by default."),
        ("aquarium-pump", "animal care", "Life-support pump log", "Aquarium staff inspect a pump restart whose controller can time out or report a verified no-effect result.", "Silence and timeout do not establish the opposite outcome."),
        ("warehouse-lift", "warehouse safety", "Lift reset worksheet", "A dock lift reset produces operational entries that may be inconclusive or may establish no effect.", "Separate unresolved outcome from an authenticated failure."),
        ("observatory-dome", "observatory operations", "Dome rotation exception", "The dome-control service records a rotation attempt, including timeout versus live no-effect status.", "A recorded attempt can still leave the result unknown."),
        ("greenhouse-mister", "horticulture", "Mister restart report", "A greenhouse misting zone is restarted under a named run, with evidence that distinguishes no answer from no effect.", "Failure requires its specified positive evidence."),
        ("library-sorter", "library automation", "Sorter recovery slip", "A returns sorter recovery run may stop at timeout or produce a trusted live no-effect record.", "The absence of established completion is not itself a failed run."),
        ("tram-door", "transit maintenance", "Door-cycle troubleshooting log", "A tram door-cycle test carries a run identifier and an outcome record that can remain unresolved.", "Use three-way outcome reasoning: completion, failure, or unknown."),
        ("brewery-chiller", "beverage production", "Chiller restart notebook", "A fermentation chiller restart is logged once by the controller, with timeout and no-effect treated differently.", "Only qualifying live evidence establishes the failure predicate."),
    ],
    "attribution_and_endorsement": [
        ("field-radio", "field research", "Radio relay transcript", "A field coordinator relays a surveyor's completion statement and marks whether the coordinator adopts it.", "Quoting a completion statement differs from endorsing it."),
        ("news-desk", "publishing", "Copy-desk attribution sheet", "An editor records a source's claim that a correction shipped and separately marks editorial endorsement.", "Attribution preserves who asserted what; endorsement records adoption."),
        ("embassy-pouch", "diplomatic logistics", "Pouch handoff cable", "A desk officer repeats that a pouch was transferred, with an explicit adopted-or-not flag.", "A repeated assertion remains a claim even without endorsement."),
        ("storm-spotter", "weather operations", "Spotter relay bulletin", "A dispatcher relays a spotter's report that a beacon was activated and can endorse that report separately.", "The bulletin distinguishes carrying a report from standing behind it."),
        ("film-dailies", "film production", "Dailies processing note", "A production assistant records that a lab says the reel was synced, then indicates whether production endorses the statement.", "The source's assertion and the recorder's adoption are separate facts."),
        ("court-exhibit", "legal operations", "Exhibit transfer minute", "A clerk enters counsel's statement that an exhibit was sealed and separately records adoption of that statement.", "Do not infer endorsement from faithful transcription."),
        ("expedition-cache", "expedition logistics", "Base-camp relay card", "Base camp repeats a scout's report that a supply cache was marked, with an explicit endorsement marker.", "Relay provenance matters even when the wording says completed."),
        ("factory-shift", "industrial operations", "Shift-change narration", "An outgoing foreperson says a guard was installed; the incoming foreperson may merely quote or explicitly adopt it.", "The record makes assertion and endorsement independently visible."),
    ],
    "entity_binding": [
        ("api-tenant", "cloud operations", "Tenant recovery trace", "A recovery event succeeds for a neighboring tenant before the queried tenant's record is corrected.", "Bind success to the exact target, not to a nearby entity."),
        ("patient-chart", "clinical administration", "Chart merge audit", "A merge succeeds on a similarly named patient chart, then the target chart identifier is examined.", "Entity identity is part of the completion condition."),
        ("cargo-bay", "air freight", "Cargo-bay latch log", "A latch controller reports success for bay B while the query concerns bay A.", "A success on the wrong bay is operationally real but irrelevant to the queried entity."),
        ("classroom-projector", "campus support", "Classroom device ticket", "A remote command succeeds on the projector next door before technicians inspect the requested room.", "Match the target device exactly."),
        ("forest-sensor", "environmental monitoring", "Sensor reset ledger", "A reset succeeds on a neighboring watershed sensor rather than the named sensor.", "Nearby equipment does not satisfy the queried binding."),
        ("hotel-room", "hospitality", "Room-control service note", "A privacy-mode command succeeds for an adjacent room and is later compared with the requested room.", "The room identifier is not interchangeable context."),
        ("robot-bin", "fulfillment", "Picking-cell run sheet", "A robot confirms a placement into bin 18 while the query names bin 19.", "Completion follows the exact object binding."),
        ("vaccine-lot", "medical logistics", "Lot quarantine register", "A quarantine command succeeds for a sibling vaccine lot before the specified lot is checked.", "A correct action on the wrong lot remains a decoy success."),
    ],
    "action_binding": [
        ("dam-gate", "hydrology", "Spillway command history", "The controller successfully tests a gate while the query asks whether it opened the gate.", "Match the operation verb as well as the equipment."),
        ("printer-queue", "office systems", "Print-service trace", "A service successfully pauses a queue when the requested operation is to purge it.", "Success for another operation does not transfer."),
        ("drone-camera", "aerial survey", "Payload command sheet", "A drone successfully focuses its camera while the query concerns capturing an image.", "Keep action identity attached to the run."),
        ("kiln-cycle", "ceramics", "Kiln controller journal", "The kiln successfully vents even though the operator asks about starting the firing cycle.", "A successful neighboring command is still the wrong action."),
        ("canal-lock", "canal operations", "Lock-house event strip", "A lock controller acknowledges filling while the query asks about draining the chamber.", "Operation binding decides relevance."),
        ("backup-vault", "data protection", "Backup appliance history", "An appliance verifies a snapshot while the requested action is replication to the vault.", "Do not collapse related maintenance verbs."),
        ("farm-feeder", "animal husbandry", "Automated feeder log", "The feeder successfully weighs a ration when the query concerns dispensing it.", "The same actor and target do not rescue a mismatched operation."),
        ("telescope-filter", "astronomy", "Instrument wheel trace", "A telescope calibrates its filter wheel while the requested operation is changing filters.", "Action names remain binding even inside one instrument."),
    ],
    "temporal_scope": [
        ("festival-sign", "event operations", "Festival sign timeline", "A wayfinding sign is active at the checkpoint and has a later state by closing time.", "Answer earlier and current-state questions at their own cutoffs."),
        ("ice-rink", "recreation", "Rink refrigeration timeline", "The refrigeration loop has an agreed state at inspection time and another at the window end.", "Later evidence may change current state without rewriting the earlier state."),
        ("research-cluster", "computing", "Cluster maintenance chronology", "A compute partition is active at the query checkpoint, followed by a later state event.", "Use the latest authentic report at each requested time."),
        ("market-alarm", "public safety", "Market alarm chronology", "A market alarm is reported active during inspection and has a subsequent end-of-window report.", "Historical and current predicates can diverge."),
        ("river-beacon", "navigation", "Beacon status timeline", "A river beacon's checkpoint state is followed by a later status transmission.", "The time cutoff belongs to the predicate, not to the prose order."),
        ("nursery-heater", "horticulture", "Propagation-house chronology", "A nursery heater is active at an early checkpoint and later receives a closing state report.", "Retain both the earlier state and the current state."),
        ("arena-screen", "venue operations", "Scoreboard state history", "An arena screen has an active pre-event report and a later report before the window ends.", "Select the latest qualifying event separately for each cutoff."),
        ("coastal-siren", "emergency management", "Coastal siren state history", "A warning siren is active at the exercise checkpoint and later reports its end state.", "A later reversal changes now, not what was true earlier."),
    ],
    "reversal_and_current_state": [
        ("bridge-lane", "traffic control", "Bridge lane-control journal", "A lane closure succeeds and may later be reversed by reopening the same run.", "Historical completion can coexist with a no-longer-effective state."),
        ("database-freeze", "data operations", "Write-freeze change log", "A database write freeze completes, then a later unfreeze may explicitly reverse it.", "The reversal does not erase the original completion evidence."),
        ("gallery-rope", "visitor safety", "Gallery barrier record", "A gallery barrier is installed successfully and can later be removed under a linked reversal.", "Judge completed, reversed, and effective separately."),
        ("runway-closure", "airport operations", "Runway status action log", "A runway closure succeeds before a linked reopening event may undo its effect.", "Current effectiveness depends on a qualifying reversal."),
        ("shipping-hold", "commerce", "Shipment hold chronology", "A hold is applied to a shipment and can later be released by an event naming the original run.", "A released hold was still historically applied."),
        ("lab-quarantine", "biosafety", "Quarantine action ledger", "A sample freezer is quarantined, then a later clearance may reverse that exact action.", "Reversal requires the explicit run link and valid timing."),
        ("school-lockdown", "school operations", "Door policy event book", "A building lockdown completes and is later eligible for a linked all-clear reversal.", "Do not confuse past completion with present effectiveness."),
        ("pipeline-bypass", "energy operations", "Bypass command chronology", "A pipeline bypass engages successfully and may later be disengaged by a linked command.", "The ledger supports both historical and current-state questions."),
    ],
    "negation_and_quantifiers": [
        ("mail-trays", "postal operations", "Mail-tray reconciliation", "Four outgoing trays have individually reported statuses, including unresolved and no-effect entries.", "Quantifiers range only over the listed tray identifiers."),
        ("seed-batches", "agriculture", "Seed-batch treatment roll", "A treatment roll lists four seed batches whose verified outcomes may be success, no effect, or unresolved.", "Unknown status does not count as either success or failure."),
        ("lab-aliquots", "laboratory operations", "Aliquot processing census", "Four aliquots form the declared scope for a processing-status census.", "All means every scoped identifier and also requires a complete inventory."),
        ("bus-inspections", "transit safety", "Bus inspection roster", "A roster names four buses with verified inspection outcomes or missing resolution.", "Negated and existential questions use the same explicit roster."),
        ("orchard-zones", "agriculture", "Orchard-zone survey", "Four irrigation zones are scoped, each with a verified outcome when one is available.", "A missing row remains unknown rather than becoming no_effect."),
        ("museum-cases", "conservation", "Display-case seal census", "A conservator audits four display cases and records known success or no-effect status per case.", "Inventory completeness and item resolution are distinct."),
        ("relief-pallets", "disaster logistics", "Relief-pallet checklist", "Four pallets make up the delivery scope, with some statuses resolved and others absent.", "Count only scoped identifiers, even if other observations exist."),
        ("solar-strings", "energy maintenance", "Solar-string restart census", "A crew reviews four solar strings whose restart outcomes can be successful, no effect, or unknown.", "Existential success and universal success are different claims."),
    ],
    "ordered_rubrics": [
        ("water-turbidity", "water quality", "Turbidity threshold card", "A treatment outlet has one finite turbidity observation to compare with two explicit ordered scales.", "Select the highest threshold level satisfied."),
        ("battery-charge", "energy storage", "Battery reserve card", "A field battery's reserve measurement is scored under two different threshold schedules.", "The same number can receive different rubric levels."),
        ("wind-speed", "weather operations", "Wind advisory ruler", "A mast reports a finite wind-speed reading for two ordered advisory rubrics.", "Threshold definitions, not intuition, determine the level."),
        ("grain-moisture", "agriculture", "Grain moisture grading slip", "A silo sample carries one numeric moisture reading and two supplied grading scales.", "Use numeric comparisons exactly as written."),
        ("queue-depth", "computing", "Queue-pressure gauge", "A worker queue exposes one depth measurement to two ordered escalation rubrics.", "Higher levels correspond to higher stated cutoffs."),
        ("room-occupancy", "building operations", "Occupancy response scale", "A room counter provides a finite occupancy value for two response schedules.", "Apply every threshold and retain the highest satisfied index."),
        ("vibration-level", "industrial maintenance", "Bearing vibration scale", "A bearing monitor produces one vibration reading to classify under two explicit ladders.", "Missing intuition is irrelevant because every level is defined."),
        ("snow-load", "infrastructure", "Roof snow-load ruler", "A roof sensor reports a finite load value evaluated by two ordered action scales.", "A rubric change can move the score while evidence stays fixed."),
    ],
}


STYLE_NAMES = ("chronicle", "docket", "brief", "transcript", "worksheet", "dispatch", "audit", "logbook")


def _frames():
    frames = []
    for category in CATEGORIES:
        for index, (slug, domain, heading, intro, caution) in enumerate(FRAME_BANK[category]):
            frame_id = f"fresh-v1/{category}/{slug}"
            frames.append({
                "frame_id": frame_id,
                "family_id": frame_id,
                "category": category,
                "index": index,
                "domain": domain,
                "heading": heading,
                "intro": intro,
                "caution": caution,
                "style": STYLE_NAMES[(index + CATEGORIES.index(category)) % len(STYLE_NAMES)],
                "author": "blind-fresh-author",
            })
    return frames


def _assign_splits(frames):
    by_category = defaultdict(list)
    for frame in frames:
        by_category[frame["category"]].append(frame)
    for category, rows in by_category.items():
        ordered = sorted(rows, key=lambda row: hashlib.sha256(("fresh-split-v1/" + row["frame_id"]).encode()).hexdigest())
        for position, frame in enumerate(ordered):
            frame["split"] = "validation" if position < 2 else "calibration" if position < 4 else "confirmation"
            frame["split_hash"] = hashlib.sha256(("fresh-split-v1/" + frame["frame_id"]).encode()).hexdigest()
    return frames


def _identity(frame):
    slug = frame["frame_id"].rsplit("/", 1)[-1]
    words = slug.split("-")
    return {
        "actor": f"{words[0]} operator",
        "operation": " ".join(words[1:] or ["operate"]),
        "target": f"{slug} asset",
        "run_id": f"{slug}-run-{frame['index'] + 1}",
    }


def _base(frame):
    query = _identity(frame)
    return {
        "query": query,
        "window": [10, 30],
        "query_time": 18,
        "trusted_sources": [f"{frame['frame_id'].rsplit('/', 1)[-1]} ledger", "independent inspector"],
        "statements": [],
        "executions": [],
        "approvals": [],
        "reversals": [],
        "corroborations": [],
        "state_events": [],
        "item_ids": ["unit-a", "unit-b", "unit-c", "unit-d"],
        "items": [],
        "inventory_complete": True,
        "measurement": None,
    }


def _execution(record, **changes):
    return {
        **record["query"],
        "source": record["trusted_sources"][0],
        "time": 22,
        "verified": True,
        "mode": "live",
        "outcome": "success",
        **changes,
    }


def _statement(record, **changes):
    row = {
        "actor": record["query"]["actor"],
        "operation": record["query"]["operation"],
        "target": record["query"]["target"],
        "source": "named witness",
        "time": 16,
        "tense": "completed",
        "polarity": "positive",
        "endorsed": False,
        **changes,
    }
    return row


def _records(frame):
    category, index = frame["category"], frame["index"]
    left = _base(frame)
    right = copy.deepcopy(left)
    if category == "claim_vs_completion":
        left["statements"] = [_statement(left, endorsed=index % 3 == 0)]
        right = copy.deepcopy(left)
        right["executions"] = [_execution(right)]
    elif category == "permission_vs_execution":
        approval = _execution(left, decision="allow", time=14)
        approval = {key: approval[key] for key in ("actor", "operation", "target", "source", "time", "verified", "decision")}
        left["approvals"] = [approval]
        right = copy.deepcopy(left)
        right["executions"] = [_execution(right)]
    elif category == "unknown_vs_failure":
        unresolved = (
            {"outcome": "timeout"},
            {"outcome": "no_effect", "mode": "simulation"},
            {"outcome": "no_effect", "verified": False},
            {"outcome": "no_effect", "source": "untrusted observer"},
            {"outcome": "no_effect", "time": 35},
            {"outcome": "no_effect", "run_id": "neighbor-run"},
            {"outcome": "timeout"},
            {"outcome": "no_effect", "mode": "simulation"},
        )[index]
        left["executions"] = [_execution(left, **unresolved)]
        right = copy.deepcopy(left)
        # Each pair changes only the field that prevents the no_effect row from
        # qualifying (or changes timeout to no_effect for timeout frames).
        if left["executions"][0]["outcome"] == "timeout":
            right["executions"][0]["outcome"] = "no_effect"
        elif left["executions"][0]["mode"] == "simulation":
            right["executions"][0]["mode"] = "live"
        elif left["executions"][0]["verified"] is False:
            right["executions"][0]["verified"] = True
        elif left["executions"][0]["source"] == "untrusted observer":
            right["executions"][0]["source"] = right["trusted_sources"][0]
        elif left["executions"][0]["time"] > right["window"][1]:
            right["executions"][0]["time"] = 22
        else:
            right["executions"][0]["run_id"] = right["query"]["run_id"]
    elif category == "attribution_and_endorsement":
        left["statements"] = [_statement(left, source=f"relay source {index + 1}", endorsed=False)]
        left["executions"] = [_execution(left)]
        right = copy.deepcopy(left)
        right["statements"][0]["endorsed"] = True
    elif category == "entity_binding":
        left["executions"] = [_execution(left, target=f"neighboring {left['query']['target']}")]
        if index % 2:
            left["executions"].append(_execution(left, actor="background operator", target="background asset", run_id="background-run"))
        right = copy.deepcopy(left)
        right["executions"][0]["target"] = right["query"]["target"]
    elif category == "action_binding":
        left["executions"] = [_execution(left, operation=f"inspect {left['query']['operation']}")]
        if index % 2:
            left["executions"].append(_execution(left, actor="background operator", operation="unrelated action", target="background asset", run_id="background-run"))
        right = copy.deepcopy(left)
        right["executions"][0]["operation"] = right["query"]["operation"]
    elif category == "temporal_scope":
        left["state_events"] = [
            {"target": left["query"]["target"], "source": left["trusted_sources"][0], "time": 16, "verified": True, "active": True},
            {"target": left["query"]["target"], "source": left["trusted_sources"][0], "time": 26, "verified": True, "active": False},
        ]
        right = copy.deepcopy(left)
        right["state_events"][1]["active"] = True
    elif category == "reversal_and_current_state":
        left["executions"] = [_execution(left)]
        if index % 2 == 0:
            left["corroborations"] = [_execution(left, source=left["trusted_sources"][1], time=24)]
        right = copy.deepcopy(left)
        reversal = _execution(right, time=27, reverses=right["query"]["run_id"])
        right["reversals"] = [reversal]
    elif category == "negation_and_quantifiers":
        left["items"] = [
            {"id": "unit-a", "status": "no_effect", "verified": True},
            {"id": "unit-b", "status": "no_effect", "verified": True},
        ]
        right = copy.deepcopy(left)
        right["items"].append({"id": "unit-c", "status": "success", "verified": True})
    elif category == "ordered_rubrics":
        left["measurement"] = 24 + index * 3
        right = copy.deepcopy(left)
        right["measurement"] = 64 + index * 3
    else:
        raise ValueError(category)
    return [left, right]


def _truth_table_choice(frame, first, second, first_label, second_label):
    prefix = frame["frame_id"].rsplit("/", 1)[-1].replace("-", " ").title()
    return {
        f"{prefix}: both": AND(first, second),
        f"{prefix}: {first_label} only": AND(first, NOT(second)),
        f"{prefix}: {second_label} only": AND(NOT(first), second),
        f"{prefix}: neither": AND(NOT(first), NOT(second)),
    }


def _noul_and_choice(frame):
    category, index = frame["category"], frame["index"]
    configurations = {
        "claim_vs_completion": ([F("claim"), F("completed"), AND(F("claim"), F("completed")), F("endorsement"), F("unknown")], (F("claim"), F("completed"), "claim", "completion")),
        "permission_vs_execution": ([F("approved"), F("completed"), AND(F("approved"), F("completed")), F("invocation"), F("forbidden")], (F("approved"), F("completed"), "permission", "completion")),
        "unknown_vs_failure": ([F("unknown"), F("failed"), F("completed"), F("failure_report"), F("invocation")], (F("unknown"), F("failed"), "unknown", "failure")),
        "attribution_and_endorsement": ([F("claim"), F("endorsement"), F("completed"), AND(F("claim"), F("endorsement")), F("corroborated")], (F("endorsement"), F("completed"), "endorsement", "completion")),
        "entity_binding": ([F("actor_seen"), F("completed"), F("other_live_success"), F("target_seen"), F("success_report")], (F("completed"), F("other_live_success"), "matching completion", "other-entity success")),
        "action_binding": ([F("operation_seen"), F("completed"), F("other_live_success"), F("actor_seen"), F("success_report")], (F("completed"), F("other_live_success"), "matching completion", "other-action success")),
        "temporal_scope": ([F("active_earlier"), F("active_now"), F("ever_active"), F("inactive_now"), F("current_known")], (F("active_earlier"), F("active_now"), "active earlier", "active now")),
        "reversal_and_current_state": ([F("completed"), F("effective"), F("reversed"), F("corroborated"), F("unknown")], (F("effective"), F("reversed"), "effective", "reversed")),
        "negation_and_quantifiers": ([F("any_success"), F("any_failure"), F("item_unknown"), F("all_success"), F("inventory_complete")], (F("any_failure"), F("item_unknown"), "known failure", "unresolved item")),
        "ordered_rubrics": ([F("has_measurement"), CMP("measurement", ">=", 40 + index), CMP("measurement", "<", 80 + index), CMP("measurement", ">=", 60 + index), CMP("measurement", "<", 30 + index)], (CMP("measurement", ">=", 40 + index), CMP("measurement", ">=", 70 + index), "middle threshold", "high threshold")),
    }
    pool, choice = configurations[category]
    # The first two are deliberate semantic foils for this category.  The third
    # rotates among additional predicates so frames vary without losing the
    # within-record question contrast that the design requires.
    extra_index = 2 + (index % 2 if category == "action_binding" else index % 3)
    nouls = [copy.deepcopy(pool[0]), copy.deepcopy(pool[1]), copy.deepcopy(pool[extra_index])]
    choices = _truth_table_choice(frame, choice[0], choice[1], choice[2], choice[3])
    return nouls, choices


def _first_conditions(category, index):
    keys = {
        "claim_vs_completion": ("claim", "completed", "endorsement", "corroborated"),
        "permission_vs_execution": ("approved", "completed", "invocation", "corroborated"),
        "unknown_vs_failure": ("actor_seen", "failed", "invocation", "unknown"),
        "attribution_and_endorsement": ("claim", "endorsement", "completed", "corroborated"),
        "entity_binding": ("actor_seen", "completed", "target_seen", "other_live_success"),
        "action_binding": ("actor_seen", "completed", "operation_seen", "other_live_success"),
        "temporal_scope": ("active_earlier", "active_now", "current_known", "ever_active"),
        "reversal_and_current_state": ("completed", "effective", "reversed", "corroborated"),
    }[category]
    true_key, false_key, third, fourth = keys
    shape = index % 4
    true_condition = (
        F(true_key),
        AND(F(true_key), NOT(F("conflict"))),
        OR(F(true_key), F(third)),
        {"at_least": {"count": 1, "of": [F(true_key), F(fourth)]}},
    )[shape]
    false_condition = (
        F(false_key),
        AND(F(false_key), F(true_key)),
        AND(F(false_key), NOT(F("conflict"))),
        {"at_least": {"count": 2, "of": [F(false_key), F(true_key)]}},
    )[shape]
    tail = [F(third), F(fourth), NOT(F("conflict")), OR(F(third), F(fourth))]
    shift = index % len(tail)
    return true_condition, false_condition, tail[shift:] + tail[:shift]


def _score_programs(frame):
    category, index = frame["category"], frame["index"]
    width = (2, 3, 4, 5)[index % 4]
    if category == "ordered_rubrics":
        offset = index * 3
        checklists = (
            [CMP("measurement", ">=", threshold + offset) for threshold in (20, 40, 60, 80)],
            [CMP("measurement", ">=", threshold + offset) for threshold in (30, 50, 70, 90)],
        )
    else:
        names = {
            "claim_vs_completion": ("claim", "completed", "endorsement", "unknown", "effective", "denied_claim", "invocation", "corroborated"),
            "permission_vs_execution": ("approved", "completed", "invocation", "unknown", "forbidden", "approval_unknown", "effective", "corroborated"),
            "unknown_vs_failure": ("unknown", "failed", "invocation", "failure_report", "actor_seen", "completed", "conflict", "success_report"),
            "attribution_and_endorsement": ("claim", "endorsement", "completed", "unknown", "denied_claim", "intent", "corroborated", "effective"),
            "entity_binding": ("actor_seen", "target_seen", "completed", "other_live_success", "success_report", "invocation", "unknown", "any_live_success"),
            "action_binding": ("actor_seen", "operation_seen", "completed", "other_live_success", "success_report", "invocation", "unknown", "any_live_success"),
            "temporal_scope": ("active_earlier", "active_now", "inactive_now", "earlier_known", "current_known", "ever_active", "inactive_earlier", "current_known"),
            "reversal_and_current_state": ("completed", "effective", "reversed", "corroborated", "unknown", "invocation", "success_report", "any_live_success"),
            "negation_and_quantifiers": ("any_success", "any_failure", "item_unknown", "all_success", "inventory_complete", "unknown", "completed", "conflict"),
        }[category]
        atoms = [F(name) for name in names]
        candidates = atoms + [
            AND(atoms[0], atoms[1]),
            AND(atoms[0], NOT(atoms[2])),
            OR(atoms[1], atoms[3]),
            AND(atoms[2], NOT(atoms[3])),
            OR(atoms[4], atoms[5]),
            {"at_least": {"count": 2, "of": [atoms[0], atoms[1], atoms[4]]}},
        ]
        if category == "negation_and_quantifiers":
            candidates.extend([
                CMP("success_count", ">=", 1), CMP("success_count", ">=", 2),
                CMP("unknown_count", "<=", 3), CMP("failure_count", ">=", 1),
            ])
        unique_candidates = {canonical(rule): rule for rule in candidates}
        candidates = [unique_candidates[key] for key in sorted(unique_candidates)]
        combinations = list(itertools.combinations(candidates, 4))
        # A checklist is a set of four independently stated requirements.
        # Sorting by a frame-specific digest chooses genuinely different sets,
        # rather than obtaining a new hash by permuting the same requirements.
        combinations.sort(key=lambda checklist: hashlib.sha256(
            (frame["frame_id"] + "/" + canonical(checklist)).encode()
        ).hexdigest())
        features = [derive_features(raw) for raw in _records(frame)]

        def outcomes(checklist):
            return tuple(min(sum(eval_rule(rule, values) for rule in checklist), width - 1) for values in features)

        first = combinations[0]
        first_outcomes = outcomes(first)
        second = next(
            checklist for checklist in combinations[1:]
            if set(map(canonical, checklist)) != set(map(canonical, first)) and outcomes(checklist) != first_outcomes
        )
        checklists = (list(first), list(second))
    programs = []
    for rubric_index, checklist in enumerate(checklists):
        rules = [True] + [
            {"at_least": {"count": threshold, "of": copy.deepcopy(checklist)}}
            for threshold in range(1, width)
        ]
        levels = [f"Level {level}: {describe(rule)}." for level, rule in enumerate(rules)]
        programs.append({
            "rules": rules,
            "levels": levels,
            "instruction": (
                f"Apply checklist rubric {rubric_index + 1}. It contains four explicit requirements: "
                + "; ".join(f"({position + 1}) {describe(rule)}" for position, rule in enumerate(checklist))
                + ". Evaluate the level definitions and choose the highest numbered applicable level."
            ),
            "ordinality": "Higher numbered levels require a larger count of the same four explicit requirements to hold.",
            "selection": "Level 0 is the explicit fallback; otherwise select the highest applicable index.",
        })
    return programs


def _format_row(kind, row):
    ordered = {
        "statement": ("actor", "operation", "target", "source", "time", "tense", "polarity", "endorsed"),
        "execution": ("actor", "operation", "target", "run_id", "source", "time", "verified", "mode", "outcome"),
        "approval": ("actor", "operation", "target", "source", "time", "verified", "decision"),
        "reversal": ("actor", "operation", "target", "run_id", "source", "time", "verified", "mode", "outcome", "reverses"),
        "corroboration": ("actor", "operation", "target", "run_id", "source", "time", "verified", "mode", "outcome"),
        "state event": ("target", "source", "time", "verified", "active"),
        "item": ("id", "status", "verified"),
    }[kind]
    values = "; ".join(f"{key}={str(row[key]).lower() if type(row[key]) is bool else row[key]}" for key in ordered)
    return f"{kind.title()} — {values}."


def _render_evidence(raw, language):
    entries = []
    for key, kind in (
        ("statements", "statement"), ("executions", "execution"), ("approvals", "approval"),
        ("reversals", "reversal"), ("corroborations", "corroboration"),
        ("state_events", "state event"), ("items", "item"),
    ):
        entries.extend(_format_row(kind, row) for row in raw[key])
    if not entries:
        entries = ["No statement, execution, approval, reversal, corroboration, state-event, or item-status rows were supplied."]
    style = language["style"]
    if style == "chronicle":
        evidence = {"chronicle_heading": language["heading"], "opening": language["intro"], "dated_entries": entries}
    elif style == "docket":
        evidence = {"docket_title": language["heading"], "matter": language["intro"], "numbered_facts": [f"Fact {i + 1}: {row}" for i, row in enumerate(entries)]}
    elif style == "brief":
        evidence = {"brief_title": language["heading"], "situation": language["intro"], "record_extracts": entries}
    elif style == "transcript":
        evidence = {"transcript_label": language["heading"], "context_line": language["intro"], "record_lines": [f"RECORD {i + 1} | {row}" for i, row in enumerate(entries)]}
    elif style == "worksheet":
        evidence = {"worksheet": language["heading"], "scenario": language["intro"], "observed_fields": entries}
    elif style == "dispatch":
        evidence = {"dispatch_subject": language["heading"], "dispatch_text": language["intro"], "attachments": entries}
    elif style == "audit":
        evidence = {"audit_caption": language["heading"], "scope_note": language["intro"], "evidence_rows": entries}
    else:
        evidence = {"logbook_name": language["heading"], "preface": language["intro"], "log_entries": entries}
    evidence["reader_note"] = language["caution"]
    evidence["inventory"] = {
        "scoped_item_ids": copy.deepcopy(raw["item_ids"]),
        "inventory_complete": raw["inventory_complete"],
    }
    evidence["measurement_observation"] = "missing" if raw["measurement"] is None else raw["measurement"]
    return evidence


PUBLIC_RULES = (
    "Judge only the supplied record. A source is trusted only when named in trusted_sources, and an entry is authentic only "
    "when verified=true and its source is trusted. Execution outcome predicates require the exact queried actor, operation, "
    "target, and run_id; an inclusive window time; authentic evidence; and mode=live. A matching success without a matching "
    "no_effect establishes completion. A matching no_effect without matching success establishes failure. Both together are a "
    "conflict. Timeout, absence, simulation, an untrusted source, an unverified entry, or a wrong binding establishes neither. "
    "Statements match actor, operation, and target; future wording is intent, completed positive wording is a claim, and endorsed "
    "marks adoption rather than quotation. The latest authentic approval time decides allow or deny; disagreement at that latest "
    "time is unknown. State predicates match target and use the latest authentic reports at or before their stated cutoff; a tied "
    "disagreement is unknown. A reversal must be an authentic in-window live success matching actor, operation, and target, name "
    "an established success run in reverses, and occur no earlier than that success. Item scope is exactly item_ids; an item is "
    "known only when its verified success/no_effect rows yield one unambiguous status. Public boolean false means the record does "
    "not establish that predicate; it does not assert an opposite unobserved world fact."
)


def render_request(private_case):
    raw = private_case["raw"]
    program = private_case["program"]
    language = private_case["language"]
    features = derive_features(raw)
    choice_definitions = {label: describe(rule) for label, rule in program["choice_rules"].items()}
    score = program["score"]
    questions = {
        f"n{index + 1}": {
            "type": "noul",
            "instructions": "Does the supplied record establish this predicate: " + describe(rule) + "?",
            "criteria": {
                "true": "The public rules and observations establish the predicate.",
                "false": "The record does not establish the predicate; this does not assert the opposite unobserved fact.",
            },
        }
        for index, rule in enumerate(program["noul_rules"])
    }
    questions["c1"] = {
        "type": "choice",
        "instructions": "Which one of the supplied mutually exclusive and exhaustive definitions applies?",
        "criteria": choice_definitions,
    }
    questions["s1"] = {
        "type": "score",
        "instructions": score["instruction"],
        "criteria": score["levels"],
    }
    state = {
        "case_file": {"heading": language["heading"], "domain": language["domain"], "presentation": language["style"]},
        "scope": {
            "query": copy.deepcopy(raw["query"]),
            "window_inclusive": copy.deepcopy(raw["window"]),
            "query_time": raw["query_time"],
            "trusted_sources": copy.deepcopy(raw["trusted_sources"]),
        },
        "evidence": _render_evidence(raw, language),
        "public_rules": PUBLIC_RULES,
        "choice_definitions": choice_definitions,
        "score_rule": score["instruction"],
        "score_ordinality": score["ordinality"],
        "score_selection": score["selection"],
        "score_levels": score["levels"],
    }
    expected = {f"n{i + 1}": eval_rule(rule, features) for i, rule in enumerate(program["noul_rules"])}
    expected["c1"] = select_choice(program["choice_rules"], features)
    expected["s1"] = select_level(score["rules"], features)
    return {"state": state, "questions": questions}, expected


def _relation(family_id, cases, axis, a, qa, b, qb, kind):
    return {
        "id": f"{family_id}/{axis}/{a}-{qa}-{b}-{qb}",
        "family_id": family_id,
        "kind": kind,
        "contrast_axis": axis,
        "expected_equal": kind in {"invariant", "layout_invariant"},
        "left": {"case_id": cases[a]["id"], "question_id": qa},
        "right": {"case_id": cases[b]["id"], "question_id": qb},
    }


def _build_family(frame):
    raws = _records(frame)
    nouls, choices = _noul_and_choice(frame)
    scores = _score_programs(frame)
    language = {key: frame[key] for key in ("frame_id", "domain", "heading", "intro", "caution", "style")}
    cases, private_cases = [], {}
    for evidence_index, rubric_index in itertools.product(range(2), range(2)):
        case_id = f"{frame['family_id']}/e{evidence_index}r{rubric_index}"
        program = {
            "noul_rules": copy.deepcopy(nouls),
            "choice_rules": copy.deepcopy(choices),
            "score": copy.deepcopy(scores[rubric_index]),
        }
        private = {
            "family_id": frame["family_id"],
            "category": frame["category"],
            "split": frame["split"],
            "frame_id": frame["frame_id"],
            "evidence_index": evidence_index,
            "rubric_index": rubric_index,
            "raw": copy.deepcopy(raws[evidence_index]),
            "program": program,
            "language": copy.deepcopy(language),
        }
        request, expected = render_request(private)
        private["derived_features"] = derive_features(private["raw"])
        private["raw_sha256"] = canonical_sha256(private["raw"])
        private["program_sha256"] = canonical_sha256(private["program"])
        private["request_sha256"] = canonical_sha256(request)
        private_cases[case_id] = private
        cases.append({
            "id": case_id,
            "family_id": frame["family_id"],
            "category": frame["category"],
            "domain": frame["domain"],
            "variant": f"e{evidence_index}r{rubric_index}",
            "layout": frame["style"],
            "predicate_tags": [frame["category"], "fresh-independent-authoring"],
            "request": request,
            "expected": expected,
            "rationale": {qid: "Derived from the raw observations using the explicit public definition." for qid in QUESTION_IDS},
        })
    relations = []
    # Evidence comparisons: e0r0/e1r0 and e0r1/e1r1.
    for a, b in ((0, 2), (1, 3)):
        for qid in QUESTION_IDS:
            equal = cases[a]["expected"][qid] == cases[b]["expected"][qid]
            relations.append(_relation(frame["family_id"], cases, "invariant" if equal else "evidence", a, qid, b, qid, "invariant" if equal else "flip"))
    # Rubric comparisons preserve evidence and answer space; only score can differ.
    for a, b in ((0, 1), (2, 3)):
        equal = cases[a]["expected"]["s1"] == cases[b]["expected"]["s1"]
        relations.append(_relation(frame["family_id"], cases, "invariant" if equal else "rubric", a, "s1", b, "s1", "invariant" if equal else "flip"))
    # Within-case semantic question contrasts and one invariant when available.
    for case_index, case in enumerate(cases):
        for qa, qb in itertools.combinations(("n1", "n2", "n3"), 2):
            equal = case["expected"][qa] == case["expected"][qb]
            axis, kind = ("invariant", "invariant") if equal else ("question", "question_contrast")
            relations.append(_relation(frame["family_id"], cases, axis, case_index, qa, case_index, qb, kind))
    return cases, relations, private_cases


def build_artifacts():
    frames = _assign_splits(_frames())
    suites = {
        split: {
            "name": f"intermediate-supervision-v1/fresh-{split}",
            "cases": [],
            "relations": [],
            "metadata": {"version": 1, "authoring": "blind-independent", "split": split},
            "ordered_family_ids": [],
        }
        for split in SPLITS
    }
    raw_records = {"version": 1, "cases": {}, "families": {}}
    manifest_frames = []
    for frame in sorted(frames, key=lambda row: row["frame_id"]):
        cases, relations, private_cases = _build_family(frame)
        suite = suites[frame["split"]]
        suite["cases"].extend(cases)
        suite["relations"].extend(relations)
        suite["ordered_family_ids"].append(frame["family_id"])
        raw_records["cases"].update(private_cases)
        raw_records["families"][frame["family_id"]] = {
            "frame_id": frame["frame_id"],
            "split": frame["split"],
            "category": frame["category"],
            "case_ids": [case["id"] for case in cases],
            "raw_hashes": [canonical_sha256(raw) for raw in _records(frame)],
            "program_hashes": [canonical_sha256(score) for score in _score_programs(frame)],
        }
        language = {key: frame[key] for key in ("frame_id", "domain", "heading", "intro", "caution", "style")}
        manifest_frames.append({
            "frame_id": frame["frame_id"], "family_id": frame["family_id"], "category": frame["category"],
            "split": frame["split"], "split_hash": frame["split_hash"], "language_sha256": canonical_sha256(language),
        })
    for split, suite in suites.items():
        suite["metadata"]["families"] = len(suite["ordered_family_ids"])
        suite["metadata"]["cases"] = len(suite["cases"])
    manifest = {
        "version": 1,
        "assignment": "Within each category, sort SHA256('fresh-split-v1/' + frame_id); assign 2 validation, 2 calibration, 4 confirmation.",
        "frames": manifest_frames,
        "splits": {split: {"family_ids": suites[split]["ordered_family_ids"]} for split in SPLITS},
        "limits": (
            "These are 80 independently written scenario/language frames over the same declared raw-observation ontology. "
            "They are not claimed to implement 80 independent reasoning mechanisms."
        ),
    }
    artifacts = {"suites": suites, "raw_records": raw_records, "manifest": manifest}
    validate_artifacts(artifacts)
    return artifacts


def validate_artifacts(artifacts):
    if set(artifacts) != {"suites", "raw_records", "manifest"} or set(artifacts["suites"]) != set(SPLITS):
        raise ValueError("malformed fresh artifact container")
    frames = artifacts["manifest"]["frames"]
    if len(frames) != 80 or len({row["frame_id"] for row in frames}) != 80:
        raise ValueError("fresh authoring requires 80 distinct frames")
    if Counter(row["category"] for row in frames) != Counter({category: 8 for category in CATEGORIES}):
        raise ValueError("category frame counts differ")
    split_counts = Counter(row["split"] for row in frames)
    if split_counts != Counter(validation=20, calibration=20, confirmation=40):
        raise ValueError("split frame counts differ")
    raw_cases = artifacts["raw_records"]["cases"]
    seen_cases, seen_families = set(), set()
    for split, suite in artifacts["suites"].items():
        validate_public_suite(suite)
        families = set(suite["ordered_family_ids"])
        if seen_families & families:
            raise ValueError("family overlap across fresh splits")
        seen_families |= families
        grouped = defaultdict(list)
        for case in suite["cases"]:
            if case["id"] in seen_cases or case["id"] not in raw_cases:
                raise ValueError("case overlap or missing private record")
            seen_cases.add(case["id"])
            grouped[case["family_id"]].append(case)
            private = raw_cases[case["id"]]
            request, expected = render_request(private)
            if request != case["request"] or canonical_sha256(request) != private["request_sha256"]:
                raise ValueError("public request rerender mismatch")
            if expected != case["expected"]:
                raise ValueError("gold target rerender mismatch")
            if private["split"] != split or private["program_sha256"] != canonical_sha256(private["program"]):
                raise ValueError("private binding mismatch")
            state = case["request"]["state"]
            forbidden = set(private["derived_features"]) | {"expected", "rationale", "program", "derived_features"}
            if set(state) & forbidden:
                raise ValueError("oracle feature or label leaked into STATE")
        if set(grouped) != families or any(len(rows) != 4 for rows in grouped.values()):
            raise ValueError("family/case cardinality mismatch")
        for family_id, cases in grouped.items():
            related = [row for row in suite["relations"] if row["family_id"] == family_id]
            axes = {row["contrast_axis"] for row in related}
            if axes != {"evidence", "rubric", "question", "invariant"}:
                raise ValueError("family lacks required semantic contrast or invariant")
            if len({len(case["request"]["questions"]["s1"]["criteria"]) for case in cases}) != 1:
                raise ValueError("rubric answer spaces differ inside family")
    if seen_cases != set(raw_cases):
        raise ValueError("private/public case inventories differ")
    return True


def emit(root=Path(".")):
    root = Path(root)
    artifacts = build_artifacts()
    output = root / OUTPUT
    authoring = root / AUTHORING
    output.mkdir(parents=True, exist_ok=True)
    authoring.mkdir(parents=True, exist_ok=True)
    for split, suite in artifacts["suites"].items():
        (output / f"{split}-suite.json").write_text(canonical(suite) + "\n")
    (output / "raw-records.json").write_text(canonical(artifacts["raw_records"]) + "\n")
    manifest = copy.deepcopy(artifacts["manifest"])
    manifest["files"] = {
        f"fresh-draft/{split}-suite.json": canonical_sha256(artifacts["suites"][split]) for split in SPLITS
    }
    manifest["files"]["fresh-draft/raw-records.json"] = canonical_sha256(artifacts["raw_records"])
    (authoring / "manifest.json").write_text(canonical(manifest) + "\n")
    (authoring / "frames.json").write_text(canonical({"frames": _assign_splits(_frames())}) + "\n")
    return artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    artifacts = emit(args.root)
    print(canonical({split: len(suite["cases"]) for split, suite in artifacts["suites"].items()}))


if __name__ == "__main__":
    main()
