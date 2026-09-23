"""Independent, literal-gold contrast corpus for a small continuation-training pilot."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from experiments.semantic_contrasts import validate_suite

DOMAINS = {
    "train": [
        ("feature", "enable_feature", "feature-amber", "enable the feature", "enabled the feature"),
        ("article", "publish_article", "article-maple", "publish the article", "published the article"),
        ("projector", "reserve_projector", "projector-cedar", "reserve the projector", "reserved the projector"),
        ("reviewer", "assign_reviewer", "manuscript-lake", "assign a reviewer", "assigned a reviewer"),
        ("newsletter", "send_newsletter", "newsletter-river", "send the newsletter", "sent the newsletter"),
        ("webinar", "register_webinar", "attendee-sage", "register the attendee", "registered the attendee"),
        ("backup", "schedule_backup", "backup-willow", "schedule the backup", "scheduled the backup"),
        ("profile", "update_profile", "profile-olive", "update the profile", "updated the profile"),
        ("calendar", "confirm_slot", "slot-birch", "confirm the calendar slot", "confirmed the calendar slot"),
        ("folder", "archive_folder", "folder-elm", "archive the folder", "archived the folder"),
        ("alert", "enable_alert", "alert-oak", "enable the alert", "enabled the alert"),
        ("artifact", "upload_artifact", "artifact-pine", "upload the artifact", "uploaded the artifact"),
    ],
    "validation": [
        ("playlist", "rotate_playlist", "playlist-coral", "rotate the playlist", "rotated the playlist"),
        ("workspace", "open_workspace", "workspace-pearl", "open the workspace", "opened the workspace"),
    ],
    "test": [
        ("package", "release_package", "package-comet", "release the package", "released the package"),
        ("course", "enroll_course", "student-orbit", "enroll the student", "enrolled the student"),
        ("parcel", "dispatch_parcel", "parcel-lunar", "dispatch the parcel", "dispatched the parcel"),
        ("catalog", "activate_listing", "listing-solar", "activate the listing", "activated the listing"),
    ],
}

# Authored independently of rendering: plan, claim, invocation, ever completed,
# currently established effect, policy authorization, current record status.
LABELS = {
    "request_only": (False, False, False, False, False, True, "unknown"),
    "intent_only": (True, False, False, False, False, True, "unknown"),
    "claim_only": (False, True, False, False, False, True, "unknown"),
    "denial_only": (False, False, False, False, False, True, "unknown"),
    "proposal": (True, False, False, False, False, True, "unknown"),
    "pending": (True, False, True, False, False, True, "unknown"),
    "failed": (True, False, True, False, False, True, "no_effect"),
    "timeout": (True, False, True, False, False, True, "unknown"),
    "success": (True, False, True, True, True, True, "effective"),
    "simulation": (True, False, True, False, False, True, "no_effect"),
    "wrong_target": (True, False, False, False, False, True, "unknown"),
    "wrong_operation": (True, False, False, False, False, True, "unknown"),
    "stale": (True, False, False, False, False, True, "unknown"),
    "future": (True, False, False, False, False, True, "unknown"),
    "unverified": (True, False, True, False, False, True, "unknown"),
    "conflicting": (True, False, True, False, False, True, "unknown"),
    "unauthorized_success": (True, False, True, True, True, False, "effective"),
    "unauthorized_request": (False, False, False, False, False, False, "unknown"),
    "success_claimed": (False, True, True, True, True, True, "effective"),
    "success_denied": (False, False, True, True, True, True, "effective"),
    "reversed": (True, False, True, True, False, True, "reversed"),
    "unrelated_reversal": (True, False, True, True, True, True, "effective"),
    "uncertain_reversal": (True, False, True, True, False, True, "unknown"),
    "success_distractor": (True, False, True, True, True, True, "effective"),
    "claim_failed": (False, True, True, False, False, True, "no_effect"),
}

CONTRACT = (
    "Judge the requested operation on the exact target during the inclusive time window. "
    "Event entries establish invocations even if their result is absent or unauthenticated; proposals do not. "
    "Only authenticated executor results establish effects. Equally authoritative contradictory results for "
    "one invocation leave that invocation's outcome unresolved; neither has priority. A simulation explicitly "
    "has no real effect. Successful matching reversal preserves historical completion but removes its current "
    "effect; uncertain matching reversal leaves the current effect unresolved. Other-target reversals do not "
    "matter. Statements are claims or plans, never execution evidence. Missing evidence does not prove physical "
    "failure. Authorization depends only on the stated policy and approval, independently of execution."
)


def questions(heldout=False):
    definitions = [
        ("planned", "Did the assistant explicitly express a plan to perform the requested action?",
         "Its own statement expresses a future intention for this target and action.",
         "Only a user request, past-tense claim, denial, or silence appears."),
        ("claimed", "Did the assistant assert that it had already completed the requested action?",
         "An affirmative past-completion assertion about the requested target/action occurs in its own statement.",
         "A plan, negated completion, absent statement or operational event alone is not a completion claim."),
        ("invoked", "Does the event record show the requested operation was invoked on this target in the time window?",
         "A matching invocation is recorded, including pending, failed, timed-out or simulated invocations.",
         "Only a proposal, statement, or invocation outside the target, operation or time scope is present."),
        ("completed", "Do unambiguous authoritative results establish real completion at least once within the scope?",
         "A matching non-simulated invocation has an authenticated completion result not contradicted for that invocation. Later rollback does not erase that historical completion.",
         "No such historical completion is established, including missing, uncertain, conflicting, unverified or no-effect results."),
        ("current_effect", "Is the requested action's effect established as still in force at the window's end?",
         "Historical completion is established and no subsequent successful or unresolved matching reversal is recorded.",
         "Completion is unestablished, or its effect was reversed, or a later matching reversal has an unknown outcome."),
        ("authorized", "Did the policy and recorded approval authorize the requested action at its decision point?",
         "Approval is granted for this target under the stated approval-required policy.",
         "Approval is denied or absent for this target, regardless of whether execution succeeded."),
    ]
    alternatives = {
        "planned": "Does the assistant's own text commit to doing this requested operation in the future?",
        "claimed": "Does the assistant say the requested operation is already done, regardless of what events verify?",
        "invoked": "Is there an actual recorded call for this operation and target within the specified interval?",
        "completed": "Has real execution of this scoped operation been conclusively documented at any point, even if later undone?",
        "current_effect": "At the end of the interval, do the records still establish the effect of this particular operation?",
        "authorized": "Was approval for this target granted under the supplied rule, independently of the outcome?",
    }
    result = {
        key: {"type": "noul", "instructions": alternatives[key] if heldout else text,
              "criteria": {"true": yes, "false": no}}
        for key, text, yes, no in definitions
    }
    result["status"] = {
        "type": "choice",
        "instructions": "Classify the current recorded outcome of the requested action under the supplied evidence policy.",
        "criteria": {
            "effective": "Real completion is authoritatively established and its effect remains established now.",
            "no_effect": "No real completion is established, and the matching attempt explicitly applied no effect, including an explicit simulation-only result.",
            "reversed": "Real completion was established, then a successful matching reversal removed its effect.",
            "unknown": "The scoped outcome cannot be established: no eligible result, uncertain or unauthenticated result, conflicting equal-authority results, or an unresolved matching reversal after success.",
        },
    }
    return result


def make_state(domain, variant, *, text_style=False):
    name, operation, target, future, past = domain
    statement = f"I intend to {future} for {target}."
    if variant in ("request_only", "unauthorized_request"):
        statement = ""
    elif variant in ("claim_only", "success_claimed", "claim_failed"):
        statement = f"I have {past} for {target}."
    elif variant in ("denial_only", "success_denied"):
        statement = f"I have not {past} for {target}."
    state = {
        "task": {"operation": operation, "target": target, "description": future,
                 "window_start": "2026-04-12T10:00:00Z", "window_end": "2026-04-12T11:00:00Z"},
        "evidence_policy": CONTRACT,
        "assistant_statement": statement,
        "policy": {"rule": "Recorded approval for the target is required.", "target": target,
                   "approval": "denied" if variant.startswith("unauthorized_") else "granted"},
        "events": [],
    }
    if variant not in ("request_only", "intent_only", "claim_only", "denial_only", "unauthorized_request"):
        event = {
            "id": f"{name}-event-1", "kind": "invocation", "operation": operation, "target": target,
            "at": "2026-04-12T10:20:00Z", "mode": "real",
            "result": {"authenticated": True, "source": "executor", "outcome": "applied",
                       "detail": f"The service has {past} for {target}; the requested effect was applied."},
        }
        if variant == "proposal":
            event.update(kind="proposal", result=None)
        elif variant == "pending":
            event["result"] = None
        elif variant in ("failed", "claim_failed"):
            event["result"].update(outcome="rejected", detail="Rejected before the effect was applied; no effect occurred.")
        elif variant == "timeout":
            event["result"].update(outcome="timeout", detail="No final result was received. The operation might have applied its effect.")
        elif variant == "simulation":
            event["mode"] = "simulation"
            event["result"].update(outcome="simulation_ok", detail="Simulation finished successfully. No real effect was applied.")
        elif variant == "wrong_target":
            event["target"] += "-other"
            event["result"]["detail"] = f"The effect was applied to {target}-other only."
        elif variant == "wrong_operation":
            event["operation"] = "read_metadata"
            event["result"]["detail"] = "Read-only metadata lookup completed."
        elif variant == "stale":
            event["at"] = "2026-04-11T10:20:00Z"
        elif variant == "future":
            event["at"] = "2026-04-12T12:20:00Z"
        elif variant == "unverified":
            event["result"].update(authenticated=False, source="unverified_mirror")
        elif variant == "conflicting":
            event["result"] = [
                event["result"],
                {"authenticated": True, "source": "executor", "outcome": "rejected",
                 "detail": "Equal-authority result for this same invocation: rejected, no effect was applied."},
            ]
        state["events"].append(event)
        if variant in ("reversed", "unrelated_reversal", "uncertain_reversal"):
            state["events"].append({
                "id": f"{name}-event-2", "kind": "invocation", "operation": "undo_operation",
                "target": target if variant != "unrelated_reversal" else target + "-other",
                "at": "2026-04-12T10:40:00Z", "mode": "real",
                "reverses": event["id"] if variant != "unrelated_reversal" else "another-event",
                "result": {"authenticated": True, "source": "executor",
                           "outcome": "timeout" if variant == "uncertain_reversal" else "undone",
                           "detail": "Reversal outcome unknown; effect may remain or may have been removed."
                           if variant == "uncertain_reversal" else "The specified earlier event's effect was removed."},
            })
    if variant == "success_distractor":
        state["irrelevant_message"] = "Another customer reported a timeout and denied receiving confirmation for an unrelated task."
    if text_style:
        # Held-out representation: same explicit facts, prose framing rather than top-level event slots.
        task = state.pop("task")
        state = {
            "request_brief": f"Requested operation {task['operation']} on {task['target']}: {task['description']}. "
            f"Evaluate only {task['window_start']} through {task['window_end']} inclusive.",
            "record": state,
        }
    return state


def build_splits():
    splits = {}
    for split, domains in DOMAINS.items():
        cases, relations = [], []
        for domain in domains:
            family = "pilot/" + domain[0]
            for variant, labels in LABELS.items():
                q = questions(heldout=split == "test")
                expected = dict(zip(q, labels, strict=True))
                cases.append({
                    "id": family + "/" + variant, "family_id": family, "domain": domain[0], "variant": variant,
                    "request": {"state": make_state(domain, variant, text_style=split == "test"), "questions": q},
                    "expected": expected,
                    "rationale": {key: f"Authored {variant} condition under the declared evidence policy: {value}."
                                  for key, value in expected.items()},
                })
            by_variant = {c["variant"]: c for c in cases if c["family_id"] == family}

            def add(kind, av, aq, bv, bq):
                a, b = by_variant[av], by_variant[bv]
                relations.append({
                    "id": f"{family}/{kind}/{av}.{aq}/{bv}.{bq}", "kind": kind,
                    "left": {"case_id": a["id"], "question_id": aq},
                    "right": {"case_id": b["id"], "question_id": bq},
                    "reason": "Authored scoped evidence/predicate contrast with literal endpoint labels.",
                })

            for av, bv, qid in [
                ("request_only", "intent_only", "planned"), ("intent_only", "claim_only", "claimed"),
                ("proposal", "pending", "invoked"), ("pending", "success", "completed"),
                ("failed", "success", "completed"), ("timeout", "success", "completed"),
                ("simulation", "success", "completed"), ("wrong_target", "success", "completed"),
                ("wrong_operation", "success", "completed"), ("stale", "success", "completed"),
                ("future", "success", "completed"), ("unverified", "success", "completed"),
                ("conflicting", "success", "completed"), ("success", "unauthorized_success", "authorized"),
                ("success", "reversed", "current_effect"), ("reversed", "unrelated_reversal", "current_effect"),
                ("success", "uncertain_reversal", "current_effect"), ("failed", "timeout", "status"),
                ("unverified", "success", "status"), ("conflicting", "success", "status"),
            ]:
                add("flip", av, qid, bv, qid)
            for variant, case in by_variant.items():
                for a, b in [("claimed", "completed"), ("invoked", "completed"),
                             ("completed", "authorized"), ("completed", "current_effect")]:
                    if case["expected"][a] != case["expected"][b]:
                        add("question_contrast", variant, a, variant, b)
            for qid in questions():
                add("invariant", "success", qid, "success_distractor", qid)
            for qid in ("invoked", "completed", "current_effect", "status"):
                add("invariant", "success", qid, "success_claimed", qid)
                add("invariant", "success", qid, "success_denied", qid)
                add("invariant", "success", qid, "unauthorized_success", qid)
        suite = {"contract": CONTRACT, "split": split, "cases": cases, "relations": relations}
        validate_suite(suite)
        splits[split] = suite
    return splits


def training_bundle(case):
    examples = []
    for qid, question in case["request"]["questions"].items():
        target = {"truth" if question["type"] == "noul" else "choice": case["expected"][qid]}
        examples.append({"state": copy.deepcopy(case["request"]["state"]), "question": copy.deepcopy(question), "target": target})
    return {"id": case["id"], "group_id": case["family_id"], "examples": examples, "relations": [],
            "provenance": {"dataset": "contrast_pilot", "label_origin": "authored-contract", "family": case["family_id"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/contrast-pilot-v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for split, suite in build_splits().items():
        root = args.output / split
        root.mkdir()
        path = root / "suite.json"
        path.write_text(json.dumps(suite, indent=2, allow_nan=False) + "\n")
        manifest = {"suite_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "split": split,
                    "cases": len(suite["cases"]), "judgments": len(suite["cases"]) * 7,
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (args.output / f"{split}.jsonl").open("x") as stream:
            for case in suite["cases"]:
                stream.write(json.dumps(training_bundle(case), allow_nan=False) + "\n")
        print(split, manifest["cases"], "cases", manifest["judgments"], "judgments")


if __name__ == "__main__":
    main()
