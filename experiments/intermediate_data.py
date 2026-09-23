"""CPU-only observable auxiliary annotations and source-bound pilot emission.

Regression targets are raw units; the trainer divides once by declared scale.
No final targets, rubric rules, or hidden world facts enter annotate_case.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from experiments.contrast_factorial_data import (
    CATEGORIES,
    canonical,
    digest,
    file_sha256,
    render_family,
    training_bundles,
)
from experiments.contrast_scaling_logic import derive_features, eval_rule, select_choice, select_level
from experiments.revised_metrics import validate_suite

EXEC_CATEGORIES = tuple(
    c for c in CATEGORIES if c not in {"temporal_scope", "negation_and_quantifiers", "ordered_rubrics"}
)
AUTH_CATEGORIES = tuple(c for c in EXEC_CATEGORIES if c not in {"entity_binding", "action_binding"})
DEFAULT_SOURCE = Path("data/contrast-factorial-v1/train/v1")
DEFAULT_OUTPUT = Path("data/intermediate-supervision-v1")


def schema_features():
    """Candidate features, ordered stably; fit_schema filters using training only."""
    rows = []

    def add(name, description, categories, kind="binary", scale=1, condition="category"):
        rows.append(
            {
                "name": name,
                "kind": kind,
                "scale": scale,
                "description": description,
                "categories": list(categories),
                "applicability": condition,
            }
        )

    for suffix, text in [
        ("claim", "positive past-completion assertion"),
        ("intent", "positive future intention"),
        ("denial", "negative past-completion assertion"),
        ("endorsed", "explicitly adopted positive completion assertion"),
    ]:
        add(
            "statement_" + suffix,
            "A statement matching actor/action/target is a " + text + "; no trust/time restriction.",
            EXEC_CATEGORIES,
        )
    add(
        "execution_present",
        "At least one execution record is present, regardless of binding or outcome.",
        EXEC_CATEGORIES,
    )
    for field in ["actor", "operation", "target", "run"]:
        add(
            "execution_" + field + "_seen",
            f"One execution row names the queried {field}; other fields need not match.",
            EXEC_CATEGORIES,
        )
    for name, text in [
        (
            "execution_joint_identity",
            "One execution row jointly matches actor/action/target/run; time/trust/mode unrestricted.",
        ),
        ("execution_joint_in_window", "A jointly matching execution row is inside the inclusive window."),
        ("execution_joint_authentic", "An in-window jointly matching row is verified and from a trusted source."),
        ("execution_joint_live", "An authentic in-window jointly matching row is live, regardless of outcome."),
        ("scoped_untrusted_execution", "A jointly matching in-window row is unverified or from an untrusted source."),
        ("scoped_nonlive_execution", "A jointly matching in-window row has a mode other than live."),
        ("scoped_success_report", "A jointly matching in-window row reports success, without a trust/mode test."),
        ("scoped_no_effect_report", "A jointly matching in-window row reports no_effect, without a trust/mode test."),
        (
            "live_success_receipt",
            "An authentic live in-window jointly matching row reports success; conflicting reports do not erase it.",
        ),
        (
            "live_no_effect_receipt",
            "An authentic live in-window jointly matching row reports no_effect; absence/timeout is not no_effect.",
        ),
        ("live_timeout_receipt", "An authentic live in-window jointly matching row reports timeout."),
        (
            "other_live_success_receipt",
            "An authentic live in-window success row differs on at least one query identity field.",
        ),
    ]:
        add(name, text, EXEC_CATEGORIES)
    for value in ["allow", "deny"]:
        add(
            "permission_latest_" + value + "_report",
            f"An authentic in-window actor/action/target permission row at the latest such timestamp says {value}; ties can contain both decisions.",
            AUTH_CATEGORIES,
        )
    add(
        "linked_corroboration_receipt",
        "Authentic scoped success corroboration from a different source at or after a matching live success receipt; no inference of conflict resolution.",
        AUTH_CATEGORIES,
    )
    add(
        "linked_reversal_success_receipt",
        "Authentic live scoped successful reversal names a matching live success run and occurs no earlier; competing failure receipts do not erase this link.",
        ["reversal_and_current_state"],
    )
    for when in ["earlier", "current"]:
        cutoff = "query_time" if when == "earlier" else "window end"
        add(
            "state_" + when + "_known",
            f"Latest authentic target-state reports at/before {cutoff} agree on a boolean; absent/conflicting is unknown.",
            ["temporal_scope"],
        )
        add(
            "state_" + when + "_active",
            f"Agreed active value at/before {cutoff}; unknown values are masked.",
            ["temporal_scope"],
            condition="category and corresponding state known",
        )
    add(
        "inventory_complete",
        "The public inventory_complete flag is true; not inferred from item success.",
        ["negation_and_quantifiers"],
    )
    for name, text in [
        ("scope", "distinct publicly scoped identities"),
        ("success", "scoped identities with one verified success status and no verified no_effect"),
        ("no_effect", "scoped identities with one verified no_effect status and no verified success"),
        ("unknown", "scoped identities with absent or conflicting verified success/no_effect status"),
    ]:
        add(
            "item_" + name + "_count",
            "Number of " + text + ".",
            ["negation_and_quantifiers"],
            kind="regression",
            scale=4,
        )
    add("measurement_known", "A finite numeric measurement is publicly recorded.", ["ordered_rubrics"])
    add(
        "measurement_value",
        "The recorded measurement in raw units; absent measurements are masked, never zero-imputed as observations.",
        ["ordered_rubrics"],
        kind="regression",
        scale=100,
        condition="category and finite measurement present",
    )
    return rows


def annotate_case(raw, category, *, features=None):
    """Return targets/masks in feature order, with zero placeholders for masked facts."""
    if category not in CATEGORIES:
        raise ValueError("unknown category")
    q = raw["query"]
    start, end = raw["window"]
    if start > end:
        raise ValueError("window endpoints reversed")
    trusted = set(raw["trusted_sources"])

    def matches(row, run=True):
        fields = ("actor", "operation", "target", "run_id") if run else ("actor", "operation", "target")
        return all(row.get(k) == q[k] for k in fields)

    def within(row):
        return type(row.get("time")) in (int, float) and start <= row["time"] <= end

    def authentic(row):
        return row.get("verified") is True and row.get("source") in trusted

    executions = raw.get("executions", [])
    joint = [r for r in executions if matches(r)]
    scoped = [r for r in joint if within(r)]
    auth = [r for r in scoped if authentic(r)]
    live = [r for r in auth if r.get("mode") == "live"]
    success = [r for r in live if r.get("outcome") == "success"]
    statements = [r for r in raw.get("statements", []) if matches(r, False)]
    claims = [r for r in statements if r.get("tense") == "completed" and r.get("polarity") == "positive"]
    v = {
        "statement_claim": bool(claims),
        "statement_intent": any(r.get("tense") == "future" and r.get("polarity") == "positive" for r in statements),
        "statement_denial": any(r.get("tense") == "completed" and r.get("polarity") == "negative" for r in statements),
        "statement_endorsed": any(r.get("endorsed") is True for r in claims),
        "execution_present": bool(executions),
        "execution_joint_identity": bool(joint),
        "execution_joint_in_window": bool(scoped),
        "execution_joint_authentic": bool(auth),
        "execution_joint_live": bool(live),
        "scoped_untrusted_execution": any(not authentic(r) for r in scoped),
        "scoped_nonlive_execution": any(r.get("mode") != "live" for r in scoped),
        "scoped_success_report": any(r.get("outcome") == "success" for r in scoped),
        "scoped_no_effect_report": any(r.get("outcome") == "no_effect" for r in scoped),
        "live_success_receipt": bool(success),
        "live_no_effect_receipt": any(r.get("outcome") == "no_effect" for r in live),
        "live_timeout_receipt": any(r.get("outcome") == "timeout" for r in live),
        "other_live_success_receipt": any(
            not matches(r) and within(r) and authentic(r) and r.get("mode") == "live" and r.get("outcome") == "success"
            for r in executions
        ),
    }
    for name, field in [("actor", "actor"), ("operation", "operation"), ("target", "target"), ("run", "run_id")]:
        v["execution_" + name + "_seen"] = any(r.get(field) == q[field] for r in executions)
    permissions = [r for r in raw.get("approvals", []) if matches(r, False) and within(r) and authentic(r)]
    latest = max((r["time"] for r in permissions), default=None)
    for decision in ["allow", "deny"]:
        v["permission_latest_" + decision + "_report"] = any(
            r["time"] == latest and r.get("decision") == decision for r in permissions
        )
    v["linked_corroboration_receipt"] = any(
        matches(r)
        and within(r)
        and authentic(r)
        and r.get("outcome") == "success"
        and any(r["source"] != s["source"] and r["time"] >= s["time"] for s in success)
        for r in raw.get("corroborations", [])
    )
    v["linked_reversal_success_receipt"] = any(
        matches(r, False)
        and within(r)
        and authentic(r)
        and r.get("mode") == "live"
        and r.get("outcome") == "success"
        and any(r.get("reverses") == s["run_id"] and r["time"] >= s["time"] for s in success)
        for r in raw.get("reversals", [])
    )
    known = {}
    for when, cutoff in [("earlier", raw["query_time"]), ("current", end)]:
        events = [
            r
            for r in raw.get("state_events", [])
            if r.get("target") == q["target"]
            and authentic(r)
            and type(r.get("time")) in (int, float)
            and r["time"] <= cutoff
        ]
        latest = max((r["time"] for r in events), default=None)
        values = [r.get("active") for r in events if r["time"] == latest]
        known[when] = bool(values) and all(type(x) is bool for x in values) and len(set(values)) == 1
        v["state_" + when + "_known"] = known[when]
        v["state_" + when + "_active"] = values[0] if known[when] else False
    # Only public item/measurement categories expose these fields. Other categories'
    # private base-record defaults must never become supervision.
    v["inventory_complete"] = raw.get("inventory_complete") is True
    scope = list(dict.fromkeys(raw.get("item_ids", [])))
    counts = Counter()
    for identity in scope:
        statuses = {
            r.get("status")
            for r in raw.get("items", [])
            if r.get("id") == identity and r.get("verified") is True and r.get("status") in {"success", "no_effect"}
        }
        counts[next(iter(statuses)) if len(statuses) == 1 else "unknown"] += 1
    v.update(
        item_scope_count=len(scope),
        item_success_count=counts["success"],
        item_no_effect_count=counts["no_effect"],
        item_unknown_count=counts["unknown"],
    )
    measurement = raw.get("measurement")
    if (
        category == "ordered_rubrics"
        and measurement is not None
        and (type(measurement) not in (int, float) or not math.isfinite(measurement))
    ):
        raise ValueError("measurement must be finite numeric or missing")
    v["measurement_known"] = measurement is not None
    v["measurement_value"] = measurement if measurement is not None else 0
    targets, masks = [], []
    for f in features if features is not None else schema_features():
        mask = category in f["categories"]
        if f["name"] == "measurement_value":
            mask = mask and measurement is not None
        if f["name"] in {"state_earlier_active", "state_current_active"}:
            mask = mask and known[f["name"].split("_")[1]]
        targets.append(float(v[f["name"]]) if mask else 0.0)
        masks.append(bool(mask))
    return targets, masks


def fit_schema(records):
    """Freeze training-only eligibility: >=2 per class, or >=2 distinct numeric values."""
    candidates = schema_features()
    columns = [[] for _ in candidates]
    for raw, category in records:
        targets, masks = annotate_case(raw, category)
        for i, (v, mask) in enumerate(zip(targets, masks)):
            if mask:
                columns[i].append(v)
    features, coverage, dropped = [], {}, []
    for f, values in zip(candidates, columns):
        counts = Counter(values)
        eligible = (counts[0] >= 2 and counts[1] >= 2) if f["kind"] == "binary" else len(counts) >= 2
        coverage[f["name"]] = {
            "applicable": len(values),
            "distinct": len(counts),
            "zeros": counts[0],
            "ones": counts[1],
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "eligible": eligible,
        }
        (features if eligible else dropped).append(f)
    return {
        "version": "intermediate-supervision-v1",
        "target_units": "raw; divide regression targets by scale once",
        "eligibility_policy": "Training raw evidence states only (one per e, not repeated rubric cases): binary >=2 zeros and >=2 ones; regression >=2 distinct observed values. Category masks fixed before eligibility. No evaluation outcomes used.",
        "features": features,
        "dropped_features": dropped,
        "coverage": coverage,
    }


def select_families(assignments, seed=42):
    """Choose 20/category, one/blueprint, five/K using only provenance hashes."""
    allowed = ["family_id", "category", "category_index", "low_blueprint", "high_blueprint", "narrow_k", "broad_k"]
    rows = [{k: r[k] for k in allowed} for r in assignments]
    chosen = set()

    def order(value):
        return hashlib.sha256(f"intermediate-supervision-v1/{seed}/{value}".encode()).hexdigest()

    for category in sorted(CATEGORIES):
        groups = defaultdict(list)
        for row in rows:
            if row["category"] == category:
                groups[row["low_blueprint"]].append(row)
        if len(groups) != 20:
            raise ValueError("expected 20 low-language blueprints per category")
        options = [sorted(groups[key], key=lambda r: order(r["family_id"])) for key in sorted(groups, key=order)]

        @lru_cache(None)
        def solve(i, quotas):
            if i == len(options):
                return () if not any(quotas) else None
            for row in options[i]:
                k = row["broad_k"] - 2
                if quotas[k] == 0:
                    continue
                next_quota = list(quotas)
                next_quota[k] -= 1
                tail = solve(i + 1, tuple(next_quota))
                if tail is not None:
                    return (row["family_id"], *tail)
            return None

        selection = solve(0, (5, 5, 5, 5))
        if selection is None:
            raise ValueError("no unique-blueprint balanced selection exists; no silent relaxation")
        chosen.update(selection)
    return [row for row in rows if row["family_id"] in chosen]


def verify_training_family(plan, card, cases, bundle, assignment):
    """Bind every public question/state/final target to exact old raw plan and prose."""
    if plan["plan_sha256"] != digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("private plan fingerprint mismatch")
    if plan["canonical_fact_hashes"] != [digest(raw) for raw in plan["records"]]:
        raise ValueError("raw-record fingerprint mismatch")
    if (
        assignment["family_id"] != plan["family_id"]
        or assignment["category"] != plan["category"]
        or assignment["low_blueprint"] != card["id"]
        or assignment["broad_k"] != plan["broad_k"]
    ):
        raise ValueError("family/source blueprint binding mismatch")
    rendered = render_family(plan, card, "broad")
    if canonical(cases) != canonical(rendered["cases"]):
        raise ValueError("public case/state/question/target binding mismatch")
    expected_bundle = training_bundles({"families": [rendered]}, "B")[0]
    if canonical(bundle) != canonical(expected_bundle):
        raise ValueError("training state/question/target binding mismatch")
    return rendered


def _write_new(path, value):
    path = Path(path)
    with path.open("x") as stream:
        stream.write(canonical(value) + "\n")


def _auxiliary(suite, raw_cases, schema, schema_sha256):
    cases = {}
    for case in suite["cases"]:
        source = raw_cases[case["id"]]
        targets, mask = annotate_case(source["raw"], case["category"], features=schema["features"])
        cases[case["id"]] = {
            "targets": targets,
            "mask": mask,
            "family_id": case["family_id"],
            "raw_sha256": digest(source["raw"]),
            "request_sha256": digest(case["request"]),
        }
    return {"schema_sha256": schema_sha256, "target_units": "raw", "cases": cases}


def emit_train(
    *, source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT, bank=Path("data/contrast-factorial-v1/language/reviewed-bank.json")
):
    """Copy the exact selected historical B rows to new paths, retaining all sources."""
    source, output, bank = Path(source), Path(output), Path(bank)
    destination_paths = [
        "train-suite.json",
        "train.jsonl",
        "train-raw-records.json",
        "train-aux.json",
        "feature-schema.json",
        "train-binding-manifest.json",
    ]
    if any((output / path).exists() for path in destination_paths):
        raise FileExistsError("training output exists; preserve previous attempt and choose a new directory")
    source_paths = [source / "B/train-suite.json", source / "B/train.jsonl", source / "private-audit.json", bank]
    source_hashes = {str(path.resolve()): file_sha256(path) for path in source_paths}
    suite = json.loads(source_paths[0].read_text())
    train = {row["id"]: row for row in (json.loads(line) for line in source_paths[1].read_text().splitlines())}
    audit = json.loads(source_paths[2].read_text())
    cards = {row["id"]: row for row in json.loads(bank.read_text())["blueprints"]}
    plans = {row["family_id"]: row for row in audit["plans"]}
    selected = select_families(audit["assignment"])
    selected_ids = [row["family_id"] for row in selected]
    case_groups = defaultdict(list)
    for case in suite["cases"]:
        case_groups[case["family_id"]].append(case)
    cases, relations, bundles, raw_cases, provenance = [], [], [], {}, {}
    for assignment in selected:
        family_id = assignment["family_id"]
        plan, card = plans[family_id], cards[assignment["low_blueprint"]]
        family = verify_training_family(plan, card, case_groups[family_id], train[family_id], assignment)
        original_relations = [r for r in suite["relations"] if r["family_id"] == family_id]
        if canonical(original_relations) != canonical(family["relations"]):
            raise ValueError("public relation source binding mismatch")
        cases.extend(case_groups[family_id])
        relations.extend(original_relations)
        bundles.append(train[family_id])
        provenance[family_id] = {"assignment": assignment, "plan": plan, "blueprint": card}
        for case in case_groups[family_id]:
            variant = case["variant"]
            e, r = int(variant[1]), int(variant[3])
            program = {
                "noul_rules": plan["noul_rules"],
                "choice_rules": plan["choice_rules"],
                "score_rules": plan["programs"]["broad"][r]["rules"],
            }
            raw_cases[case["id"]] = {
                "family_id": family_id,
                "category": case["category"],
                "evidence_index": e,
                "rubric_index": r,
                "raw": plan["records"][e],
                "program": program,
                "program_sha256": digest(program),
                "request_sha256": digest(case["request"]),
            }
    train_suite = {
        "name": "intermediate-supervision-v1/train",
        "cases": cases,
        "relations": relations,
        "metadata": {"source_arm": "B", "seed": 42},
        "ordered_family_ids": selected_ids,
    }
    validate_suite(train_suite)
    raw_records = {"version": 1, "cases": raw_cases, "families": provenance}
    schema = fit_schema(
        (source["raw"], source["category"]) for source in raw_cases.values() if source["rubric_index"] == 0
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_new(output / "feature-schema.json", schema)
    _write_new(output / "train-suite.json", train_suite)
    with (output / "train.jsonl").open("x") as stream:
        for bundle in bundles:
            stream.write(canonical(bundle) + "\n")
    _write_new(output / "train-raw-records.json", raw_records)
    _write_new(
        output / "train-aux.json",
        _auxiliary(train_suite, raw_cases, schema, file_sha256(output / "feature-schema.json")),
    )
    manifest = {
        "version": 1,
        "partition": "train",
        "source_sha256": source_hashes,
        "selection_policy": "SHA256 intermediate-supervision-v1/42/identity ordered backtracking over low-blueprints, 20/category, 5 per broad K2..5, exactly one per low-blueprint; original assignment order retained. No model outcomes consumed.",
        "selected_families": selected,
        "ordered_family_ids": selected_ids,
        "counts": {"families": len(selected), "cases": len(cases), "judgments": len(cases) * 5},
        "files_sha256": {p: file_sha256(output / p) for p in destination_paths if p != "train-binding-manifest.json"},
    }
    for path, fingerprint in source_hashes.items():
        if file_sha256(path) != fingerprint:
            raise ValueError("historical source changed during emission")
    _write_new(output / "train-binding-manifest.json", manifest)
    return manifest


def _score_rules(program):
    return program["score_rules"] if "score_rules" in program else program["score"]["rules"]


def _verify_case_program(case, source):
    if source["family_id"] != case["family_id"] or source["category"] != case["category"]:
        raise ValueError("raw family/category binding mismatch")
    if source["request_sha256"] != digest(case["request"]) or source["program_sha256"] != digest(source["program"]):
        raise ValueError("request/program fingerprint mismatch")
    f, p = derive_features(source["raw"]), source["program"]
    expected = {f"n{i + 1}": eval_rule(rule, f) for i, rule in enumerate(p["noul_rules"])}
    expected.update(c1=select_choice(p["choice_rules"], f), s1=select_level(_score_rules(p), f))
    if canonical(case["expected"]) != canonical(expected):
        raise ValueError("raw oracle target binding mismatch")


def preflight(output=DEFAULT_OUTPUT):
    """Recompute alignment/masks and every final oracle target, without a model."""
    output = Path(output)
    schema_path = output / "feature-schema.json"
    schema = json.loads(schema_path.read_text())
    report = {
        "version": 1,
        "schema_sha256": file_sha256(schema_path),
        "features": len(schema["features"]),
        "partitions": {},
    }
    for partition in ["train", "validation", "calibration", "confirmation"]:
        manifest_path = output / f"{partition}-binding-manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        for relative, fingerprint in manifest["files_sha256"].items():
            if file_sha256(output / relative) != fingerprint:
                raise ValueError("output hash binding mismatch: " + relative)
        for source, fingerprint in manifest["source_sha256"].items():
            if file_sha256(source) != fingerprint:
                raise ValueError("source hash binding mismatch: " + source)
        suite = json.loads((output / f"{partition}-suite.json").read_text())
        raw_records = json.loads((output / f"{partition}-raw-records.json").read_text())
        aux = json.loads((output / f"{partition}-aux.json").read_text())
        if aux["schema_sha256"] != report["schema_sha256"]:
            raise ValueError("auxiliary schema fingerprint mismatch")
        ids = {c["id"] for c in suite["cases"]}
        if ids != set(raw_records["cases"]) or ids != set(aux["cases"]) or len(ids) != len(suite["cases"]):
            raise ValueError("case ID binding mismatch")
        validate_suite(suite)
        columns = [[] for _ in schema["features"]]
        for case in suite["cases"]:
            source = raw_records["cases"][case["id"]]
            _verify_case_program(case, source)
            recomputed = _auxiliary({"cases": [case]}, raw_records["cases"], schema, report["schema_sha256"])["cases"][
                case["id"]
            ]
            if aux["cases"][case["id"]] != recomputed:
                raise ValueError("auxiliary annotation/mask binding mismatch")
            for i, (target, mask) in enumerate(zip(recomputed["targets"], recomputed["mask"])):
                if mask:
                    columns[i].append(target)
        feature_coverage = {
            f["name"]: {
                "applicable": len(values),
                "zeros": values.count(0),
                "ones": values.count(1),
                "distinct": len(set(values)),
            }
            for f, values in zip(schema["features"], columns)
        }
        counts = Counter(case["family_id"] for case in suite["cases"])
        if any(count != 4 for count in counts.values()):
            raise ValueError("incomplete four-case family")
        report["partitions"][partition] = {
            "families": len(counts),
            "cases": len(ids),
            "judgments": sum(len(c["request"]["questions"]) for c in suite["cases"]),
            "feature_coverage": feature_coverage,
        }
    return report


def _rule_primitives(value):
    if isinstance(value, dict):
        return (
            {value["feature"]}
            if set(value) == {"feature"}
            else set().union(*(_rule_primitives(v) for v in value.values()))
        )
    if isinstance(value, list):
        return set().union(*(_rule_primitives(v) for v in value))
    return set()


def audit_program_overlap(training_plans, partitions):
    """Audit executable rules, allowing shared primitives but withholding score pairs."""
    train_rules = [p["rules"] for plan in training_plans for p in plan["programs"]["broad"]]
    train_rule_hashes = {digest(r) for r in train_rules}
    train_pairs = {digest(sorted(digest(p["rules"]) for p in plan["programs"]["broad"])) for plan in training_plans}
    train_primitives = _rule_primitives(train_rules)
    reports, partition_pairs = {}, {}
    for name, records in partitions.items():
        family_rules = defaultdict(dict)
        for source in records["cases"].values():
            fid, r = source["family_id"], source["rubric_index"]
            rules = _score_rules(source["program"])
            if r in family_rules[fid] and family_rules[fid][r] != rules:
                raise ValueError("rubric program changed across evidence states")
            family_rules[fid][r] = rules
        if any(set(variants) != {0, 1} for variants in family_rules.values()):
            raise ValueError("score pair requires both rubric variants")
        pairs = {
            fid: digest(sorted(digest(rule) for rule in variants.values())) for fid, variants in family_rules.items()
        }
        hashes = {digest(rule) for variants in family_rules.values() for rule in variants.values()}
        primitives = _rule_primitives([rule for variants in family_rules.values() for rule in variants.values()])
        partition_pairs[name] = set(pairs.values())
        reports[name] = {
            "families": len(pairs),
            "unique_score_pairs": len(set(pairs.values())),
            "score_pair_hash_by_family": pairs,
            "score_pair_overlap_with_training": len(set(pairs.values()) & train_pairs),
            "individual_score_rules_shared_with_training": len(hashes & train_rule_hashes),
            "individual_score_rules_distinct": len(hashes),
            "shared_primitives_with_training": sorted(primitives & train_primitives),
            "primitives": sorted(primitives),
        }
    confirmation = partition_pairs.get("confirmation", set())
    earlier = train_pairs | partition_pairs.get("validation", set()) | partition_pairs.get("calibration", set())
    if confirmation & earlier:
        raise ValueError("confirmation score-rule pair overlaps training/validation/calibration")
    return {
        "baseline_training_families": len(training_plans),
        "baseline_unique_score_pairs": len(train_pairs),
        "pair_equivalence": "unordered pair of SHA256 canonical executable score-rule lists; wording ignored",
        "limitation": "Fresh combinations can share primitive predicates and rule subexpressions; disjoint pair hashes do not imply independent reasoning mechanisms.",
        "partitions": reports,
        "confirmation_score_pairs_withheld": True,
    }


def verify_fresh_partition(suite, records, render_request):
    """Check the author's renderer against public inputs and independently derive gold."""
    validate_suite(suite)
    ids = {c["id"] for c in suite["cases"]}
    if len(ids) != len(suite["cases"]) or ids != set(records["cases"]):
        raise ValueError("fresh case set binding mismatch")
    for case in suite["cases"]:
        source = records["cases"][case["id"]]
        rendered = render_request(source)
        if isinstance(rendered, tuple):
            rendered, expected = rendered
            if canonical(expected) != canonical(case["expected"]):
                raise ValueError("fresh renderer target binding mismatch")
        if canonical(rendered) != canonical(case["request"]):
            raise ValueError("fresh public request/raw/language binding mismatch")
        _verify_case_program(case, source)
        if case["variant"] != f"e{source['evidence_index']}r{source['rubric_index']}":
            raise ValueError("fresh evidence/rubric index binding mismatch")
    return True


def integrate_fresh(*, draft=None, output=DEFAULT_OUTPUT, source=DEFAULT_SOURCE):
    """Validate separately authored frozen drafts, then annotate using TRAIN schema only."""
    from experiments.intermediate_fresh import render_request

    output, source = Path(output), Path(source)
    draft = Path(draft) if draft is not None else output / "fresh-draft"
    if any(
        (output / f"{split}-binding-manifest.json").exists() for split in ["validation", "calibration", "confirmation"]
    ):
        raise FileExistsError("fresh annotation attempt exists; preserve it and choose a new output")
    schema = json.loads((output / "feature-schema.json").read_text())
    schema_sha256 = file_sha256(output / "feature-schema.json")
    all_records = json.loads((draft / "raw-records.json").read_text())
    partitions, suites, seen_families, seen_frames = {}, {}, set(), set()
    for split, required_families in [("validation", 20), ("calibration", 20), ("confirmation", 40)]:
        suite = json.loads((draft / f"{split}-suite.json").read_text())
        ids = {c["id"] for c in suite["cases"]}
        family_ids = {c["family_id"] for c in suite["cases"]}
        records = {
            "version": 1,
            "cases": {key: all_records["cases"][key] for key in ids},
            "families": {key: all_records["families"][key] for key in family_ids},
        }
        frame_ids = {r["frame_id"] for r in records["families"].values()}
        if (
            len(family_ids) != required_families
            or len(frame_ids) != required_families
            or family_ids & seen_families
            or frame_ids & seen_frames
        ):
            raise ValueError("fresh authoring frames/families are not disjoint and correctly sized")
        seen_families |= family_ids
        seen_frames |= frame_ids
        category_counts = Counter(c["category"] for c in suite["cases"])
        if set(category_counts) != set(CATEGORIES) or set(category_counts.values()) != {required_families * 4 // 10}:
            raise ValueError("fresh category stratification differs from protocol")
        if any(r["split"] != split for r in records["cases"].values()):
            raise ValueError("fresh partition binding mismatch")
        verify_fresh_partition(suite, records, render_request)
        partitions[split], suites[split] = records, suite
    train_manifest = json.loads((output / "train-binding-manifest.json").read_text())
    training_ids = set(train_manifest["ordered_family_ids"])
    training_blueprints = {r["low_blueprint"] for r in train_manifest["selected_families"]}
    if seen_families & training_ids or seen_frames & training_blueprints:
        raise ValueError("fresh authoring identity overlaps training")
    plans = json.loads((source / "private-audit.json").read_text())["plans"]
    overlap = audit_program_overlap(plans, partitions)
    _write_new(output / "fresh-program-overlap.json", overlap)
    for split, records in partitions.items():
        suite = suites[split]
        source_paths = [draft / "raw-records.json", draft / f"{split}-suite.json", source / "private-audit.json"]
        source_hashes = {str(path.resolve()): file_sha256(path) for path in source_paths}
        _write_new(output / f"{split}-suite.json", suite)
        _write_new(output / f"{split}-raw-records.json", records)
        _write_new(output / f"{split}-aux.json", _auxiliary(suite, records["cases"], schema, schema_sha256))
        paths = [
            f"{split}-suite.json",
            f"{split}-raw-records.json",
            f"{split}-aux.json",
            "feature-schema.json",
            "fresh-program-overlap.json",
        ]
        _write_new(
            output / f"{split}-binding-manifest.json",
            {
                "version": 1,
                "partition": split,
                "source_sha256": source_hashes,
                "files_sha256": {path: file_sha256(output / path) for path in paths},
                "ordered_family_ids": suite["ordered_family_ids"],
                "counts": {
                    "families": len(records["families"]),
                    "cases": len(suite["cases"]),
                    "judgments": len(suite["cases"]) * 5,
                },
                "schema_eligibility": "training-only frozen schema reused without alteration",
            },
        )
    return preflight(output)


def emit_repaired_train(*, original=DEFAULT_OUTPUT, output=None):
    """Root-approved six-family common-data repair; preserve all original artifacts."""
    from experiments.contrast_factorial_data import bind_plan_fingerprints

    original = Path(original)
    output = Path(output) if output is not None else original / "prepared-v2"
    if output.exists():
        raise FileExistsError("repair output already exists; preserve previous attempt")
    preflight(original)
    suite = json.loads((original / "train-suite.json").read_text())
    records = json.loads((original / "train-raw-records.json").read_text())
    manifest = json.loads((original / "train-binding-manifest.json").read_text())
    bundles = {
        row["id"]: row for row in (json.loads(line) for line in (original / "train.jsonl").read_text().splitlines())
    }
    source_hashes = {
        str((original / name).resolve()): file_sha256(original / name)
        for name in [
            "train-suite.json",
            "train-raw-records.json",
            "train.jsonl",
            "feature-schema.json",
            "train-aux.json",
            "train-binding-manifest.json",
        ]
    }
    repairs, attempts, replacements = [], [], {}

    def order(value):
        return digest(["intermediate-supervision-v1", "coverage-repair-v2", 42, value])

    measured = sorted(
        {
            r["raw"]["measurement"]
            for r in records["cases"].values()
            if r["category"] == "ordered_rubrics" and r["raw"]["measurement"] is not None
        },
        key=order,
    )
    requests = [
        (c, "window_boundary", 1)
        for c in ["claim_vs_completion", "entity_binding", "action_binding", "unknown_vs_failure"]
    ]
    requests.append(("ordered_rubrics", "measurement_missingness", 2))
    for category, kind, count in requests:
        candidates = sorted(
            (fid for fid, provenance in records["families"].items() if provenance["plan"]["category"] == category),
            key=order,
        )
        found = 0
        for fid in candidates:
            provenance = records["families"][fid]
            old_plan, card = provenance["plan"], provenance["blueprint"]
            accepted = None
            for base_index in sorted([0, 1], key=lambda e: order([fid, e])):
                for value in measured if kind == "measurement_missingness" else [None]:
                    first = copy.deepcopy(old_plan["records"][base_index])
                    changed_index = None
                    if kind == "window_boundary":
                        q = first["query"]
                        # Preserve all wrong-identity decoys, replace jointly bound rows
                        # with one explicit trusted live receipt. The pair then differs
                        # in exactly that receipt's timestamp, never a world outcome.
                        first["executions"] = [
                            r
                            for r in first["executions"]
                            if not all(r.get(k) == q[k] for k in ["actor", "operation", "target", "run_id"])
                        ]
                        changed_index = len(first["executions"])
                        first["executions"].append(
                            {
                                **q,
                                "source": first["trusted_sources"][0],
                                "verified": True,
                                "mode": "live",
                                "outcome": "no_effect" if category == "unknown_vs_failure" else "success",
                                "time": (first["window"][0] + first["window"][1]) // 2,
                            }
                        )
                    else:
                        first["measurement"] = value
                    second = copy.deepcopy(first)
                    if kind == "window_boundary":
                        second["executions"][changed_index]["time"] = second["window"][1] + 1
                    else:
                        second["measurement"] = None
                    new_plan = copy.deepcopy(old_plan)
                    new_plan.update(family_id=f"{fid}@intermediate-coverage-v2", records=[first, second])
                    bind_plan_fingerprints(new_plan)
                    attempt = {
                        "source_family_id": fid,
                        "kind": kind,
                        "base_evidence_index": base_index,
                        "measurement_candidate": value,
                        "candidate_records": [first, second],
                    }
                    try:
                        family = render_family(new_plan, card, "broad")
                    except ValueError as exc:
                        attempt.update(accepted=False, reason=str(exc))
                        attempts.append(attempt)
                        continue
                    attempt.update(accepted=True, reason="All evidence/rubric/question contrast axes preserved.")
                    attempts.append(attempt)
                    accepted = new_plan, family, changed_index, base_index
                    break
                if accepted is not None:
                    break
            if accepted is None:
                continue
            new_plan, family, changed_index, base_index = accepted
            repair = {
                "source_family_id": fid,
                "derived_family_id": family["id"],
                "kind": kind,
                "category": category,
                "base_evidence_index": base_index,
                "changed_execution_index": changed_index,
                "before_records": old_plan["records"],
                "after_records": new_plan["records"],
                "before_sha256": digest(old_plan["records"]),
                "after_sha256": digest(new_plan["records"]),
                "unchanged_programs_sha256": digest(old_plan["programs"]),
                "blueprint_id": card["id"],
                "score_k": old_plan["broad_k"],
            }
            if new_plan["programs"] != old_plan["programs"]:
                raise ValueError("coverage repair changed supplied programs")
            repairs.append(repair)
            replacements[fid] = family, new_plan
            found += 1
            if found == count:
                break
        if found != count:
            # Preserve the construction record even when no admissible six-family repair exists.
            output.mkdir(parents=True)
            _write_new(
                output / "coverage-repair-failed-attempts.json",
                {"category": category, "required": count, "constructed": found, "attempts": attempts},
            )
            raise ValueError(f"Cannot preserve contrast axes for {count} {category} repairs; constructed {found}")
    new_cases, new_relations, new_bundles, new_raw, new_families, selected = [], [], [], {}, {}, []
    old_cases = defaultdict(list)
    for case in suite["cases"]:
        old_cases[case["family_id"]].append(case)
    for assignment in manifest["selected_families"]:
        fid = assignment["family_id"]
        updated_assignment = copy.deepcopy(assignment)
        if fid in replacements:
            family, plan = replacements[fid]
            updated_assignment.update(
                family_id=family["id"], source_family_id=fid, derivation_version="coverage-repair-v2"
            )
            cases, relations = family["cases"], family["relations"]
            bundle = training_bundles({"families": [family]}, "B")[0]
            bundle["provenance"].update(
                dataset="intermediate-supervision-v1", source_family_id=fid, derivation_version="coverage-repair-v2"
            )
            provenance = {
                "assignment": updated_assignment,
                "plan": plan,
                "blueprint": records["families"][fid]["blueprint"],
                "source_family_id": fid,
                "derivation_version": "coverage-repair-v2",
            }
        else:
            cases, relations = old_cases[fid], [r for r in suite["relations"] if r["family_id"] == fid]
            bundle = bundles[fid]
            provenance = records["families"][fid]
            plan = provenance["plan"]
        selected.append(updated_assignment)
        new_cases.extend(cases)
        new_relations.extend(relations)
        new_bundles.append(bundle)
        new_families[updated_assignment["family_id"]] = provenance
        for case in cases:
            e, r = int(case["variant"][1]), int(case["variant"][3])
            program = {
                "noul_rules": plan["noul_rules"],
                "choice_rules": plan["choice_rules"],
                "score_rules": plan["programs"]["broad"][r]["rules"],
            }
            new_raw[case["id"]] = {
                "family_id": case["family_id"],
                "category": case["category"],
                "evidence_index": e,
                "rubric_index": r,
                "raw": plan["records"][e],
                "program": program,
                "program_sha256": digest(program),
                "request_sha256": digest(case["request"]),
            }
    new_suite = {
        **suite,
        "cases": new_cases,
        "relations": new_relations,
        "ordered_family_ids": [r["family_id"] for r in selected],
        "metadata": {**suite["metadata"], "derivation_version": "coverage-repair-v2", "changed_families": 6},
    }
    validate_suite(new_suite)
    schema = fit_schema((r["raw"], r["category"]) for r in new_raw.values() if r["rubric_index"] == 0)
    output.mkdir(parents=True)
    _write_new(output / "feature-schema.json", schema)
    _write_new(output / "train-suite.json", new_suite)
    with (output / "train.jsonl").open("x") as stream:
        for bundle in new_bundles:
            stream.write(canonical(bundle) + "\n")
    _write_new(output / "train-raw-records.json", {"version": 2, "cases": new_raw, "families": new_families})
    _write_new(
        output / "train-aux.json", _auxiliary(new_suite, new_raw, schema, file_sha256(output / "feature-schema.json"))
    )
    _write_new(
        output / "coverage-repair-manifest.json",
        {
            "version": 2,
            "reason": "Prepare both arms with explicit identity/window separation and numeric missingness labels before freezing.",
            "source_sha256": source_hashes,
            "repairs": repairs,
            "construction_attempts": attempts,
            "selection_policy": "Hash-ordered candidate families/base states/observed values, accepting only source-program-preserving meaningful contrast families. No model outcomes.",
            "cost": {
                "repaired_families": 6,
                "repaired_cases": 24,
                "unchanged_families": 194,
                "extra_training_examples": 0,
                "teacher_calls": 0,
            },
        },
    )
    paths = [
        "feature-schema.json",
        "train-suite.json",
        "train.jsonl",
        "train-raw-records.json",
        "train-aux.json",
        "coverage-repair-manifest.json",
    ]
    result = {
        **manifest,
        "version": 2,
        "selected_families": selected,
        "ordered_family_ids": new_suite["ordered_family_ids"],
        "source_sha256": {**manifest["source_sha256"], **source_hashes},
        "files_sha256": {path: file_sha256(output / path) for path in paths},
        "derivation_version": "coverage-repair-v2",
        "repair_manifest": "coverage-repair-manifest.json",
    }
    _write_new(output / "train-binding-manifest.json", result)
    preflight(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["emit-train", "repair-train", "integrate-fresh", "preflight"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draft", type=Path)
    args = parser.parse_args()
    result = (
        {
            "emit-train": lambda: emit_train(output=args.output),
            "repair-train": lambda: emit_repaired_train(original=args.output),
            "integrate-fresh": lambda: integrate_fresh(output=args.output, draft=args.draft),
            "preflight": lambda: preflight(args.output),
        }
    )[args.command]()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
