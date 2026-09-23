"""Versioned, semantics-preserving compression of fresh Score level descriptions.

Only duplicated checklist prose is replaced. Every level explicitly refers to
its unchanged, fully defined scoring instruction in STATE.score_rule. Raw facts,
executable programs, targets, identities, partitions and training bytes stay fixed.
This module never loads model weights or uses a GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path

from experiments.intermediate_data import (
    canonical,
    digest,
    file_sha256,
    integrate_fresh,
    preflight,
    verify_fresh_partition,
)
from experiments.intermediate_fresh import describe, render_request

BASE = Path("data/intermediate-supervision-v1")
SPLITS = ("validation", "calibration", "confirmation")


def compact_score(score):
    """Reference the exact shared checklist, deriving thresholds from each AST."""
    rules, levels = score["rules"], score["levels"]
    if len(rules) != len(levels) or len(rules) < 2 or rules[0] is not True:
        raise ValueError("score checklist requires an explicit true fallback and aligned levels")
    checklist, thresholds = None, []
    for rule in rules[1:]:
        if not isinstance(rule, dict) or set(rule) != {"at_least"}:
            raise ValueError("score level is not a counted checklist")
        body = rule["at_least"]
        if not isinstance(body, dict) or set(body) != {"count", "of"}:
            raise ValueError("score checklist needs explicit count and requirements")
        if type(body["count"]) is not int or not 1 <= body["count"] <= 4 or len(body["of"]) != 4:
            raise ValueError("score references must count four explicit requirements")
        if checklist is None:
            checklist = body["of"]
        if body["of"] != checklist:
            raise ValueError("score levels do not share the same four requirements")
        thresholds.append(body["count"])
    instruction = score["instruction"]
    prefix = re.match(r"^Apply checklist rubric [12]\. It contains four explicit requirements: ", instruction)
    if prefix is None:
        raise ValueError("score instruction does not explicitly define the checklist")
    expected = (
        prefix.group()
        + "; ".join(f"({i + 1}) {describe(rule)}" for i, rule in enumerate(checklist))
        + ". Evaluate the level definitions and choose the highest numbered applicable level."
    )
    if instruction != expected:
        raise ValueError("score instruction differs from the exact four AST checklist requirements")
    result = copy.deepcopy(score)
    result["levels"] = [levels[0]] + [
        f"Level {index}: at least {threshold} of the four numbered requirements in the scoring instruction (score_rule) hold."
        for index, threshold in enumerate(thresholds, start=1)
    ]
    return result


def compact_artifacts(suites, records):
    """Return new artifacts, enforcing the two allowed public-field changes only."""
    compact_suites, compact_records = copy.deepcopy(suites), copy.deepcopy(records)
    changes = []
    all_ids = set()
    for split in SPLITS:
        suite = suites[split]
        ids = {case["id"] for case in suite["cases"]}
        if all_ids & ids:
            raise ValueError("fresh partitions have overlapping case identities")
        all_ids |= ids
        verify_fresh_partition(suite, {"cases": {key: records["cases"][key] for key in ids}}, render_request)
        for old_case, new_case in zip(suite["cases"], compact_suites[split]["cases"], strict=True):
            case_id = old_case["id"]
            original = records["cases"][case_id]
            private = compact_records["cases"][case_id]
            private["program"]["score"] = compact_score(original["program"]["score"])
            request, expected = render_request(private)
            if expected != old_case["expected"]:
                raise ValueError("compact rendering changed final targets")
            allowed = copy.deepcopy(old_case["request"])
            allowed["state"]["score_levels"] = private["program"]["score"]["levels"]
            allowed["questions"]["s1"]["criteria"] = private["program"]["score"]["levels"]
            if request != allowed:
                raise ValueError("compact renderer changed a field beyond the two level-description copies")
            new_case["request"] = request
            private["program_sha256"] = digest(private["program"])
            private["request_sha256"] = digest(request)
            changes.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "changed_public_fields": ["request.state.score_levels", "request.questions.s1.criteria"],
                    "before_levels": original["program"]["score"]["levels"],
                    "after_levels": private["program"]["score"]["levels"],
                    "unchanged_raw_sha256": digest(original["raw"]),
                    "unchanged_score_ast_sha256": digest(original["program"]["score"]["rules"]),
                    "before_program_sha256": original["program_sha256"],
                    "after_program_sha256": private["program_sha256"],
                    "before_request_sha256": original["request_sha256"],
                    "after_request_sha256": private["request_sha256"],
                }
            )
    if all_ids != set(records["cases"]):
        raise ValueError("fresh raw/public case populations differ")
    for fid, family in compact_records["families"].items():
        family["program_hashes"] = [
            digest(compact_records["cases"][f"{fid}/e0r{r}"]["program"]["score"]) for r in (0, 1)
        ]
        if any(
            compact_records["cases"][f"{fid}/e0r{r}"]["program"] != compact_records["cases"][f"{fid}/e1r{r}"]["program"]
            for r in (0, 1)
        ):
            raise ValueError("rubric programs differ across evidence states")
    return compact_suites, compact_records, changes


def _tree_hashes(root):
    return {str(path.resolve()): file_sha256(path) for path in sorted(Path(root).rglob("*")) if path.is_file()}


def _write(path, data):
    with Path(path).open("x") as stream:
        stream.write(canonical(data) + "\n")


def emit_compact(
    *,
    source_prepared=BASE / "prepared-v2",
    source_draft=BASE / "fresh-draft",
    draft=BASE / "fresh-draft-compact-v3",
    prepared=BASE / "prepared-v3",
):
    source_prepared, source_draft, draft, prepared = map(Path, (source_prepared, source_draft, draft, prepared))
    if draft.exists() or prepared.exists():
        raise FileExistsError("compact attempt output exists; preserve it and choose a new version")
    originals = {**_tree_hashes(source_prepared), **_tree_hashes(source_draft)}
    preflight(source_prepared)
    suites = {split: json.loads((source_draft / f"{split}-suite.json").read_text()) for split in SPLITS}
    records = json.loads((source_draft / "raw-records.json").read_text())
    compact_suites, compact_records, changes = compact_artifacts(suites, records)
    draft.mkdir(parents=True)
    for split, suite in compact_suites.items():
        _write(draft / f"{split}-suite.json", suite)
    _write(draft / "raw-records.json", compact_records)
    training_manifest = json.loads((source_prepared / "train-binding-manifest.json").read_text())
    train_files = sorted({*training_manifest["files_sha256"], "train-binding-manifest.json"})
    prepared.mkdir(parents=True)
    for name in train_files:
        shutil.copyfile(source_prepared / name, prepared / name)
    audit = integrate_fresh(draft=draft, output=prepared)
    if originals != {path: file_sha256(path) for path in originals}:
        raise ValueError("original source bytes changed during compact emission")
    if any(file_sha256(prepared / name) != file_sha256(source_prepared / name) for name in train_files):
        raise ValueError("training copy differs from original bytes")
    # Evidence-derived auxiliary values and masks must be identical; only their
    # request hashes should change when the public Score prose changes.
    for split in SPLITS:
        before = json.loads((source_prepared / f"{split}-aux.json").read_text())
        after = json.loads((prepared / f"{split}-aux.json").read_text())
        for case_id, annotation in before["cases"].items():
            if {k: v for k, v in annotation.items() if k != "request_sha256"} != {
                k: v for k, v in after["cases"][case_id].items() if k != "request_sha256"
            }:
                raise ValueError("compact rendering changed auxiliary facts/masks")
    manifest = {
        "version": 3,
        "cases_changed": len(changes),
        "source_sha256": originals,
        "training_copied_byte_identical": True,
        "training_files_sha256": {name: file_sha256(prepared / name) for name in train_files},
        "compact_draft_files_sha256": {path.name: file_sha256(path) for path in sorted(draft.iterdir())},
        "allowed_public_changes": ["request.state.score_levels", "request.questions.s1.criteria"],
        "preserved": [
            "all raw records",
            "all executable AST rules",
            "all final/aux targets and masks",
            "family/case IDs and order",
            "splits",
            "relations",
            "Noul/Choice definitions",
            "score instruction, ordinality, selection and level-zero fallback",
            "training/schema bytes",
        ],
        "changes": changes,
        "data_preflight": audit,
    }
    _write(draft / "compact-rendering-manifest.json", manifest)
    shutil.copyfile(draft / "compact-rendering-manifest.json", prepared / "compact-rendering-manifest.json")
    return manifest


def tokenizer_audit(*, before=BASE / "prepared-v2", after=BASE / "prepared-v3", tokenizer=None):
    """Measure exact chat units and enforce unchanged <=1536 serving input gate."""
    from openjev.judgment_model import MODEL_ID, REVISION, JudgmentEngine
    from openjev.judgments import compile_request, render_unit_messages

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    engine = JudgmentEngine(None, tokenizer, max_input_tokens=1536, unit_batch_size=4)
    result = {
        "tokenizer": {"model_id": MODEL_ID, "revision": REVISION, "local_files_only": True},
        "max_input_tokens": 1536,
        "truncation": False,
        "partitions": {},
        "passed": True,
    }
    for split in ("train", *SPLITS):
        before_path, after_path = Path(before) / f"{split}-suite.json", Path(after) / f"{split}-suite.json"
        old = json.loads(before_path.read_text())
        new = json.loads(after_path.read_text())
        if [c["id"] for c in old["cases"]] != [c["id"] for c in new["cases"]]:
            raise ValueError("tokenizer audit requires identical ordered case populations")
        rows, all_before, all_after = [], [], []
        for old_case, new_case in zip(old["cases"], new["cases"], strict=True):

            def lengths(case):
                return [
                    len(
                        tokenizer.apply_chat_template(
                            render_unit_messages(unit),
                            tokenize=True,
                            return_dict=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )
                    for unit in compile_request(case["request"]).units
                ]

            old_lengths, new_lengths = lengths(old_case), lengths(new_case)
            if len(old_lengths) != len(new_lengths):
                raise ValueError("compact rendering changed compiled unit cardinality")
            rows.append(
                {
                    "case_id": old_case["id"],
                    "before": old_lengths,
                    "after": new_lengths,
                    "before_max": max(old_lengths),
                    "after_max": max(new_lengths),
                }
            )
            all_before.extend(old_lengths)
            all_after.extend(new_lengths)
            if max(new_lengths) <= 1536:
                _, prompts, _ = engine.prepare(new_case["request"])
                if list(map(len, prompts)) != new_lengths:
                    raise ValueError("direct tokenizer lengths differ from serving-engine prepare")
        bad = [row["case_id"] for row in rows if row["after_max"] > 1536]
        result["passed"] = result["passed"] and not bad
        result["partitions"][split] = {
            "cases": len(rows),
            "units": len(all_after),
            "before_max_tokens": max(all_before),
            "after_max_tokens": max(all_after),
            "before_over_limit_cases": sum(row["before_max"] > 1536 for row in rows),
            "after_over_limit_cases": bad,
            "before_total_tokens": sum(all_before),
            "after_total_tokens": sum(all_after),
            "before_suite_sha256": file_sha256(before_path),
            "after_suite_sha256": file_sha256(after_path),
            "case_lengths": rows,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["emit", "tokenizer-audit"])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = emit_compact() if args.command == "emit" else tokenizer_audit()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write(args.report, result)
    if args.command == "tokenizer-audit" and not result["passed"]:
        raise ValueError("compact fresh units still exceed1536; preserve report and do not freeze")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("version", "cases_changed", "training_copied_byte_identical", "passed")
                if k in result
            }
        )
    )


if __name__ == "__main__":
    main()
