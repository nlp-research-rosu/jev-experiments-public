"""Post-hoc reporting for the locked contrast-scaling experiment.

This module never selects checkpoints, evaluates models, or mutates experiment
artifacts.  The CLI fails closed until selection is locked and every deduplicated
test checkpoint has complete outputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from experiments.contrast_scaling_evaluation import unique_test_checkpoints
from experiments.paired_language_metrics import matrix_summary
from experiments.revised_metrics import answer_space, relation_metrics, target_label


REQUIRED_TEST_SUITES = ("new", "broad", "semantic114", "prior_paired_clarified")
ENDPOINT_ORDER = ("data_200", "data_1000", "data_5000", "repeat_200")
SELECTED_ORDER = ENDPOINT_ORDER


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _role_points(selection):
    rows = [("baseline", selection["baseline"])]
    rows.extend((f"endpoint/{name}", selection["endpoints"][name]) for name in ENDPOINT_ORDER)
    rows.extend((f"selected/{name}", selection["selected"][name]) for name in SELECTED_ORDER)
    return rows


def _validate_metadata(selection, evaluation, training, manifest):
    if not isinstance(selection, dict) or not selection.get("locked_utc"):
        raise ValueError("checkpoint selection must be durably locked before reporting")
    if evaluation.get("status") != "complete" or evaluation.get("selection") != selection:
        raise ValueError("evaluation must be complete and match the locked selection")
    if training.get("status") != "complete":
        raise ValueError("training must be complete before reporting")
    for name, updates in (("primary", 5000), ("repeat_200", 1000)):
        stream = training.get(name, {})
        if stream.get("status") != "complete" or stream.get("completed_updates") != updates:
            raise ValueError(f"training stream {name} is incomplete")
    if manifest.get("study") != "contrast-scaling-v1" or manifest.get("status") != "frozen":
        raise ValueError("training data manifest must be frozen")

    jobs = {job["checkpoint_id"]: job for job in unique_test_checkpoints(selection)}
    tests = evaluation.get("tests", {})
    if set(tests) != set(jobs):
        raise ValueError("complete deduplicated checkpoint outputs are required")
    for identity, job in jobs.items():
        actual = tests[identity]
        if actual.get("checkpoint_id") != identity or set(actual.get("roles", ())) != set(job["roles"]):
            raise ValueError("checkpoint role aliases differ from the locked selection")
        results = actual.get("results", {})
        if set(results) != set(REQUIRED_TEST_SUITES) or any("summary" not in results[name] for name in REQUIRED_TEST_SUITES):
            raise ValueError("every retained test suite must be complete")
    return jobs


def _validate_inputs(selection, evaluation, training, manifest, rows_by_checkpoint):
    jobs = _validate_metadata(selection, evaluation, training, manifest)
    if set(rows_by_checkpoint) != set(jobs):
        raise ValueError("complete deduplicated checkpoint outputs are required")
    return jobs


def _check_recorded_summary(matrix, recorded):
    actual, expected = matrix["overall"], recorded["summary"]["overall"]
    for key in ("questions", "correct"):
        if actual[key] != expected.get(key):
            raise ValueError("recomputed predictions differ from the recorded summary")
    for key in ("accuracy", "nll", "brier"):
        value = expected.get(key)
        if type(value) not in (int, float) or not math.isclose(actual[key], value, rel_tol=0, abs_tol=1e-12):
            raise ValueError("recomputed predictions differ from the recorded summary")


def _validate_full_distributions(suite, rows):
    cases = {case["id"]: case for case in suite["cases"]}
    expected_keys = {(case["id"], qid) for case in suite["cases"] for qid in case["expected"]}
    actual_keys = [(row.get("case_id"), row.get("question_id")) for row in rows]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise ValueError("reference prediction coverage differs from frozen request IDs")
    for row in rows:
        case = cases[row["case_id"]]
        question = case["request"]["questions"][row["question_id"]]
        prediction = row.get("prediction", {})
        probabilities = prediction.get("probabilities")
        labels = answer_space(question)
        expected = target_label(case["expected"][row["question_id"]])
        if (
            not isinstance(probabilities, dict) or set(probabilities) != labels
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                   for value in probabilities.values())
            or not math.isclose(math.fsum(probabilities.values()), 1, rel_tol=0, abs_tol=1e-6)
            or prediction.get("expected") != expected
        ):
            raise ValueError("reference must provide a complete valid probability distribution")
        peak = max(probabilities.values())
        winners = [label for label, value in probabilities.items() if abs(value - peak) <= 1e-12]
        predicted = winners[0] if len(winners) == 1 else None
        if prediction.get("predicted") != predicted or prediction.get("correct") is not (predicted == expected):
            raise ValueError("reference distribution winner metadata is inconsistent")
        computed_nll = -math.log(max(probabilities[expected], 1e-12))
        computed_brier = math.fsum((value - float(label == expected)) ** 2 for label, value in probabilities.items())
        if (
            prediction.get("primitive") != question["type"]
            or type(prediction.get("nll")) not in (int, float)
            or not math.isclose(prediction["nll"], computed_nll, rel_tol=0, abs_tol=1e-12)
            or type(prediction.get("brier")) not in (int, float)
            or not math.isclose(prediction["brier"], computed_brier, rel_tol=0, abs_tol=1e-12)
            or type(prediction.get("max_probability")) not in (int, float)
            or not math.isclose(prediction["max_probability"], peak, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("reference distribution metrics are inconsistent")
        if question["type"] == "score":
            target = case["expected"][row["question_id"]]
            mean = math.fsum(int(label) * value for label, value in probabilities.items())
            error = abs(mean - target)
            if (
                type(prediction.get("score_absolute_error")) not in (int, float)
                or not math.isclose(prediction["score_absolute_error"], error, rel_tol=0, abs_tol=1e-12)
                or type(prediction.get("score_normalized_absolute_error")) not in (int, float)
                or not math.isclose(prediction["score_normalized_absolute_error"], error / max(1, len(probabilities) - 1),
                                    rel_tol=0, abs_tol=1e-12)
            ):
                raise ValueError("reference Score distribution metrics are inconsistent")


def _error_overlap(local_rows, reference_rows):
    local = {(row["case_id"], row["question_id"]): row["prediction"] for row in local_rows}
    reference = {(row["case_id"], row["question_id"]): row["prediction"] for row in reference_rows}
    if local.keys() != reference.keys():
        raise ValueError("reference overlap population differs")
    counts, distances = Counter(), []
    local_nll, reference_nll = [], []
    disagreements = 0
    for key, left in local.items():
        right = reference[key]
        if left["expected"] != right["expected"] or left["probabilities"].keys() != right["probabilities"].keys():
            raise ValueError("reference overlap labels or answer spaces differ")
        state = ("correct" if left["correct"] else "wrong", "correct" if right["correct"] else "wrong")
        counts[state] += 1
        distances.append(.5 * math.fsum(abs(left["probabilities"][label] - right["probabilities"][label])
                                        for label in left["probabilities"]))
        disagreements += left["predicted"] != right["predicted"]
        local_nll.append(-math.log(max(left["probabilities"][left["expected"]], 1e-12)))
        reference_nll.append(-math.log(max(right["probabilities"][right["expected"]], 1e-12)))
    return {
        "questions": len(local), "both_correct": counts[("correct", "correct")],
        "local_only_correct": counts[("correct", "wrong")],
        "reference_only_correct": counts[("wrong", "correct")], "both_wrong": counts[("wrong", "wrong")],
        "prediction_disagreements": disagreements, "mean_total_variation": statistics.mean(distances),
        "local_probability_nll_floor_1e12": statistics.mean(local_nll),
        "reference_probability_nll_floor_1e12": statistics.mean(reference_nll),
        "nll_comparison_policy": "Both sides are recomputed from full probabilities with a shared 1e-12 floor.",
    }


def _jev_section(suite, reference, local_rows):
    report = reference.get("report", {})
    expected_hash = reference.get("expected_suite_sha256")
    case_ids = [case["id"] for case in suite["cases"]]
    if report.get("status") != "complete" or report.get("metadata", {}).get("suite_sha256") != expected_hash:
        raise ValueError("Jev reference suite hash or completion status differs")
    if reference.get("request_ids") != case_ids:
        raise ValueError("Jev reference request IDs differ from the frozen suite")
    questions = sum(len(case["expected"]) for case in suite["cases"])
    if (report.get("requests_planned") != len(case_ids) or report.get("requests_completed") != len(case_ids)
            or report.get("questions_planned") != questions):
        raise ValueError("Jev reference is incomplete")
    rows = reference.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Jev reference predictions are missing")
    _validate_full_distributions(suite, rows)
    matrix = matrix_summary(suite, rows)
    return {
        "designation": "external_reference_observation_not_gold",
        "returned_models": report.get("returned_models", []),
        "suite_sha256": expected_hash,
        "metric_policy": report.get("metric_policy", {}),
        "matrix": matrix, "headline": _headline(matrix),
        "relation_contrast_axis": relation_contrast_axis_matrix(suite, rows),
        "error_overlap_by_checkpoint": {
            identity: _error_overlap(values, rows) for identity, values in local_rows.items()
        },
    }


def _relation_block(relations, lookup):
    metrics = [relation_metrics(relation, lookup) for relation in relations]
    result = {
        "pairs": len(metrics),
        "both_correct": sum(row["both_correct"] for row in metrics),
        "both_correct_rate": statistics.mean(row["both_correct"] for row in metrics),
        "mean_total_variation": statistics.mean(row["total_variation"] for row in metrics),
    }
    if all("direction_correct" in row for row in metrics):
        result.update(direction_correct=sum(row["direction_correct"] for row in metrics),
                      direction_correct_rate=statistics.mean(row["direction_correct"] for row in metrics),
                      mean_directional_margin=statistics.mean(row["directional_margin"] for row in metrics))
    if all("within_tolerance" in row for row in metrics):
        result.update(within_tolerance=sum(row["within_tolerance"] for row in metrics),
                      within_tolerance_rate=statistics.mean(row["within_tolerance"] for row in metrics))
    return result


def relation_contrast_axis_matrix(suite, rows):
    """Summarize relation performance jointly by relation kind and declared axis."""
    cases = {case["id"]: case for case in suite["cases"]}
    lookup = {(row["case_id"], row["question_id"]): row["prediction"] for row in rows}
    grouped, categories = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for relation in suite["relations"]:
        axis = relation.get("contrast_axis", "unspecified")
        key = f"{relation['kind']}/{axis}"
        try:
            category = cases[relation["left"]["case_id"]]["category"]
            right_category = cases[relation["right"]["case_id"]]["category"]
        except KeyError as error:
            raise ValueError("relation endpoint is absent from the reporting suite") from error
        if category != right_category:
            raise ValueError("relation crosses reporting categories")
        grouped[key].append(relation)
        categories[category][key].append(relation)
    return {
        "overall": {key: _relation_block(value, lookup) for key, value in sorted(grouped.items())},
        "by_category": {
            category: {key: _relation_block(value, lookup) for key, value in sorted(groups.items())}
            for category, groups in sorted(categories.items())
        },
    }


def _family_summaries(suite, rows):
    cases = {case["id"]: case for case in suite["cases"]}
    expected = {(case["id"], qid) for case in suite["cases"] for qid in case["expected"]}
    actual = [(row["case_id"], row["question_id"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Prediction coverage differs from the complete suite")
    summaries = defaultdict(lambda: Counter(questions=0, correct=0, nll=0.0, brier=0.0,
                                             scores=0, score_error=0.0, unknown=0, unknown_correct=0,
                                             relations=0, relations_both_correct=0))
    lookup = {(row["case_id"], row["question_id"]): row["prediction"] for row in rows}
    for row in rows:
        family = cases[row["case_id"]]["family_id"]
        prediction = row["prediction"]
        value = summaries[family]
        value["questions"] += 1
        value["correct"] += int(prediction["correct"])
        value["nll"] += prediction["nll"]
        value["brier"] += prediction["brier"]
        if prediction["primitive"] == "score":
            value["scores"] += 1
            value["score_error"] += prediction["score_absolute_error"]
        if prediction["primitive"] == "choice" and prediction["expected"].lower() == "unknown":
            value["unknown"] += 1
            value["unknown_correct"] += int(prediction["correct"])
    for relation in suite["relations"]:
        family = cases[relation["left"]["case_id"]]["family_id"]
        metrics = relation_metrics(relation, lookup)
        summaries[family]["relations"] += 1
        summaries[family]["relations_both_correct"] += int(metrics["both_correct"])
    categories = defaultdict(list)
    for case in suite["cases"]:
        if case["family_id"] not in categories[case["category"]]:
            categories[case["category"]].append(case["family_id"])
    return summaries, categories


def _aggregate_family_metric(summary, sampled, metric):
    if metric in ("accuracy", "nll", "brier"):
        numerator_name = {"accuracy": "correct", "nll": "nll", "brier": "brier"}[metric]
        numerator = sum(summary[identity][numerator_name] for identity in sampled)
        denominator = sum(summary[identity]["questions"] for identity in sampled)
    elif metric == "score_mae":
        numerator = sum(summary[identity]["score_error"] for identity in sampled)
        denominator = sum(summary[identity]["scores"] for identity in sampled)
    elif metric == "unknown_recall":
        numerator = sum(summary[identity]["unknown_correct"] for identity in sampled)
        denominator = sum(summary[identity]["unknown"] for identity in sampled)
    elif metric == "relation_both_correct":
        numerator = sum(summary[identity]["relations_both_correct"] for identity in sampled)
        denominator = sum(summary[identity]["relations"] for identity in sampled)
    else:
        raise ValueError(f"unknown bootstrap metric {metric}")
    return numerator / denominator if denominator else None


def paired_family_bootstrap(suite, left, right, *, draws=2000, seed=42):
    """Bootstrap paired differences by resampling whole families within category."""
    if type(draws) is not int or draws < 1 or type(seed) is not int:
        raise ValueError("positive bootstrap draws and an integer seed are required")
    left_summary, categories = _family_summaries(suite, left)
    right_summary, right_categories = _family_summaries(suite, right)
    if categories != right_categories:
        raise ValueError("paired systems must cover identical categorized families")
    all_ids = [identity for families in categories.values() for identity in families]
    metrics = ("accuracy", "nll", "brier", "score_mae", "unknown_recall", "relation_both_correct")

    def difference(sampled, metric):
        a = _aggregate_family_metric(left_summary, sampled, metric)
        b = _aggregate_family_metric(right_summary, sampled, metric)
        return None if a is None or b is None else b - a

    samples = {metric: [] for metric in metrics}
    rng = random.Random(seed)
    for _ in range(draws):
        sampled = [rng.choice(families) for families in categories.values() for _ in families]
        for metric in metrics:
            value = difference(sampled, metric)
            if value is not None:
                samples[metric].append(value)
    result = {
        "families": len(all_ids), "draws": draws, "seed": seed,
        "resampling_unit": "complete_family", "stratification": "category", "direction": "right minus left",
        "training_variance": "Not estimated: the experiment has one training seed.",
    }
    for metric in metrics:
        values = sorted(samples[metric])
        interval = None
        if values:
            interval = [values[math.floor(.025 * (len(values) - 1))], values[math.ceil(.975 * (len(values) - 1))]]
        result[metric] = {"difference": difference(all_ids, metric), "interval_95": interval}
    return result


def _headline(matrix):
    overall, unknown = matrix["overall"], matrix["unknown"]
    return {
        "questions": overall["questions"], "accuracy": overall["accuracy"],
        "nll": overall["nll"], "probability_nll_floor_1e12": overall["probability_nll_floor_1e12"],
        "brier": overall["brier"], "confident_wrong_90": overall["confident_wrong_90"],
        "confident_wrong_95": overall["confident_wrong_95"],
        "unknown_precision": unknown["precision"], "unknown_recall": unknown["recall"],
        "score_mae": overall.get("score_mean_absolute_error"),
        "whole_case_accuracy": matrix["whole_cases"]["all_correct"] / matrix["whole_cases"]["cases"],
    }


def _metric_delta(left, right):
    return {key: right.get(key) - left.get(key) if left.get(key) is not None and right.get(key) is not None else None
            for key in ("accuracy", "nll", "brier", "unknown_precision", "unknown_recall", "score_mae", "whole_case_accuracy")}


def _retention_delta(actual, baseline):
    result = {}
    for suite_name in ("broad", "semantic114", "prior_paired_clarified"):
        left = baseline["results"][suite_name]["summary"]
        right = actual["results"][suite_name]["summary"]
        block = {
            "accuracy": right["overall"]["accuracy"],
            "accuracy_delta": right["overall"]["accuracy"] - left["overall"]["accuracy"],
            "nll": right["overall"]["nll"],
            "nll_delta": right["overall"]["nll"] - left["overall"]["nll"],
            "brier": right["overall"].get("brier"),
            "brier_delta": right["overall"].get("brier") - left["overall"].get("brier")
            if right["overall"].get("brier") is not None and left["overall"].get("brier") is not None else None,
        }
        score_sources = {}
        for source, reference in left.get("by_domain", {}).items():
            if "score_mean_absolute_error" not in reference:
                continue
            value = right["by_domain"][source]
            score_sources[source] = {
                "accuracy_delta": value["accuracy"] - reference["accuracy"],
                "score_mae_delta": value["score_mean_absolute_error"] - reference["score_mean_absolute_error"],
            }
        if score_sources:
            block["score_sources"] = score_sources
        result[suite_name] = block
    return result


def _exposure(role, point):
    step = point["step"]
    repeat = point.get("stream") == "repeat-200"
    unique = min(step, 200) if repeat else step
    return {
        "role": role, "stream": point.get("stream"), "step": step, "unique_new_families": unique,
        "family_presentations": step, "new_judgments_presented": 20 * step,
        "replay_judgments_presented": 3 * step,
        "epochs_over_unique_families": step / unique if unique else 0,
    }


def _history_step(histories, stream, step):
    for row in histories.get(stream, ()):
        if row.get("step") == step:
            return row.get("training_seconds")
    return None


def _compact_manifest(manifest):
    diversity = {key: value for key, value in manifest.get("diversity", {}).items()
                 if key != "normalized_signature_counts"}
    return {
        "study": manifest.get("study"), "status": manifest.get("status"), "frozen_utc": manifest.get("frozen_utc"),
        "prefixes": manifest.get("prefixes", {}), "diversity": diversity,
        "labels_by_category_question": manifest.get("labels_by_category_question", {}),
        "relations_by_kind_axis": manifest.get("relations_by_kind_axis", {}),
        "source_bank_sha256": manifest.get("source_bank_sha256", {}),
        "data_sha256": manifest.get("data_sha256", {}), "review_sha256": manifest.get("review_sha256", {}),
        "limitations": manifest.get("metadata", {}).get("limitations"),
    }


def _validate_exposure(exposure, training):
    if exposure is None:
        return {
            "status": "not_recorded",
            "reason": "The training summary records family/judgment exposure and wall time but not aggregate prompt tokens.",
        }
    fingerprint = training.get("fingerprint", {})
    if exposure.get("status") != "complete" or exposure.get("study") != "contrast-scaling-v1" or exposure.get("model_weights_loaded") is not False:
        raise ValueError("exposure artifact must be complete, CPU-only, and weight-free")
    for key in ("train_sha256", "manifest_sha256", "replay_sha256", "prepared_prompt_fingerprint",
                "max_units", "unit_batch_size"):
        if exposure.get(key) != fingerprint.get(key):
            raise ValueError("exposure fingerprint differs from completed training")
    if not isinstance(exposure.get("tokenizer"), dict) or not exposure["tokenizer"].get("model_id") or not exposure["tokenizer"].get("revision"):
        raise ValueError("exposure tokenizer identity is incomplete")
    if type(exposure.get("new_candidate_units")) is not int or exposure["new_candidate_units"] < 1:
        raise ValueError("exposure candidate-unit population is invalid")
    token_range = exposure.get("new_token_range")
    if (not isinstance(token_range, list) or len(token_range) != 2
            or any(type(value) is not int or value < 0 for value in token_range) or token_range[0] > token_range[1]):
        raise ValueError("exposure token range is invalid")
    fields = {"prompt_tokens", "candidate_units", "padded_forward_tokens", "forward_calls", "judgments"}

    def validate_block(block):
        if not isinstance(block, dict) or set(block) != {"new", "old"}:
            raise ValueError("exposure block must split new and old presentations")
        for values in block.values():
            if set(values) != fields or any(type(value) is not int or value < 0 for value in values.values()):
                raise ValueError("exposure counters must be complete nonnegative integers")

    expected_steps = {
        "primary_by_step": {"0", "200", "400", "600", "800", "1000", "2000", "3000", "4000", "5000"},
        "repeat_by_step": {"0", "200", "400", "600", "800", "1000"},
    }
    for name, steps in expected_steps.items():
        values = exposure.get(name)
        if not isinstance(values, dict) or set(values) != steps:
            raise ValueError("exposure checkpoint coverage is incomplete")
        for block in values.values():
            validate_block(block)
        for origin in ("new", "old"):
            for field in fields:
                series = [values[str(step)][origin][field] for step in sorted(map(int, steps))]
                if series != sorted(series):
                    raise ValueError("cumulative exposure counters must be monotonic")
    validate_block(exposure.get("repeat_extension"))
    for origin in ("new", "old"):
        for field in fields:
            expected = exposure["repeat_by_step"]["1000"][origin][field] - exposure["repeat_by_step"]["200"][origin][field]
            if exposure["repeat_extension"][origin][field] != expected:
                raise ValueError("repeat exposure extension differs from cumulative checkpoints")
    return copy.deepcopy(exposure)


def build_report(selection, evaluation, training, manifest, suite, rows_by_checkpoint, *, histories=None,
                 draws=2000, seed=42, provenance=None, jev_reference=None, exposure=None):
    """Build the complete machine-readable report from locked, complete artifacts."""
    jobs = _validate_inputs(selection, evaluation, training, manifest, rows_by_checkpoint)
    histories = histories or {}
    roles = _role_points(selection)
    role_aliases = {role: point["checkpoint_id"] for role, point in roles}
    evaluations = {}
    for identity in jobs:
        rows = rows_by_checkpoint[identity]
        matrix = matrix_summary(suite, rows)
        _check_recorded_summary(matrix, evaluation["tests"][identity]["results"]["new"])
        evaluations[identity] = {
            "roles": sorted(jobs[identity]["roles"]), "matrix": matrix,
            "headline": _headline(matrix),
            "relation_contrast_axis": relation_contrast_axis_matrix(suite, rows),
            "retained_test_summaries": evaluation["tests"][identity]["results"],
        }

    def table_row(role, point):
        identity = point["checkpoint_id"]
        return {"role": role, "checkpoint_id": identity, **_exposure(role, point),
                "metrics": evaluations[identity]["headline"]}

    endpoint_rows = [table_row("baseline", selection["baseline"])] + [
        table_row(f"endpoint/{name}", selection["endpoints"][name]) for name in ENDPOINT_ORDER
    ]
    selected_rows = [table_row(f"selected/{name}", selection["selected"][name]) for name in SELECTED_ORDER]

    comparisons = [
        ("endpoint_data_200_minus_baseline", "baseline", "endpoint/data_200"),
        ("endpoint_data_1000_minus_data_200", "endpoint/data_200", "endpoint/data_1000"),
        ("endpoint_data_5000_minus_data_1000", "endpoint/data_1000", "endpoint/data_5000"),
        ("fixed_compute_primary_1000_minus_repeat_200", "endpoint/repeat_200", "endpoint/data_1000"),
    ]
    comparisons.extend((f"selected_{name}_minus_baseline", "baseline", f"selected/{name}") for name in SELECTED_ORDER)
    bootstraps = {}
    for name, left_role, right_role in comparisons:
        left_id, right_id = role_aliases[left_role], role_aliases[right_role]
        bootstraps[name] = {
            "left_role": left_role, "right_role": right_role,
            **paired_family_bootstrap(suite, rows_by_checkpoint[left_id], rows_by_checkpoint[right_id], draws=draws, seed=seed),
        }

    repeat_id, primary_id = role_aliases["endpoint/repeat_200"], role_aliases["endpoint/data_1000"]
    fixed_compute = {
        "comparison": "endpoint/data_1000 minus endpoint/repeat_200",
        "interpretation": selection.get("compute_control"),
        "metric_differences": _metric_delta(evaluations[repeat_id]["headline"], evaluations[primary_id]["headline"]),
        "paired_family_bootstrap": bootstraps["fixed_compute_primary_1000_minus_repeat_200"],
        "limitation": "The 5000-family endpoint has no matched 5000-update repeat-200 control.",
    }

    baseline_entry = evaluation["tests"][role_aliases["baseline"]]
    retention = {
        role: _retention_delta(evaluation["tests"][identity], baseline_entry)
        for role, identity in role_aliases.items()
    }
    shared_seconds = _history_step(histories, "primary", 200)
    repeat_extension = training["repeat_200"].get("training_seconds")
    training_summary = {
        "exposure_by_role": {role: _exposure(role, point) for role, point in roles},
        "timing_seconds": {
            "primary_5000": training["primary"].get("training_seconds"),
            "shared_primary_200_prefix": shared_seconds,
            "repeat_200_extension_800_updates": repeat_extension,
            "repeat_total_including_shared_prefix": shared_seconds + repeat_extension
            if shared_seconds is not None and repeat_extension is not None else None,
        },
        "data": _compact_manifest(manifest),
        "token_exposure": _validate_exposure(exposure, training),
    }
    limitations = [
        "One training seed; family bootstrap measures held-out family variation, not training-run variance.",
        "Evaluation families are model-authored and reviewed, not human-certified ground truth.",
        "The 5000-family endpoint combines more unique data with more optimizer updates; only the 1000-update control is compute matched.",
        "Selected-checkpoint tables supplement complete-pass endpoint results and do not replace them.",
    ]
    if training_summary["token_exposure"].get("status") == "not_recorded":
        limitations.append("Aggregate training prompt-token exposure was not recorded by the runner; judgment exposure is exact.")
    result = {
        "contract": "contrast-scaling-report-v1", "status": "complete",
        "selection_locked_utc": selection["locked_utc"], "evaluation_finished_utc": evaluation.get("finished_utc"),
        "role_aliases": role_aliases, "evaluations": evaluations,
        "tables": {"endpoints": endpoint_rows, "selected": selected_rows},
        "fixed_compute": fixed_compute, "paired_family_bootstrap": bootstraps,
        "retention": retention, "training": training_summary,
        "provenance": provenance or {},
        "limitations": limitations,
    }
    if jev_reference is not None:
        result["jev_reference"] = _jev_section(suite, jev_reference, rows_by_checkpoint)
        result["limitations"].append("Jev is an external reference observation, not a source of gold labels or training targets.")
    return result


def _pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def _num(value):
    return "—" if value is None else f"{value:.4f}"


def _table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ])


def render_markdown(report):
    def role_table(rows):
        return _table(
            ["Role", "Checkpoint", "Step", "Unique families", "Accuracy", "NLL", "Brier", "Unknown P/R", "Wrong ≥.90/.95", "Score MAE"],
            [[row["role"], row["checkpoint_id"], row["step"], row["unique_new_families"],
              _pct(row["metrics"]["accuracy"]), _num(row["metrics"]["nll"]), _num(row["metrics"]["brier"]),
              f"{_pct(row['metrics']['unknown_precision'])} / {_pct(row['metrics']['unknown_recall'])}",
              f"{row['metrics']['confident_wrong_90']} / {row['metrics']['confident_wrong_95']}",
              _num(row["metrics"]["score_mae"])] for row in rows],
        )

    fixed = report["fixed_compute"]
    fixed_boot = fixed["paired_family_bootstrap"]
    endpoint_ids = [report["role_aliases"][row["role"]] for row in report["tables"]["endpoints"]]
    categories = sorted(report["evaluations"][endpoint_ids[0]]["matrix"]["by_category"])
    category_rows = []
    for category in categories:
        category_rows.append([category, *[
            _pct(report["evaluations"][identity]["matrix"]["by_category"][category]["overall"]["accuracy"])
            for identity in endpoint_ids
        ]])
    axis_rows = []
    for identity in dict.fromkeys(endpoint_ids):
        for axis, value in report["evaluations"][identity]["relation_contrast_axis"]["overall"].items():
            axis_rows.append([identity, axis, value["pairs"], _pct(value["both_correct_rate"]), _num(value["mean_total_variation"])])
    primitive_rows = []
    for row in report["tables"]["endpoints"]:
        matrix = report["evaluations"][row["checkpoint_id"]]["matrix"]
        for primitive in ("noul", "choice", "score"):
            value = matrix["by_primitive"][primitive]
            primitive_rows.append([row["role"], primitive, value["questions"], _pct(value["accuracy"]),
                                   _num(value["nll"]), _num(value["brier"]),
                                   _num(value.get("score_mean_absolute_error"))])
    retention_rows = []
    for row in report["tables"]["endpoints"] + report["tables"]["selected"]:
        retained = report["retention"][row["role"]]
        retention_rows.append([
            row["role"], _num(retained["broad"]["accuracy_delta"]), _num(retained["broad"]["nll_delta"]),
            _num(retained["semantic114"]["accuracy_delta"]),
            _num(retained["prior_paired_clarified"]["accuracy_delta"]),
        ])
    exposure_rows = []
    for row in report["tables"]["endpoints"] + report["tables"]["selected"]:
        exposure = report["training"]["exposure_by_role"][row["role"]]
        exposure_rows.append([row["role"], exposure["unique_new_families"], exposure["family_presentations"],
                              exposure["new_judgments_presented"], exposure["replay_judgments_presented"],
                              _num(exposure["epochs_over_unique_families"])])
    bootstrap_rows = []
    for name, comparison in report["paired_family_bootstrap"].items():
        for metric in ("accuracy", "nll", "brier", "score_mae", "unknown_recall", "relation_both_correct"):
            value = comparison[metric]
            bootstrap_rows.append([name, metric, _num(value["difference"]), value["interval_95"]])
    diversity = report["training"]["data"].get("diversity", {})
    diversity_text = (
        f"Frozen data diversity: {diversity.get('normalized_semantic_fact_signatures', '—')} normalized fact-only signatures, "
        f"{diversity.get('semantic_program_signatures', '—')} program-only signatures, and "
        f"{diversity.get('normalized_joint_fact_program_signatures', '—')} joint signatures."
    )
    jev_sections = []
    if "jev_reference" in report:
        jev = report["jev_reference"]
        headline = jev["headline"]
        overlap_rows = []
        for identity, value in jev["error_overlap_by_checkpoint"].items():
            overlap_rows.append([
                identity, value["both_correct"], value["local_only_correct"], value["reference_only_correct"],
                value["both_wrong"], value["prediction_disagreements"],
                _num(value["local_probability_nll_floor_1e12"]),
                _num(value["reference_probability_nll_floor_1e12"]),
            ])
        jev_sections = [
            "## Jev external reference",
            f"Jev is reported as an external reference observation, not gold. Accuracy {_pct(headline['accuracy'])}; "
            f"shared-floor probability NLL {_num(headline['probability_nll_floor_1e12'])}; Brier {_num(headline['brier'])}; "
            f"Unknown precision/recall {_pct(headline['unknown_precision'])} / {_pct(headline['unknown_recall'])}; "
            f"Score MAE {_num(headline['score_mae'])}.",
            _table(["Local checkpoint", "Both correct", "Local only", "Jev only", "Both wrong", "Prediction disagreements",
                    "Local probability NLL", "Jev probability NLL"], overlap_rows),
            "Cross-system NLL in this table is recomputed from both complete probability vectors with the same 1e-12 floor; local raw-logit NLL is not used for this comparison.",
        ]
    sections = [
        "# Contrast-scaling v1 results",
        f"Checkpoint selection locked at `{report['selection_locked_utc']}`. Complete-pass endpoints remain the headline scaling comparison; validation-selected checkpoints are reported separately.",
        "## Endpoint checkpoints", role_table(report["tables"]["endpoints"]),
        "## Selected checkpoints", role_table(report["tables"]["selected"]),
        "## Fixed-compute contrast",
        f"Primary 1,000 unique families minus repeat-200 at 1,000 total updates: accuracy {_num(fixed['metric_differences']['accuracy'])}, NLL {_num(fixed['metric_differences']['nll'])}, Brier {_num(fixed['metric_differences']['brier'])}. "
        f"The paired family-bootstrap accuracy interval is `{fixed_boot['accuracy']['interval_95']}`. {fixed['limitation']}",
        "## Accuracy by category",
        _table(["Category", *[row["role"] for row in report["tables"]["endpoints"]]], category_rows),
        "## Primitive metrics",
        _table(["Role", "Primitive", "Questions", "Accuracy", "NLL", "Brier", "Score MAE"], primitive_rows),
        "## Relations by contrast axis",
        _table(["Checkpoint", "Kind/axis", "Pairs", "Both correct", "Mean probability distance"], axis_rows),
        *jev_sections,
        "## Category-stratified paired family uncertainty",
        _table(["Comparison", "Metric", "Right − left", "Exploratory 95% interval"], bootstrap_rows),
        "Bootstrap draws resample complete held-out families within category. The experiment has one training seed, so these intervals do not estimate training-run variance.",
        "## Exposure, diversity, and timing",
        _table(["Role", "Unique families", "Family presentations", "New judgments", "Replay judgments", "Family epochs"], exposure_rows),
        f"{diversity_text} Primary training recorded {_num(report['training']['timing_seconds']['primary_5000'])} seconds. The repeat extension recorded {_num(report['training']['timing_seconds']['repeat_200_extension_800_updates'])} seconds; including the shared 200-update prefix gives {_num(report['training']['timing_seconds']['repeat_total_including_shared_prefix'])} seconds when the prefix timing is available. Token exposure status: `{report['training']['token_exposure']['status']}`. Full prefix and diversity metadata are preserved in `report.json`.",
        "## Retention",
        _table(["Role", "Broad accuracy Δ", "Broad NLL Δ", "Semantic accuracy Δ", "Prior paired accuracy Δ"], retention_rows),
        "## Limitations", "\n".join(f"- {value}" for value in report["limitations"]),
    ]
    return "\n\n".join(sections) + "\n"


def write_outputs(report, output):
    output = Path(output)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    markdown = render_markdown(report)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(payload)
    (output / "RESULTS.md").write_text(markdown)


def _read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_jev_reference(directory, suite_path):
    directory, suite_path = Path(directory), Path(suite_path)
    paths = {
        "report": directory / "report.json",
        "predictions": directory / "predictions.json",
        "requests": directory / "requests.jsonl",
    }
    requests = _read_jsonl(paths["requests"])
    return {
        "reference": {
            "expected_suite_sha256": _sha(suite_path),
            "request_ids": [row.get("case_id") for row in requests],
            "rows": _read(paths["predictions"]), "report": _read(paths["report"]),
        },
        "provenance": {name: {"path": str(path), "sha256": _sha(path)} for name, path in paths.items()},
    }


def build_from_paths(evaluation_dir, training_dir, data_dir, *, draws=2000, seed=42,
                     jev_reference_dir=None, exposure_path=None):
    evaluation_dir, training_dir, data_dir = map(Path, (evaluation_dir, training_dir, data_dir))
    paths = {
        "selection": evaluation_dir / "selection-locked.json",
        "evaluation": evaluation_dir / "report.json",
        "training": training_dir / "report.json",
        "manifest": data_dir / "manifest.json",
        "test_suite": data_dir / "test" / "suite.json",
    }
    # Read and validate the lock/status artifacts before touching prediction outcomes.
    selection = _read(paths["selection"])
    evaluation = _read(paths["evaluation"])
    training = _read(paths["training"])
    manifest = _read(paths["manifest"])
    _validate_metadata(selection, evaluation, training, manifest)
    suite = _read(paths["test_suite"])
    rows, prediction_paths = {}, {}
    for job in unique_test_checkpoints(selection):
        identity = job["checkpoint_id"]
        try:
            prediction_path = Path(evaluation["tests"][identity]["predictions_root"]) / "new" / "predictions.json"
        except KeyError as error:
            raise ValueError("complete deduplicated test outputs are required") from error
        rows[identity] = _read(prediction_path)
        prediction_paths[identity] = prediction_path
    histories = {
        "primary": _read_jsonl(training_dir / "primary" / "history.jsonl"),
        "repeat_200": _read_jsonl(training_dir / "repeat-200" / "history.jsonl"),
    }
    provenance = {name: {"path": str(path), "sha256": _sha(path)} for name, path in paths.items()}
    provenance["reporting_source_sha256"] = _sha(__file__)
    provenance["predictions"] = {
        identity: {"path": str(path), "sha256": _sha(path)} for identity, path in prediction_paths.items()
    }
    exposure = _read(exposure_path) if exposure_path is not None else None
    if exposure_path is not None:
        provenance["exposure"] = {"path": str(exposure_path), "sha256": _sha(exposure_path)}
    # Build and verify every local matrix before opening optional external outcomes.
    result = build_report(selection, evaluation, training, manifest, suite, rows, histories=histories,
                          draws=draws, seed=seed, provenance=provenance, exposure=exposure)
    if jev_reference_dir is not None:
        jev = _load_jev_reference(jev_reference_dir, paths["test_suite"])
        result["jev_reference"] = _jev_section(suite, jev["reference"], rows)
        result["limitations"].append("Jev is an external reference observation, not a source of gold labels or training targets.")
        result["provenance"]["jev_reference"] = jev["provenance"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jev-reference", type=Path)
    parser.add_argument("--exposure", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    report = build_from_paths(args.evaluation, args.training, args.data,
                              draws=args.bootstrap_draws, seed=args.seed,
                              jev_reference_dir=args.jev_reference, exposure_path=args.exposure)
    write_outputs(report, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))
    return report


if __name__ == "__main__":
    main()
