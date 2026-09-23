import copy
import json
import math

import pytest


def _suite():
    cases, relations = [], []
    for category in ("claim_vs_completion", "entity_binding"):
        for index in range(2):
            family = f"{category}/family-{index}"
            case_id = f"{family}/v0"
            expected = {"n1": True, "n2": False, "n3": True, "c1": "unknown" if index == 0 else "done", "s1": 1}
            cases.append({
                "id": case_id, "family_id": family, "domain": category.split("_")[0], "category": category,
                "variant": "v0", "layout": "structured", "expected": expected,
                "predicate_tags": {"n1": "claim", "n2": "completion", "n3": "claim", "c1": "outcome", "s1": "rubric"},
                "request": {"state": {}, "questions": {
                    "n1": {"type": "noul"}, "n2": {"type": "noul"}, "n3": {"type": "noul"},
                    "c1": {"type": "choice", "criteria": {"unknown": "Unknown", "done": "Done"}},
                    "s1": {"type": "score", "criteria": ["low", "middle", "high"]},
                }},
            })
            relations.extend([
                {"id": f"{family}/flip", "kind": "flip", "contrast_axis": "evidence",
                 "left": {"case_id": case_id, "question_id": "n1"},
                 "right": {"case_id": case_id, "question_id": "n2"}},
                {"id": f"{family}/question", "kind": "question_contrast", "contrast_axis": "question",
                 "left": {"case_id": case_id, "question_id": "n1"},
                 "right": {"case_id": case_id, "question_id": "n2"}},
                {"id": f"{family}/invariant", "kind": "invariant", "contrast_axis": "rubric",
                 "left": {"case_id": case_id, "question_id": "n1"},
                 "right": {"case_id": case_id, "question_id": "n3"}},
            ])
    return {"cases": cases, "relations": relations}


def _prediction(primitive, expected, correct, *, false_unknown=False):
    if primitive == "noul":
        other = "false" if expected == "true" else "true"
        probabilities = {expected: 0.8 if correct else 0.1, other: 0.2 if correct else 0.9}
    elif primitive == "choice":
        other = "unknown" if false_unknown else ("done" if expected == "unknown" else "unknown")
        probabilities = {expected: 0.8 if correct else 0.1, other: 0.2 if correct else 0.9}
    else:
        other = "0" if expected != "0" else "2"
        probabilities = {"0": 0.1, "1": 0.8, "2": 0.1} if correct else {other: 0.8, expected: 0.1, "2": 0.1}
    maximum = max(probabilities.values())
    predicted = next(key for key, value in probabilities.items() if value == maximum)
    result = {
        "primitive": primitive, "expected": expected, "predicted": predicted, "probabilities": probabilities,
        "correct": predicted == expected, "max_probability": maximum, "nll": -math.log(probabilities[expected]),
        "brier": sum((value - float(key == expected)) ** 2 for key, value in probabilities.items()),
    }
    if primitive == "score":
        result.update(score_absolute_error=abs(int(predicted) - int(expected)),
                      score_normalized_absolute_error=abs(int(predicted) - int(expected)) / 2)
    return result


def _rows(suite, correct_families, *, false_unknown_family=None):
    rows = []
    for case in suite["cases"]:
        correct = case["family_id"] in correct_families
        for qid, target in case["expected"].items():
            primitive = "noul" if qid.startswith("n") else "choice" if qid == "c1" else "score"
            expected = ("true" if target else "false") if type(target) is bool else str(target)
            rows.append({
                "case_id": case["id"], "question_id": qid, "family_id": case["family_id"],
                "domain": case["domain"], "variant": case["variant"], "layout": case["layout"],
                "prediction": _prediction(primitive, expected, correct,
                                          false_unknown=case["family_id"] == false_unknown_family and expected != "unknown"),
            })
    return rows


def _point(checkpoint_id, stream, step):
    return {"checkpoint_id": checkpoint_id, "checkpoint": f"/checkpoints/{checkpoint_id}", "stream": stream, "step": step,
            "status": "complete", "objective": 1 / (step + 1)}


def _selection():
    p0 = _point("ck0", "primary", 0)
    p200 = _point("ck200", "primary", 200)
    p1000 = _point("ck1000", "primary", 1000)
    p5000 = _point("ck5000", "primary", 5000)
    repeat = _point("ck-repeat", "repeat-200", 1000)
    return {
        "locked_utc": "2026-09-21T12:00:00+00:00", "baseline": p0,
        "endpoints": {"data_200": p200, "data_1000": p1000, "data_5000": p5000, "repeat_200": repeat},
        "selected": {"data_200": p0, "data_1000": p200, "data_5000": p5000, "repeat_200": repeat},
        "primary_interpretation": "complete-pass endpoints", "compute_control": "equal 1000 updates",
    }


def _result_summary(accuracy, nll, score_mae=0.25):
    return {"summary": {"overall": {"questions": 20, "correct": round(20 * accuracy), "accuracy": accuracy,
                                     "nll": nll, "brier": nll / 2},
                        "by_domain": {"source": {"accuracy": accuracy, "nll": nll,
                                                   "score_mean_absolute_error": score_mae}}},
            "max_unit_tokens": 100}


def _inputs():
    from experiments.contrast_scaling_evaluation import unique_test_checkpoints

    suite, selection = _suite(), _selection()
    family_ids = [case["family_id"] for case in suite["cases"]]
    qualities = {
        "ck0": set(family_ids[:2]), "ck200": set(family_ids[:3]), "ck1000": set(family_ids),
        "ck5000": set(family_ids), "ck-repeat": set(family_ids[::2]),
    }
    rows = {identity: _rows(suite, correct) for identity, correct in qualities.items()}
    tests = {}
    for job in unique_test_checkpoints(selection):
        accuracy = sum(row["prediction"]["correct"] for row in rows[job["checkpoint_id"]]) / len(rows[job["checkpoint_id"]])
        predictions = [row["prediction"] for row in rows[job["checkpoint_id"]]]
        new_result = {"summary": {"overall": {
            "questions": len(predictions), "correct": sum(value["correct"] for value in predictions),
            "accuracy": accuracy, "nll": sum(value["nll"] for value in predictions) / len(predictions),
            "brier": sum(value["brier"] for value in predictions) / len(predictions),
        }}, "max_unit_tokens": 100}
        tests[job["checkpoint_id"]] = {
            **job, "predictions_root": f"/predictions/{job['checkpoint_id']}",
            "results": {"new": new_result, **{name: _result_summary(accuracy, 1 - accuracy + 0.1) for name in
                        ("broad", "semantic114", "prior_paired_clarified")}},
        }
    evaluation = {"status": "complete", "selection": copy.deepcopy(selection), "tests": tests,
                  "finished_utc": "2026-09-21T13:00:00+00:00"}
    training = {"status": "complete", "families": 5000,
                "primary": {"status": "complete", "completed_updates": 5000, "history_updates": 5000, "training_seconds": 100.0},
                "repeat_200": {"status": "complete", "completed_updates": 1000, "history_updates": 800, "training_seconds": 20.0}}
    manifest = {"study": "contrast-scaling-v1", "status": "frozen", "frozen_utc": "2026-09-21T09:00:00+00:00",
                "prefixes": {str(size): {"families": size, "cases": size * 4, "judgments": size * 20} for size in (200, 1000, 5000)},
                "diversity": {"normalized_semantic_fact_signatures": 3000,
                              "semantic_program_signatures": 4636,
                              "normalized_joint_fact_program_signatures": 5000},
                "source_bank_sha256": {"sol": "a", "terra": "b"}, "data_sha256": {"train": "c"}}
    histories = {"primary": [{"step": 200, "training_seconds": 4.0}, {"step": 5000, "training_seconds": 100.0}],
                 "repeat_200": [{"step": 1000, "training_seconds": 20.0}]}
    return suite, selection, evaluation, training, manifest, rows, histories


def test_report_deduplicates_checkpoint_aliases_and_builds_endpoint_selected_tables():
    from experiments.contrast_scaling_reporting import build_report

    suite, selection, evaluation, training, manifest, rows, histories = _inputs()
    report = build_report(selection, evaluation, training, manifest, suite, rows, histories=histories, draws=100, seed=7)
    assert report["status"] == "complete"
    assert len(report["evaluations"]) == 5
    assert report["role_aliases"]["endpoint/data_200"] == report["role_aliases"]["selected/data_1000"] == "ck200"
    assert report["role_aliases"]["baseline"] == report["role_aliases"]["selected/data_200"] == "ck0"
    assert [row["role"] for row in report["tables"]["endpoints"]] == [
        "baseline", "endpoint/data_200", "endpoint/data_1000", "endpoint/data_5000", "endpoint/repeat_200",
    ]
    assert len(report["tables"]["selected"]) == 4
    assert report["fixed_compute"]["comparison"] == "endpoint/data_1000 minus endpoint/repeat_200"
    assert report["training"]["timing_seconds"]["repeat_total_including_shared_prefix"] == 24.0
    assert report["training"]["token_exposure"]["status"] == "not_recorded"


def test_report_contains_category_primitive_relation_axis_and_probability_metrics():
    from experiments.contrast_scaling_reporting import build_report

    suite, selection, evaluation, training, manifest, rows, histories = _inputs()
    report = build_report(selection, evaluation, training, manifest, suite, rows, histories=histories, draws=50)
    matrix = report["evaluations"]["ck200"]["matrix"]
    assert set(matrix["by_category"]) == {"claim_vs_completion", "entity_binding"}
    assert set(matrix["by_primitive"]) == {"noul", "choice", "score"}
    assert matrix["overall"]["nll"] > 0 and matrix["overall"]["brier"] > 0
    assert matrix["overall"]["confident_wrong_90"] > 0
    assert matrix["unknown"]["precision"] is not None and matrix["unknown"]["recall"] is not None
    assert matrix["overall"]["score_mean_absolute_error"] >= 0
    axes = report["evaluations"]["ck200"]["relation_contrast_axis"]
    assert set(axes["overall"]) == {"flip/evidence", "question_contrast/question", "invariant/rubric"}
    assert set(axes["by_category"]) == {"claim_vs_completion", "entity_binding"}
    assert report["retention"]["endpoint/data_200"]["broad"]["accuracy_delta"] > 0


def test_family_bootstrap_resamples_paired_whole_families_within_category():
    from experiments.contrast_scaling_reporting import paired_family_bootstrap

    suite = _suite()
    family_ids = [case["family_id"] for case in suite["cases"]]
    left = _rows(suite, set())
    right = _rows(suite, set(family_ids[:2]))
    result = paired_family_bootstrap(suite, left, right, draws=500, seed=11)
    assert result["resampling_unit"] == "complete_family"
    assert result["stratification"] == "category"
    assert result["direction"] == "right minus left"
    assert result["accuracy"]["difference"] == pytest.approx(0.5)
    assert result["accuracy"]["interval_95"] == pytest.approx([0.5, 0.5])
    assert result["nll"]["difference"] < 0


def test_reporting_rejects_unlocked_incomplete_or_partial_inputs():
    from experiments.contrast_scaling_reporting import build_report

    suite, selection, evaluation, training, manifest, rows, histories = _inputs()
    unlocked = copy.deepcopy(selection)
    unlocked.pop("locked_utc")
    with pytest.raises(ValueError, match="locked"):
        build_report(unlocked, evaluation, training, manifest, suite, rows, histories=histories, draws=10)
    incomplete = copy.deepcopy(evaluation)
    incomplete["status"] = "running"
    with pytest.raises(ValueError, match="complete"):
        build_report(selection, incomplete, training, manifest, suite, rows, histories=histories, draws=10)
    partial = dict(rows)
    partial["ck200"] = partial["ck200"][:-1]
    with pytest.raises(ValueError, match="coverage"):
        build_report(selection, evaluation, training, manifest, suite, partial, histories=histories, draws=10)
    mismatch = copy.deepcopy(evaluation)
    mismatch["tests"]["ck200"]["results"]["new"]["summary"]["overall"]["accuracy"] = 0.123
    with pytest.raises(ValueError, match="recorded summary"):
        build_report(selection, mismatch, training, manifest, suite, rows, histories=histories, draws=10)


def test_writes_standalone_json_and_markdown(tmp_path):
    from experiments.contrast_scaling_reporting import build_report, write_outputs

    suite, selection, evaluation, training, manifest, rows, histories = _inputs()
    report = build_report(selection, evaluation, training, manifest, suite, rows, histories=histories, draws=20)
    output = tmp_path / "report"
    write_outputs(report, output)
    saved = json.loads((output / "report.json").read_text())
    markdown = (output / "RESULTS.md").read_text()
    assert saved["contract"] == "contrast-scaling-report-v1"
    assert "Fixed-compute contrast" in markdown
    assert "Endpoint checkpoints" in markdown and "Selected checkpoints" in markdown
    assert "Primitive metrics" in markdown and "Broad accuracy Δ" in markdown
    assert "one training seed" in markdown.lower()


def test_cli_fails_before_creating_output_when_selection_is_not_locked(tmp_path):
    from experiments.contrast_scaling_reporting import main

    evaluation, training, data = tmp_path / "evaluation", tmp_path / "training", tmp_path / "data"
    evaluation.mkdir(); training.mkdir(); data.mkdir()
    (evaluation / "report.json").write_text(json.dumps({"status": "running"}))
    output = tmp_path / "output"
    with pytest.raises((FileNotFoundError, ValueError)):
        main(["--evaluation", str(evaluation), "--training", str(training), "--data", str(data), "--output", str(output)])
    assert not output.exists()


def _jev_reference(suite, rows):
    return {
        "expected_suite_sha256": "frozen-suite-sha",
        "request_ids": [case["id"] for case in suite["cases"]],
        "rows": copy.deepcopy(rows),
        "report": {
            "status": "complete", "requests_planned": len(suite["cases"]),
            "requests_completed": len(suite["cases"]),
            "questions_planned": sum(len(case["expected"]) for case in suite["cases"]),
            "metadata": {"suite_sha256": "frozen-suite-sha"},
            "returned_models": ["Jev synthetic"],
        },
    }


def test_optional_jev_reference_validates_hash_ids_distributions_and_reports_overlap():
    from experiments.contrast_scaling_reporting import build_report, render_markdown

    suite, selection, evaluation, training, manifest, rows, histories = _inputs()
    jev = _jev_reference(suite, rows["ck1000"])
    report = build_report(selection, evaluation, training, manifest, suite, rows, histories=histories,
                          draws=20, jev_reference=jev)
    reference = report["jev_reference"]
    assert reference["designation"] == "external_reference_observation_not_gold"
    assert set(reference["matrix"]["by_category"]) == {"claim_vs_completion", "entity_binding"}
    assert set(reference["matrix"]["by_primitive"]) == {"noul", "choice", "score"}
    assert set(reference["relation_contrast_axis"]["overall"]) == {
        "flip/evidence", "question_contrast/question", "invariant/rubric",
    }
    assert reference["error_overlap_by_checkpoint"]["ck1000"]["both_correct"] == 20
    assert "not gold" in render_markdown(report)

    bad_hash = copy.deepcopy(jev)
    bad_hash["report"]["metadata"]["suite_sha256"] = "wrong"
    with pytest.raises(ValueError, match="suite hash"):
        build_report(selection, evaluation, training, manifest, suite, rows, histories=histories,
                     draws=10, jev_reference=bad_hash)
    bad_ids = copy.deepcopy(jev)
    bad_ids["request_ids"] = bad_ids["request_ids"][:-1]
    with pytest.raises(ValueError, match="request IDs"):
        build_report(selection, evaluation, training, manifest, suite, rows, histories=histories,
                     draws=10, jev_reference=bad_ids)
    bad_distribution = copy.deepcopy(jev)
    bad_distribution["rows"][0]["prediction"]["probabilities"].pop("false")
    with pytest.raises(ValueError, match="distribution"):
        build_report(selection, evaluation, training, manifest, suite, rows, histories=histories,
                     draws=10, jev_reference=bad_distribution)
    bad_nll = copy.deepcopy(jev)
    bad_nll["rows"][0]["prediction"]["nll"] = 999
    with pytest.raises(ValueError, match="distribution"):
        build_report(selection, evaluation, training, manifest, suite, rows, histories=histories,
                     draws=10, jev_reference=bad_nll)


def _exposure(training):
    fingerprint = training["fingerprint"]
    fields = ("prompt_tokens", "candidate_units", "padded_forward_tokens", "forward_calls", "judgments")

    def block(value):
        return {origin: {name: value for name in fields} for origin in ("new", "old")}

    result = {
        "status": "complete", "study": "contrast-scaling-v1", "model_weights_loaded": False,
        "train_sha256": fingerprint["train_sha256"], "manifest_sha256": fingerprint["manifest_sha256"],
        "replay_sha256": fingerprint["replay_sha256"],
        "prepared_prompt_fingerprint": fingerprint["prepared_prompt_fingerprint"],
        "max_units": fingerprint["max_units"], "unit_batch_size": fingerprint["unit_batch_size"],
        "tokenizer": {"model_id": "synthetic", "revision": "fixed"},
        "new_candidate_units": 1, "new_token_range": [1, 2],
        "definitions": {"prompt_tokens": "unpadded", "padded_forward_tokens": "padded", "forward_calls": "calls"},
        "primary_by_step": {str(step): block(step) for step in (0, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000)},
        "repeat_by_step": {str(step): block(step) for step in (0, 200, 400, 600, 800, 1000)},
        "repeat_extension": block(800),
    }
    return result


def test_optional_exposure_is_separate_and_fingerprint_bound():
    from experiments.contrast_scaling_reporting import build_report

    suite, selection, evaluation, training, manifest, rows, histories = _inputs()
    training["fingerprint"] = {
        "train_sha256": "train", "manifest_sha256": "manifest", "replay_sha256": "replay",
        "prepared_prompt_fingerprint": "prompts", "max_units": 12, "unit_batch_size": 12,
    }
    exposure = _exposure(training)
    report = build_report(selection, evaluation, training, manifest, suite, rows, histories=histories,
                          draws=20, exposure=exposure)
    assert report["training"]["token_exposure"]["status"] == "complete"
    assert report["training"]["token_exposure"]["primary_by_step"]["1000"]["new"]["prompt_tokens"] == 1000
    assert "token_exposure" not in training

    incompatible = copy.deepcopy(exposure)
    incompatible["replay_sha256"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        build_report(selection, evaluation, training, manifest, suite, rows, histories=histories,
                     draws=10, exposure=incompatible)
    bad_extension = copy.deepcopy(exposure)
    bad_extension["repeat_extension"]["new"]["prompt_tokens"] += 1
    with pytest.raises(ValueError, match="extension"):
        build_report(selection, evaluation, training, manifest, suite, rows, histories=histories,
                     draws=10, exposure=bad_extension)
