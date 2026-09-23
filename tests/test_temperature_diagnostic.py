import copy
import math

import pytest

from experiments.revised_metrics import prediction_view
from experiments.temperature_diagnostic import (
    assign_folds,
    balanced_weights,
    calibrate_record,
    crossfit,
    fit_temperature,
)


def record(group, target, logit=math.log(4), origin="new", category="a", primitive="noul"):
    question = {"type": primitive, "criteria": {"true": "yes", "false": "no"}, "instructions": "Judge."}
    answer = {"noul": 1 / (1 + math.exp(-logit)), "details": {"raw_logit": logit}}
    if primitive == "score":
        target = int(target)
        question = {"type": "score", "criteria": ["low", "high"], "instructions": "Score."}
        answer = {
            "probabilities": {"0": 1 / (1 + math.exp(logit)), "1": 1 / (1 + math.exp(-logit))},
            "score": 1 / (1 + math.exp(-logit)),
            "details": {"raw_logits": {"0": 0.0, "1": logit}},
        }
    if primitive == "choice":
        target = "true" if target else "false"
        answer = {
            "probabilities": {"false": 1 / (1 + math.exp(logit)), "true": 1 / (1 + math.exp(-logit))},
            "details": {"raw_logits": {"false": 0.0, "true": logit}},
        }
    pred = prediction_view(question, target, answer)
    labels = ["0", "1"] if primitive == "score" else ["false", "true"]
    row = {
        "case_id": group + "/" + str(target),
        "question_id": "q",
        "family_id": group,
        "domain": category,
        "prediction": pred,
    }
    return {
        "origin": origin,
        "category": category,
        "group_id": group,
        "primitive": primitive,
        "logits": [0.0, logit],
        "labels": labels,
        "target_index": labels.index(pred["expected"]),
        "target": target,
        "question": question,
        "row": row,
    }


def test_binary_temperature_fit_matches_known_frequency_optimum():
    rows = [record(str(i), i < 3) for i in range(4)]
    result = fit_temperature(rows)
    assert result["temperature"] == pytest.approx(math.log(4) / math.log(3), rel=1e-6)
    assert result["objective"] == pytest.approx(-0.75 * math.log(0.75) - 0.25 * math.log(0.25), abs=1e-10)


def test_binary_log_odds_are_not_normalized_twice():
    row = calibrate_record(record("a", False), 1.0)
    assert row["prediction"]["probabilities"]["true"] == pytest.approx(0.8)
    assert row["prediction"]["nll"] == pytest.approx(math.log(5))


def test_temperature_preserves_score_mode_but_changes_expected_score():
    original = record("a", 1, primitive="score")
    scaled = calibrate_record(original, 2.0)["prediction"]
    assert scaled["predicted"] == original["row"]["prediction"]["predicted"] == "1"
    assert scaled["score_mean"] == pytest.approx(2 / 3)
    assert scaled["returned_score"] == pytest.approx(2 / 3)
    assert scaled["score_absolute_error"] == pytest.approx(1 / 3)


def test_family_folds_ignore_labels_and_keep_siblings_together():
    rows = [
        record(f"{category}/{family}", value, category=category)
        for category in ("a", "b")
        for family in range(4)
        for value in (True, False)
    ]
    folds = assign_folds(rows)
    assert len(folds) == 8
    for category in ("a", "b"):
        assert {folds["new", f"{category}/{i}"] for i in range(4)} == {0, 1, 2, 3}
    changed = copy.deepcopy(rows)
    for r in changed:
        r["target_index"] = 1 - r["target_index"]
    assert assign_folds(changed) == folds


def test_crossfit_heldout_labels_cannot_change_their_fitted_temperature():
    rows = [record(f"{origin}/{i}", i % 3 != 0, origin=origin) for origin in ("new", "broad") for i in range(8)]
    folds = assign_folds(rows)
    first = crossfit(rows, folds, variant="global")
    changed = copy.deepcopy(rows)
    for r in changed:
        if folds[r["origin"], r["group_id"]] == 0:
            r["target_index"] = 1 - r["target_index"]
            r["target"] = bool(r["target_index"])
            r["row"]["prediction"]["expected"] = "true" if r["target"] else "false"
            r["row"]["prediction"]["correct"] = (
                r["row"]["prediction"]["predicted"] == r["row"]["prediction"]["expected"]
            )
    second = crossfit(changed, folds, variant="global")
    assert first["fits"]["0"] == second["fits"]["0"]
    for before, after, r in zip(first["rows"], second["rows"], rows, strict=True):
        if folds[r["origin"], r["group_id"]] == 0:
            assert before["prediction"]["probabilities"] == after["prediction"]["probabilities"]


def test_nonpositive_or_nonfinite_temperatures_are_rejected():
    for value in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="temperature"):
            calibrate_record(record("a", True), value)


def test_exact_categorical_ties_remain_ties():
    r = record("tie", 1, logit=0.0, primitive="score")
    assert r["row"]["prediction"]["predicted"] is None
    assert calibrate_record(r, 7.0)["prediction"]["predicted"] is None


def test_large_common_logit_offsets_do_not_change_fitted_temperature():
    rows = [record(str(i), i < 3, logit=2.0) for i in range(4)]
    shifted = copy.deepcopy(rows)
    for row in shifted:
        row["logits"] = [1e16, 1e16 + 2.0]
    a, b = fit_temperature(rows), fit_temperature(shifted)
    assert b["temperature"] == pytest.approx(a["temperature"], rel=1e-8)
    assert b["objective"] == pytest.approx(a["objective"], abs=1e-12)


def test_macro_weights_balance_origins_and_primitives_not_row_counts():
    rows = []
    for origin, counts in [("new", [12, 4, 4]), ("broad", [1, 1, 1])]:
        for kind, count in zip(["noul", "choice", "score"], counts, strict=True):
            rows.extend({"origin": origin, "primitive": kind} for _ in range(count))
    weights = balanced_weights(rows)
    assert weights[0] == pytest.approx(1 / 72)
    assert weights[-1] == pytest.approx(1 / 6)
    for origin in ("new", "broad"):
        for kind in ("noul", "choice", "score"):
            assert sum(
                w for w, r in zip(weights, rows, strict=True) if (r["origin"], r["primitive"]) == (origin, kind)
            ) == pytest.approx(1 / 6)


def test_per_primitive_crossfit_recovers_known_different_scales():
    rows = []
    for origin in ("new", "broad"):
        for family in range(4):
            for kind, scale in [("noul", 1), ("choice", 2), ("score", 3)]:
                for index in range(4):
                    r = record(
                        f"{origin}/{family}", index < 3, logit=scale * math.log(4), origin=origin, primitive=kind
                    )
                    r["row"]["case_id"] += f"/{kind}/{index}"
                    rows.append(r)
    result = crossfit(rows, assign_folds(rows), variant="per_primitive")
    for fitted in result["fits"].values():
        for kind, scale in [("noul", 1), ("choice", 2), ("score", 3)]:
            assert fitted[kind]["temperature"] == pytest.approx(scale * math.log(4) / math.log(3), rel=1e-6)


def test_ragged_class_groups_do_not_add_fake_candidates_to_normalization():
    rows = [record(str(i), i < 3) for i in range(4)]
    rows.append({"logits": [0.0, 0.0, 0.0], "target_index": 0})
    fitted = fit_temperature(rows)
    assert fitted["temperature"] == pytest.approx(math.log(4) / math.log(3), rel=1e-6)
    assert fitted["objective"] == pytest.approx(
        0.8 * (-0.75 * math.log(0.75) - 0.25 * math.log(0.25)) + 0.2 * math.log(3), abs=1e-10
    )


def test_numerical_tie_threshold_crossing_fails_instead_of_changing_accuracy():
    with pytest.raises(ValueError, match="winner/tie"):
        calibrate_record(record("near-tie", True, logit=1e-11), 100.0)
