"""Protect split isolation, decision semantics, and numerical calibration behavior."""

import math
from dataclasses import replace

import numpy as np
import pytest


def samples():
    from experiments.contrast_calibration_probe import Sample

    return [
        Sample(f"{family}/{i}", "q", family, generator, z, y)
        for family, generator in [("a", "execution"), ("b", "execution"), ("c", "destination")]
        for i, (z, y) in enumerate([(-3.0, False), (-1.0, True), (1.0, False), (3.0, True)])
    ]


@pytest.mark.parametrize("scheme,expected_folds", [("family", 3), ("generator", 2)])
def test_folds_exclude_all_heldout_families_and_predict_each_sample_once(scheme, expected_folds):
    from experiments.contrast_calibration_probe import make_folds

    rows = samples()
    folds = make_folds(rows, scheme)
    assert len(folds) == expected_folds
    assert sorted(i for f in folds for i in f.test_indices) == list(range(12))
    for fold in folds:
        assert not set(fold.train_indices) & set(fold.test_indices)
        assert not {rows[i].family_id for i in fold.train_indices} & {
            rows[i].family_id for i in fold.test_indices
        }


def test_changing_heldout_targets_cannot_change_fitted_coefficients():
    from experiments.contrast_calibration_probe import evaluate_fold, make_folds

    rows = samples()
    fold = make_folds(rows, "family")[0]
    before, _ = evaluate_fold(rows, fold, "affine")
    changed = [replace(row, target=not row.target) if i in fold.test_indices else row for i, row in enumerate(rows)]
    after, _ = evaluate_fold(changed, fold, "affine")
    assert (before["a"], before["b"]) == (after["a"], after["b"])


def test_affine_map_preserves_logit_order_and_rejects_nonpositive_scales():
    from experiments.contrast_calibration_probe import transform_logits

    assert np.all(np.diff(transform_logits([-1000, -1, 0, 1, 1000], 0.001, -20)) > 0)
    for a in [0, -1]:
        with pytest.raises(ValueError):
            transform_logits([0, 1], a, 0)


def test_temperature_never_changes_binary_decisions_at_one_half():
    from experiments.contrast_calibration_probe import binary_decisions, transform_logits

    logits = [-1000.0, -0.01, 0.0, 0.01, 1000.0]
    for scale in [0.001, 0.1, 1.0, 20.0]:
        assert binary_decisions(transform_logits(logits, scale, 0)).tolist() == [False, False, True, True, True]


def test_saturated_logits_have_finite_correct_loss_and_probability():
    from experiments.contrast_calibration_probe import sigmoid, stable_bce

    assert stable_bce([1000.0, -1000.0, 0.0], [False, True, True]).tolist() == pytest.approx(
        [1000.0, 1000.0, math.log(2)]
    )
    assert sigmoid([1000.0, -1000.0, 0.0]).tolist() == [1.0, 0.0, 0.5]


def test_case_weights_do_not_favor_cases_with_more_binary_fields():
    from experiments.contrast_calibration_probe import Sample, case_weights

    rows = [Sample("six", str(i), "a", "execution", 1.0, False) for i in range(6)]
    rows += [Sample("five", str(i), "b", "destination", 1.0, False) for i in range(5)]
    weights = case_weights(rows)
    assert weights[:6].sum() == pytest.approx(0.5)
    assert weights[6:].sum() == pytest.approx(0.5)


def test_fit_improves_training_objective_without_changing_rank():
    from experiments.contrast_calibration_probe import fit_calibrator

    result = fit_calibrator(samples(), "affine")
    assert 0 < result["a"] < 1
    assert result["objective"] < result["identity_objective"]
    assert result["projected_gradient_inf"] <= 1e-8
    assert result["converged"]


def test_binary_metrics_report_two_class_brier_and_high_confidence_errors():
    from experiments.contrast_calibration_probe import binary_metrics

    result = binary_metrics([1000.0, -1000.0, 0.0], [False, True, True])
    assert result["high_probability_errors"] == 2
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["balanced_binary_accuracy"] == pytest.approx(0.25)
    assert result["brier"] == pytest.approx(1.5)
    assert result["nll"] == pytest.approx((2000 + math.log(2)) / 3)


def test_relation_both_correct_is_not_substituted_with_correct_order():
    from experiments.contrast_calibration_probe import relation_results

    relation = {
        "id": "pair", "kind": "flip",
        "left": {"case_id": "a", "question_id": "q"},
        "right": {"case_id": "b", "question_id": "q"},
    }
    predictions = [
        {"case_id": "a", "question_id": "q", "family_id": "f", "fold_id": "f", "target": False, "logit": -2.0},
        {"case_id": "b", "question_id": "q", "family_id": "f", "fold_id": "f", "target": True, "logit": -1.0},
    ]
    result = relation_results([relation], predictions)
    assert result[0]["direction_correct"] is True
    assert result[0]["both_correct"] is False
    predictions[1]["fold_id"] = "different"
    with pytest.raises(ValueError, match="same held-out"):
        relation_results([relation], predictions)


def test_generator_folds_never_split_families():
    from experiments.contrast_calibration_probe import make_folds

    rows = samples()
    rows[0] = replace(rows[0], generator="destination")
    with pytest.raises(ValueError, match="family"):
        make_folds(rows, "generator")
