"""CPU-only retrospective calibration diagnostic; never changes deployed calibration.

Run: .venv/bin/python -m experiments.contrast_calibration_probe
Inputs are frozen case labels and previously saved Noul logits, never new model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCALE_BOUNDS = (0.001, 20.0)
OFFSET_BOUNDS = (-20.0, 20.0)
RIDGE = 1e-4
GRADIENT_TOLERANCE = 1e-8
MAX_CYCLES = 2000


@dataclass(frozen=True)
class Sample:
    case_id: str
    question_id: str
    family_id: str
    generator: str
    raw_logit: float
    target: bool


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def sigmoid(logits):
    z = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -z))


def stable_bce(logits, targets):
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    return np.logaddexp(0.0, (1.0 - 2.0 * y) * z)


def transform_logits(logits, a, b):
    if not np.isfinite(a) or a <= 0 or not np.isfinite(b):
        raise ValueError("calibration must have finite positive scale and finite offset")
    return a * np.asarray(logits, dtype=np.float64) + b


def binary_decisions(logits):
    # Exact zero is assigned true; using logits avoids probability rounding at 0.5.
    return np.asarray(logits, dtype=np.float64) >= 0.0


def case_weights(samples):
    counts = Counter(row.case_id for row in samples)
    if not counts:
        raise ValueError("empty fitting data")
    return np.asarray([1.0 / (len(counts) * counts[row.case_id]) for row in samples], dtype=np.float64)


def make_folds(samples, scheme):
    if scheme not in {"family", "generator"}:
        raise ValueError("unknown split scheme")
    family_generators = defaultdict(set)
    for row in samples:
        family_generators[row.family_id].add(row.generator)
    if any(len(values) != 1 for values in family_generators.values()):
        raise ValueError("each family must belong to exactly one generator")
    groups = [row.family_id if scheme == "family" else row.generator for row in samples]
    if len(set(groups)) < 2:
        raise ValueError("at least two groups are needed")
    return [
        Fold(group, tuple(i for i, value in enumerate(groups) if value != group),
             tuple(i for i, value in enumerate(groups) if value == group))
        for group in sorted(set(groups))
    ]


def fit_calibrator(samples, method):
    if method not in {"temperature", "affine"}:
        raise ValueError("unknown calibration method")
    z = np.asarray([row.raw_logit for row in samples], dtype=np.float64)
    y = np.asarray([row.target for row in samples], dtype=np.float64)
    weights = case_weights(samples)
    if not np.isfinite(z).all():
        raise ValueError("nonfinite raw logits")
    parameters = np.asarray([1.0, 0.0], dtype=np.float64)
    bounds = [SCALE_BOUNDS, OFFSET_BOUNDS]
    coordinates = range(2 if method == "affine" else 1)

    def objective(theta):
        loss = np.dot(weights, stable_bce(theta[0] * z + theta[1], y))
        return float(loss + 0.5 * RIDGE * ((theta[0] - 1) ** 2 + theta[1] ** 2))

    def gradient(theta):
        residual = weights * (sigmoid(theta[0] * z + theta[1]) - y)
        return np.asarray([np.dot(residual, z) + RIDGE * (theta[0] - 1),
                           residual.sum() + RIDGE * theta[1]])

    initial_objective = objective(parameters)
    # Each coordinate objective is strictly convex. Its monotone derivative is
    # solved by bracketed bisection, including an exact check for bound optima.
    for cycle in range(1, MAX_CYCLES + 1):
        for coordinate in coordinates:
            low, high = bounds[coordinate]
            trial = parameters.copy()
            trial[coordinate] = low
            low_gradient = gradient(trial)[coordinate]
            trial[coordinate] = high
            high_gradient = gradient(trial)[coordinate]
            if low_gradient >= 0:
                parameters[coordinate] = low
            elif high_gradient <= 0:
                parameters[coordinate] = high
            else:
                for _ in range(60):
                    middle = (low + high) / 2
                    trial[coordinate] = middle
                    if gradient(trial)[coordinate] > 0:
                        high = middle
                    else:
                        low = middle
                parameters[coordinate] = (low + high) / 2
        projected = gradient(parameters)
        if method == "temperature":
            projected[1] = 0.0
        for coordinate in coordinates:
            low, high = bounds[coordinate]
            if (parameters[coordinate] == low and projected[coordinate] > 0) or (
                parameters[coordinate] == high and projected[coordinate] < 0
            ):
                projected[coordinate] = 0.0
        gradient_norm = float(np.max(np.abs(projected)))
        if gradient_norm <= GRADIENT_TOLERANCE:
            break
    converged = gradient_norm <= GRADIENT_TOLERANCE
    if not converged:
        raise RuntimeError(f"calibration optimizer did not converge: gradient={gradient_norm}")
    a, b = map(float, parameters)
    return {
        "a": a, "b": b, "temperature": 1 / a,
        "raw_logit_decision_threshold": -b / a,
        "objective": objective(parameters), "identity_objective": initial_objective,
        "case_weighted_training_bce": float(np.dot(weights, stable_bce(a * z + b, y))),
        "projected_gradient_inf": gradient_norm, "cycles": cycle, "converged": converged,
        "at_scale_bound": a in SCALE_BOUNDS, "at_offset_bound": b in OFFSET_BOUNDS,
    }


def binary_metrics(logits, targets, weights=None):
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(targets, dtype=bool)
    p = sigmoid(z)
    decisions = binary_decisions(z)
    correct = decisions == y
    weights = np.full(len(z), 1 / len(z)) if weights is None else np.asarray(weights, dtype=np.float64)
    recalls = [float(np.mean(correct[y == label])) for label in (False, True) if np.any(y == label)]
    positives, negatives = z[y], z[~y]
    auc = None
    if len(positives) and len(negatives):
        differences = positives[:, None] - negatives[None, :]
        auc = float(np.mean((differences > 0) + 0.5 * (differences == 0)))
    return {
        "questions": len(z), "correct": int(correct.sum()),
        "accuracy": float(np.dot(weights, correct)),
        "balanced_binary_accuracy": float(np.mean(recalls)),
        "nll": float(np.dot(weights, stable_bce(z, y))),
        "brier": float(np.dot(weights, 2 * (p - y) ** 2)),
        "scalar_binary_brier": float(np.dot(weights, (p - y) ** 2)),
        "high_probability_errors": int(np.sum((~correct) & (np.maximum(p, 1 - p) >= 0.95))),
        "false_positive": int(np.sum(decisions & ~y)), "false_negative": int(np.sum(~decisions & y)),
        "expected_true": int(y.sum()), "predicted_true": int(decisions.sum()),
        "exact_zero_logits": int(np.sum(z == 0)), "auroc": auc,
    }


def evaluate_fold(samples, fold, method):
    train = [samples[i] for i in fold.train_indices]
    test = [samples[i] for i in fold.test_indices]
    if set(fold.train_indices) & set(fold.test_indices) or (
        {row.family_id for row in train} & {row.family_id for row in test}
    ):
        raise ValueError("training and held-out families overlap")
    fit = {"a": 1.0, "b": 0.0} if method == "raw" else fit_calibrator(train, method)
    fit.update(
        fold_id=fold.fold_id, method=method,
        training_case_ids=sorted({row.case_id for row in train}),
        test_case_ids=sorted({row.case_id for row in test}),
        training_families=sorted({row.family_id for row in train}),
        test_families=sorted({row.family_id for row in test}),
        training_question_ids=sorted({row.question_id for row in train}),
        test_question_ids=sorted({row.question_id for row in test}),
        training_binary_fields=len(train), test_binary_fields=len(test),
    )
    predictions = []
    for row in test:
        value = float(transform_logits(row.raw_logit, fit["a"], fit["b"]))
        predictions.append({
            "case_id": row.case_id, "question_id": row.question_id, "family_id": row.family_id,
            "generator": row.generator, "target": row.target, "fold_id": fold.fold_id,
            "raw_logit": row.raw_logit, "logit": value, "probability_true": float(sigmoid(value)),
            "predicted": bool(binary_decisions(value)),
        })
    return fit, predictions


def relation_results(relations, predictions):
    lookup = {(row["case_id"], row["question_id"]): row for row in predictions}
    results = []
    for relation in relations:
        endpoints = [(relation[side]["case_id"], relation[side]["question_id"]) for side in ("left", "right")]
        if not all(endpoint in lookup for endpoint in endpoints):
            continue  # Categorical relations are ineligible.
        a, b = [lookup[endpoint] for endpoint in endpoints]
        if a["family_id"] != b["family_id"] or a["fold_id"] != b["fold_id"]:
            raise ValueError("binary relation endpoints must use the same held-out family calibrator")
        item = {
            "id": relation["id"], "kind": relation["kind"], "family_id": a["family_id"],
            "fold_id": a["fold_id"],
            "both_correct": bool((a["logit"] >= 0) == a["target"] and (b["logit"] >= 0) == b["target"]),
            "total_variation": abs(float(sigmoid(a["logit"]) - sigmoid(b["logit"]))),
        }
        if relation["kind"] != "invariant":
            item["direction_correct"] = bool((b["logit"] - a["logit"]) * (2 * b["target"] - 1) > 0)
        results.append(item)
    return results


def summarize(predictions, relations):
    def metrics(rows):
        return binary_metrics([r["logit"] for r in rows], [r["target"] for r in rows])

    result = {"overall": metrics(predictions)}
    # Pooled OOF AUC is intentionally omitted: different folds use different maps.
    result["overall"].pop("auroc")
    for group in ("family_id", "generator", "fold_id"):
        result["by_" + group] = {
            value: metrics([r for r in predictions if r[group] == value])
            for value in sorted({r[group] for r in predictions})
        }
        if group == "generator":
            for value in result["by_" + group].values():
                value.pop("auroc")
    result["relations"] = {}
    for kind in sorted({r["kind"] for r in relations}):
        rows = [r for r in relations if r["kind"] == kind]
        item = {
            "pairs": len(rows), "both_correct": sum(r["both_correct"] for r in rows),
            "both_correct_rate": float(np.mean([r["both_correct"] for r in rows])),
            "mean_total_variation": float(np.mean([r["total_variation"] for r in rows])),
        }
        if kind != "invariant":
            item["direction_correct"] = sum(r["direction_correct"] for r in rows)
            item["direction_correct_rate"] = float(np.mean([r["direction_correct"] for r in rows]))
        result["relations"][kind] = item
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_samples(suite, response_path):
    responses = [json.loads(line) for line in response_path.read_text().splitlines() if line.strip()]
    by_case = {row["case_id"]: row["response"] for row in responses}
    if len(by_case) != len(responses) or set(by_case) != {case["id"] for case in suite["cases"]}:
        raise ValueError("response cases must match frozen cases exactly once")
    samples = []
    for case in suite["cases"]:
        family = case["family_id"]
        generator = "execution" if family.startswith("action/") else "destination"
        for qid, question in case["request"]["questions"].items():
            if question["type"] != "noul":
                continue
            target = case["expected"][qid]
            if type(target) is not bool:
                raise ValueError("Noul target must be a hard boolean")
            answer = by_case[case["id"]]["answers"][qid]
            z = answer["details"]["raw_logit"]
            if answer["type"] != "noul" or not np.isfinite(z):
                raise ValueError("invalid binary response")
            if not np.isclose(float(sigmoid(z)), answer["noul"], atol=1e-12, rtol=1e-12):
                raise ValueError("saved response is not the identity-calibrated raw logit")
            samples.append(Sample(case["id"], qid, family, generator, float(z), target))
    if len({(row.case_id, row.question_id) for row in samples}) != len(samples):
        raise ValueError("duplicate binary field")
    return samples


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(suite_dir, response_dir, output_dir):
    suite_path, manifest_path = suite_dir / "suite.json", suite_dir / "manifest.json"
    suite = json.loads(suite_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if sha256(suite_path) != manifest["suite_sha256"]:
        raise ValueError("frozen suite hash mismatch")
    inputs = [suite_path, manifest_path] + [response_dir / f"{model}-responses.jsonl" for model in ("untrained", "trained")]
    source = Path(__file__).resolve()
    tests = source.parent.parent / "tests" / "test_contrast_calibration_probe.py"
    input_hashes = {str(path): sha256(path) for path in inputs}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "Retrospective diagnostic on previously inspected templates; not production calibration or untouched heldout performance.",
        "method": {
            "dtype": "CPU numpy float64", "loss": "stable binary cross-entropy plus ridge",
            "scale_bounds": SCALE_BOUNDS, "offset_bounds": OFFSET_BOUNDS, "ridge": RIDGE,
            "ridge_formula": "0.5 * ridge * ((a - 1)^2 + b^2); b fixed zero for temperature",
            "fitting_weights": "Each original case has equal total weight; its Noul fields share that weight equally.",
            "metric_weights": "Each binary question has equal weight in primary metrics; case-weighted NLL/Brier/accuracy also reported.",
            "brier_definition": "Sum of squared errors over false and true, equal to 2*(p_true-y)^2; scalar form also provided.",
            "decision_rule": "true iff calibrated logit >= 0; exact zero assigned true; raw zero counts reported",
            "optimizer": "Convex cyclic coordinate minimization, 60 derivative-bisection steps per free coordinate",
            "max_cycles": MAX_CYCLES, "projected_gradient_tolerance": GRADIENT_TOLERANCE,
            "selection_policy": "Bounds, regularization, grouping and optimizer fixed before observing diagnostic fold scores; no score-based tuning.",
            "discrimination": "Within-family/fold AUROC and non-invariant relation direction use logits; monotone maps cannot change their ordering.",
        },
        "input_sha256": input_hashes,
        "source_sha256": {str(path): sha256(path) for path in (source, tests)},
        "suite_cases": len(suite["cases"]), "results": {}, "fits": [],
    }
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite an existing diagnostic report directory")
    output_dir.mkdir(parents=True)
    with (output_dir / "predictions.jsonl").open("w") as prediction_file, (
        output_dir / "relations.jsonl"
    ).open("w") as relation_file:
        for model in ("untrained", "trained"):
            samples = load_samples(suite, response_dir / f"{model}-responses.jsonl")
            report["results"][model] = {}
            for scheme in ("family", "generator"):
                folds = make_folds(samples, scheme)
                methods = {}
                for method in ("raw", "temperature", "affine"):
                    predictions = []
                    for fold in folds:
                        fit, predicted = evaluate_fold(samples, fold, method)
                        report["fits"].append({"model": model, "scheme": scheme, **fit})
                        predictions.extend(predicted)
                    if len(predictions) != len(samples) or len({(r["case_id"], r["question_id"]) for r in predictions}) != len(samples):
                        raise ValueError("OOF predictions must cover every binary field exactly once")
                    relations = relation_results(suite["relations"], predictions)
                    summary = summarize(predictions, relations)
                    counts = Counter(row["case_id"] for row in predictions)
                    weights = np.asarray([1 / (len(counts) * counts[r["case_id"]]) for r in predictions])
                    weighted = binary_metrics([r["logit"] for r in predictions], [r["target"] for r in predictions], weights)
                    summary["case_weighted"] = {key: weighted[key] for key in ("nll", "brier", "scalar_binary_brier", "accuracy")}
                    methods[method] = summary
                    for row in predictions:
                        prediction_file.write(json.dumps({"model": model, "scheme": scheme, "method": method, **row}, allow_nan=False) + "\n")
                    for row in relations:
                        relation_file.write(json.dumps({"model": model, "scheme": scheme, "method": method, **row}, allow_nan=False) + "\n")
                report["results"][model][scheme] = methods
    if input_hashes != {str(path): sha256(path) for path in inputs}:
        raise RuntimeError("an input changed during the diagnostic")
    report["output_sha256"] = {
        name: sha256(output_dir / name) for name in ("predictions.jsonl", "relations.jsonl")
    }
    write_json(output_dir / "summary.json", report)
    write_json(output_dir / "manifest.json", {
        "role": report["role"], "input_sha256": input_hashes,
        "source_sha256": report["source_sha256"],
        "output_sha256": {**report["output_sha256"], "summary.json": sha256(output_dir / "summary.json")},
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("data/semantic-contrasts-v1"))
    parser.add_argument("--responses", type=Path, default=Path("reports/semantic-contrasts-v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/contrast-calibration-v1"))
    args = parser.parse_args()
    report = run(args.suite, args.responses, args.output)
    for model, schemes in report["results"].items():
        for scheme, methods in schemes.items():
            for method, summary in methods.items():
                values = summary["overall"]
                print(model, scheme, method, json.dumps(values))


if __name__ == "__main__":
    main()
