"""Validation-only, family-preserving temperature diagnostics for saved logits."""

import copy
import hashlib
import math
from collections import Counter, defaultdict

import numpy as np

from experiments.revised_metrics import prediction_view


def assign_folds(records, *, folds=4, seed=42):
    groups, strata = {}, defaultdict(set)
    for record in records:
        key = record["origin"], record["group_id"]
        stratum = record["origin"], record["category"]
        if key in groups and groups[key] != stratum:
            raise ValueError("a family spans different calibration strata")
        groups[key] = stratum
        strata[stratum].add(record["group_id"])
    result = {}
    for (origin, category), identities in sorted(strata.items()):
        if len(identities) < folds:
            raise ValueError("each calibration stratum needs at least one family per fold")
        ordered = sorted(identities, key=lambda identity: hashlib.sha256(
            f"temperature-v1/{seed}/{origin}/{category}/{identity}".encode()
        ).hexdigest())
        for index, identity in enumerate(ordered):
            result[origin, identity] = index % folds
    return result


def balanced_weights(records):
    origins = {record["origin"] for record in records}
    kinds = {record["primitive"] for record in records}
    counts = Counter((record["origin"], record["primitive"]) for record in records)
    if origins != {"new", "broad"} or any(not counts[o, k] for o in origins for k in kinds):
        raise ValueError("calibration fitting needs both populations and every chosen primitive")
    return np.array([1 / (2 * len(kinds) * counts[r["origin"], r["primitive"]]) for r in records])


def _packed(records):
    if not records:
        raise ValueError("calibration needs nonempty records")
    width = max(len(record["logits"]) for record in records)
    logits = np.full((len(records), width), -np.inf, dtype=np.float64)
    targets = []
    for index, record in enumerate(records):
        values, target = record["logits"], record["target_index"]
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("calibration logits must be finite")
        if type(target) is not int or not 0 <= target < len(values):
            raise ValueError("calibration target index is invalid")
        logits[index, :len(values)] = values
        targets.append(target)
    # Remove the arbitrary per-question offset before temperature division;
    # otherwise a large common utility can erase finite loss differences.
    logits -= np.max(logits, axis=1)[:, None]
    return logits, np.asarray(targets, dtype=np.int64)


def fit_temperature(records, weights=None, *, lower=.05, upper=100.):
    """Minimize stable full-vector CE over a predeclared positive interval."""
    if not 0 < lower < 1 < upper or not math.isfinite(upper):
        raise ValueError("temperature bounds must bracket one and be positive/finite")
    logits, targets = _packed(records)
    weights = np.full(len(records), 1 / len(records)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(records),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("invalid calibration weights")
    weights = weights / weights.sum()
    indices = np.arange(len(records))

    def objective(log_temperature):
        scaled = logits / math.exp(log_temperature)
        losses = np.log(np.exp(scaled).sum(axis=1)) - scaled[indices, targets]
        return float(weights @ losses)

    # CE is convex in inverse temperature; this monotone reparameterization
    # keeps the one-dimensional objective unimodal on a finite positive range.
    left, right = math.log(lower), math.log(upper)
    ratio = (math.sqrt(5) - 1) / 2
    a, b = right - ratio * (right - left), left + ratio * (right - left)
    fa, fb = objective(a), objective(b)
    for _ in range(80):
        if fa <= fb:
            right, b, fb = b, a, fa
            a = right - ratio * (right - left)
            fa = objective(a)
        else:
            left, a, fa = a, b, fb
            b = left + ratio * (right - left)
            fb = objective(b)
    candidates = (math.log(lower), 0., math.log(upper), (left + right) / 2)
    best = min(candidates, key=lambda value: (objective(value), abs(value)))
    temperature = math.exp(best)
    return {"temperature": temperature, "objective": objective(best), "identity_objective": objective(0.),
            "at_boundary": abs(best - math.log(lower)) < 1e-6 or abs(best - math.log(upper)) < 1e-6,
            "fit_records": len(records), "bounds": [lower, upper]}


def calibrate_record(record, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    peak = max(record["logits"])
    logits = [(float(value) - peak) / temperature for value in record["logits"]]
    kind, labels = record["primitive"], record["labels"]
    if kind == "noul":
        z = logits[1] - logits[0]
        value = math.exp(-abs(z))
        p = 1 / (1 + value) if z >= 0 else value / (1 + value)
        answer = {"noul": p, "details": {"calibrated_logit": z}}
    else:
        peak = max(logits)
        weights = [math.exp(value - peak) for value in logits]
        probabilities = {label: value / sum(weights) for label, value in zip(labels, weights, strict=True)}
        answer = {"probabilities": probabilities, "details": {"calibrated_logits": dict(zip(labels, logits, strict=True))}}
        if kind == "score":
            answer["score"] = sum(int(label) * value for label, value in probabilities.items())
    result = copy.deepcopy(record["row"])
    result["prediction"] = prediction_view(record["question"], record["target"], answer)
    if result["prediction"]["predicted"] != record["row"]["prediction"]["predicted"]:
        raise ValueError("positive temperature changed the observed winner/tie; inspect numerical tolerance")
    return result


def crossfit(records, fold_map, *, variant, folds=4):
    if variant not in ("global", "per_primitive"):
        raise ValueError("unknown calibration variant")
    kinds = sorted({record["primitive"] for record in records})
    results, fits = [None] * len(records), {}
    for fold in range(folds):
        training = [r for r in records if fold_map[r["origin"], r["group_id"]] != fold]
        heldout = [(index, r) for index, r in enumerate(records) if fold_map[r["origin"], r["group_id"]] == fold]
        if not training or not heldout:
            raise ValueError("every fold needs distinct fitting and assessment records")
        if variant == "global":
            fitted = {"all": fit_temperature(training, balanced_weights(training))}
            temperatures = {kind: fitted["all"]["temperature"] for kind in kinds}
        else:
            fitted = {}
            for kind in kinds:
                chosen = [r for r in training if r["primitive"] == kind]
                fitted[kind] = fit_temperature(chosen, balanced_weights(chosen))
            temperatures = {kind: fitted[kind]["temperature"] for kind in kinds}
        fits[str(fold)] = fitted
        for index, record in heldout:
            row = calibrate_record(record, temperatures[record["primitive"]])
            row["calibration"] = {"variant": variant, "heldout_fold": fold,
                                  "temperature": temperatures[record["primitive"]]}
            results[index] = row
    if any(row is None for row in results):
        raise ValueError("cross-fitting did not assess every record exactly once")
    return {"rows": results, "fits": fits}
