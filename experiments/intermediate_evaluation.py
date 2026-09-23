"""Evaluation helpers for matched intermediate supervision; no label enters a model input."""

import math
from collections import Counter, defaultdict
from dataclasses import replace

import torch

from experiments.calibrated_targets import labels_for
from experiments.revised_metrics import prediction_view
from experiments.temperature_diagnostic import balanced_weights, fit_temperature
from openjev.consistent_cached_inference import score_canonical_cached
from openjev.judgments import assemble_response, compile_request


def _tiles(prompts, prefixes):
    groups = defaultdict(lambda: defaultdict(list))
    for i, (prompt, prefix) in enumerate(zip(prompts, prefixes, strict=True)):
        bucket = ((len(prompt) - prefix + 64) // 64) * 64
        groups[tuple(prompt[:prefix])][bucket].append(i)
    return [indices[start:start + 4] for _, buckets in sorted(groups.items())
            for _, indices in sorted(buckets.items()) for start in range(0, len(indices), 4)]


@torch.inference_mode()
def capture_case(engine, case, auxiliary=None):
    """Observe the exact hidden states consumed by canonical cached scalar readouts.

    The hook delegates to the original readout first, does not alter its output,
    and restores even on failure. Root schedules this synchronous helper under
    the shared GPU lock; concurrent calls on this model are not supported.
    """
    compiled, prompts, kinds = engine.prepare(case["request"])
    prefixes = engine.prefix_lengths(compiled, prompts)
    tiles = _tiles(prompts, prefixes)
    predictions = [None] * len(prompts) if auxiliary is not None else None
    original = engine.model._read
    had_instance = "_read" in engine.model.__dict__
    prior_instance = engine.model.__dict__.get("_read")
    position = 0

    def observe(hidden, readout_kinds):
        nonlocal position
        result = original(hidden, readout_kinds)
        if position >= len(tiles) or hidden.shape[0] != 4:
            raise RuntimeError("cached readout call differs from pinned B4 tile plan")
        values = auxiliary(hidden.float())
        if values.ndim != 2 or values.shape[0] != 4 or not torch.isfinite(values).all():
            raise ValueError("auxiliary readout must return finite B4 feature rows")
        rows = values.detach().float().cpu().tolist()
        for row, index in enumerate(tiles[position]):
            predictions[index] = rows[row]
        position += 1
        return result

    try:
        if auxiliary is not None:
            engine.model._read = observe
        logits, stats = score_canonical_cached(engine.model, prompts, kinds, prefixes,
                                               unit_batch_size=engine.unit_batch_size,
                                               max_input_tokens=engine.max_input_tokens)
    finally:
        if auxiliary is not None:
            if had_instance:
                engine.model._read = prior_instance
            else:
                del engine.model._read
    if auxiliary is not None and (position != len(tiles) or any(x is None for x in predictions)):
        raise RuntimeError("not all logical hidden-state rows were captured")
    scores = logits.detach().float().cpu().tolist()
    response = assemble_response(compiled, scores, details=True)
    response.update(execution_policy=dict(engine.policy), source_model=engine.source_model_id)
    return {"case_id": case["id"], "logits": scores, "auxiliary": predictions,
            "response": response, "stats": stats, "prefix_lengths": prefixes}


def _sigmoid(z):
    value = math.exp(-abs(z))
    return 1 / (1 + value) if z >= 0 else value / (1 + value)


def fact_metrics(suite, captures, annotations, schema):
    """Equal logical-question weighting; candidate multiplicity is averaged away."""
    features = schema["features"]
    records = defaultdict(list)
    joint = []
    for case in suite["cases"]:
        capture = captures[case["id"]]
        values = capture["auxiliary"]
        targets = annotations["cases"][case["id"]]["targets"]
        mask = annotations["cases"][case["id"]]["mask"]
        compiled = compile_request(case["request"])
        if (values is None or len(values) != len(compiled.units) or len(targets) != len(features)
                or len(mask) != len(features) or any(type(v) is not bool for v in mask)
                or any(len(row) != len(features) or any(not math.isfinite(v) for v in row) for row in values)):
            raise ValueError("fact target/mask/prediction dimensions or values invalid")
        for question in compiled.questions:
            qid = question.path[0]
            binary_checks = []
            for index, feature in enumerate(features):
                if not mask[index]:
                    continue
                target = targets[index]
                if not math.isfinite(target):
                    raise ValueError("active auxiliary targets must be finite")
                zs = [values[i][index] for i in question.unit_indices]
                base = {"case_id": case["id"], "question_id": qid, "family_id": case["family_id"],
                        "category": case["category"], "primitive": question.primitive, "target": target}
                if feature["kind"] == "binary":
                    if target not in (0, 1):
                        raise ValueError("binary fact targets must be 0/1")
                    probabilities = [_sigmoid(z) for z in zs]
                    probability = sum(probabilities) / len(probabilities)
                    predicted = None if abs(probability - .5) <= 1e-12 else int(probability > .5)
                    unit_correct = [abs(p - .5) > 1e-12 and int(p > .5) == target for p in probabilities]
                    loss = sum(max(z, 0) - z * target + math.log1p(math.exp(-abs(z))) for z in zs) / len(zs)
                    records[feature["name"]].append({**base, "probability": probability, "predicted": predicted,
                        "correct": predicted == target, "logloss": loss, "all_units_correct": all(unit_correct)})
                    binary_checks.append(all(unit_correct))
                elif feature["kind"] == "regression":
                    scale = feature["scale"]
                    if not math.isfinite(scale) or scale <= 0:
                        raise ValueError("positive finite numeric scale required")
                    predictions = [z * scale for z in zs]
                    records[feature["name"]].append({**base, "prediction": sum(predictions) / len(predictions),
                        "absolute_error": sum(abs(z - target) for z in predictions) / len(predictions)})
                else:
                    raise ValueError("unknown auxiliary feature kind")
            if binary_checks:
                final = prediction_view(case["request"]["questions"][qid], case["expected"][qid],
                                        capture["response"]["answers"][qid])
                joint.append({"case_id": case["id"], "question_id": qid, "family_id": case["family_id"],
                              "category": case["category"], "primitive": question.primitive, "facts_correct": all(binary_checks),
                              "final_correct": final["correct"]})
    summary = {}
    for feature in features:
        xs = records[feature["name"]]
        value = {"kind": feature["kind"], "questions": len(xs)}
        if feature["kind"] == "binary":
            positive, negative = sum(x["target"] == 1 for x in xs), sum(x["target"] == 0 for x in xs)
            tp = sum(x["target"] == 1 and x["correct"] for x in xs)
            tn = sum(x["target"] == 0 and x["correct"] for x in xs)
            value.update(correct=sum(x["correct"] for x in xs), positives=positive, negatives=negative,
                         true_positives=tp, true_negatives=tn, ambiguous=sum(x["predicted"] is None for x in xs),
                         false_positives=sum(x['target'] == 0 and x['predicted'] == 1 for x in xs),
                         false_negatives=sum(x['target'] == 1 and x['predicted'] == 0 for x in xs),
                         ambiguous_positive=sum(x['target'] == 1 and x['predicted'] is None for x in xs),
                         ambiguous_negative=sum(x['target'] == 0 and x['predicted'] is None for x in xs),
                         accuracy=sum(x["correct"] for x in xs) / len(xs) if xs else None,
                         balanced_accuracy=(tp / positive + tn / negative) / 2 if positive and negative else None,
                         logloss=sum(x["logloss"] for x in xs) / len(xs) if xs else None)
        else:
            value.update(mean_absolute_error=sum(x["absolute_error"] for x in xs) / len(xs) if xs else None,
                         scale=feature["scale"])
        summary[feature["name"]] = value
    balanced = [x["balanced_accuracy"] for x in summary.values()
                if x["kind"] == "binary" and x["balanced_accuracy"] is not None]
    return {"features": summary, "binary_macro_balanced_accuracy": sum(balanced) / len(balanced) if balanced else None,
            "eligible_balanced_features": len(balanced),
            "eligible_balanced_feature_names": [name for name, x in summary.items()
                                                if x['kind'] == 'binary' and x['balanced_accuracy'] is not None],
            "joint_binary": {
                "questions": len(joint), "all_units_facts_correct": sum(x["facts_correct"] for x in joint),
                "facts_and_final_correct": sum(x["facts_correct"] and x["final_correct"] for x in joint)},
            "feature_rows": dict(records), "joint_rows": joint,
            "definition": "Per-question pooled binary predictions; equal candidate mean loss/MAE; joint requires every applicable binary fact on every candidate branch."}


def calibration_records(suite, captures, *, origin):
    rows = []
    for case in suite["cases"]:
        compiled = compile_request(case["request"])
        scores = captures[case["id"]]["logits"]
        if len(scores) != len(compiled.units):
            raise ValueError("logit/input unit mismatch")
        for question in compiled.questions:
            qid = question.path[0]
            labels = labels_for(case["request"]["questions"][qid])
            target = case["expected"][qid]
            label = str(target).lower() if type(target) is bool else str(target)
            logits = [float(scores[i]) for i in question.unit_indices]
            if question.primitive == "noul":
                logits = [0., logits[0]]
            rows.append({"case_id": case["id"], "question_id": qid, "origin": origin,
                         "primitive": question.primitive, "labels": labels, "logits": logits,
                         "target_index": labels.index(label)})
    return rows


def fit_calibration(records):
    counts = Counter((r["origin"], r["primitive"]) for r in records)
    if set(counts) != {(origin, primitive) for origin in ("new", "broad") for primitive in ("noul", "choice", "score")}:
        raise ValueError("calibration needs every primitive in separate new and broad populations")
    return {"global": fit_temperature(records, balanced_weights(records)), "per_primitive": {
        kind: fit_temperature([r for r in records if r["primitive"] == kind],
                              balanced_weights([r for r in records if r["primitive"] == kind]))
        for kind in ("noul", "choice", "score")}}


def calibrated_responses(suite, captures, temperatures):
    responses = {}
    for case in suite["cases"]:
        capture = captures[case["id"]]
        compiled = replace(compile_request(case["request"]), model=capture["response"]["model"])
        response = assemble_response(compiled, capture["logits"], details=True, temperatures=temperatures)
        for key in ("execution_policy", "source_model"):
            if key in capture["response"]:
                response[key] = capture["response"][key]
        # The hashed execution policy identifies the raw GPU computation. Applied
        # calibration is a separate, explicitly recorded transformation.
        response['probability_transform'] = {'method': 'temperature-v1', 'temperatures': dict(temperatures),
                                             'source_execution_policy_is_raw': True}
        responses[case["id"]] = response
    return responses
