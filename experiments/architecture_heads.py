"""CPU-only, frozen-H0-feature readout probes for architecture diagnostics D4."""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.calibrated_targets import file_sha256, labels_for, request_sha256
from experiments.revised_metrics import target_label
from experiments.temperature_diagnostic import fit_temperature
from openjev.judgments import assemble_response, compile_request

VARIANTS = ("frozen-original", "linear-refit", "split-choice-score", "residual-mlp128")
SPLITS = ("fit", "calibration", "development", "retention")
PRIMITIVES = {"choice": 0, "score": 1, "noul": 2}


def unit_primitives(cache):
    kinds = torch.full((len(cache["hidden"]),), -1, dtype=torch.long)
    for group in cache["groups"]:
        kinds[group["indices"]] = PRIMITIVES[group["primitive"]]
    if (kinds < 0).any():
        raise ValueError("features contain unassigned units")
    return kinds


class HeadProbe(nn.Module):
    """Shared scalar compatibility, split scalar compatibility, or residual readout."""

    def __init__(self, heads, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("unknown head variant")
        self.variant = variant
        width = heads["compatibility.weight"].shape[1]
        self.compatibility = nn.Linear(width, 1, bias=False, dtype=torch.float32)
        self.binary = nn.Linear(width, 1, bias=True, dtype=torch.float32)
        with torch.no_grad():
            self.compatibility.weight.copy_(heads["compatibility.weight"])
            self.binary.weight.copy_(heads["binary.weight"])
            self.binary.bias.copy_(heads["binary.bias"])
        if variant == "split-choice-score":
            self.score = nn.Linear(width, 1, bias=False, dtype=torch.float32)
            with torch.no_grad():
                self.score.weight.copy_(heads["compatibility.weight"])
        if variant == "residual-mlp128":
            # Fixed per-unit LayerNorm bounds the MLP input scale; original linear
            # heads still consume untouched H0 features. No data-fitted statistics.
            self.residuals = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(width, elementwise_affine=False, eps=1e-5),
                              nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))
                for _ in range(2)
            ])
            for residual in self.residuals:
                nn.init.zeros_(residual[-1].weight)
                nn.init.zeros_(residual[-1].bias)
        if variant == "frozen-original":
            self.requires_grad_(False)

    def forward(self, hidden, primitives):
        if hidden.device.type != "cpu" or hidden.dtype != torch.float32:
            raise ValueError("head probes require CPU float32 frozen features")
        hidden = hidden.detach()
        result = torch.where(primitives == 2, self.binary(hidden).squeeze(-1),
                             self.compatibility(hidden).squeeze(-1))
        if self.variant == "split-choice-score":
            result = torch.where(primitives == 1, self.score(hidden).squeeze(-1), result)
        if self.variant == "residual-mlp128":
            for kind, residual in enumerate(self.residuals):
                mask = (primitives == 2) if kind else (primitives != 2)
                if mask.any():
                    result = result.index_add(0, mask.nonzero().flatten(), residual(hidden[mask]).squeeze(-1))
        return result


def head_reproduction(cache):
    """Audit separate CPU reproduction and diagnostic-only CPU/GPU tolerances.

    Original GPU logits remain untouched. Probabilities use the same stable
    scalar transforms and 1e-12 top-tie rule as the original response metrics.
    """
    cpu = cache["cpu_reference_logits"].detach().double()
    gpu = cache["original_logits"].detach().double()
    kinds = unit_primitives(cache)
    with torch.no_grad():
        reconstructed = HeadProbe(cache["heads"], "frozen-original")(cache["hidden"], kinds).double()
        hidden = cache["hidden"].detach().double()
        heads = {key: value.detach().double() for key, value in cache["heads"].items()}
        fp64 = torch.where(kinds == 2,
                           F.linear(hidden, heads["binary.weight"], heads["binary.bias"]).squeeze(-1),
                           F.linear(hidden, heads["compatibility.weight"]).squeeze(-1))
    cpu_error = float((reconstructed - cpu).abs().max())
    logit_errors = (cpu - gpu).abs()
    logit_gate = bool((logit_errors <= 1e-5 + 1e-6 * gpu.abs()).all())

    def probabilities(values, primitive):
        if primitive == "noul":
            z = values[0]
            p = 1. / (1. + math.exp(-z)) if z >= 0 else math.exp(z) / (1. + math.exp(z))
            return [1. - p, p]
        peak = max(values)
        weights = [math.exp(value - peak) for value in values]
        return [value / sum(weights) for value in weights]

    rows = []
    for group in cache["groups"]:
        cpu_p = probabilities(cpu[group["indices"]].tolist(), group["primitive"])
        gpu_p = probabilities(gpu[group["indices"]].tolist(), group["primitive"])
        cpu_top = [label for label, p in zip(group["labels"], cpu_p, strict=True) if abs(p - max(cpu_p)) <= 1e-12]
        gpu_top = [label for label, p in zip(group["labels"], gpu_p, strict=True) if abs(p - max(gpu_p)) <= 1e-12]
        rows.append({"case_id": group["case_id"], "question_id": group["question_id"],
                     "primitive": group["primitive"], "labels": group["labels"],
                     "cpu_probabilities": cpu_p, "gpu_probabilities": gpu_p,
                     "max_absolute_probability_error": max(abs(a - b) for a, b in zip(cpu_p, gpu_p, strict=True)),
                     "cpu_top_labels": cpu_top, "gpu_top_labels": gpu_top, "decision_changed": cpu_top != gpu_top})
    probability_error = max(row["max_absolute_probability_error"] for row in rows)
    decision_changes = sum(row["decision_changed"] for row in rows)
    return {"policy": {"scope": "architecture-diagnostics-v1 CPU/GPU readout only",
                       "cpu_reconstruction_atol": 1e-6, "cpu_reconstruction_rtol": 0.,
                       "gpu_logit_atol": 1e-5, "gpu_logit_rtol": 1e-6, "gpu_logit_relative_reference": "original GPU",
                       "probability_atol": 1e-6, "top_tie_probability_atol": 1e-12,
                       "allowed_decision_changes": 0},
            "cpu_reconstruction": {"max_absolute_logit_error": cpu_error, "passed": cpu_error <= 1e-6},
            "cpu_vs_gpu": {"max_absolute_logit_error": float(logit_errors.max()),
                           "logits_within_tolerance": logit_gate,
                           "max_absolute_probability_error": probability_error, "decision_changes": decision_changes,
                           "passed": logit_gate and probability_error <= 1e-6 and decision_changes == 0,
                           "groups": rows},
            "fp64_reference": {"definition": "FP64 scalar linear readout of the same cached FP32 features and heads",
                               "cpu_max_absolute_error": float((cpu - fp64).abs().max()),
                               "gpu_max_absolute_error": float((gpu - fp64).abs().max())}}


def validate_cache(cache, suite_path):
    """Bind every feature row and target to the original request, in exact order."""
    if not isinstance(cache, dict) or cache.get("version") != 1:
        raise ValueError("invalid feature cache version")
    if cache.get("source_suite_sha256") != file_sha256(suite_path):
        raise ValueError("feature cache suite hash mismatch")
    for key in ("checkpoint", "checkpoint_id"):
        if not isinstance(cache.get(key), str) or not cache[key]:
            raise ValueError("feature checkpoint identity missing")
    if not isinstance(cache.get("sourcepins"), dict) or not cache["sourcepins"]:
        raise ValueError("feature source pins missing")
    suite = json.loads(Path(suite_path).read_text())
    expected, case_ids, offset = [], set(), 0
    if not suite.get("cases"):
        raise ValueError("suite has no cases")
    for case in suite["cases"]:
        cid, request = case["id"], case["request"]
        if not isinstance(cid, str) or not cid or cid in case_ids:
            raise ValueError("duplicate or invalid case identity")
        case_ids.add(cid)
        compiled = compile_request(request)
        if set(case["expected"]) != set(request["questions"]):
            raise ValueError("missing or extra question targets")
        for question in compiled.questions:
            if len(question.path) != 1 or not isinstance(question.path[0], str):
                raise ValueError("D4 suites require named top-level questions")
            qid = question.path[0]
            labels = labels_for(request["questions"][qid])
            target = case["expected"][qid]
            valid = ((question.primitive == "noul" and type(target) is bool)
                     or (question.primitive == "score" and type(target) is int)
                     or (question.primitive == "choice" and isinstance(target, str)))
            if not valid or target_label(target) not in labels:
                raise ValueError("invalid target value")
            expected.append({"case_id": cid, "question_id": qid, "primitive": question.primitive,
                             "labels": labels, "target_index": labels.index(target_label(target)),
                             "indices": [offset + i for i in question.unit_indices],
                             "request_sha256": request_sha256(request)})
        offset += len(compiled.units)
    groups = cache.get("groups")
    if not isinstance(groups, list) or len(groups) != len(expected):
        raise ValueError("missing or extra feature groups")
    for actual, wanted in zip(groups, expected, strict=True):
        if not isinstance(actual, dict) or any(actual.get(key) != value for key, value in wanted.items()):
            raise ValueError("feature groups/labels/indices/targets/request identity mismatch")
        if type(actual["target_index"]) is not int or any(type(i) is not int for i in actual["indices"]):
            raise ValueError("feature indices must be integers")

    def checked_tensor(value, shape):
        return (isinstance(value, torch.Tensor) and value.dtype == torch.float32
                and tuple(value.shape) == shape and value.device.type == "cpu" and torch.isfinite(value).all())

    if not checked_tensor(cache.get("hidden"), (offset, 2048)):
        raise ValueError("hidden features must be finite CPU float32[Nunits,2048]")
    heads = cache.get("heads")
    shapes = {"compatibility.weight": (1, 2048), "binary.weight": (1, 2048), "binary.bias": (1,)}
    if not isinstance(heads, dict) or set(heads) != set(shapes):
        raise ValueError("original readout tensors missing or unexpected")
    if any(not checked_tensor(heads[key], shape) for key, shape in shapes.items()):
        raise ValueError("invalid original readout shape/dtype/value")
    if not checked_tensor(cache.get("original_logits"), (offset,)):
        raise ValueError("original reference logits missing or invalid")
    if not checked_tensor(cache.get("cpu_reference_logits"), (offset,)):
        raise ValueError("CPU reference logits missing or invalid")
    clean = {**cache, "hidden": cache["hidden"].detach(),
             "heads": {key: value.detach() for key, value in heads.items()}}
    evidence = head_reproduction(clean)
    if not evidence["cpu_reconstruction"]["passed"]:
        raise ValueError("frozen CPU readout does not reproduce CPU reference logits within 1e-6")
    if not evidence["cpu_vs_gpu"]["passed"]:
        raise ValueError("CPU/GPU readout differs beyond declared logit/probability/decision gates")
    clean["head_reproduction"] = evidence
    return clean


def check_common_source(caches):
    source = next(iter(caches.values()))
    for cache in caches.values():
        for key in ("checkpoint", "checkpoint_id", "sourcepins"):
            if cache[key] != source[key]:
                raise ValueError("feature splits use different checkpoint/source pins")
        if set(cache["heads"]) != set(source["heads"]) or any(
            not torch.equal(cache["heads"][key], value) for key, value in source["heads"].items()
        ):
            raise ValueError("feature splits use different original readouts")


def _balanced_weights(groups):
    counts = Counter(group["primitive"] for group in groups)
    if not counts:
        raise ValueError("nonempty groups required")
    return [1 / (len(counts) * counts[group["primitive"]]) for group in groups]


def _loss_plan(groups):
    width = max(len(group["labels"]) for group in groups)
    indices, zero, padding, targets = [], [], [], []
    for group in groups:
        ids = group["indices"]
        binary = group["primitive"] == "noul"
        ids = [ids[0], ids[0]] if binary else ids
        n = len(ids)
        indices.append(ids + [0] * (width - n))
        zero.append(([True, False] if binary else [False] * n) + [False] * (width - n))
        padding.append([False] * n + [True] * (width - n))
        targets.append(group["target_index"])
    return (torch.tensor(indices), torch.tensor(zero), torch.tensor(padding), torch.tensor(targets),
            torch.tensor(_balanced_weights(groups)))


def balanced_loss(logits, groups, *, plan=None):
    indices, zero, padding, targets, weights = _loss_plan(groups) if plan is None else plan
    matrix = logits[indices].masked_fill(zero, 0.).masked_fill(padding, -torch.inf)
    return (F.cross_entropy(matrix, targets, reduction="none") * weights).sum()


def train_probe(model, cache, *, updates=300, max_seconds=300.):
    if type(updates) is not int or not 1 <= updates <= 300 or not 0 < max_seconds <= 300:
        raise ValueError("training must remain within the 300 update/300 second protocol cap")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    hidden, groups = cache["hidden"].detach(), cache["groups"]
    kinds, plan = unit_primitives(cache), _loss_plan(groups)
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-4, weight_decay=0.) if parameters else None
    start = time.monotonic()
    with torch.no_grad():
        initial = float(balanced_loss(model(hidden, kinds), groups, plan=plan))
    history = [{"update": 0, "loss": initial, "elapsed_seconds": time.monotonic() - start}]
    completed = 0
    reason = "frozen-reference" if optimizer is None else "fixed-update-cap"
    if optimizer is not None:
        for update in range(1, updates + 1):
            if time.monotonic() - start >= max_seconds:
                reason = "walltime-cap"
                break
            optimizer.zero_grad(set_to_none=True)
            loss = balanced_loss(model(hidden, kinds), groups, plan=plan)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite head training loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            completed = update
            history.append({"update": update, "pre_update_loss": float(loss.detach()),
                            "gradient_norm_before_clip": float(norm), "elapsed_seconds": time.monotonic() - start})
    model.eval()
    with torch.no_grad():
        history[-1]["loss"] = float(balanced_loss(model(hidden, kinds), groups, plan=plan))
    return {"history": history, "updates_completed": completed, "requested_updates": updates,
            "elapsed_seconds": time.monotonic() - start, "max_seconds": max_seconds, "stop_reason": reason,
            "endpoint_policy": "fixed-update-or-walltime-cap", "seed": 42, "cpu_threads": 4,
            "optimizer": {"name": "AdamW", "learning_rate": 5e-4, "weight_decay": 0., "clip_norm": 1.},
            "loss": "mean of primitive means of hard-target cross entropy"}


def calibrate(logits, groups):
    records = []
    for group in groups:
        values = logits[group["indices"]].detach().tolist()
        records.append({"primitive": group["primitive"], "target_index": group["target_index"],
                        "logits": [0., values[0]] if group["primitive"] == "noul" else values})
    return {"global": fit_temperature(records, _balanced_weights(groups), lower=.05, upper=100.),
            "per_primitive": {kind: fit_temperature([row for row in records if row["primitive"] == kind],
                                                     lower=.05, upper=100.)
                              for kind in sorted({row["primitive"] for row in records})},
            "fit_source": "calibration only", "objective": "mean of primitive mean NLL"}


def responses_for(suite, logits, *, temperatures=None):
    responses, offset = {}, 0
    for case in suite["cases"]:
        compiled = compile_request(case["request"])
        end = offset + len(compiled.units)
        responses[case["id"]] = assemble_response(compiled, logits[offset:end].detach().tolist(), details=True,
                                                 temperatures=temperatures)
        offset = end
    if offset != len(logits):
        raise ValueError("response feature count differs from suite")
    return responses


def _write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run(data, features, output, *, updates=300, max_seconds=300.):
    from experiments.architecture_diagnostics import suite_metrics

    data, features, output = Path(data).resolve(), Path(features).resolve(), Path(output).resolve()
    torch.set_num_threads(4)
    suites, caches, sources = {}, {}, {}
    for split in SPLITS:
        suite_path, cache_path = data / f"{split}.json", features / f"{split}.pt"
        suites[split] = json.loads(suite_path.read_text())
        caches[split] = validate_cache(torch.load(cache_path, map_location="cpu", weights_only=True), suite_path)
        sources[split] = {"suite": str(suite_path), "suite_sha256": file_sha256(suite_path),
                          "cache": str(cache_path), "cache_sha256": file_sha256(cache_path),
                          "feature_shape": list(caches[split]["hidden"].shape),
                          "groups": len(caches[split]["groups"])}
    check_common_source(caches)
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            a, b = suites[left]["cases"], suites[right]["cases"]
            if ({c["id"] for c in a} & {c["id"] for c in b}
                    or {c["family_id"] for c in a} & {c["family_id"] for c in b}):
                raise ValueError("fit/calibration/development/retention cases and families must be disjoint")
    required = {g["primitive"] for split in SPLITS for g in caches[split]["groups"]}
    if {g["primitive"] for g in caches["calibration"]["groups"]} != required:
        raise ValueError("calibration is missing a primitive")
    output.mkdir(parents=True, exist_ok=False)
    report = {"version": 1, "protocol": "architecture-diagnostics-v1/D4", "sources": sources,
              "checkpoint": caches["fit"]["checkpoint"], "checkpoint_id": caches["fit"]["checkpoint_id"],
              "sourcepins": caches["fit"]["sourcepins"], "implementation_sha256": file_sha256(__file__),
              "head_reproduction": {split: cache["head_reproduction"] for split, cache in caches.items()},
              "torch_version": str(torch.__version__), "feature_dtype": "float32", "device": "cpu",
              "initialization": "warm H0 hardCE step-0400; body features frozen; fit cases previously seen by H0",
              "variants": {}, "interpretation": "finite-set fit/accessibility; not fresh learning or rule generalization"}
    _write_json(output / "sources.json", {key: value for key, value in report.items() if key != "variants"})
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = HeadProbe(caches["fit"]["heads"], variant)
        initial_errors = {}
        with torch.no_grad():
            for split, cache in caches.items():
                initial = model(cache["hidden"], unit_primitives(cache))
                error = float((initial - cache["cpu_reference_logits"]).abs().max())
                if error > 1e-6:
                    raise ValueError("variant initial logits differ from CPU reference readout")
                initial_errors[split] = error
        history = train_probe(model, caches["fit"], updates=updates, max_seconds=max_seconds)
        directory = output / variant
        directory.mkdir()
        _write_json(directory / "history.json", history)
        with torch.no_grad():
            logits = {split: model(cache["hidden"], unit_primitives(cache)) for split, cache in caches.items()}
        logits_path = directory / "raw-logits.pt"
        torch.save({"version": 1, "variant": variant, "checkpoint_id": report["checkpoint_id"],
                    "logits": logits, "groups": {split: cache["groups"] for split, cache in caches.items()},
                    "initial_references": {split: {"gpu": cache["original_logits"], "cpu": cache["cpu_reference_logits"]}
                                           for split, cache in caches.items()},
                    "sources": sources}, logits_path)
        temperatures = calibrate(logits["calibration"], caches["calibration"]["groups"])
        _write_json(directory / "calibration.json", temperatures)
        state_path = directory / "head-state.pt"
        torch.save({"version": 1, "variant": variant, "checkpoint_id": report["checkpoint_id"],
                    "state_dict": model.state_dict(), "calibration": temperatures,
                    "configuration": {"hidden_width": 2048, "residual_width": 128,
                                      "residual_normalization": "per-unit LayerNorm eps=1e-5, no affine",
                                      "residual_readout_kinds": ["compatibility", "binary"]},
                    "sources": sources, "updates_completed": history["updates_completed"]}, state_path)
        result = {"training": history, "initial_max_absolute_logit_error": initial_errors,
                  "initial_reference": "cpu_reference_logits",
                  "parameters": {"total": sum(p.numel() for p in model.parameters()),
                                 "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                 "shapes": {name: list(value.shape) for name, value in model.state_dict().items()}},
                  "state_path": str(state_path), "state_sha256": file_sha256(state_path),
                  "raw_logits_path": str(logits_path), "raw_logits_sha256": file_sha256(logits_path), "splits": {}}
        for split in ("fit", "development", "retention"):
            modes = {"raw": None, "GT": {kind: temperatures["global"]["temperature"] for kind in required},
                     "PT": {kind: fit["temperature"] for kind, fit in temperatures["per_primitive"].items()}}
            result["splits"][split] = {}
            for mode, scaling in modes.items():
                responses = responses_for(suites[split], logits[split], temperatures=scaling)
                _write_json(directory / f"{split}-{mode}-responses.json", responses)
                result["splits"][split][mode] = suite_metrics(suites[split], responses)
        _write_json(directory / "results.json", result)
        report["variants"][variant] = result
        print(json.dumps({"variant": variant, "updates": history["updates_completed"],
                          "stop_reason": history["stop_reason"], "elapsed_seconds": history["elapsed_seconds"]}), flush=True)
    # Read-only inputs are checked again after all numerical fits.
    for source in sources.values():
        if file_sha256(source["cache"]) != source["cache_sha256"] or file_sha256(source["suite"]) != source["suite_sha256"]:
            raise ValueError("source files changed during head fitting")
    report["input_files_unchanged"] = True
    _write_json(output / "results.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/architecture-diagnostics-v1"))
    parser.add_argument("--features", type=Path, default=Path("reports/architecture-diagnostics-v1/features"))
    parser.add_argument("--output", type=Path, default=Path("reports/architecture-diagnostics-v1/heads"))
    args = parser.parse_args(argv)
    run(args.data, args.features, args.output)


if __name__ == "__main__":
    main()
