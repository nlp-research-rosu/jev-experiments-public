"""D3: a finite fit-only rank-eight diagnostic, never a generalization claim."""

import argparse
import hashlib
import json
import math
import os
import random
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.contrast_scaling_training import (
    _complete_microbatches,
    _group_loss,
    _package_versions,
    apply_determinism,
    sha,
    validate_family_coverage,
    write,
)
from experiments.judgment_pipeline import optimizer_for
from openjev.judgment_cli import load_judgment_engine
from openjev.judgment_model import frozen_digest
from openjev.judgment_training import load_bundles, prepare_bundle, target_vector

CHECKPOINT = Path("checkpoints/calibrated-screen-v1/run-v1/H0/step-0400")
ROOT = Path(__file__).resolve().parents[1]
KINDS = ("noul", "choice", "score")
MILESTONES = (0, 10, 40, 80)
SPLITS = ("fit", "calibration", "development", "retention")
SETTINGS = {
    "seed": 42, "adapter_lr": 5e-5, "head_lr": 2.5e-5, "weight_decay": .01,
    "gradient_clip": 1., "max_units": 8, "unit_batch_size": 4, "max_input_tokens": 1536,
    "loss": "equal mean hard CE across three primitives; fit only; no replay",
}


class FitTimeLimit(Exception):
    """A family was abandoned before the optimizer changed any weights."""


def backward_fit_family(model, family, *, max_units=8, unit_batch_size=4, deadline=None):
    """Release each whole-candidate graph after backward with weights 1/(3*Nkind)."""
    if not 1 <= max_units <= 8 or not 1 <= unit_batch_size <= 4:
        raise ValueError("diagnostic microbatch and unit-batch bounds exceeded")
    counts = {kind: 0 for kind in KINDS}
    for group in family.groups:
        if group.primitive not in counts:
            raise ValueError("unknown primitive")
        key = {"noul": "truth", "choice": "choice", "score": "level_index"}[group.primitive]
        if set(group.target) != {key}:
            raise ValueError("fit diagnostic requires hard targets")
        target_vector(group)
        counts[group.primitive] += 1
    if not all(counts.values()):
        raise ValueError("every fit family needs all three primitives")
    if len(family.prompts) != len(family.kinds) or sorted(
        i for group in family.groups for i in group.indices
    ) != list(range(len(family.prompts))):
        raise ValueError("complete groups must cover each prompt exactly once")
    # Validate every group before touching gradients. These lists hold only token IDs.
    batches = list(_complete_microbatches(family, max_units))
    totals = {kind: 0. for kind in KINDS}
    for prompts, kinds, groups in batches:
        if deadline is not None and time.monotonic() >= deadline:
            raise FitTimeLimit("training time cap reached during family")
        logits = model.score_prompts(prompts, kinds, unit_batch_size=unit_batch_size)
        losses = [(group.primitive, _group_loss(logits[list(local)], group)) for group, local in groups]
        micro = sum(loss / (3 * counts[kind]) for kind, loss in losses)
        if not torch.isfinite(micro):
            raise RuntimeError("non-finite fit loss")
        micro.backward()
        for kind, loss in losses:
            totals[kind] += loss.detach().item()
        del micro, losses, logits
    means = {kind: totals[kind] / counts[kind] for kind in KINDS}
    return {"loss": sum(means.values()) / 3, "components": means, "counts": counts}


def validate_limits(max_updates, max_seconds):
    if type(max_updates) is not int or not 1 <= max_updates <= 80:
        raise ValueError("max_updates must be between 1 and 80")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 900:
        raise ValueError("training seconds must be positive and at most 900")


def fit_stop_reason(completed, elapsed, *, max_updates, max_seconds, perfect_fit=False):
    if completed in MILESTONES and perfect_fit:
        return "fit_success"
    if elapsed >= max_seconds:
        return "time_cap_inconclusive"
    if completed >= max_updates:
        return "update_cap_inconclusive"
    return None


def save_boundary(path, model, optimizer, state, fingerprint):
    """Persist actual trainable weights, Adam, all used RNG, history and identity atomically.

    The original H0 supplies the fingerprinted frozen body. This diagnostic's
    training.pt is restored onto a fresh H0 instance by restore_boundary.
    """
    if state.get("update_phase") != "boundary":
        raise ValueError("only complete update boundary states are resumable")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.parent / f".{path.name}.pending-{uuid.uuid4().hex}"
    pending.mkdir()
    payload = {
        "version": 1, "fingerprint": fingerprint, "state": state,
        "trainable": {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
    }
    # Failed writes deliberately remain .pending-* for inspection, never resume.
    torch.save(payload, pending / "training.pt")
    write(pending / "checkpoint.json", {
        "format": "openjev-architecture-fit-v1", "training_sha256": sha(pending / "training.pt"),
        "fingerprint": fingerprint, "state": state,
    })
    pending.rename(path)


def restore_boundary(path, model, optimizer, fingerprint):
    path = Path(path)
    manifest = json.loads((path / "checkpoint.json").read_text())
    if manifest.get("format") != "openjev-architecture-fit-v1" or manifest.get("fingerprint") != fingerprint:
        raise ValueError("resume fingerprint or format mismatch")
    if sha(path / "training.pt") != manifest.get("training_sha256"):
        raise ValueError("resume state file hash mismatch")
    payload = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    if payload["fingerprint"] != fingerprint or payload["state"] != manifest["state"]:
        raise ValueError("resume fingerprint or state mismatch")
    if payload["state"].get("update_phase") != "boundary":
        raise ValueError("resume requires a complete update boundary")
    for name in ("progress.json", "result.json", "failure.json"):
        latest_path = path.parent / name
        if latest_path.exists():
            latest = json.loads(latest_path.read_text())
            if latest.get("reason"):
                raise ValueError("capped or completed diagnostic cannot be resumed")
            if latest["completed_updates"] > payload["state"].get("completed_updates", 0):
                raise ValueError("resume checkpoint is stale relative to recorded updates")
            if latest.get("error", {}).get("phase") == "optimizer":
                raise ValueError("stale state before a failed optimizer step cannot be resumed")
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(trainable) != set(payload["trainable"]):
        raise ValueError("resume trainable parameter names differ")
    for name, value in payload["trainable"].items():
        if value.shape != trainable[name].shape or value.dtype != trainable[name].dtype:
            raise ValueError("resume trainable parameter shape/dtype differs")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(payload["trainable"][name])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    random.setstate(payload["python_rng"])
    return payload["state"]


def diagnostic_identity(model, fingerprint):
    """Adapted responses must not claim to be the unchanged original H0 weights."""
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode())
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return "openjev-architecture-fit-v1/sha256-" + digest.hexdigest()


def perfect_fit(metrics):
    overall, pairs = metrics["overall"], metrics["primary"]
    return (overall["questions"] > 0 and overall["correct"] == overall["questions"]
            and pairs["pairs"] > 0 and pairs["both_correct"] == pairs["pairs"])


def check_training_contract(model, optimizer):
    """Check current trainability and actual optimizer groups, including restored LRs."""
    named = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if any(not (".lora_" in name or name.startswith(("binary.", "compatibility."))) for name in named):
        raise ValueError("unexpected trainable body parameter")
    if not any(".lora_" in name for name in named) or not {
        "binary.weight", "binary.bias", "compatibility.weight"
    } <= set(named) or model.lora_settings.get("rank") != 8:
        raise ValueError("rank-eight adapter and all current heads must be trainable")
    if len(optimizer.param_groups) != 2 or [group["lr"] for group in optimizer.param_groups] != [5e-5, 2.5e-5]:
        raise ValueError("current optimizer learning rates differ from fixed protocol")
    if any(group["weight_decay"] != .01 for group in optimizer.param_groups):
        raise ValueError("optimizer weight decay differs from original configuration")
    expected = [[p for name, p in named.items() if ".lora_" in name],
                [p for name, p in named.items() if name.startswith(("binary.", "compatibility."))]]
    if any([id(p) for p in group["params"]] != [id(p) for p in wanted]
           for group, wanted in zip(optimizer.param_groups, expected, strict=True)):
        raise ValueError("optimizer groups do not exactly cover trainable adapters and heads")
    return {"trainable_parameters": sum(p.numel() for p in named.values()), "trainable_names": list(named),
            "learning_rates": [group["lr"] for group in optimizer.param_groups], "weight_decay": .01}


def evaluate_suites(split_names, engine, suites, directory):
    """Retain every completed independent response even if a later case fails."""
    from experiments.architecture_diagnostics import suite_metrics

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in split_names:
        started = time.monotonic()
        responses = {}
        with (directory / f"{split}-responses.jsonl").open("x") as stream:
            for case in suites[split]["cases"]:
                response = engine.evaluate(case["request"], details=True, cached=False)
                responses[case["id"]] = response
                stream.write(json.dumps({"case_id": case["id"], "response": response}, allow_nan=False) + "\n")
                stream.flush()
        metrics = suite_metrics(suites[split], responses)
        metrics["evaluation_seconds"] = time.monotonic() - started
        metrics["probability_policy"] = "raw; no temperature fit"
        metrics["input_tokens"] = sum(r["usage"]["input_tokens"] for r in responses.values())
        write(directory / f"{split}-metrics.json", metrics)
        results[split] = metrics
    return results


def run_fit(engine, prepared, suites, output, fingerprint, *, max_updates=80, max_training_seconds=900, resume=None):
    """Cycle fixed family order, with capped optimization time and safe boundary saves.

    The optimization budget includes forward/backward/clip/step and synchronization;
    preparation, evaluation, digesting and checkpoint IO are recorded separately by
    the outer wall clock. A cap detected between microbatches discards that family.
    """
    validate_limits(max_updates, max_training_seconds)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = engine.model
    optimizer = optimizer_for(model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
    contract = check_training_contract(model, optimizer)
    before = frozen_digest(model)
    started = time.monotonic()
    state = {
        "completed_updates": 0, "training_seconds": 0., "update_phase": "boundary",
        "history": [], "evaluations": {}, "reason": None, "frozen_body_before": before,
        "settings": SETTINGS, "max_updates": max_updates, "max_training_seconds": max_training_seconds,
        "contract": contract, "family_order": [bundle.id for bundle in prepared],
        "interpretation": "H0 already trained on these B families; finite fitting is not generalization",
    }
    if not prepared or len(set(state["family_order"])) != len(prepared):
        raise ValueError("prepared family order must be nonempty and unique")
    if resume:
        state = restore_boundary(resume, model, optimizer, fingerprint)
        check_training_contract(model, optimizer)
        if state["frozen_body_before"] != before or state["family_order"] != [bundle.id for bundle in prepared]:
            raise ValueError("resume frozen body or family order mismatch")
        if state["max_updates"] != max_updates or state["max_training_seconds"] != max_training_seconds:
            raise ValueError("resume cannot change diagnostic caps")
        if state.get("reason"):
            raise ValueError("completed or capped diagnostic cannot be extended by resume")
    engine.model_id = diagnostic_identity(model, fingerprint)
    try:
        if not state["evaluations"]:
            state["evaluations"]["0"] = evaluate_suites(SPLITS, engine, suites, output / "evaluation-0000")
        completed = state["completed_updates"]
        state["reason"] = fit_stop_reason(
            completed, state["training_seconds"], max_updates=max_updates, max_seconds=max_training_seconds,
            perfect_fit=perfect_fit(state["evaluations"].get(str(completed), {}).get("fit", {
                "overall": {"questions": 0, "correct": 0}, "primary": {"pairs": 0, "both_correct": 0}
            })),
        )
        save_boundary(output / f"step-{completed:04d}", model, optimizer, state, fingerprint)
        while not state["reason"]:
            bundle = prepared[state["completed_updates"] % len(prepared)]
            check_training_contract(model, optimizer)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            engine.synchronize()
            step_started = time.monotonic()
            deadline = step_started + max_training_seconds - state["training_seconds"]
            state["update_phase"] = "backward"
            write(output / "progress.json", state)
            row = None
            try:
                loss = backward_fit_family(model, bundle, deadline=deadline)
                engine.synchronize()
                if time.monotonic() >= deadline:
                    raise FitTimeLimit("time cap before optimizer update")
                parameters = [p for p in model.parameters() if p.requires_grad]
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise RuntimeError("missing or non-finite trainable gradient")
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True).item()
                if time.monotonic() >= deadline:
                    raise FitTimeLimit("time cap during gradient validation")
                state["update_phase"] = "optimizer"
                optimizer.step()
                engine.synchronize()
                state["completed_updates"] += 1
                state["update_phase"] = "boundary"
                row = {"update": state["completed_updates"], "family_id": bundle.id, **loss,
                       "gradient_norm_before_clip": norm, "finite_gradients": True,
                       "learning_rates": [group["lr"] for group in optimizer.param_groups]}
                state["history"].append(row)
            except FitTimeLimit:
                state["update_phase"] = "boundary"
                state["reason"] = "time_cap_inconclusive"
                state["abandoned_family"] = bundle.id
            finally:
                engine.synchronize()
                duration = time.monotonic() - step_started
                state["training_seconds"] += duration
                if row is not None:
                    row["training_seconds"] = duration
                optimizer.zero_grad(set_to_none=True)
            completed = state["completed_updates"]
            if row is not None and completed in MILESTONES:
                engine.model_id = diagnostic_identity(model, fingerprint)
                state["evaluations"][str(completed)] = evaluate_suites(
                    ("fit",), engine, suites, output / f"evaluation-{completed:04d}"
                )
            fit_metrics = state["evaluations"].get(str(completed), {}).get("fit")
            state["reason"] = state["reason"] or fit_stop_reason(
                completed, state["training_seconds"], max_updates=max_updates, max_seconds=max_training_seconds,
                perfect_fit=fit_metrics is not None and perfect_fit(fit_metrics),
            )
            write(output / "progress.json", state)
            if row is not None and completed in MILESTONES:
                save_boundary(output / f"step-{completed:04d}", model, optimizer, state, fingerprint)
        completed = state["completed_updates"]
        engine.model_id = diagnostic_identity(model, fingerprint)
        state["endpoint_model_id"] = engine.model_id
        evaluated = state["evaluations"].setdefault(str(completed), {})
        missing = [split for split in SPLITS if split not in evaluated]
        evaluated.update(evaluate_suites(missing, engine, suites, output / f"evaluation-{completed:04d}"))
        state["frozen_body_after"] = frozen_digest(model)
        state["frozen_body_unchanged"] = before == state["frozen_body_after"]
        if not state["frozen_body_unchanged"]:
            raise RuntimeError("frozen body digest changed")
        state["wall_seconds"] = time.monotonic() - started
        state["status"] = "FIT_SUCCESS" if state["reason"] == "fit_success" else "INCONCLUSIVE"
        save_boundary(output / f"endpoint-{completed:04d}", model, optimizer, state, fingerprint)
        write(output / "result.json", state)
        write(output / "progress.json", state)
        return state
    except BaseException as error:
        failed_phase = state["update_phase"]
        state["error"] = {"type": type(error).__name__, "message": str(error), "phase": failed_phase}
        state["status"] = "INCONCLUSIVE"
        state["wall_seconds"] = time.monotonic() - started
        try:
            state["frozen_body_after"] = frozen_digest(model)
            state["frozen_body_unchanged"] = before == state["frozen_body_after"]
        except Exception as digest_error:
            state["frozen_body_check_error"] = str(digest_error)
        if failed_phase in ("backward", "boundary"):
            # No optimizer mutation occurred in an abandoned backward family.
            state["update_phase"] = "boundary"
            save_boundary(output / f"failure-{state['completed_updates']:04d}", model, optimizer, state, fingerprint)
        write(output / "failure.json", state)
        raise


def validate_output(output, checkpoint):
    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    if output.exists() or output == checkpoint or checkpoint in output.parents or output in checkpoint.parents:
        raise ValueError("choose a new output outside the immutable original checkpoint")


def tree_hashes(path):
    path = Path(path)
    return {str(file.relative_to(path)): sha(file) for file in sorted(path.rglob("*")) if file.is_file()}


def make_fingerprint(data, checkpoint, prepared, *, max_updates, max_training_seconds):
    sources = (
        "experiments/architecture_fit.py", "experiments/architecture_diagnostics.py",
        "experiments/contrast_scaling_training.py", "experiments/judgment_pipeline.py",
        "experiments/contrast_factorial_reporting.py", "experiments/revised_metrics.py",
        "src/openjev/judgment_training.py", "src/openjev/judgment_model.py",
        "src/openjev/judgment_cli.py", "src/openjev/judgments.py", "src/openjev/runtime.py",
        "reports/architecture-diagnostics-v1/PROTOCOL.md",
    )
    serialized = [{"id": b.id, "prompts": b.prompts, "kinds": b.kinds,
                   "groups": [vars(group) for group in b.groups], "relations": b.relations} for b in prepared]
    return {
        "study": "architecture-diagnostics-v1/D3", "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_files": tree_hashes(checkpoint),
        "data_files": {name: sha(Path(data) / name) for name in
                       [*(f"{split}.json" for split in SPLITS), "fit-bundles.jsonl", "manifest.json"]},
        "source_files": {name: sha(ROOT / name) for name in sources},
        "prepared_sha256": hashlib.sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest(),
        "settings": SETTINGS, "max_updates": max_updates, "max_training_seconds": max_training_seconds,
        "runtime": _package_versions(),
    }


def validate_population(records, suites):
    if len(records) != 10:
        raise ValueError("fit requires ten complete training families")
    identities = [record["id"] for record in records]
    validate_family_coverage(records, identities)
    families = {}
    for split in SPLITS:
        cases = suites[split]["cases"]
        questions = sum(len(case["request"]["questions"]) for case in cases)
        if questions != (60 if split == "retention" else 200):
            raise ValueError(f"{split} judgment population differs from protocol")
        families[split] = {case["family_id"] for case in cases}
        if split != "retention" and (len(cases) != 40 or len(families[split]) != 10):
            raise ValueError(f"{split} requires ten four-case families")
        if suites[split].get("set_relations"):
            raise ValueError("candidate-set sentinels cannot enter the fit or holdout cohorts")
    if any(families[left] & families[right] for left, right in
           (("fit", "calibration"), ("fit", "development"), ("calibration", "development"))):
        raise ValueError("fit, calibration and development families must be disjoint")
    if families["fit"] != set(identities):
        raise ValueError("fit suite and training family IDs differ")
    by_case = {case["id"]: case for case in suites["fit"]["cases"]}
    members = set()
    for record in records:
        for example in record["examples"]:
            case = by_case[example["case_id"]]
            qid = example["question_id"]
            question = example["question"]
            expected = case["expected"][qid]
            key = {"noul": "truth", "choice": "choice", "score": "level_index"}[question["type"]]
            if (case["family_id"] != record["id"] or case["request"]["state"] != example["state"]
                    or case["request"]["questions"][qid] != question or example["target"] != {key: expected}):
                raise ValueError("fit record and evaluation question/state/target differ")
            members.add((case["id"], qid))
    if members != {(case["id"], qid) for case in by_case.values() for qid in case["request"]["questions"]}:
        raise ValueError("fit records omit evaluation members")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/architecture-diagnostics-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/architecture-diagnostics-v1/tiny-fit")
    parser.add_argument("--max-updates", type=int, default=80)
    parser.add_argument("--max-training-seconds", type=float, default=900)
    parser.add_argument("--resume", type=Path, help="complete diagnostic boundary; requires a new output directory")
    args = parser.parse_args()
    try:
        validate_limits(args.max_updates, args.max_training_seconds)
        validate_output(args.output, ROOT / CHECKPOINT)
    except ValueError as error:
        parser.error(str(error))
    from experiments.architecture_diagnostics import GPU_lock, load_suite

    suites = {split: load_suite(args.data / f"{split}.json") for split in SPLITS}
    records = load_bundles(args.data / "fit-bundles.jsonl")
    validate_population(records, suites)
    manifest = json.loads((args.data / "manifest.json").read_text())
    for split in SPLITS:
        if sha(args.data / f"{split}.json") != manifest["splits"][split]["sha256"]:
            raise ValueError("suite differs from fixed cohort manifest")
    if ([record["id"] for record in records] != manifest["fit_bundles"]["ordered_ids"]
            or sha(args.data / "fit-bundles.jsonl") != manifest["fit_bundles"]["sha256"]):
        raise ValueError("fit bundles/order differ from fixed cohort manifest")
    # Enforce cached-only loading before transformers/huggingface_hub are imported.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with GPU_lock():
        apply_determinism(42)
        original = tree_hashes(ROOT / CHECKPOINT)
        engine = load_judgment_engine(checkpoint=ROOT / CHECKPOINT, device="cuda", backend="fla",
                                      unit_batch_size=4, max_input_tokens=1536, trainable=True)
        prepared = [prepare_bundle(engine, record) for record in records]
        fingerprint = make_fingerprint(args.data, ROOT / CHECKPOINT, prepared,
                                       max_updates=args.max_updates, max_training_seconds=args.max_training_seconds)
        args.output.mkdir(parents=True)
        write(args.output / "fingerprint.json", fingerprint)
        try:
            result = run_fit(engine, prepared, suites, args.output, fingerprint,
                             max_updates=args.max_updates, max_training_seconds=args.max_training_seconds,
                             resume=args.resume)
        finally:
            unchanged = original == tree_hashes(ROOT / CHECKPOINT)
            write(args.output / "original-checkpoint-integrity.json", {"unchanged": unchanged, "before": original})
            if not unchanged:
                raise RuntimeError("original H0 checkpoint changed during diagnostic")
        print(json.dumps({key: result[key] for key in ("status", "reason", "completed_updates", "training_seconds")},
                         allow_nan=False))


if __name__ == "__main__":
    main()
