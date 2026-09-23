"""Locked calibration, assessment, and reporting for contrast-factorial-v1.

The module deliberately separates endpoint validation, calibration locking, and
assessment reads.  Production defaults load one FLA judgment engine at a time;
tests may supply a tiny engine factory through the same callable boundary.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib
import json
import math
import os
import weakref
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from experiments.contrast_factorial_reporting import estimate_factorial_effects, summarize_factorial
from experiments.revised_evaluation import broad_suite
from experiments.revised_metrics import prediction_view, relation_metrics, validate_suite
from experiments.temperature_diagnostic import balanced_weights, calibrate_record, fit_temperature
from openjev.judgment_cli import load_judgment_engine
from openjev.judgment_model import checkpoint_identity

STUDY = "contrast-factorial-v1"
MODELS = ("BASE", "A", "B", "C", "D")
ARMS = tuple("ABCD")
PRIMITIVES = ("noul", "choice", "score")
MAX_INPUT_TOKENS = 1536
UNIT_BATCH_SIZE = 12
LOCK_FD_ENV = "OPENJEV_CONTRAST_FACTORIAL_LOCK_FD"
REPORTING_PROTOCOL = {
    "primary": "category-macro family contrast both-correct",
    "axes": ["evidence", "rubric", "question"],
    "invariants": "separate",
    "calibration_variants": ["raw", "global", "per_primitive"],
    "bootstrap_samples": 2000,
    "bootstrap_seed": 42,
    "ci": 0.95,
    "temperature_bounds": [0.05, 100.0],
    "selection": "all endpoints are complete update-400 checkpoints; no assessment selection",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path):
    digest = hashlib.sha256()
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"checkpoint directory is missing: {root}")
    files = [item for item in sorted(root.rglob("*")) if item.is_file()]
    if not files:
        raise ValueError(f"checkpoint directory is empty: {root}")
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pending-{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def study_gpu_lock(output_root):
    """Hold the one study lock, or verify a lock descriptor inherited from the launcher."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "gpu.lock"
    inherited = os.environ.get(LOCK_FD_ENV)
    if inherited is not None:
        try:
            descriptor = int(inherited)
            descriptor_stat = os.fstat(descriptor)
            path_stat = path.stat()
        except (OSError, ValueError) as error:
            raise RuntimeError("inherited study GPU lock descriptor is invalid") from error
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
            raise RuntimeError("inherited study GPU lock descriptor points to a different file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("inherited study GPU lock is not exclusively owned") from error
        yield path
        return
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("study GPU lock is already held by another process") from error
        yield path


def _read_json(path):
    return json.loads(Path(path).read_text())


def _repo_path(repo_root, relative, label):
    repo_root = Path(repo_root).resolve()
    value = Path(relative)
    if value.is_absolute():
        raise ValueError(f"{label} must be repository-relative")
    path = (repo_root / value).resolve()
    if path != repo_root and repo_root not in path.parents:
        raise ValueError(f"{label} escapes repository")
    return path


def _verify_file(repo_root, spec, digest, label):
    path = _repo_path(repo_root, spec, label)
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{label} hash mismatch")
    return path


def _checkpoint_info(path, *, model_id, revision):
    import torch
    from safetensors.torch import load_file

    path = Path(path)
    try:
        info = _read_json(path / "checkpoint.json")
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid checkpoint metadata: {path}") from error
    if info.get("format") != "openjev-judgment-v0.2":
        raise ValueError("endpoint checkpoint format mismatch")
    if info.get("model_id") != model_id or info.get("revision") != revision:
        raise ValueError("endpoint base model identity mismatch")
    identity = checkpoint_identity(path, info)
    if info.get("checkpoint_id") != identity:
        raise ValueError("endpoint checkpoint identity mismatch")
    lora = info.get("lora")
    if not isinstance(lora, dict) or lora.get("rank") != 8 or lora.get("alpha") != 16:
        raise ValueError("endpoint checkpoint LoRA contract mismatch")
    if not isinstance(lora.get("targets"), list) or not lora["targets"]:
        raise ValueError("endpoint checkpoint LoRA targets are missing")
    readout_path = path / "readouts.safetensors"
    adapter_path = path / "adapter" / "adapter_model.safetensors"
    adapter_config_path = path / "adapter" / "adapter_config.json"
    if not readout_path.is_file() or not adapter_path.is_file() or not adapter_config_path.is_file():
        raise ValueError("endpoint checkpoint is missing readout or adapter weights")
    try:
        readouts = load_file(str(readout_path), device="cpu")
        adapters = load_file(str(adapter_path), device="cpu")
        adapter_config = _read_json(adapter_config_path)
    except (OSError, ValueError, RuntimeError) as error:
        raise ValueError("endpoint checkpoint weight files are invalid") from error
    expected_readouts = {"compatibility.weight", "binary.weight", "binary.bias"}
    if set(readouts) != expected_readouts:
        raise ValueError("endpoint checkpoint readout tensor inventory mismatch")
    hidden = readouts["compatibility.weight"].shape
    if (len(hidden) != 2 or hidden[0] != 1 or readouts["binary.weight"].shape != hidden
            or readouts["binary.bias"].shape != (1,)):
        raise ValueError("endpoint checkpoint readout tensor shapes are invalid")
    if not adapters or any(".lora_A." not in key and ".lora_B." not in key for key in adapters):
        raise ValueError("endpoint checkpoint adapter tensor inventory is invalid")
    a_keys = {key.replace(".lora_A.", ".lora_.") for key in adapters if ".lora_A." in key}
    b_keys = {key.replace(".lora_B.", ".lora_.") for key in adapters if ".lora_B." in key}
    if not a_keys or a_keys != b_keys:
        raise ValueError("endpoint checkpoint adapter A/B tensors are not paired")
    for prefix in a_keys:
        a = adapters[prefix.replace(".lora_.", ".lora_A.")]
        b = adapters[prefix.replace(".lora_.", ".lora_B.")]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != 8 or b.shape[1] != 8:
            raise ValueError("endpoint checkpoint adapter tensor shapes are invalid")
    tensors = {**readouts, **adapters}
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() or value.numel() < 1
           for value in tensors.values()):
        raise ValueError("endpoint checkpoint trainable tensors must be finite FP32")
    if (adapter_config.get("r") != 8 or adapter_config.get("lora_alpha") != 16
            or set(adapter_config.get("target_modules", [])) != set(lora["targets"])):
        raise ValueError("endpoint checkpoint adapter config differs from checkpoint metadata")
    return info, {key: tuple(value.shape) for key, value in tensors.items()}


def _validate_optimizer_payload(payload, trainable_shapes, completed):
    import torch

    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise ValueError("endpoint Adam state is missing")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 2:
        raise ValueError("endpoint Adam parameter groups are missing")
    common = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01, "amsgrad": False,
              "maximize": False, "foreach": None, "capturable": False, "differentiable": False,
              "fused": None, "decoupled_weight_decay": True}
    expected_groups = ({**common, "lr": 5e-5}, {**common, "lr": 2.5e-5})
    parameter_ids = []
    expected_shapes = (
        Counter(shape for name, shape in trainable_shapes.items() if ".lora_" in name),
        Counter(shape for name, shape in trainable_shapes.items()
                if name in {"compatibility.weight", "binary.weight", "binary.bias"}),
    )
    for index, (group, expected) in enumerate(zip(groups, expected_groups, strict=True)):
        if not isinstance(group, dict) or not isinstance(group.get("params"), list) or not group["params"]:
            raise ValueError("endpoint Adam parameter group is invalid")
        if any(group.get(key) != value for key, value in expected.items()):
            raise ValueError("endpoint AdamW parameter-group recipe mismatch")
        state_shapes = Counter(tuple(optimizer["state"][identity]["exp_avg"].shape)
                               for identity in group["params"] if identity in optimizer["state"])
        if state_shapes != expected_shapes[index]:
            raise ValueError("endpoint Adam parameter groups do not match adapter/head tensors")
        parameter_ids.extend(group["params"])
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != set(optimizer["state"]):
        raise ValueError("endpoint Adam parameter coverage is incomplete")
    if len(parameter_ids) != len(trainable_shapes):
        raise ValueError("endpoint Adam state does not cover every saved trainable tensor")
    optimizer_shapes = []
    for identity in parameter_ids:
        state = optimizer["state"][identity]
        if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise ValueError("endpoint Adam moment state is incomplete")
        step = state["step"]
        raw_step = step.item() if isinstance(step, torch.Tensor) and step.numel() == 1 else step
        if type(raw_step) not in (int, float) or not math.isfinite(raw_step) or raw_step != completed:
            raise ValueError("endpoint Adam step counter differs from completed updates")
        first, second = state["exp_avg"], state["exp_avg_sq"]
        if (not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor)
                or first.shape != second.shape or first.dtype != torch.float32 or second.dtype != torch.float32
                or not torch.isfinite(first).all() or not torch.isfinite(second).all()):
            raise ValueError("endpoint Adam moments are invalid")
        optimizer_shapes.append(tuple(first.shape))
    if Counter(optimizer_shapes) != Counter(trainable_shapes.values()):
        raise ValueError("endpoint Adam moment shapes differ from saved trainable tensors")
    cpu_rng = payload.get("torch_rng")
    cuda_rng = payload.get("cuda_rng")
    if not isinstance(cpu_rng, torch.Tensor) or cpu_rng.dtype != torch.uint8 or cpu_rng.numel() < 1:
        raise ValueError("endpoint CPU RNG state is invalid")
    if (not isinstance(cuda_rng, list) or not cuda_rng
            or any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.numel() < 1
                   for value in cuda_rng)):
        raise ValueError("endpoint CUDA RNG state is missing or invalid")


def _validate_approval(repo_root, approval_path):
    approval_path = Path(approval_path).resolve()
    approval = _read_json(approval_path)
    required = {
        "base_checkpoint", "base_checkpoint_sha256", "base_checkpoint_id", "model_id", "revision",
        "evaluation_freeze_path", "evaluation_freeze_sha256", "contract_path", "contract_sha256",
        "expected_frozen_body_sha256", "arms", "updates_per_arm",
    }
    if approval.get("study") != STUDY or approval.get("status") != "approved" or not required.issubset(approval):
        raise ValueError("STUDY_APPROVAL is missing the approved study identity")
    if approval["arms"] != list(ARMS) or approval["updates_per_arm"] != 400:
        raise ValueError("STUDY_APPROVAL does not authorize four 400-update arms")
    freeze = _verify_file(repo_root, approval["evaluation_freeze_path"], approval["evaluation_freeze_sha256"], "freeze")
    contract = _verify_file(repo_root, approval["contract_path"], approval["contract_sha256"], "contract")
    base = _repo_path(repo_root, approval["base_checkpoint"], "base checkpoint")
    if tree_sha256(base) != approval["base_checkpoint_sha256"]:
        raise ValueError("approved base checkpoint tree hash mismatch")
    info, _ = _checkpoint_info(base, model_id=approval["model_id"], revision=approval["revision"])
    if info["checkpoint_id"] != approval["base_checkpoint_id"]:
        raise ValueError("approved base checkpoint ID mismatch")
    freeze_value = _read_json(freeze)
    if freeze_value.get("study") != STUDY or freeze_value.get("status") != "frozen":
        raise ValueError("evaluation freeze is not frozen for this study")
    if freeze_value.get("unresolved_review_flags") != 0 or freeze_value.get("no_model_predictions_used") is not True:
        raise ValueError("evaluation freeze review status is invalid")
    return approval, freeze_value, base, freeze, contract


def _verify_manifest(repo_root, manifest_path, manifest, approval, approval_path):
    if manifest.get("study") != STUDY or set(manifest.get("arms", {})) != set(ARMS):
        raise ValueError("training manifest requires exactly A/B/C/D")
    if manifest.get("ordered_latent_ids") is None or len(manifest["ordered_latent_ids"]) != 400:
        raise ValueError("training manifest requires 400 ordered latent families")
    if len(set(manifest["ordered_latent_ids"])) != 400:
        raise ValueError("training manifest latent families are not unique")
    bindings = (
        ("base_checkpoint", "base_checkpoint_sha256"),
        ("evaluation_freeze_path", "evaluation_freeze_sha256"),
    )
    for path_key, hash_key in bindings:
        if manifest.get(path_key) != approval[path_key] or manifest.get(hash_key) != approval[hash_key]:
            raise ValueError(f"manifest {path_key} differs from STUDY_APPROVAL")
    if manifest.get("study_approval_path") != Path(approval_path).resolve().relative_to(Path(repo_root).resolve()).as_posix():
        raise ValueError("manifest does not identify STUDY_APPROVAL")
    if manifest.get("study_approval_sha256") != sha256_file(approval_path):
        raise ValueError("manifest STUDY_APPROVAL hash mismatch")
    for mapping_name in ("source_fingerprints", "data_fingerprint"):
        mapping = manifest.get(mapping_name)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"manifest {mapping_name} is missing")
        for relative, digest in mapping.items():
            _verify_file(repo_root, relative, digest, mapping_name)
    ids = manifest["ordered_latent_ids"]
    for arm in ARMS:
        spec = manifest["arms"][arm]
        if spec.get("ordered_family_ids") != ids:
            raise ValueError(f"arm {arm} family order differs")
        _verify_file(repo_root, spec.get("train_file"), spec.get("sha256"), f"arm {arm} train")
        _verify_file(repo_root, spec.get("suite_file"), spec.get("suite_sha256"), f"arm {arm} suite")
    if not Path(manifest_path).is_file():
        raise ValueError("training manifest is missing")


def _history_totals(history):
    return {
        "candidate_units": sum(row["exposure"]["candidate_units"] for row in history),
        "real_tokens": sum(row["exposure"]["unpadded_tokens"] for row in history),
        "padded_tokens": sum(row["exposure"]["padded_tokens"] for row in history),
        "wall_time_seconds": history[-1].get("total_time_seconds") if history else 0,
        "peak_memory_bytes": max((row.get("peak_memory_bytes", 0) for row in history), default=0),
    }


def _validate_start_evidence(value, approval):
    if not isinstance(value, dict) or set(value) != {
        "checkpoint_id", "trainable_weights_sha256", "cpu_rng_sha256", "cuda_rng_sha256",
    }:
        raise ValueError("arm start evidence is incomplete")
    if value["checkpoint_id"] != approval["base_checkpoint_id"]:
        raise ValueError("arm start checkpoint differs from the approved base")
    for key in ("trainable_weights_sha256", "cpu_rng_sha256"):
        if not isinstance(value[key], str) or len(value[key]) != 64:
            raise ValueError(f"arm start {key} is invalid")
    if (not isinstance(value["cuda_rng_sha256"], list) or not value["cuda_rng_sha256"]
            or any(not isinstance(digest, str) or len(digest) != 64 for digest in value["cuda_rng_sha256"])):
        raise ValueError("arm start CUDA RNG evidence is invalid")
    return value


def _validate_arm_endpoint(path, *, arm, manifest, manifest_sha, approval, coordinator_start):
    from experiments.contrast_factorial_training import _checkpoint_state, _history_hash

    path = Path(path).resolve()
    if path.name != "step-0400":
        raise ValueError(f"arm {arm} endpoint is not the update-400 checkpoint")
    payload = _checkpoint_state(path)
    state = payload["state"]
    if any(state.get(key) != value for key, value in {
        "study": STUDY, "arm": arm, "status": "complete", "completed_updates": 400,
        "next_position": 400, "complete_boundary": True, "resumable": True,
        "update_phase": "boundary", "frozen_verified_at": 400,
    }.items()):
        raise ValueError(f"arm {arm} is not a complete update-400 endpoint")
    history = state.get("history")
    if not isinstance(history, list) or len(history) != 400 or [row.get("step") for row in history] != list(range(1, 401)):
        raise ValueError(f"arm {arm} history is not contiguous through update 400")
    if state.get("history_sha256") != _history_hash(history):
        raise ValueError(f"arm {arm} history hash mismatch")
    frozen = approval["expected_frozen_body_sha256"]
    if state.get("frozen_before") != frozen or state.get("frozen_after") != frozen:
        raise ValueError(f"arm {arm} frozen body differs from approval")
    fp = state.get("fingerprint", {})
    arm_spec = manifest["arms"][arm]
    expected = {
        "study": STUDY, "arm": arm, "manifest_sha256": manifest_sha,
        "base_checkpoint_sha256": approval["base_checkpoint_sha256"],
        "evaluation_freeze_sha256": approval["evaluation_freeze_sha256"],
        "train_sha256": arm_spec["sha256"], "suite_sha256": arm_spec["suite_sha256"],
        "source_fingerprints": manifest["source_fingerprints"], "data_fingerprint": manifest["data_fingerprint"],
        "backend": "fla", "max_input_tokens": 1536, "max_units": 12, "unit_batch_size": 12,
    }
    if any(fp.get(key) != value for key, value in expected.items()):
        raise ValueError(f"arm {arm} training fingerprint mismatch")
    determinism = {"seed": 42, "deterministic_algorithms": True, "cudnn_deterministic": True,
                   "cudnn_benchmark": False, "tf32": False, "cublas_workspace_config": ":4096:8"}
    if fp.get("determinism") != determinism:
        raise ValueError(f"arm {arm} determinism fingerprint mismatch")
    if state.get("arm_start") != coordinator_start:
        raise ValueError(f"arm {arm} checkpoint start evidence differs from coordinator")
    for field in ("schedule_sha256", "replay_position_sha256", "replay_sha256", "prepared_prompt_fingerprint"):
        if not isinstance(fp.get(field), str) or len(fp[field]) != 64:
            raise ValueError(f"arm {arm} fingerprint lacks {field}")
    production_sources = fp.get("production_source_sha256")
    if not isinstance(production_sources, dict) or not production_sources:
        raise ValueError(f"arm {arm} production source fingerprints are invalid")
    for relative, digest in production_sources.items():
        _verify_file(Path(approval["_repo_root"]), relative, digest, f"arm {arm} production source")
    expected_counts = {"new/noul": 12, "new/choice": 4, "new/score": 4,
                       "old/noul": 1, "old/choice": 1, "old/score": 1}
    for position, row in enumerate(history):
        exposure = row.get("exposure", {})
        if row.get("component_counts") != expected_counts:
            raise ValueError(f"arm {arm} update {position + 1} scalar counts differ")
        if exposure.get("family_id") != manifest["ordered_latent_ids"][position]:
            raise ValueError(f"arm {arm} family exposure differs at position {position}")
        if set(exposure.get("replay_ids", {})) != set(PRIMITIVES):
            raise ValueError(f"arm {arm} replay exposure is incomplete")
        for key in ("candidate_units", "unpadded_tokens", "padded_tokens"):
            if type(exposure.get(key)) is not int or exposure[key] < 1:
                raise ValueError(f"arm {arm} exposure {key} is invalid")
        if exposure["padded_tokens"] < exposure["unpadded_tokens"]:
            raise ValueError(f"arm {arm} padded token exposure is invalid")
    info, trainable_shapes = _checkpoint_info(path, model_id=approval["model_id"], revision=approval["revision"])
    _validate_optimizer_payload(payload, trainable_shapes, completed=400)
    return {
        "checkpoint": str(path), "checkpoint_id": info["checkpoint_id"], "tree_sha256": tree_sha256(path),
        "history_sha256": state["history_sha256"], "fingerprint_sha256": canonical_hash(fp),
        "fingerprint": fp, "exposure": _history_totals(history),
        "arm_start": coordinator_start,
        "family_ids": [row["exposure"]["family_id"] for row in history],
        "replay_ids": [row["exposure"]["replay_ids"] for row in history],
    }


def _partition_specs(repo_root, freeze, legacy_manifest_path):
    suites = freeze.get("suites", {})
    if set(suites) != {"assessment", "calibration"}:
        raise ValueError("freeze must identify assessment and calibration")
    legacy_manifest_path = Path(legacy_manifest_path).resolve()
    legacy = _read_json(legacy_manifest_path)
    legacy_source = _repo_path(repo_root, legacy.get("source", ""), "legacy validation source")
    if (legacy_source.name != "validation.jsonl"
            or any(part.lower() in {"test", "tests", "finaltest"} for part in legacy_source.parts)
            or not legacy.get("selection")):
        raise ValueError("legacy manifest does not identify a validation-only source")
    if not legacy_source.is_file() or sha256_file(legacy_source) != legacy.get("source_sha256"):
        raise ValueError("legacy validation source hash mismatch")
    result = {}
    for name in ("assessment", "calibration"):
        spec = suites[name]
        expected = {"assessment": (80, 320, 1600), "calibration": (40, 160, 800)}[name]
        if tuple(spec.get(key) for key in ("families", "cases", "judgments")) != expected:
            raise ValueError(f"{name} frozen population declaration is invalid")
        _verify_file(repo_root, spec["path"], spec["suite_sha256"], name)
        result[name] = {"path": spec["path"], "sha256": spec["suite_sha256"],
                        **{key: spec.get(key) for key in ("families", "cases", "judgments")}}
    for source, name in (("calibration", "legacy_calibration"), ("retention", "legacy_retention")):
        spec = legacy.get("splits", {}).get(source, {})
        expected_records = 150 if source == "calibration" else 300
        if spec.get("records") != expected_records or len(spec.get("ids", [])) != expected_records:
            raise ValueError(f"{name} population declaration is invalid")
        _verify_file(repo_root, spec.get("file"), spec.get("sha256"), name)
        result[name] = {"path": spec["file"], "sha256": spec["sha256"], "records": spec.get("records"),
                        "record_ids_sha256": canonical_hash(spec.get("ids", [])) if spec.get("ids") else None}
    result["legacy_manifest"] = {
        "path": legacy_manifest_path.relative_to(Path(repo_root).resolve()).as_posix(),
        "sha256": sha256_file(legacy_manifest_path),
    }
    return result


def create_endpoint_lock(*, repo_root, approval_path, manifest_path, training_root, output_root,
                         legacy_manifest_path):
    """Validate the five endpoints and atomically bind them before inference."""
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root).resolve()
    lock_path = output_root / "endpoint-lock.json"
    if lock_path.exists():
        existing = _load_lock(lock_path, kind="endpoint")
        expected_paths = {
            "repo_root": str(repo_root), "training_root": str(Path(training_root).resolve()),
        }
        if any(existing.get(key) != value for key, value in expected_paths.items()):
            raise ValueError("existing endpoint lock belongs to different inputs")
        if existing.get("approval", {}).get("path") != str(Path(approval_path).resolve()):
            raise ValueError("existing endpoint lock belongs to a different approval")
        if existing.get("manifest", {}).get("path") != str(Path(manifest_path).resolve()):
            raise ValueError("existing endpoint lock belongs to a different manifest")
        legacy = existing.get("partitions", {}).get("legacy_manifest", {})
        expected_legacy = Path(legacy_manifest_path).resolve().relative_to(repo_root).as_posix()
        if legacy.get("path") != expected_legacy:
            raise ValueError("existing endpoint lock belongs to a different legacy manifest")
        _verify_endpoint_files(existing)
        return existing
    approval, freeze, base, _, _ = _validate_approval(repo_root, approval_path)
    approval["_repo_root"] = str(repo_root)
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    _verify_manifest(repo_root, manifest_path, manifest, approval, approval_path)
    training_root = Path(training_root).resolve()
    coordinator = _read_json(training_root / "status.json")
    if coordinator.get("study") != STUDY or coordinator.get("status") != "complete" or coordinator.get("completed_arms") != list(ARMS):
        raise ValueError("four-arm coordinator is not complete")
    arm_starts = coordinator.get("arm_starts")
    if not isinstance(arm_starts, dict) or set(arm_starts) != set(ARMS):
        raise ValueError("coordinator arm start evidence is missing")
    for arm in ARMS:
        _validate_start_evidence(arm_starts[arm], approval)
    if any(arm_starts[arm] != arm_starts["A"] for arm in ARMS[1:]):
        raise ValueError("coordinator arm starts differ")
    manifest_sha = sha256_file(manifest_path)
    endpoints = {
        "BASE": {"checkpoint": str(base), "checkpoint_id": approval["base_checkpoint_id"],
                 "tree_sha256": approval["base_checkpoint_sha256"], "exposure": None}
    }
    for arm in ARMS:
        summary = coordinator.get("summaries", {}).get(arm, {})
        checkpoint = summary.get("checkpoint")
        if summary.get("status") != "complete" or summary.get("completed_updates") != 400 or not checkpoint:
            raise ValueError(f"coordinator summary for arm {arm} is incomplete")
        endpoint = _validate_arm_endpoint(checkpoint, arm=arm, manifest=manifest,
                                          manifest_sha=manifest_sha, approval=approval,
                                          coordinator_start=arm_starts[arm])
        progress = _read_json(training_root / arm / "progress.json")
        for field in ("status", "completed_updates", "next_position", "history_sha256", "frozen_before", "frozen_after"):
            expected = "complete" if field == "status" else (400 if field in {"completed_updates", "next_position"}
                                                               else _read_json(Path(checkpoint) / "state.json")[field])
            if progress.get(field) != expected:
                raise ValueError(f"arm {arm} progress disagrees with checkpoint")
        endpoints[arm] = endpoint
    family_rows = {arm: endpoints[arm]["family_ids"] for arm in ARMS}
    replay_rows = {arm: endpoints[arm]["replay_ids"] for arm in ARMS}
    if any(family_rows[arm] != family_rows["A"] for arm in ARMS[1:]):
        raise ValueError("arm family exposures are not position-matched")
    if any(replay_rows[arm] != replay_rows["A"] for arm in ARMS[1:]):
        raise ValueError("arm replay exposures are not position-matched")
    for arm in ARMS:
        endpoints[arm].pop("family_ids")
        endpoints[arm].pop("replay_ids")
    partitions = _partition_specs(repo_root, freeze, legacy_manifest_path)
    reporting_source = Path(__file__).with_name("contrast_factorial_reporting.py")
    lock = {
        "version": 1, "study": STUDY, "kind": "endpoint", "status": "locked", "created_utc": _utc(),
        "lock_path": str(lock_path.resolve()),
        "repo_root": str(repo_root), "approval": {"path": str(Path(approval_path).resolve()),
                                                     "sha256": sha256_file(approval_path)},
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "training_root": str(training_root), "gpu_lock": str((output_root / "gpu.lock").resolve()),
        "endpoints": endpoints, "partitions": partitions,
        "common_start": arm_starts["A"],
        "training_protocol": {"updates_per_arm": 400, "new_judgments_per_update": 20,
                              "legacy_judgments_per_update": 3,
                              "scalar_groups": ["new/noul", "new/choice", "new/score",
                                                "old/noul", "old/choice", "old/score"],
                              "scalar_group_weight": "1/6 each; bound by audited production source hashes"},
        "reporting": {"source": str(reporting_source.resolve()), "sha256": sha256_file(reporting_source),
                      "protocol": REPORTING_PROTOCOL},
    }
    lock["lock_sha256"] = canonical_hash(lock)
    _atomic_json(lock_path, lock)
    return lock


def _load_lock(path, *, kind):
    path = Path(path)
    value = _read_json(path)
    digest = value.get("lock_sha256")
    body = {key: item for key, item in value.items() if key != "lock_sha256"}
    if value.get("study") != STUDY or value.get("kind") != kind or value.get("status") != "locked":
        raise ValueError(f"invalid {kind} lock")
    if digest != canonical_hash(body):
        raise ValueError(f"{kind} lock hash mismatch")
    return value


def _locked_stage(stage):
    return {key: stage[key] for key in (
        "status", "origin", "binding", "binding_sha256", "completed_cases", "judgments",
        "max_unit_tokens", "artifacts", "directory",
    )}


def _validate_completed_stage(stage, expected_binding, *, origin, label):
    required_artifacts = {"responses.jsonl", "predictions.jsonl", "predictions.json", "calibration-records.json"}
    if not isinstance(stage, dict) or stage.get("status") != "complete" or stage.get("origin") != origin:
        raise ValueError(f"{label} completed stage is missing")
    if stage.get("binding") != expected_binding or stage.get("binding_sha256") != canonical_hash(expected_binding):
        raise ValueError(f"{label} stage binding mismatch")
    directory = Path(stage.get("directory", ""))
    state_path = directory / "stage.json"
    if not state_path.is_file():
        raise ValueError(f"{label} stage state is missing")
    disk = _read_json(state_path)
    if (disk.get("status") != "complete" or disk.get("origin") != origin
            or disk.get("binding") != expected_binding or disk.get("binding_sha256") != canonical_hash(expected_binding)):
        raise ValueError(f"{label} on-disk stage binding mismatch")
    for key in ("completed_cases", "judgments", "max_unit_tokens"):
        if stage.get(key) != disk.get(key):
            raise ValueError(f"{label} stage completion metadata mismatch")
    artifacts = stage.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != required_artifacts or disk.get("artifacts") != artifacts:
        raise ValueError(f"{label} stage artifact inventory mismatch")
    for name, digest in artifacts.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"{label} calibration/prediction artifact hash mismatch")
    return _read_json(directory / "calibration-records.json")


def _validate_calibration_lock(endpoint, calibration):
    if calibration.get("endpoint_lock_sha256") != endpoint["lock_sha256"]:
        raise ValueError("calibration lock endpoint linkage mismatch")
    if set(calibration.get("models", {})) != set(MODELS):
        raise ValueError("calibration lock model population mismatch")
    expected_protocol = {"objective": "0.5 new macro-primitive NLL + 0.5 broad macro-primitive NLL",
                         "variants": ["global", "per_primitive"], "bounds": [0.05, 100.0]}
    if calibration.get("protocol") != expected_protocol:
        raise ValueError("calibration lock protocol mismatch")
    fits = {}
    for model in MODELS:
        endpoint_spec = endpoint["endpoints"][model]
        spec = calibration["models"][model]
        checkpoint_id = endpoint_spec["checkpoint_id"]
        if spec.get("checkpoint_id") != checkpoint_id:
            raise ValueError(f"calibration lock {model} checkpoint identity mismatch")
        common = {"endpoint_lock_sha256": endpoint["lock_sha256"], "model": model,
                  "checkpoint_id": checkpoint_id}
        stages = spec.get("stages", {})
        if set(stages) != {"new", "broad"}:
            raise ValueError(f"calibration lock {model} selected stages are missing")
        new_records = _validate_completed_stage(
            stages["new"], {**common, "partition": endpoint["partitions"]["calibration"]["sha256"]},
            origin="new", label=f"{model} new calibration",
        )
        broad_records = _validate_completed_stage(
            stages["broad"], {**common, "partition": endpoint["partitions"]["legacy_calibration"]["sha256"]},
            origin="broad", label=f"{model} broad calibration",
        )
        fit_path = Path(spec.get("fit_path", ""))
        if not fit_path.is_file() or sha256_file(fit_path) != spec.get("fit_sha256"):
            raise ValueError(f"locked {model} temperature fit hash mismatch")
        fit = _read_json(fit_path)
        expected_partitions = {"new": endpoint["partitions"]["calibration"]["sha256"],
                               "broad": endpoint["partitions"]["legacy_calibration"]["sha256"]}
        expected_predictions = {"new": stages["new"]["artifacts"]["calibration-records.json"],
                                "broad": stages["broad"]["artifacts"]["calibration-records.json"]}
        if (fit.get("study") != STUDY or fit.get("model") != model or fit.get("checkpoint_id") != checkpoint_id
                or fit.get("endpoint_lock_sha256") != endpoint["lock_sha256"]
                or fit.get("source_partition_sha256") != expected_partitions
                or fit.get("prediction_sha256") != expected_predictions
                or spec.get("prediction_sha256") != expected_predictions
                or fit.get("temperatures") != spec.get("temperatures")):
            raise ValueError(f"locked {model} temperature fit binding mismatch")
        if len(new_records) + len(broad_records) != fit["global"]["fit_records"]:
            raise ValueError(f"locked {model} temperature fit population mismatch")
        recomputed = _fit(new_records + broad_records)
        if any(fit.get(key) != recomputed[key] for key in ("global", "per_primitive", "temperatures")):
            raise ValueError(f"locked {model} temperature fit does not match its saved logits")
        fits[model] = fit
    return fits


def _validate_evaluation_status(endpoint, calibration, evaluation):
    if (evaluation.get("study") != STUDY or evaluation.get("status") != "complete"
            or evaluation.get("endpoint_lock_sha256") != endpoint["lock_sha256"]
            or evaluation.get("calibration_lock_sha256") != calibration["lock_sha256"]
            or evaluation.get("endpoint_lock") != endpoint.get("lock_path")
            or evaluation.get("calibration_lock") != calibration.get("lock_path")
            or set(evaluation.get("models", {})) != set(MODELS)):
        raise ValueError("evaluation status lock linkage mismatch")
    records = {}
    for model in MODELS:
        checkpoint_id = endpoint["endpoints"][model]["checkpoint_id"]
        common = {"endpoint_lock_sha256": endpoint["lock_sha256"],
                  "calibration_lock_sha256": calibration["lock_sha256"],
                  "model": model, "checkpoint_id": checkpoint_id}
        stages = evaluation["models"][model]
        if set(stages) != {"assessment", "retention"}:
            raise ValueError(f"evaluation {model} stage population mismatch")
        assessment = _validate_completed_stage(
            stages["assessment"], {**common, "partition": endpoint["partitions"]["assessment"]["sha256"]},
            origin="assessment", label=f"{model} assessment",
        )
        retention = _validate_completed_stage(
            stages["retention"], {**common, "partition": endpoint["partitions"]["legacy_retention"]["sha256"]},
            origin="retention", label=f"{model} retention",
        )
        records[model] = {"assessment": assessment, "retention": retention}
    return records


def _verify_endpoint_files(lock):
    """Recheck every locked artifact without parsing evaluation narratives."""
    if sha256_file(lock["approval"]["path"]) != lock["approval"]["sha256"]:
        raise ValueError("STUDY_APPROVAL changed after endpoint lock")
    if sha256_file(lock["manifest"]["path"]) != lock["manifest"]["sha256"]:
        raise ValueError("training manifest changed after endpoint lock")
    if sha256_file(lock["reporting"]["source"]) != lock["reporting"]["sha256"]:
        raise ValueError("reporting source changed after endpoint lock")
    approval = _read_json(lock["approval"]["path"])
    for path_key, hash_key, label in (
        ("evaluation_freeze_path", "evaluation_freeze_sha256", "evaluation freeze"),
        ("contract_path", "contract_sha256", "contract"),
    ):
        _verify_file(lock["repo_root"], approval[path_key], approval[hash_key], label)
    for name, spec in lock["partitions"].items():
        _verify_file(lock["repo_root"], spec["path"], spec["sha256"], name)
    for name, endpoint in lock["endpoints"].items():
        if tree_sha256(endpoint["checkpoint"]) != endpoint["tree_sha256"]:
            raise ValueError(f"endpoint {name} changed after locking")


def _answer_logits(question, answer):
    kind = question["type"]
    details = answer.get("details", {})
    if answer.get("type") != kind or details.get("link") != ("sigmoid" if kind == "noul" else "softmax"):
        raise ValueError("response primitive diagnostics differ from the question")
    if details.get("temperature") != 1.0 or details.get("calibration_id") != "identity-uncalibrated":
        raise ValueError("response lacks identity-uncalibrated temperature diagnostics")
    if kind == "noul":
        value = details.get("raw_logit")
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("Noul raw_logit is missing or non-finite")
        effective = details.get("calibrated_logit")
        if type(effective) not in (float, int) or not math.isclose(effective, value, rel_tol=0, abs_tol=1e-12):
            raise ValueError("identity Noul effective logit differs from raw logit")
        return [0.0, float(value)], ["false", "true"]
    raw = details.get("raw_logits")
    effective = details.get("calibrated_logits")
    labels = [str(index) for index in range(len(question["criteria"]))] if kind == "score" else list(question["criteria"])
    if not isinstance(raw, dict) or set(raw) != set(labels) or not isinstance(effective, dict) or set(effective) != set(labels):
        raise ValueError(f"{kind} raw logits differ from the answer space")
    values = [raw[label] for label in labels]
    if any(type(value) not in (float, int) or not math.isfinite(value) for value in values):
        raise ValueError(f"{kind} raw logits must be finite")
    if any(type(effective[label]) not in (float, int)
           or not math.isclose(effective[label], raw[label], rel_tol=0, abs_tol=1e-12) for label in labels):
        raise ValueError(f"identity {kind} effective logits differ from raw logits")
    return list(map(float, values)), labels


def _raw_answer(question, logits, labels):
    kind = question["type"]
    if kind == "noul":
        value = logits[1]
        tail = math.exp(-abs(value))
        probability = 1 / (1 + tail) if value >= 0 else tail / (1 + tail)
        return {"noul": probability, "details": {"raw_logit": value}}
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    probabilities = {label: weight / sum(weights) for label, weight in zip(labels, weights, strict=True)}
    result = {"probabilities": probabilities, "details": {"raw_logits": dict(zip(labels, logits, strict=True))}}
    if kind == "score":
        result["score"] = sum(int(label) * probability for label, probability in probabilities.items())
    return result


def _target(question, expected):
    if question["type"] == "noul":
        return bool(expected), int(bool(expected))
    if question["type"] == "score":
        return int(expected), int(expected)
    labels = list(question["criteria"])
    return expected, labels.index(expected)


def _append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _next_attempt(path):
    path = Path(path)
    number = 2
    while path.with_name(f"{path.name}.attempt-{number:04d}").exists():
        number += 1
    return path.with_name(f"{path.name}.attempt-{number:04d}")


def run_prediction_stage(engine, suite, output, *, binding, origin):
    """Evaluate a complete population, preserving every response and raw logit."""
    validate_suite(suite)
    output = Path(output)
    binding_hash = canonical_hash(binding)
    if output.exists():
        candidates = [output, *sorted(output.parent.glob(f"{output.name}.attempt-*"), reverse=True)]
        for candidate in candidates:
            state_path = candidate / "stage.json"
            if not state_path.is_file():
                continue
            state = _read_json(state_path)
            if state.get("status") != "complete":
                continue
            if state.get("binding_sha256") != binding_hash:
                raise ValueError("completed prediction stage binding mismatch")
            for name, digest in state.get("artifacts", {}).items():
                if sha256_file(candidate / name) != digest:
                    raise ValueError("completed prediction stage artifact hash mismatch")
            return {**state, "directory": str(candidate.resolve()), "resumed": True}
        output = _next_attempt(output)
    output.mkdir(parents=True, exist_ok=False)
    state = {"version": 1, "study": STUDY, "status": "incomplete", "origin": origin,
             "binding": binding, "binding_sha256": binding_hash, "completed_cases": 0}
    _atomic_json(output / "stage.json", state)
    rows, records = [], []
    try:
        for index, case in enumerate(suite["cases"], 1):
            response = engine.evaluate(case["request"], cached=False, details=True)
            if response.get("model") != binding.get("checkpoint_id"):
                raise ValueError("response model identity differs from the prediction binding")
            if set(response.get("answers", {})) != set(case["request"]["questions"]):
                raise ValueError("response question population differs from the case")
            usage = response.get("usage", {})
            if (type(usage.get("max_unit_tokens")) is not int or usage["max_unit_tokens"] < 1
                    or usage["max_unit_tokens"] > MAX_INPUT_TOKENS):
                raise ValueError("evaluation exceeded the 1536-token cap")
            if usage.get("truncated") not in (None, False) or usage.get("truncated_units", 0) != 0:
                raise ValueError("evaluation reported truncation")
            _append_jsonl(output / "responses.jsonl", {"case_id": case["id"], "response": response})
            for qid, question in case["request"]["questions"].items():
                answer = response["answers"][qid]
                logits, labels = _answer_logits(question, answer)
                raw_answer = _raw_answer(question, logits, labels)
                raw_probabilities = (raw_answer["probabilities"] if question["type"] != "noul" else
                                     {"false": 1 - raw_answer["noul"], "true": raw_answer["noul"]})
                response_probabilities = answer.get("probabilities")
                if not isinstance(response_probabilities, dict):
                    raise ValueError("response probability vector is missing")
                if not _probabilities_equal(raw_probabilities, response_probabilities):
                    raise ValueError("response probabilities differ from saved raw logits")
                if question["type"] == "choice":
                    winner = min(labels, key=lambda label: (-raw_probabilities[label], label))
                    if answer.get("choice") != winner:
                        raise ValueError("response choice differs from saved raw logits")
                if question["type"] == "score" and not math.isclose(
                        raw_answer["score"], answer.get("score", math.nan), rel_tol=0, abs_tol=1e-12):
                    raise ValueError("response score differs from saved raw logits")
                prediction = prediction_view(question, case["expected"][qid], raw_answer)
                row = {"case_id": case["id"], "question_id": qid, "family_id": case["family_id"],
                       "domain": case["domain"], "variant": case["variant"],
                       "layout": case.get("layout", "unspecified"), "prediction": prediction}
                target, target_index = _target(question, case["expected"][qid])
                record = {"origin": origin, "category": case.get("category", case["domain"]),
                          "group_id": case["family_id"], "primitive": question["type"], "logits": logits,
                          "labels": labels, "target_index": target_index, "target": target,
                          "question": question, "row": row}
                # T=1 must reproduce the uncalibrated prediction; this also catches
                # accidentally treating probabilities as logits.
                calibrated = calibrate_record(record, 1.0)["prediction"]
                if not _probabilities_equal(calibrated["probabilities"], prediction["probabilities"]):
                    raise ValueError("saved raw logits do not reproduce the uncalibrated prediction")
                rows.append(row)
                records.append(record)
                _append_jsonl(output / "predictions.jsonl", row)
            state["completed_cases"] = index
            _atomic_json(output / "stage.json", state)
    except BaseException as error:
        state["error"] = repr(error)
        _atomic_json(output / "stage.json", state)
        raise
    expected = {(case["id"], qid) for case in suite["cases"] for qid in case["expected"]}
    actual = {(row["case_id"], row["question_id"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("prediction population differs from the suite")
    _atomic_json(output / "predictions.json", rows)
    _atomic_json(output / "calibration-records.json", records)
    state.update(status="complete", completed_cases=len(suite["cases"]), judgments=len(rows),
                 max_unit_tokens=max(_read_jsonl_usage(output / "responses.jsonl"), default=0))
    state["artifacts"] = {name: sha256_file(output / name) for name in
                          ("responses.jsonl", "predictions.jsonl", "predictions.json", "calibration-records.json")}
    _atomic_json(output / "stage.json", state)
    return {**state, "directory": str(output.resolve()), "resumed": False}


def _probabilities_equal(left, right):
    return set(left) == set(right) and all(math.isclose(left[key], right[key], rel_tol=0, abs_tol=1e-12) for key in left)


def _read_jsonl_usage(path):
    for line in Path(path).read_text().splitlines():
        if line:
            yield json.loads(line)["response"]["usage"]["max_unit_tokens"]


def _stage_records(stage):
    directory = Path(stage["directory"])
    return _read_json(directory / "calibration-records.json")


def _fit(records):
    if {record["origin"] for record in records} != {"new", "broad"}:
        raise ValueError("temperature fit requires new and broad calibration populations")
    if {record["primitive"] for record in records} != set(PRIMITIVES):
        raise ValueError("temperature fit requires all three primitives")
    global_fit = fit_temperature(records, balanced_weights(records), lower=0.05, upper=100.0)
    per = {}
    for primitive in PRIMITIVES:
        chosen = [record for record in records if record["primitive"] == primitive]
        per[primitive] = fit_temperature(chosen, balanced_weights(chosen), lower=0.05, upper=100.0)
    return {"global": global_fit, "per_primitive": per,
            "temperatures": {"global": global_fit["temperature"],
                             "per_primitive": {kind: per[kind]["temperature"] for kind in PRIMITIVES}}}


def _run_with_engine(engine_loader, endpoint, operation):
    """Run one model-scoped operation and prove the real engine is no longer referenced."""
    engine = engine_loader(endpoint)
    reference = weakref.ref(engine)
    try:
        model_reference = weakref.ref(engine.model)
    except TypeError:
        model_reference = None
    if (getattr(engine, "model_id", None) != endpoint["checkpoint_id"]
            or getattr(getattr(engine, "model", None), "checkpoint_id", None) != endpoint["checkpoint_id"]):
        if hasattr(engine, "close"):
            engine.close()
        raise ValueError("loaded model identity differs from endpoint lock")
    result = None
    failure = None
    try:
        result = operation(engine)
    except BaseException as error:
        failure = error.with_traceback(None)
    finally:
        if hasattr(engine, "close"):
            engine.close()
        engine = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    if reference() is not None or (model_reference is not None and model_reference() is not None):
        raise RuntimeError("evaluation engine remained live after release") from failure
    if failure is not None:
        raise failure
    return result


def _default_engine(endpoint):
    return load_judgment_engine(checkpoint=Path(endpoint["checkpoint"]), backend="fla",
                                max_input_tokens=MAX_INPUT_TOKENS, unit_batch_size=UNIT_BATCH_SIZE)


def _load_partition(lock, name, json_loader):
    spec = lock["partitions"][name]
    path = _repo_path(lock["repo_root"], spec["path"], name)
    if name.startswith("legacy") and any(part.lower() in {"test", "tests", "finaltest"} for part in path.parts):
        raise ValueError("legacy test paths are quarantined")
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"{name} partition hash differs from endpoint lock")
    value = json_loader(path)
    if name in {"assessment", "calibration"}:
        validate_suite(value)
        families = {case["family_id"] for case in value["cases"]}
        judgments = sum(len(case["expected"]) for case in value["cases"])
        declared = (spec.get("families"), spec.get("cases"), spec.get("judgments"))
        if declared != (len(families), len(value["cases"]), judgments):
            raise ValueError(f"{name} population differs from the endpoint lock")
    elif name in {"legacy_calibration", "legacy_retention"}:
        return _normalize_legacy_partition(lock, name, value)
    return value


def _legacy_split_metadata(lock, name):
    manifest_spec = lock["partitions"].get("legacy_manifest", {})
    manifest_path = _repo_path(lock["repo_root"], manifest_spec.get("path", ""), "legacy manifest")
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_spec.get("sha256"):
        raise ValueError("legacy manifest hash differs from the endpoint lock")
    manifest = _read_json(manifest_path)
    source = _repo_path(lock["repo_root"], manifest.get("source", ""), "legacy validation source")
    if (source.name != "validation.jsonl"
            or any(part.lower() in {"test", "tests", "finaltest"} for part in source.parts)
            or not manifest.get("selection")):
        raise ValueError("legacy manifest does not identify a validation-only source")
    if not source.is_file() or sha256_file(source) != manifest.get("source_sha256"):
        raise ValueError("legacy validation source hash mismatch")
    split_name = "calibration" if name == "legacy_calibration" else "retention"
    split = manifest.get("splits", {}).get(split_name)
    spec = lock["partitions"][name]
    if (not isinstance(split, dict) or split.get("file") != spec.get("path")
            or split.get("sha256") != spec.get("sha256") or split.get("records") != spec.get("records")):
        raise ValueError(f"{name} manifest linkage differs from the endpoint lock")
    identities = split.get("ids")
    sources = split.get("sources")
    if (not isinstance(identities, list) or len(identities) != spec.get("records")
            or len(set(identities)) != len(identities)
            or canonical_hash(identities) != spec.get("record_ids_sha256")):
        raise ValueError(f"{name} manifest identity population is invalid")
    if (not isinstance(sources, dict) or not sources or any(type(count) is not int or count < 1 for count in sources.values())
            or sum(sources.values()) != len(identities)):
        raise ValueError(f"{name} manifest source population is invalid")
    return split


def _normalize_legacy_partition(lock, name, value):
    """Return one canonical evaluation suite from either frozen legacy representation."""
    split = _legacy_split_metadata(lock, name)
    identities = split["ids"]
    expected_sources = Counter(split["sources"])
    if isinstance(value, list):
        if len(value) != len(identities) or [record.get("id") for record in value] != identities:
            raise ValueError(f"{name} record order/identity differs from the frozen manifest")
        if any(record.get("provenance", {}).get("assigned_split") != "validation" for record in value):
            raise ValueError(f"{name} contains a non-validation record")
        actual_sources = Counter(record.get("provenance", {}).get("dataset") for record in value)
        if actual_sources != expected_sources:
            raise ValueError(f"{name} record sources differ from the frozen manifest")
        suite = broad_suite(value)
    elif isinstance(value, dict):
        if set(value) != {"cases", "relations", "contract"} or value.get("relations") != []:
            raise ValueError(f"{name} frozen suite envelope is invalid")
        cases = value.get("cases")
        if not isinstance(cases, list) or len(cases) != len(identities):
            raise ValueError(f"{name} population differs from the endpoint lock")
        if [case.get("id") for case in cases] != identities:
            raise ValueError(f"{name} suite order/identity differs from the frozen manifest")
        if Counter(case.get("domain") for case in cases) != expected_sources:
            raise ValueError(f"{name} suite sources differ from the frozen manifest")
        if any(case.get("variant") != "canonical" or len(case.get("request", {}).get("questions", {})) != 1
               for case in cases):
            raise ValueError(f"{name} suite is not the canonical one-question population")
        suite = value
    else:
        raise ValueError(f"{name} must be raw records or a frozen evaluation suite")
    validate_suite(suite)
    return suite


def _run_evaluation_locked(endpoint_lock_path, output_root, *, engine_loader=None, json_loader=None):
    """Calibrate all five models, lock fits, then and only then read assessment."""
    endpoint_lock_path = Path(endpoint_lock_path).resolve()
    endpoint = _load_lock(endpoint_lock_path, kind="endpoint")
    _verify_endpoint_files(endpoint)
    output_root = Path(output_root).resolve()
    if endpoint.get("gpu_lock") != str((output_root / "gpu.lock").resolve()):
        raise ValueError("endpoint lock names a different study GPU lock")
    engine_loader = engine_loader or _default_engine
    json_loader = json_loader or _read_json
    predictions_root = output_root / "predictions"

    calibration_lock_path = output_root / "calibration-lock.json"
    if calibration_lock_path.exists():
        calibration_lock = _load_lock(calibration_lock_path, kind="calibration")
        _validate_calibration_lock(endpoint, calibration_lock)
    else:
        # Loading calibration here is intentional. Assessment paths are not opened
        # until every model fit is durable and calibration-lock.json is committed.
        new_calibration = _load_partition(endpoint, "calibration", json_loader)
        legacy_calibration = _load_partition(endpoint, "legacy_calibration", json_loader)
        calibration_models = {}
        for model in MODELS:
            endpoint_spec = endpoint["endpoints"][model]
            binding = {"endpoint_lock_sha256": endpoint["lock_sha256"], "model": model,
                       "checkpoint_id": endpoint_spec["checkpoint_id"]}
            def calibration_operation(engine):
                new_stage = run_prediction_stage(
                    engine, new_calibration, predictions_root / model / "calibration-new",
                    binding={**binding, "partition": endpoint["partitions"]["calibration"]["sha256"]}, origin="new",
                )
                broad_stage = run_prediction_stage(
                    engine, legacy_calibration, predictions_root / model / "calibration-legacy",
                    binding={**binding, "partition": endpoint["partitions"]["legacy_calibration"]["sha256"]},
                    origin="broad",
                )
                return new_stage, broad_stage

            new_stage, broad_stage = _run_with_engine(engine_loader, endpoint_spec, calibration_operation)
            records = _stage_records(new_stage) + _stage_records(broad_stage)
            fit = _fit(records)
            fit_path = predictions_root / model / "temperature-fit.json"
            fit_value = {
                "version": 1, "study": STUDY, "model": model, "checkpoint_id": endpoint_spec["checkpoint_id"],
                "endpoint_lock_sha256": endpoint["lock_sha256"], "source_partition_sha256": {
                    "new": endpoint["partitions"]["calibration"]["sha256"],
                    "broad": endpoint["partitions"]["legacy_calibration"]["sha256"],
                }, "prediction_sha256": {
                    "new": new_stage["artifacts"]["calibration-records.json"],
                    "broad": broad_stage["artifacts"]["calibration-records.json"],
                }, **fit,
            }
            _atomic_json(fit_path, fit_value)
            calibration_models[model] = {
                "checkpoint_id": endpoint_spec["checkpoint_id"], "fit_path": str(fit_path),
                "fit_sha256": sha256_file(fit_path), "prediction_sha256": fit_value["prediction_sha256"],
                "temperatures": fit_value["temperatures"],
                "stages": {"new": _locked_stage(new_stage), "broad": _locked_stage(broad_stage)},
            }
        calibration_lock = {
            "version": 1, "study": STUDY, "kind": "calibration", "status": "locked", "created_utc": _utc(),
            "lock_path": str(calibration_lock_path.resolve()),
            "endpoint_lock_sha256": endpoint["lock_sha256"],
            "protocol": {"objective": "0.5 new macro-primitive NLL + 0.5 broad macro-primitive NLL",
                         "variants": ["global", "per_primitive"], "bounds": [0.05, 100.0]},
            "models": calibration_models,
        }
        calibration_lock["lock_sha256"] = canonical_hash(calibration_lock)
        _atomic_json(calibration_lock_path, calibration_lock)
        _validate_calibration_lock(endpoint, calibration_lock)

    assessment = _load_partition(endpoint, "assessment", json_loader)
    retention = _load_partition(endpoint, "legacy_retention", json_loader)
    model_results = {}
    for model in MODELS:
        endpoint_spec = endpoint["endpoints"][model]
        binding = {"endpoint_lock_sha256": endpoint["lock_sha256"],
                   "calibration_lock_sha256": calibration_lock["lock_sha256"],
                   "model": model, "checkpoint_id": endpoint_spec["checkpoint_id"]}
        def assessment_operation(engine):
            assessment_stage = run_prediction_stage(
                engine, assessment, predictions_root / model / "assessment",
                binding={**binding, "partition": endpoint["partitions"]["assessment"]["sha256"]}, origin="assessment",
            )
            retention_stage = run_prediction_stage(
                engine, retention, predictions_root / model / "retention",
                binding={**binding, "partition": endpoint["partitions"]["legacy_retention"]["sha256"]}, origin="retention",
            )
            return assessment_stage, retention_stage

        assessment_stage, retention_stage = _run_with_engine(engine_loader, endpoint_spec, assessment_operation)
        model_results[model] = {"assessment": assessment_stage, "retention": retention_stage}
    result = {"version": 1, "study": STUDY, "status": "complete", "completed_utc": _utc(),
              "endpoint_lock": str(endpoint_lock_path), "endpoint_lock_sha256": endpoint["lock_sha256"],
              "calibration_lock": str(calibration_lock_path), "calibration_lock_sha256": calibration_lock["lock_sha256"],
              "models": model_results}
    _atomic_json(output_root / "evaluation-status.json", result)
    _validate_evaluation_status(endpoint, calibration_lock, result)
    _verify_endpoint_files(endpoint)
    return result


def run_evaluation(endpoint_lock_path, output_root, *, engine_loader=None, json_loader=None):
    """Acquire the pinned study lock, then run the locked calibration/assessment protocol."""
    with study_gpu_lock(output_root):
        return _run_evaluation_locked(endpoint_lock_path, output_root,
                                      engine_loader=engine_loader, json_loader=json_loader)


def _calibrated_rows(records, temperatures, variant):
    rows = []
    for record in records:
        temperature = temperatures["global"] if variant == "global" else temperatures["per_primitive"][record["primitive"]]
        row = calibrate_record(record, temperature)
        row["calibration"] = {"variant": variant, "temperature": temperature}
        rows.append(row)
    return rows


def _relation_rows(suite, rows):
    lookup = {(row["case_id"], row["question_id"]): row["prediction"] for row in rows}
    return [{**relation, "metrics": relation_metrics(relation, lookup)} for relation in suite["relations"]]


def build_final_report(output_root):
    """Generate locked JSON and Markdown results from saved logits only."""
    output_root = Path(output_root).resolve()
    endpoint = _load_lock(output_root / "endpoint-lock.json", kind="endpoint")
    _verify_endpoint_files(endpoint)
    calibration = _load_lock(output_root / "calibration-lock.json", kind="calibration")
    fits = _validate_calibration_lock(endpoint, calibration)
    evaluation = _read_json(output_root / "evaluation-status.json")
    evaluated_records = _validate_evaluation_status(endpoint, calibration, evaluation)
    assessment = _load_partition(endpoint, "assessment", _read_json)
    retention = _load_partition(endpoint, "legacy_retention", _read_json)
    models = {}
    for model in MODELS:
        fit = fits[model]
        temperatures = fit["temperatures"]
        assessment_records = evaluated_records[model]["assessment"]
        retention_records = evaluated_records[model]["retention"]
        variants = {
            "raw": ([record["row"] for record in assessment_records], [record["row"] for record in retention_records]),
            "global": (_calibrated_rows(assessment_records, temperatures, "global"),
                       _calibrated_rows(retention_records, temperatures, "global")),
            "per_primitive": (_calibrated_rows(assessment_records, temperatures, "per_primitive"),
                              _calibrated_rows(retention_records, temperatures, "per_primitive")),
        }
        models[model] = {
            "checkpoint_id": endpoint["endpoints"][model]["checkpoint_id"],
            "temperatures": temperatures,
            "variants": {name: {"assessment": summarize_factorial(assessment, pair[0]),
                                "assessment_relation_rows": _relation_rows(assessment, pair[0]),
                                "retention": summarize_factorial(retention, pair[1]),
                                "retention_relation_rows": _relation_rows(retention, pair[1])}
                         for name, pair in variants.items()},
            "compute": endpoint["endpoints"][model].get("exposure"),
        }
    effects = {}
    for variant in ("raw", "global", "per_primitive"):
        arm_values = {arm: models[arm]["variants"][variant]["assessment"]["primary"]["family_values"] for arm in ARMS}
        categories = models["A"]["variants"][variant]["assessment"]["primary"]["family_categories"]
        effects[variant] = estimate_factorial_effects(arm_values, categories, bootstrap_samples=2000, seed=42)
    report = {
        "version": 1, "study": STUDY, "status": "complete", "generated_utc": _utc(),
        "endpoint_lock_sha256": endpoint["lock_sha256"], "calibration_lock_sha256": calibration["lock_sha256"],
        "reporting_source_sha256": endpoint["reporting"]["sha256"], "protocol": endpoint["reporting"]["protocol"],
        "models": models, "factorial_effects": effects,
        "limitations": [
            "One training seed; intervals describe paired corpus uncertainty, not training-seed variance.",
            "Synthetic fixed semantic programs limit external validity.",
            "Raw and calibrated metrics answer different probability-quality questions.",
            "Independent authorship does not imply statistical independence.",
            "The rubric factor bundles Score K with rule diversity.",
        ],
        "decision": "No automatic promotion or follow-on experiment is authorized by this report.",
    }
    final = output_root / "final-report"
    _atomic_json(final / "report.json", report)
    lines = [
        "# Contrast factorial v1 results", "", "All five locked endpoints were evaluated after calibration was frozen.", "",
        "## Primary assessment metric", "",
        "| Model | Raw | Global T | Per-primitive T |", "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        values = [models[model]["variants"][variant]["assessment"]["primary"]["category_macro"]
                  for variant in ("raw", "global", "per_primitive")]
        lines.append(f"| {model} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    lines.extend(["", "## Factorial effects", "",
                  "| Calibration | Effect | Estimate | 95% paired family interval |", "|---|---|---:|---:|"])
    for variant in ("raw", "global", "per_primitive"):
        for name in ("rubric_at_low", "language_at_narrow", "rubric_at_high", "language_at_broad", "interaction"):
            value = effects[variant]["effects"][name]
            lines.append(f"| {variant} | {name} | {value['estimate']:.4f} | "
                         f"[{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}] |")
    lines.extend(["", "## Raw retention accuracy", "", "| Model | Accuracy |", "|---|---:|"])
    for model in MODELS:
        accuracy = models[model]["variants"]["raw"]["retention"]["overall"]["accuracy"]
        lines.append(f"| {model} | {accuracy:.4f} |")
    lines.extend(["", "## Interpretation limits", "",
                  "Equal updates do not imply equal compute; candidate units, real and padded tokens, time, and memory are retained per arm.",
                  "The study uses one training seed, synthetic fixed semantic programs, and independently authored data that are not statistically independent. "
                  "Raw and calibrated metrics differ, and the rubric factor bundles Score K with rule diversity.", "",
                  "No automatic promotion or future technique is selected.", "", "## Artifacts", "",
                  f"- Endpoint lock: {str((output_root / 'endpoint-lock.json').resolve())}",
                  f"- Calibration lock: {str((output_root / 'calibration-lock.json').resolve())}",
                  f"- Evaluation status: {str((output_root / 'evaluation-status.json').resolve())}",
                  f"- Machine-readable report: {str((final / 'report.json').resolve())}", ""])
    final.mkdir(parents=True, exist_ok=True)
    markdown = final / "RESULTS.md"
    temporary = markdown.with_name(f".{markdown.name}.pending-{os.getpid()}")
    temporary.write_text("\n".join(lines))
    temporary.replace(markdown)
    _verify_endpoint_files(endpoint)
    return report


def load_factory(spec):
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("engine factory must be module:callable")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise ValueError("engine factory is not callable")
    return factory


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    def lock_arguments(command):
        command.add_argument("--repo-root", type=Path, default=Path("."))
        command.add_argument("--approval", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--training-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--legacy-manifest", type=Path, required=True)

    lock = sub.add_parser("lock", help="Validate complete endpoints and freeze the reporting protocol")
    lock_arguments(lock)
    evaluate = sub.add_parser("evaluate", help="Run calibration, lock temperatures, then assess")
    evaluate.add_argument("--endpoint-lock", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--engine-factory", help="Optional module:callable tiny/local engine boundary")
    report = sub.add_parser("report", help="Generate report.json and RESULTS.md from locked predictions")
    report.add_argument("--output-root", type=Path, required=True)
    complete = sub.add_parser("run", help="Lock endpoints, evaluate, and generate the final report")
    lock_arguments(complete)
    complete.add_argument("--engine-factory", help="Optional module:callable tiny/local engine boundary")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command in {"lock", "run"}:
        result = create_endpoint_lock(repo_root=args.repo_root, approval_path=args.approval,
                                      manifest_path=args.manifest, training_root=args.training_root,
                                      output_root=args.output_root, legacy_manifest_path=args.legacy_manifest)
        if args.command == "run":
            factory = load_factory(args.engine_factory) if args.engine_factory else None
            run_evaluation(Path(args.output_root) / "endpoint-lock.json", args.output_root, engine_loader=factory)
            result = build_final_report(args.output_root)
    elif args.command == "evaluate":
        factory = load_factory(args.engine_factory) if args.engine_factory else None
        result = run_evaluation(args.endpoint_lock, args.output_root, engine_loader=factory)
    else:
        result = build_final_report(args.output_root)
    print(json.dumps({"study": STUDY, "command": args.command, "status": result["status"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
