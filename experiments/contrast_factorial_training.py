"""Operational four-arm trainer for the approved contrast-factorial-v1 study.

Manifest paths are repository-relative.  Preflight reads only training/freeze
inputs.  Smoke and train load the production judgment engine, and only root is
authorized to invoke those GPU modes.
"""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import signal
import time
import uuid
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.contrast_scaling_training import (
    _complete_microbatches,
    _component_groups,
    apply_determinism,
    backward_family_loss,
    make_family_schedule,
    prepare_canonical_replay,
    prepare_families,
    prepared_prompt_fingerprint,
    production_resume_equivalence,
)
from experiments.judgment_pipeline import optimizer_for
from openjev.judgment_cli import load_judgment_engine
from openjev.judgment_model import JudgmentEngine, checkpoint_identity, frozen_digest
from openjev.judgment_training import load_bundles

STUDY = "contrast-factorial-v1"
ARMS = ("A", "B", "C", "D")
KINDS = ("noul", "choice", "score")
CHECKPOINTS = (100, 200, 300, 400)
SEED = 42
MAX_UNITS = 12
UNIT_BATCH_SIZE = 12
MAX_INPUT_TOKENS = 1536
CONTRACT_CATEGORIES = (
    "claim_vs_completion", "permission_vs_execution", "unknown_vs_failure", "attribution_and_endorsement",
    "entity_binding", "action_binding", "temporal_scope", "reversal_and_current_state",
    "negation_and_quantifiers", "ordered_rubrics",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SOURCES = (
    "experiments/contrast_factorial_training.py",
    "experiments/contrast_scaling_training.py",
    "experiments/judgment_pipeline.py",
    "src/openjev/judgment_cli.py",
    "src/openjev/judgment_model.py",
    "src/openjev/judgment_training.py",
    "src/openjev/judgments.py",
    "src/openjev/prompts.py",
    "src/openjev/runtime.py",
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path):
    """Hash a directory independent of its absolute location."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"base checkpoint is not a directory: {root}")
    digest, count = hashlib.sha256(), 0
    for item in sorted(root.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            with item.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            count += 1
    if not count:
        raise ValueError(f"base checkpoint is empty: {root}")
    return digest.hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path, value, *, durable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        if durable:
            stream.flush()
            os.fsync(stream.fileno())
    temporary.replace(path)


def _history_record_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _append_history(path, value):
    block = _history_record_bytes(value)
    with Path(path).open("ab") as stream:
        stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    return block


def _write_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _repo_path(repo_root, value, label):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{label} path must be nonempty and repo-relative")
    root = Path(repo_root).resolve()
    result = (root / value).resolve()
    if result != root and root not in result.parents:
        raise ValueError(f"{label} path escapes repo root")
    return result


def _verify_file(repo_root, relative, expected, label):
    path = _repo_path(repo_root, relative, label)
    if not path.is_file() or not isinstance(expected, str) or _sha(path) != expected:
        raise ValueError(f"{label} file is missing or hash mismatched: {relative}")
    return path


def _verify_mapping(repo_root, mapping, label):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{label} fingerprint mapping must be nonempty")
    for relative, expected in mapping.items():
        _verify_file(repo_root, relative, expected, label)


def _validate_family(record, arm):
    provenance = record.get("provenance", {})
    if record.get("id") != record.get("group_id") or provenance.get("arm") != arm:
        raise ValueError("family ID/group/provenance does not match arm")
    if provenance.get("assigned_split") != "train":
        raise ValueError("family provenance must be assigned_split=train")
    examples = record.get("examples")
    if not isinstance(examples, list) or len(examples) != 20:
        raise ValueError("family must have exactly 20 examples")
    expected = {"n1": "noul", "n2": "noul", "n3": "noul", "c1": "choice", "s1": "score"}
    members, cases = set(), {}
    for example in examples:
        case, qid = example.get("case_id"), example.get("question_id")
        question, target = example.get("question"), example.get("target")
        if not isinstance(case, str) or not case or (case, qid) in members or qid not in expected:
            raise ValueError("family has invalid canonical question members")
        if not isinstance(question, dict) or question.get("type") != expected[qid]:
            raise ValueError("family question type does not match canonical ID")
        members.add((case, qid))
        cases.setdefault(case, set()).add(qid)
        criteria = question.get("criteria")
        if qid.startswith("n"):
            valid = set(target or ()) == {"truth"} and type(target["truth"]) is bool
        elif qid == "c1":
            valid = set(target or ()) == {"choice"} and isinstance(criteria, dict) and target["choice"] in criteria
        else:
            valid = (set(target or ()) == {"level_index"} and type(target["level_index"]) is int
                     and isinstance(criteria, list) and 0 <= target["level_index"] < len(criteria))
        if not valid:
            raise ValueError(f"hard {expected[qid]} target required")
    if len(cases) != 4 or any(qids != set(expected) for qids in cases.values()):
        raise ValueError("family must contain four complete canonical cases")


def _family_score_k(record):
    widths = [len(example.get("question", {}).get("criteria", ())) for example in record.get("examples", [])
              if example.get("question", {}).get("type") == "score"]
    if len(widths) != 4 or len(set(widths)) != 1:
        raise ValueError("family must declare one consistent Score K across four cases")
    return widths[0]


def validate_manifest(manifest, repo_root=REPO_ROOT):
    """Verify the strict Task 3 manifest and return all four ordered train sets."""
    if not isinstance(manifest, dict) or manifest.get("study") != STUDY:
        raise ValueError("manifest study must be contrast-factorial-v1")
    required_top = {
        "arms", "ordered_latent_ids", "evaluation_freeze_path", "evaluation_freeze_sha256",
        "base_checkpoint", "base_checkpoint_sha256", "source_fingerprints", "data_fingerprint",
    }
    if not required_top.issubset(manifest):
        raise ValueError("manifest is missing required contract/freeze/source/data/base fields")
    if not isinstance(manifest["arms"], dict) or set(manifest["arms"]) != set(ARMS):
        raise ValueError("manifest requires exactly arms A/B/C/D")
    ids = manifest["ordered_latent_ids"]
    if not isinstance(ids, list) or len(ids) != 400 or len(set(ids)) != 400 or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("manifest requires 400 unique ordered latent IDs")
    _verify_file(repo_root, manifest["evaluation_freeze_path"], manifest["evaluation_freeze_sha256"], "freeze")
    base = _repo_path(repo_root, manifest["base_checkpoint"], "base checkpoint")
    if tree_sha256(base) != manifest["base_checkpoint_sha256"]:
        raise ValueError("base checkpoint tree hash mismatch")
    _verify_mapping(repo_root, manifest["source_fingerprints"], "source")
    _verify_mapping(repo_root, manifest["data_fingerprint"], "data")
    result = {}
    required_arm = {"train_file", "suite_file", "ordered_family_ids", "sha256", "suite_sha256"}
    for arm in ARMS:
        spec = manifest["arms"][arm]
        if not isinstance(spec, dict) or not required_arm.issubset(spec):
            raise ValueError(f"arm {arm} manifest is incomplete")
        if spec["ordered_family_ids"] != ids:
            raise ValueError(f"arm {arm} must use matching ordered 400 latent IDs")
        train = _verify_file(repo_root, spec["train_file"], spec["sha256"], f"arm {arm} train")
        suite = _verify_file(repo_root, spec["suite_file"], spec["suite_sha256"], f"arm {arm} suite")
        try:
            suite_value = json.loads(suite.read_text())
        except ValueError as error:
            raise ValueError(f"arm {arm} suite is invalid JSON") from error
        if not isinstance(suite_value, dict) or suite_value.get("ordered_family_ids") != ids:
            raise ValueError(f"arm {arm} suite ordered family IDs mismatch")
        families = suite_value.get("families")
        if families is not None and [row.get("id") for row in families] != ids:
            raise ValueError(f"arm {arm} suite family records mismatch")
        rows = load_bundles(train)
        if len(rows) != 400 or [row.get("id") for row in rows] != ids:
            raise ValueError(f"arm {arm} train must contain ordered 400 families")
        categories, score_widths = [], []
        for row in rows:
            _validate_family(row, arm)
            category = row.get("provenance", {}).get("category")
            if category not in CONTRACT_CATEGORIES:
                raise ValueError(f"arm {arm} family has an invalid provenance.category")
            categories.append(category)
            score_widths.append(_family_score_k(row))
        if {category: categories.count(category) for category in CONTRACT_CATEGORIES} != {
                category: 40 for category in CONTRACT_CATEGORIES}:
            raise ValueError(f"arm {arm} must contain 40 families per contract category")
        expected_widths = {2: 200, 3: 200} if arm in ("A", "C") else {2: 100, 3: 100, 4: 100, 5: 100}
        if {width: score_widths.count(width) for width in expected_widths} != expected_widths or set(score_widths) != set(expected_widths):
            raise ValueError(f"arm {arm} Score K distribution does not match its narrow/broad contract")
        result[arm] = rows
    return result


def make_schedule(family_ids, replay_pools, seed=SEED):
    if len(family_ids) != 400:
        raise ValueError("factorial arms require exactly 400 ordered families")
    return make_family_schedule(family_ids, replay_pools, total_steps=400, seed=seed)


def _history_hash(history):
    digest = hashlib.sha256()
    for row in history:
        digest.update(_history_record_bytes(row))
    return digest.hexdigest()


def _exposure(family, replay, row, *, max_units, unit_batch_size):
    candidate_units = unpadded_tokens = padded_tokens = 0
    for values in _component_groups(family, replay).values():
        bundle = values[0][0]
        selected = [group for _, group in values]
        for prompts, _, _ in _complete_microbatches(bundle, max_units, selected=selected):
            candidate_units += len(prompts)
            unpadded_tokens += sum(map(len, prompts))
            for start in range(0, len(prompts), unit_batch_size):
                batch = prompts[start:start + unit_batch_size]
                padded_tokens += len(batch) * max(map(len, batch))
    return {
        "family_id": row["family_id"],
        "replay_ids": {item["primitive"]: item["id"] for item in sorted(row["replay"], key=lambda x: x["primitive"])},
        "candidate_units": candidate_units,
        "unpadded_tokens": unpadded_tokens,
        "padded_tokens": padded_tokens,
    }


def _compiled_workload(bundle):
    padded_tokens = 0
    for prompts, _, _ in _complete_microbatches(bundle, MAX_UNITS):
        for start in range(0, len(prompts), UNIT_BATCH_SIZE):
            batch = prompts[start:start + UNIT_BATCH_SIZE]
            padded_tokens += len(batch) * max(map(len, batch))
    return {
        "max_prompt_tokens": max(map(len, bundle.prompts)),
        "max_group_width": max(len(group.indices) for group in bundle.groups),
        "candidate_units": len(bundle.prompts),
        "padded_tokens": padded_tokens,
    }


def select_smoke_families(records_by_arm, prepared_by_arm):
    """Select outcome-independent category/K/workload coverage from compiled inputs."""
    candidates = []
    for arm in ARMS:
        for record in records_by_arm.get(arm, []):
            identity = record.get("id")
            if identity not in prepared_by_arm.get(arm, {}):
                raise ValueError(f"smoke selection is missing compiled family {arm}/{identity}")
            category = record.get("provenance", {}).get("category")
            if category not in CONTRACT_CATEGORIES:
                raise ValueError("smoke selection requires provenance.category")
            score_k = _family_score_k(record)
            expected = {2, 3} if arm in ("A", "C") else {2, 3, 4, 5}
            if score_k not in expected:
                raise ValueError(f"smoke selection found invalid Score K for arm {arm}")
            candidates.append({
                "arm": arm, "family_id": identity, "category": category, "score_k": score_k,
                **_compiled_workload(prepared_by_arm[arm][identity]),
            })
    broad = [row for row in candidates if row["arm"] in ("B", "D")]
    selected = {}

    def add(row, reason):
        key = (row["arm"], row["family_id"])
        selected.setdefault(key, {**row, "reasons": []})["reasons"].append(reason)

    for category in CONTRACT_CATEGORIES:
        choices = sorted((row for row in broad if row["category"] == category), key=lambda row: (row["arm"], row["family_id"]))
        if not choices:
            raise ValueError(f"smoke cannot cover category {category} in a broad arm")
        add(choices[0], f"category:{category}")
    for score_k in (2, 3, 4, 5):
        choices = sorted((row for row in broad if row["score_k"] == score_k), key=lambda row: (row["arm"], row["family_id"]))
        if not choices:
            raise ValueError(f"smoke cannot cover broad Score K={score_k}")
        add(choices[0], f"broad_score_k:{score_k}")
    metrics = (
        ("max_prompt_tokens", "global_max_prompt_tokens"),
        ("max_group_width", "global_max_group_width"),
        ("padded_tokens", "global_max_padded_tokens"),
    )
    for metric, reason in metrics:
        maximum = max(row[metric] for row in candidates)
        choice = min((row for row in candidates if row[metric] == maximum), key=lambda row: (row["arm"], row["family_id"]))
        add(choice, reason)
    result = [selected[key] for key in sorted(selected)]
    if {row["category"] for row in result} != set(CONTRACT_CATEGORIES):
        raise RuntimeError("smoke category selection is incomplete")
    if {row["score_k"] for row in result if row["arm"] in ("B", "D")} != {2, 3, 4, 5}:
        raise RuntimeError("smoke broad Score K selection is incomplete")
    return result


def build_smoke_schedule(selection, prepared_by_arm, training_schedule):
    """Map cross-arm smoke families into one temporary production checkpoint stream."""
    if not selection or len(selection) >= len(training_schedule):
        raise ValueError("smoke selection must leave one update for the resume proof")
    prepared = {}
    synthetic_ids = []
    for row in selection:
        synthetic = f"smoke/{row['arm']}/{row['family_id']}"
        prepared[synthetic] = replace(prepared_by_arm[row["arm"]][row["family_id"]], id=synthetic)
        synthetic_ids.append(synthetic)
    for row in training_schedule:
        for item in row["replay"]:
            prepared[item["id"]] = prepared_by_arm[ARMS[0]][item["id"]]
    schedule = []
    for position, row in enumerate(training_schedule):
        schedule.append({**row, "family_id": synthetic_ids[position % len(synthetic_ids)]})
    return schedule, prepared


def _gradient_check(model):
    adapters = [p for name, p in model.named_parameters() if p.requires_grad and ".lora_" in name]
    heads = [p for name, p in model.named_parameters()
             if p.requires_grad and name.startswith(("binary.", "compatibility."))]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    if not adapters or not heads:
        raise RuntimeError("adapter/head gradient gate has no matching trainable parameters")
    adapter_gradients = [p.grad for p in adapters if p.grad is not None]
    head_gradients = [p.grad for p in heads if p.grad is not None]
    gradients = adapter_gradients + head_gradients
    if not adapter_gradients or not head_gradients:
        raise RuntimeError("adapter/head gradient gate has missing gradients")
    adapter_max = torch.stack([gradient.detach().abs().amax() for gradient in adapter_gradients]).amax()
    head_max = torch.stack([gradient.detach().abs().amax() for gradient in head_gradients]).amax()
    finite = torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()
    flags = torch.stack((adapter_max > 0, head_max > 0, finite)).to(device="cpu").tolist()
    result = {
        "adapters": bool(flags[0]),
        "heads": bool(flags[1]),
        "frozen": all(p.grad is None for p in frozen),
        "finite": bool(flags[2]),
    }
    if not all(result.values()):
        raise RuntimeError(f"adapter/head/frozen gradient gate failed: {result}")
    return result


def save_arm_checkpoint(path, engine, optimizer, state):
    """Use the production adapter/readout writer; never serialize the frozen body."""
    metadata = {
        "study": STUDY,
        "arm": state["arm"],
        "completed_updates": state["completed_updates"],
        "update_phase": state["update_phase"],
        "fingerprint": state["fingerprint"],
    }
    identity = engine.model.save_checkpoint(path, metadata=metadata, optimizer=optimizer, training_state=state)
    engine.model_id = identity
    _write(Path(path) / "state.json", state)
    return Path(path)


def _checkpoint_state(path):
    path = Path(path)
    if not (path / "checkpoint.json").is_file() or not (path / "training.pt").is_file() or not (path / "state.json").is_file():
        raise ValueError("resume checkpoint is missing checkpoint.json, training.pt, or state.json")
    payload = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"optimizer", "state", "torch_rng", "cuda_rng"}.issubset(payload):
        raise ValueError("resume checkpoint training state is incomplete")
    state = payload["state"]
    try:
        json_state = json.loads((path / "state.json").read_text())
        info = json.loads((path / "checkpoint.json").read_text())
    except ValueError as error:
        raise ValueError("resume checkpoint metadata is invalid JSON") from error
    if json_state != state:
        raise ValueError("resume state JSON differs from training state")
    metadata = info.get("metadata", {})
    if info.get("format") == "openjev-judgment-v0.2" and info.get("checkpoint_id") != checkpoint_identity(path, info):
        raise ValueError("resume checkpoint weight identity mismatch")
    if metadata.get("study") != STUDY or metadata.get("arm") != state.get("arm"):
        raise ValueError("resume checkpoint identity metadata mismatch")
    if (metadata.get("completed_updates") != state.get("completed_updates")
            or metadata.get("update_phase") != state.get("update_phase")
            or metadata.get("fingerprint") != state.get("fingerprint")):
        raise ValueError("resume checkpoint metadata differs from training state")
    return payload


def validate_resume_checkpoint(checkpoint, *, arm, fingerprint, output):
    """Validate every resume invariant without mutating model, optimizer, RNG, or files."""
    payload = _checkpoint_state(checkpoint)
    state = _validate_resume_core(payload, arm=arm, fingerprint=fingerprint)
    completed, history = state["completed_updates"], state["history"]
    history_path = Path(output) / "history.jsonl"
    if completed and not history_path.is_file():
        raise ValueError("resume history file is missing")
    disk = []
    if history_path.is_file():
        try:
            disk = [json.loads(line) for line in history_path.read_text().splitlines() if line]
        except ValueError as error:
            raise ValueError("existing resume history is invalid JSON") from error
    if disk != history:
        raise ValueError("existing history on disk differs from checkpoint history")
    return payload


def _optimizer_parameter_map(optimizer, model):
    names = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(parameters) != len({id(parameter) for parameter in parameters}) or set(map(id, parameters)) != set(names):
        raise ValueError("optimizer trainable parameter coverage differs from the model")
    return parameters, names


def _validate_optimizer_payload(saved, optimizer, model, completed):
    if not isinstance(saved, dict) or not isinstance(saved.get("param_groups"), list) or not isinstance(saved.get("state"), dict):
        raise ValueError("optimizer state payload is malformed")
    parameters, names = _optimizer_parameter_map(optimizer, model)
    saved_groups = saved["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("optimizer parameter-group coverage mismatch")
    bindings = []
    for saved_group, current_group in zip(saved_groups, optimizer.param_groups, strict=True):
        saved_ids, current = saved_group.get("params"), current_group["params"]
        if not isinstance(saved_ids, list) or len(saved_ids) != len(current):
            raise ValueError("optimizer parameter coverage mismatch")
        bindings.extend(zip(saved_ids, current, strict=True))
    saved_ids = [identity for identity, _ in bindings]
    if len(saved_ids) != len(set(saved_ids)) or len(bindings) != len(parameters):
        raise ValueError("optimizer parameter coverage mismatch")
    state = saved["state"]
    if completed == 0:
        if state:
            raise ValueError("zero-update optimizer state must be empty")
        return {"parameter_names": sorted(names.values()), "state_coverage_count": 0, "unique_step_counters": []}
    if set(state) != set(saved_ids):
        raise ValueError("optimizer state coverage is incomplete")
    counters = []
    for identity, parameter in bindings:
        name, values = names[id(parameter)], state[identity]
        if not isinstance(values, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(values):
            raise ValueError(f"optimizer state coverage is incomplete for {name}")
        step = values["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1 or not torch.isfinite(step).all():
                raise ValueError(f"optimizer counter is invalid for {name}")
            raw_counter = step.item()
            counter = int(raw_counter)
            if raw_counter != counter:
                raise ValueError(f"optimizer counter is invalid for {name}")
        elif type(step) is int:
            counter = step
        else:
            raise ValueError(f"optimizer counter is invalid for {name}")
        counters.append(counter)
        if counter != completed:
            raise ValueError(f"optimizer counter differs from completed updates for {name}")
        for key in ("exp_avg", "exp_avg_sq"):
            moment = values[key]
            if not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape:
                raise ValueError(f"optimizer {key} shape mismatch for {name}")
            if moment.dtype != parameter.dtype:
                raise ValueError(f"optimizer {key} dtype mismatch for {name}")
            if not torch.isfinite(moment).all():
                raise ValueError(f"optimizer {key} must be finite for {name}")
    return {
        "parameter_names": sorted(names.values()),
        "state_coverage_count": len(state),
        "unique_step_counters": sorted(set(counters)),
    }


def _validate_loaded_optimizer(optimizer, model):
    parameters, names = _optimizer_parameter_map(optimizer, model)
    if set(optimizer.state) != set(parameters):
        raise ValueError("loaded optimizer state coverage is incomplete")
    for parameter in parameters:
        for key in ("exp_avg", "exp_avg_sq"):
            moment = optimizer.state[parameter][key]
            if moment.shape != parameter.shape:
                raise ValueError(f"loaded optimizer {key} shape mismatch for {names[id(parameter)]}")
            if moment.dtype != parameter.dtype:
                raise ValueError(f"loaded optimizer {key} dtype mismatch for {names[id(parameter)]}")
            if moment.device != parameter.device:
                raise ValueError(f"loaded optimizer {key} device mismatch for {names[id(parameter)]}")


def restore_arm(checkpoint, engine, optimizer, *, arm, fingerprint, output):
    """Restore Adam and CPU/CUDA RNG after all read-only resume checks pass."""
    payload = validate_resume_checkpoint(checkpoint, arm=arm, fingerprint=fingerprint, output=output)
    completed = payload["state"]["completed_updates"]
    optimizer_verification = _validate_optimizer_payload(payload["optimizer"], optimizer, engine.model, completed)
    optimizer.load_state_dict(payload["optimizer"])
    if completed:
        _validate_loaded_optimizer(optimizer, engine.model)
    torch.set_rng_state(payload["torch_rng"])
    if not torch.equal(torch.get_rng_state(), payload["torch_rng"]):
        raise RuntimeError("CPU RNG restore verification failed")
    saved_cuda = payload["cuda_rng"]
    cuda_count = 0
    if torch.cuda.is_available():
        if len(saved_cuda) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all(saved_cuda)
        restored = torch.cuda.get_rng_state_all()
        if len(restored) != len(saved_cuda) or any(not torch.equal(a, b) for a, b in zip(restored, saved_cuda, strict=True)):
            raise RuntimeError("CUDA RNG restore verification failed")
        cuda_count = len(restored)
    elif saved_cuda:
        raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
    state = copy.deepcopy(payload["state"])
    state["restore_verification"] = {
        "optimizer_state_entries": len(optimizer.state),
        **optimizer_verification,
        "cpu_rng_states": 1,
        "cuda_rng_states": cuda_count,
    }
    return state


def _validate_resume_core(payload, *, arm, fingerprint):
    state = payload["state"]
    if state.get("arm") != arm:
        raise ValueError("resume arm identity mismatch")
    if state.get("fingerprint") != fingerprint:
        raise ValueError("resume fingerprint mismatch")
    if not state.get("complete_boundary") or not state.get("resumable") or state.get("update_phase") != "boundary":
        raise ValueError("resume requires a complete boundary checkpoint")
    completed, history = state.get("completed_updates"), state.get("history")
    if type(completed) is not int or state.get("next_position") != completed or not isinstance(history, list):
        raise ValueError("resume counter/history mismatch")
    if len(history) != completed or [row.get("step") for row in history] != list(range(1, completed + 1)):
        raise ValueError("resume history is not monotonic and contiguous")
    if state.get("history_sha256") != _history_hash(history):
        raise ValueError("resume history hash mismatch")
    return state


def _parse_history_bytes(raw):
    complete, torn = [], b""
    blocks = raw.splitlines(keepends=True)
    for index, block in enumerate(blocks):
        if not block.endswith(b"\n"):
            if index != len(blocks) - 1:
                raise ValueError("history has a non-final torn record")
            torn = block
            continue
        try:
            complete.append(json.loads(block))
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("history has an invalid complete record") from error
    return complete, torn


def _snapshot_matches(path, arm, fingerprint):
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except ValueError as error:
        raise ValueError(f"recovery snapshot is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"recovery snapshot is not an object: {path.name}")
    if value.get("arm") != arm or value.get("study") != STUDY:
        raise ValueError(f"recovery snapshot identity mismatch: {path.name}")
    has_full = "fingerprint" in value
    has_hash = "fingerprint_sha256" in value
    if not has_full and not has_hash:
        raise ValueError(f"recovery snapshot fingerprint is missing: {path.name}")
    if has_full and (not isinstance(value["fingerprint"], dict) or value["fingerprint"] != fingerprint):
        raise ValueError(f"recovery snapshot fingerprint mismatch: {path.name}")
    if has_hash and (not isinstance(value["fingerprint_sha256"], str)
                     or value["fingerprint_sha256"] != _canonical_hash(fingerprint)):
        raise ValueError(f"recovery snapshot fingerprint mismatch: {path.name}")
    return value


def recover_history_tail(checkpoint, *, arm, fingerprint, output):
    """Explicitly archive a verified uncheckpointed/torn tail and restore its checkpoint prefix."""
    checkpoint, output = Path(checkpoint), Path(output)
    payload = _checkpoint_state(checkpoint)
    state = _validate_resume_core(payload, arm=arm, fingerprint=fingerprint)
    history_path = output / "history.jsonl"
    if not history_path.is_file():
        raise ValueError("history recovery requires an existing history file")
    raw = history_path.read_bytes()
    disk, torn = _parse_history_bytes(raw)
    prefix = state["history"]
    if disk[:len(prefix)] != prefix:
        raise ValueError("checkpoint history is not an exact prefix of live history")
    tail = disk[len(prefix):]
    expected_steps = list(range(len(prefix) + 1, len(prefix) + len(tail) + 1))
    if [row.get("step") for row in tail] != expected_steps:
        raise ValueError("history tail is not monotonic and contiguous")
    if not tail and not torn:
        raise ValueError("no history tail exists to recover")
    for name in ("progress.json", "failure.json"):
        snapshot = _snapshot_matches(output / name, arm, fingerprint)
        if snapshot is not None:
            count = snapshot.get("completed_updates")
            if type(count) is not int or not len(prefix) <= count <= len(prefix) + len(tail):
                raise ValueError(f"recovery snapshot counter mismatch: {name}")

    recovery_root = output / "recovery"
    archive = recovery_root / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex)
    archive.mkdir(parents=True, exist_ok=False)
    archived = {}
    try:
        history_copy = archive / "history.original.jsonl"
        _write_bytes(history_copy, raw)
        archived[history_copy.name] = _sha(history_copy)
        for name in ("progress.json", "failure.json"):
            source = output / name
            if source.is_file():
                target = archive / name
                _write_bytes(target, source.read_bytes())
                archived[name] = _sha(target)
        checkpoint_info = json.loads((checkpoint / "checkpoint.json").read_text())
        first = len(prefix) + 1
        last_complete = len(prefix) + len(tail)
        recompute_last = last_complete + (1 if torn else 0)
        recovery = {
            "version": 1, "study": STUDY, "arm": arm,
            "checkpoint": str(checkpoint.resolve()), "checkpoint_id": checkpoint_info.get("checkpoint_id"),
            "fingerprint_sha256": _canonical_hash(fingerprint),
            "accepted_prefix_updates": len(prefix),
            "archived_complete_tail": [first, last_complete] if tail else None,
            "torn_final_line": bool(torn), "torn_bytes": len(torn),
            "recompute_updates": [first, recompute_last],
            "archived_sha256": archived,
        }
        _write(archive / "recovery.json", recovery)
        archived["recovery.json"] = _sha(archive / "recovery.json")
        _write_bytes(history_path, b"".join(_history_record_bytes(row) for row in prefix))
        for item in archive.iterdir():
            item.chmod(0o444)
        archive.chmod(0o555)
        return archive
    except BaseException:
        # The active history is not changed until every archive artifact and its
        # manifest have been durably written.
        raise


def _device(model):
    return next(model.parameters()).device


def _peak_memory(model):
    device = _device(model)
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


def _state(arm, fingerprint, history, *, history_sha256=None, **updates):
    completed = len(history)
    value = {
        "version": 2, "study": STUDY, "arm": arm, "status": "running",
        "completed_updates": completed, "next_position": completed,
        "update_phase": "boundary", "complete_boundary": True, "resumable": True,
        "history": history, "history_sha256": history_sha256 or _history_hash(history), "fingerprint": fingerprint,
    }
    value.update(updates)
    return value


def _compact_progress(state):
    result = {
        "version": 2, "study": STUDY, "arm": state["arm"], "status": state.get("status"),
        "completed_updates": state.get("completed_updates", 0), "next_position": state.get("next_position"),
        "attempted_update": state.get("attempted_update"), "update_phase": state.get("update_phase"),
        "complete_boundary": state.get("complete_boundary", False), "resumable": state.get("resumable", False),
        "history_updates": len(state.get("history", [])), "history_sha256": state.get("history_sha256"),
        "fingerprint_sha256": _canonical_hash(state["fingerprint"]),
        "total_time_seconds": state.get("total_time_seconds", 0.0),
        "peak_memory_bytes": state.get("peak_memory_bytes", 0),
    }
    for key in ("checkpoint", "pause_reason", "error", "frozen_before", "frozen_after",
                "frozen_verified_at", "gradient_check", "restore_verification", "arm_start"):
        if state.get(key) is not None:
            result[key] = state[key]
    if state.get("history"):
        result["latest_update"] = state["history"][-1]
    return result


def _safe_frozen_digest(model):
    try:
        return frozen_digest(model), None
    except BaseException as error:
        return None, repr(error)


def _validate_run_fingerprint(arm, schedule, fingerprint):
    required = {
        "study", "arm", "schedule_sha256", "replay_sha256", "manifest_sha256",
        "base_checkpoint_sha256", "source_fingerprints", "data_fingerprint", "runtime", "determinism",
        "prepared_prompt_fingerprint",
    }
    if not isinstance(fingerprint, dict) or not required.issubset(fingerprint):
        raise ValueError("run fingerprint is incomplete")
    if fingerprint["study"] != STUDY or fingerprint["arm"] != arm:
        raise ValueError("run fingerprint arm/study mismatch")
    if fingerprint["schedule_sha256"] != _canonical_hash(schedule):
        raise ValueError("run fingerprint schedule mismatch")


def run_arm(
    arm, engine, schedule, prepared, output, *, fingerprint, optimizer=None, start_state=None,
    pause_file=None, max_units=MAX_UNITS, unit_batch_size=UNIT_BATCH_SIZE, stop_after=400, progress=None,
    arm_start=None, allow_pristine_output=False,
):
    """Train one contiguous arm range and checkpoint only known complete boundaries."""
    if arm not in ARMS or len(schedule) != 400 or [row.get("step") for row in schedule] != list(range(400)):
        raise ValueError("arm requires a contiguous 400-position schedule")
    _validate_run_fingerprint(arm, schedule, fingerprint)
    if max_units != MAX_UNITS or unit_batch_size != UNIT_BATCH_SIZE or engine.unit_batch_size != UNIT_BATCH_SIZE:
        raise ValueError("factorial contract requires max_units=12 and unit_batch_size=12")
    if type(stop_after) is not int or not 1 <= stop_after <= 400:
        raise ValueError("stop_after must be an update in 1..400")
    output, pause_path = Path(output), Path(pause_file) if pause_file else None
    if start_state is None:
        if output.exists():
            if not allow_pristine_output or any(
                    path.name != "pristine-pause.json" for path in output.iterdir()):
                raise FileExistsError(output)
        else:
            output.mkdir(parents=True, exist_ok=False)
        history = []
    else:
        if not output.is_dir():
            raise ValueError("resume output directory is missing")
        history = copy.deepcopy(start_state["history"])
    if len(history) > stop_after:
        raise ValueError("resume position is beyond requested endpoint")
    model = engine.model
    if start_state is not None and start_state.get("fingerprint") != fingerprint:
        raise ValueError("run fingerprint mismatch")
    history_hasher = hashlib.sha256()
    for entry in history:
        history_hasher.update(_history_record_bytes(entry))
    state = _state(
        arm, fingerprint, history, history_sha256=history_hasher.hexdigest(), status="initializing",
        update_phase="initialization", complete_boundary=False, resumable=False, next_position=None,
        gradient_check=start_state.get("gradient_check") if start_state else None,
        restore_verification=start_state.get("restore_verification") if start_state else None,
        total_time_seconds=start_state.get("total_time_seconds", 0.0) if start_state else 0.0,
        peak_memory_bytes=start_state.get("peak_memory_bytes", 0) if start_state else 0,
        arm_start=arm_start or (start_state.get("arm_start") if start_state else None),
    )
    terminate = [False]
    previous, handler_installed = None, False
    run_started = None
    current_checkpoint = start_state.get("checkpoint") if start_state else None

    def publish(*, durable=False):
        try:
            state["peak_memory_bytes"] = max(state.get("peak_memory_bytes", 0), _peak_memory(model))
        except BaseException:
            pass
        _write(output / "progress.json", _compact_progress(state), durable=durable)

    def pause_at_boundary(reason):
        nonlocal current_checkpoint
        state.update(status="paused", pause_reason=reason)
        completed = state["completed_updates"]
        candidate = output / f"step-{completed:04d}"
        if not candidate.exists():
            save_arm_checkpoint(candidate, engine, optimizer, state)
            current_checkpoint = str(candidate)
        elif candidate.exists():
            current_checkpoint = str(candidate)
        if current_checkpoint:
            state["checkpoint"] = current_checkpoint
        publish(durable=True)
        return state

    try:
        optimizer = optimizer or optimizer_for(model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: terminate.__setitem__(0, True))
        handler_installed = True
        current_frozen = frozen_digest(model)
        frozen_before = start_state.get("frozen_before") if start_state else current_frozen
        if start_state and current_frozen != start_state.get("frozen_after", frozen_before):
            raise ValueError("resumed frozen-body hash mismatch")
        state.update(frozen_before=frozen_before, frozen_after=current_frozen)
        engine.synchronize()
        device = _device(model)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        state.update(
            status="running", update_phase="boundary", complete_boundary=True, resumable=True,
            next_position=len(history),
        )
        run_started = time.perf_counter()
        if pause_path and pause_path.exists():
            return pause_at_boundary("request-file-present-before-update")
        for row in schedule[len(history):stop_after]:
            if terminate[0] or (pause_path and pause_path.exists()):
                return pause_at_boundary("signal" if terminate[0] else "request-file")
            attempted = row["step"] + 1
            state.update(update_phase="pre-step", complete_boundary=False, resumable=False,
                         next_position=None, attempted_update=attempted)
            publish()
            family = prepared[row["family_id"]]
            replay = {item["primitive"]: prepared[item["id"]] for item in row["replay"]}
            exposure = _exposure(family, replay, row, max_units=max_units, unit_batch_size=unit_batch_size)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            engine.synchronize()
            step_started = time.perf_counter()
            result = backward_family_loss(model, family, replay, max_units=max_units, unit_batch_size=unit_batch_size)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                                   error_if_nonfinite=True)
            gradient = _gradient_check(model)
            state.update(update_phase="optimizer-step")
            optimizer.step()
            state.update(update_phase="post-step")
            engine.synchronize()
            elapsed = time.perf_counter() - step_started
            prior_total = start_state.get("total_time_seconds", 0.0) if start_state else 0.0
            total = prior_total + (time.perf_counter() - run_started)
            entry = {
                "step": attempted, "update_phase": "boundary", "loss": float(result["loss"]),
                "component_losses": {key: float(value) for key, value in result["components"].items()},
                "component_counts": result["counts"], "gradient_norm": float(norm), "exposure": exposure,
                "update_time_seconds": elapsed, "total_time_seconds": total, "peak_memory_bytes": _peak_memory(model),
            }
            block = _append_history(output / "history.jsonl", entry)
            history_hasher.update(block)
            history.append(entry)
            state = _state(
                arm, fingerprint, history, history_sha256=history_hasher.hexdigest(),
                frozen_before=frozen_before, frozen_after=state.get("frozen_after", frozen_before),
                gradient_check=gradient, total_time_seconds=total,
                restore_verification=start_state.get("restore_verification") if start_state else None,
                arm_start=arm_start or (start_state.get("arm_start") if start_state else None),
                peak_memory_bytes=max(state.get("peak_memory_bytes", 0), entry["peak_memory_bytes"]),
            )
            publish()
            if progress is not None:
                progress(arm, attempted, entry)
            requested_pause = terminate[0] or (pause_path and pause_path.exists())
            endpoint = attempted in CHECKPOINTS or attempted == stop_after or requested_pause
            if endpoint:
                state.update(update_phase="boundary-verification", complete_boundary=False, resumable=False,
                             next_position=None)
                state["frozen_after"] = frozen_digest(model)
                state["frozen_verified_at"] = attempted
                if state["frozen_before"] != state["frozen_after"]:
                    raise RuntimeError("frozen body changed during arm training")
                state.update(update_phase="boundary", complete_boundary=True, resumable=True,
                             next_position=attempted)
                state["status"] = "paused" if requested_pause else ("complete" if attempted == stop_after else "running")
                if requested_pause:
                    state["pause_reason"] = "signal" if terminate[0] else "request-file"
                checkpoint = output / f"step-{attempted:04d}"
                save_arm_checkpoint(checkpoint, engine, optimizer, state)
                current_checkpoint = str(checkpoint)
                state["checkpoint"] = current_checkpoint
                publish(durable=True)
                if requested_pause:
                    return state
        if state.get("frozen_verified_at") != len(history):
            state.update(frozen_after=frozen_digest(model), frozen_verified_at=len(history))
        state["status"] = "complete"
        if state["frozen_before"] != state["frozen_after"]:
            raise RuntimeError("frozen body changed during arm training")
        state["session_time_seconds"] = time.perf_counter() - run_started
        publish(durable=True)
        return state
    except BaseException as error:
        phase = state.get("update_phase", "unknown")
        failure_frozen, frozen_error = _safe_frozen_digest(model)
        frozen_before = state.get("frozen_before")
        resumable = phase == "boundary" and failure_frozen is not None and failure_frozen == frozen_before
        state.update(status="failed", complete_boundary=resumable, resumable=resumable,
                     next_position=len(history) if resumable else None,
                     error=repr(error), update_phase=phase, frozen_after=failure_frozen)
        recording_errors = []
        if frozen_error:
            recording_errors.append({"operation": "frozen_digest", "error": frozen_error})
        checkpoint = output / f"partial-{len(history):04d}-attempt-{state.get('attempted_update', len(history)):04d}"
        if optimizer is not None:
            try:
                save_arm_checkpoint(checkpoint, engine, optimizer, state)
                state["checkpoint"] = str(checkpoint)
            except BaseException as checkpoint_error:
                recording_errors.append({"operation": "checkpoint", "error": repr(checkpoint_error)})
        failure = _compact_progress(state)
        if recording_errors:
            failure["recording_errors"] = recording_errors
        try:
            _write(output / "failure.json", failure)
        except BaseException:
            pass
        raise
    finally:
        if handler_installed:
            try:
                signal.signal(signal.SIGTERM, previous)
            except BaseException:
                pass


def resume_arm(
    arm, checkpoint, schedule, prepared, output, *, fingerprint, engine_loader, optimizer_factory=None,
    pause_file=None, stop_after=400, progress=None,
):
    """Explicitly load production weights, Adam and RNG, then continue an arm."""
    validate_resume_checkpoint(checkpoint, arm=arm, fingerprint=fingerprint, output=output)
    engine = engine_loader(Path(checkpoint))
    optimizer = (optimizer_factory or (lambda model: optimizer_for(
        model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))))(engine.model)
    state = restore_arm(checkpoint, engine, optimizer, arm=arm, fingerprint=fingerprint, output=output)
    state["checkpoint"] = str(checkpoint)
    result = run_arm(
        arm, engine, schedule, prepared, output, fingerprint=fingerprint, optimizer=optimizer,
        start_state=state, pause_file=pause_file, stop_after=stop_after, progress=progress,
    )
    return {"engine": engine, "optimizer": optimizer, "state": result}


def _tensor_sha256(value):
    tensor = value.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _trainable_sha256(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _arm_start_identity(engine):
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "checkpoint_id": getattr(engine.model, "checkpoint_id", None) or getattr(engine, "model_id", None),
        "trainable_weights_sha256": _trainable_sha256(engine.model),
        "cpu_rng_sha256": _tensor_sha256(torch.get_rng_state()),
        "cuda_rng_sha256": [_tensor_sha256(value) for value in cuda_states],
    }


def _arm_summary(state):
    return {
        key: state.get(key)
        for key in (
            "arm", "status", "completed_updates", "next_position", "checkpoint", "pause_reason",
            "history_sha256", "total_time_seconds", "peak_memory_bytes", "frozen_before", "frozen_after",
        )
    }


def run_all_arms(
    *, engine_factory, base_checkpoint, arms, schedule, prepared_by_arm, output, fingerprints,
    optimizer_factory=None, pause_file=None, stop_after=400, progress=None, coordinator_key=None,
):
    """Load and release one fresh base model per arm; never clone a full GPU model."""
    output = Path(output)
    key = coordinator_key or _canonical_hash({
        "base_checkpoint": str(Path(base_checkpoint)), "arms": list(arms),
        "schedule_sha256": _canonical_hash(schedule), "fingerprints": fingerprints, "stop_after": stop_after,
    })
    completed_arms, arm_starts, durable_summaries = [], {}, {}
    if output.exists():
        status_path = output / "status.json"
        if not status_path.is_file():
            raise FileExistsError(output)
        prior = json.loads(status_path.read_text())
        if prior.get("status") != "paused" or prior.get("coordinator_key") != key:
            raise FileExistsError(output)
        completed_arms = list(prior.get("completed_arms", []))
        arm_starts = dict(prior.get("arm_starts", {}))
        durable_summaries = dict(prior.get("summaries", {}))
    else:
        output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    coordinator_status = "running"
    before_arm = None
    for arm in arms:
        if arm in completed_arms:
            continue
        if pause_file and Path(pause_file).exists():
            coordinator_status, before_arm = "paused", arm
            break
        apply_determinism(SEED)
        engine = optimizer = None
        engine_reference = model_reference = None
        arm_failed = False
        retained_engine = False
        try:
            engine = engine_factory(base_checkpoint)
            engine_reference = weakref.ref(engine)
            model_reference = weakref.ref(engine.model)
            optimizer = (optimizer_factory or (lambda model: optimizer_for(
                model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))))(engine.model)
            start = _arm_start_identity(engine)
            baseline = next(iter(arm_starts.values()), start)
            for field in ("checkpoint_id", "trainable_weights_sha256", "cpu_rng_sha256", "cuda_rng_sha256"):
                if start[field] != baseline[field]:
                    raise RuntimeError(f"arm {arm} start {field} differs from the common base")
            arm_starts[arm] = start
            state = run_arm(
                arm, engine, schedule, prepared_by_arm[arm], output / arm, fingerprint=fingerprints[arm],
                optimizer=optimizer, pause_file=pause_file, stop_after=stop_after, progress=progress,
                arm_start=start,
            )
            summaries[arm] = state
            durable_summaries[arm] = _arm_summary(state)
            if state.get("status") == "complete" and state.get("completed_updates") == stop_after:
                completed_arms.append(arm)
            else:
                coordinator_status = "paused" if state.get("status") == "paused" else "failed"
                before_arm = arm
        except BaseException as error:
            arm_failed = True
            coordinator_status, before_arm = "failed", arm
            _write(output / "status.json", {
                "version": 2, "study": STUDY, "status": coordinator_status, "before_arm": before_arm,
                "completed_arms": completed_arms, "coordinator_key": key, "arm_starts": arm_starts,
                "summaries": durable_summaries, "error": repr(error),
            })
            raise
        finally:
            del optimizer, engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            retained_engine = (
                not arm_failed
                and engine_reference is not None
                and (engine_reference() is not None or model_reference() is not None)
            )
        if retained_engine:
            coordinator_status, before_arm = "failed", arm
            error = RuntimeError(f"arm {arm} engine remained live after release")
            _write(output / "status.json", {
                "version": 2, "study": STUDY, "status": coordinator_status, "before_arm": before_arm,
                "completed_arms": completed_arms, "coordinator_key": key, "arm_starts": arm_starts,
                "summaries": durable_summaries, "error": repr(error),
            })
            raise error
        if coordinator_status != "running":
            break
    if coordinator_status == "running":
        coordinator_status = "complete" if completed_arms == list(arms) else "paused"
    _write(output / "status.json", {
        "version": 2, "study": STUDY, "status": coordinator_status, "before_arm": before_arm,
        "completed_arms": completed_arms, "coordinator_key": key, "arm_starts": arm_starts,
        "summaries": durable_summaries,
    })
    return summaries


def _runtime_versions():
    result = {}
    for package in ("torch", "transformers", "peft", "fla-core"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _source_hashes(repo_root):
    return {path: _sha(Path(repo_root) / path) for path in PRODUCTION_SOURCES}


def build_fingerprint(manifest_path, manifest, arm, schedule, replay_path, prepared, repo_root=REPO_ROOT):
    replay_positions = [[{key: item[key] for key in ("primitive", "source", "id")} for item in row["replay"]]
                        for row in schedule]
    return {
        "study": STUDY, "arm": arm, "manifest_sha256": _sha(manifest_path),
        "train_sha256": manifest["arms"][arm]["sha256"], "suite_sha256": manifest["arms"][arm]["suite_sha256"],
        "evaluation_freeze_sha256": manifest["evaluation_freeze_sha256"],
        "base_checkpoint": manifest["base_checkpoint"], "base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
        "source_fingerprints": manifest["source_fingerprints"], "data_fingerprint": manifest["data_fingerprint"],
        "production_source_sha256": _source_hashes(repo_root), "schedule_sha256": _canonical_hash(schedule),
        "replay_position_sha256": _canonical_hash(replay_positions),
        "replay_path": Path(replay_path).relative_to(Path(repo_root)).as_posix(), "replay_sha256": _sha(replay_path),
        "prepared_prompt_fingerprint": prepared_prompt_fingerprint(prepared),
        "runtime": _runtime_versions(), "backend": "fla", "max_input_tokens": MAX_INPUT_TOKENS,
        "max_units": MAX_UNITS, "unit_batch_size": UNIT_BATCH_SIZE,
        "determinism": {
            "seed": SEED, "deterministic_algorithms": True, "cudnn_deterministic": True,
            "cudnn_benchmark": False, "tf32": False, "cublas_workspace_config": ":4096:8",
        },
    }


def common_fingerprint(fingerprint):
    """Return the smoke-gated environment shared by all four arm fingerprints."""
    arm_specific = {"arm", "train_sha256", "suite_sha256", "prepared_prompt_fingerprint", "schedule_sha256"}
    return {key: value for key, value in fingerprint.items() if key not in arm_specific}


def validate_smoke_gate_static(gate):
    proof = gate.get("resume_equivalence", {}) if isinstance(gate, dict) else {}
    valid_proof = (
        proof.get("passed") is True
        and type(proof.get("tolerance")) in (int, float)
        and proof["tolerance"] <= 1e-7
        and type(proof.get("max_probability_difference")) in (int, float)
        and proof["max_probability_difference"] <= 1e-7
        and type(proof.get("max_trainable_difference")) in (int, float)
        and proof["max_trainable_difference"] <= 1e-7
        and proof.get("frozen_before")
        and proof.get("frozen_before") == proof.get("frozen_after")
    )
    if not valid_proof:
        raise ValueError("full training requires strict <=1e-7 production resume equivalence")
    if not (gate.get("mode") == "smoke" and gate.get("status") == "complete"
            and gate.get("passed") is True and gate.get("completed_requested_pass") is True):
        raise ValueError("full training requires a completed strict smoke gate")


def validate_smoke_gate(gate, fingerprints):
    validate_smoke_gate_static(gate)
    expected = gate.get("common_fingerprint")
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError("full training fingerprints are missing")
    if any(common_fingerprint(value) != expected for value in fingerprints.values()):
        raise ValueError("smoke fingerprint differs from the requested training run")


def _new_engine(checkpoint):
    return load_judgment_engine(
        checkpoint=checkpoint, device="cuda", backend="fla", trainable=True,
        max_input_tokens=MAX_INPUT_TOKENS, unit_batch_size=UNIT_BATCH_SIZE,
    )


def _require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("smoke/train require the root-owned CUDA production runtime")


def _new_preparation_engine(checkpoint):
    """Load tokenizer/compiler state without constructing a model or touching CUDA."""
    from transformers import AutoTokenizer

    info = json.loads((Path(checkpoint) / "checkpoint.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(info["model_id"], revision=info["revision"])
    return JudgmentEngine(
        None, tokenizer, model_id=info["checkpoint_id"],
        max_input_tokens=MAX_INPUT_TOKENS, unit_batch_size=UNIT_BATCH_SIZE,
    )


def _manifest_path(data):
    path = Path(data)
    return path if path.is_file() else path / "manifest.json"


def _static_run_key(manifest_path, manifest, args):
    return _canonical_hash({
        "study": STUDY, "manifest_sha256": _sha(manifest_path),
        "base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
        "mode": args.mode, "arm": args.arm, "resume": str(args.resume) if args.resume else None,
    })


def _record_static_pause(output, *, key, before_arm, coordinator):
    output = Path(output)
    marker = output / "pristine-pause.json"
    value = {
        "version": 1, "study": STUDY, "status": "paused", "before_arm": before_arm,
        "static_run_key": key, "model_loaded": False,
    }
    if output.exists():
        if marker.is_file() and json.loads(marker.read_text()).get("static_run_key") == key:
            return value
        status_path = output / "status.json"
        if coordinator and status_path.is_file():
            status = json.loads(status_path.read_text())
            if status.get("coordinator_key") == key and status.get("status") == "paused":
                return status
        if not coordinator:
            _write(output / "pause-status.json", value)
            return value
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _write(marker, value)
    if coordinator:
        _write(output / "status.json", {
            **value, "coordinator_key": key, "completed_arms": [], "arm_starts": {}, "summaries": {},
        })
    return value


def run(args):
    """Execute preflight, strict production smoke, fresh training, or explicit resume."""
    repo_root = Path(getattr(args, "repo_root", REPO_ROOT)).resolve()
    manifest_path = _manifest_path(args.data).resolve()
    manifest = json.loads(manifest_path.read_text())
    records = validate_manifest(manifest, repo_root)
    declared_base = _repo_path(repo_root, manifest["base_checkpoint"], "base checkpoint")
    if Path(args.base_checkpoint).resolve() != declared_base:
        raise ValueError("--base-checkpoint differs from the manifest")
    output = Path(args.output)
    if args.mode == "preflight":
        output.mkdir(parents=True, exist_ok=False)
        report = {
            "study": STUDY, "mode": "preflight", "status": "complete", "manifest_sha256": _sha(manifest_path),
            "base_checkpoint_sha256": tree_sha256(declared_base), "arms": {arm: len(rows) for arm, rows in records.items()},
            "model_loaded": False, "test_outcomes_read": False,
        }
        _write(output / "report.json", report)
        return report
    if args.arm == "all" and args.mode == "smoke":
        raise ValueError("smoke requires one explicit arm")
    if args.resume and args.arm == "all":
        raise ValueError("--resume requires one explicit arm")
    if getattr(args, "recover_history_tail", False) and not args.resume:
        raise ValueError("--recover-history-tail requires --resume")
    gate = None
    if args.mode == "train":
        if args.smoke_gate is None:
            raise ValueError("full training requires --smoke-gate")
        try:
            gate = json.loads(Path(args.smoke_gate).read_text())
        except (OSError, ValueError) as error:
            raise ValueError("full training requires a readable smoke-gate JSON") from error
        validate_smoke_gate_static(gate)
    pause_file = Path(args.pause_file) if args.pause_file else output / "PAUSE"
    run_key = _static_run_key(manifest_path, manifest, args)
    pristine_marker = output / "pristine-pause.json"
    pristine_continuation = False
    if pause_file.exists():
        before_arm = ARMS[0] if args.arm == "all" else args.arm
        return _record_static_pause(
            output, key=run_key, before_arm=before_arm, coordinator=args.arm == "all" and not args.resume,
        )
    if pristine_marker.is_file():
        marker = json.loads(pristine_marker.read_text())
        if marker.get("static_run_key") != run_key:
            raise ValueError("pristine pause belongs to a different run declaration")
        pristine_continuation = True
    _require_cuda()
    replay_path = Path(args.replay).resolve()
    if repo_root != replay_path and repo_root not in replay_path.parents:
        raise ValueError("replay path must be inside the repository")
    apply_determinism(SEED)
    preparation_checkpoint = Path(args.resume) if args.resume else declared_base
    selected_arms = ARMS if args.arm == "all" or args.mode == "smoke" else (args.arm,)
    preparation = _new_preparation_engine(preparation_checkpoint)
    try:
        prepared_by_arm = {
            arm: prepare_families(preparation, records[arm], manifest["ordered_latent_ids"])
            for arm in selected_arms
        }
        replay_prepared, replay_pools = prepare_canonical_replay(preparation, replay_path)
    finally:
        del preparation
        gc.collect()
    schedule = make_schedule(manifest["ordered_latent_ids"], replay_pools)
    for arm in selected_arms:
        prepared_by_arm[arm].update(replay_prepared)
    fingerprints = {arm: build_fingerprint(manifest_path, manifest, arm, schedule, replay_path, prepared_by_arm[arm], repo_root)
                    for arm in selected_arms}
    if args.mode == "train":
        validate_smoke_gate(gate, fingerprints)

    if args.resume:
        if getattr(args, "recover_history_tail", False):
            recover_history_tail(
                args.resume, arm=args.arm, fingerprint=fingerprints[args.arm], output=output,
            )
        validate_resume_checkpoint(
            args.resume, arm=args.arm, fingerprint=fingerprints[args.arm], output=output,
        )
        engine = optimizer = None
        try:
            apply_determinism(SEED)
            engine = _new_engine(Path(args.resume))
            optimizer = optimizer_for(engine.model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
            state = restore_arm(
                args.resume, engine, optimizer, arm=args.arm, fingerprint=fingerprints[args.arm], output=output,
            )
            return run_arm(
                args.arm, engine, schedule, prepared_by_arm[args.arm], output,
                fingerprint=fingerprints[args.arm], optimizer=optimizer, start_state=state,
                pause_file=pause_file, stop_after=400,
            )
        finally:
            del optimizer, engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if args.mode == "smoke":
        if not pristine_continuation:
            output.mkdir(parents=True, exist_ok=False)
        arm = args.arm
        selection = select_smoke_families(records, prepared_by_arm)
        smoke_schedule, smoke_prepared = build_smoke_schedule(selection, prepared_by_arm, schedule)
        smoke_steps = args.smoke_steps or len(selection)
        if smoke_steps < len(selection) or smoke_steps >= 400:
            raise ValueError(f"--smoke-steps must cover all {len(selection)} selected families and leave a proof update")
        smoke_fingerprint = build_fingerprint(
            manifest_path, manifest, arm, smoke_schedule, replay_path, smoke_prepared, repo_root
        )
        engine = optimizer = None
        try:
            apply_determinism(SEED)
            engine = _new_engine(declared_base)
            optimizer = optimizer_for(engine.model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
            state = run_arm(
                arm, engine, smoke_schedule, smoke_prepared, output / arm,
                fingerprint=smoke_fingerprint, optimizer=optimizer, pause_file=pause_file,
                stop_after=smoke_steps,
            )
            if state["status"] != "complete":
                raise RuntimeError("strict smoke did not reach a next-update resume proof boundary")
            next_row = smoke_schedule[smoke_steps]
            replay = {item["primitive"]: smoke_prepared[item["id"]] for item in next_row["replay"]}
            proof = production_resume_equivalence(
                engine, optimizer, state["checkpoint"], _new_engine,
                smoke_prepared[next_row["family_id"]], replay,
                max_units=MAX_UNITS, unit_batch_size=UNIT_BATCH_SIZE, tolerance=1e-7,
            )
            report = {
                "study": STUDY, "mode": "smoke", "status": "complete", "passed": proof["passed"],
                "arm": arm, "completed_requested_pass": True,
                "common_fingerprint": common_fingerprint(smoke_fingerprint),
                "smoke_selection": selection, "smoke_updates": smoke_steps,
                "selection_is_outcome_independent": True, "training": state, "resume_equivalence": proof,
            }
            _write(output / "report.json", report)
            return report
        finally:
            del optimizer, engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if args.arm == "all":
        return run_all_arms(
            engine_factory=_new_engine, base_checkpoint=declared_base, arms=selected_arms, schedule=schedule,
            prepared_by_arm=prepared_by_arm, output=output, fingerprints=fingerprints, pause_file=pause_file,
            coordinator_key=run_key,
        )
    engine = optimizer = None
    try:
        apply_determinism(SEED)
        engine = _new_engine(declared_base)
        optimizer = optimizer_for(engine.model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
        return run_arm(
            args.arm, engine, schedule, prepared_by_arm[args.arm], output,
            fingerprint=fingerprints[args.arm], optimizer=optimizer, pause_file=pause_file,
            allow_pristine_output=pristine_continuation,
        )
    finally:
        del optimizer, engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Manifest path or directory containing manifest.json")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train"), default="preflight")
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--resume", type=Path, help="Explicit complete-boundary checkpoint directory")
    parser.add_argument(
        "--recover-history-tail", action="store_true",
        help="Archive a verified uncheckpointed/torn history tail before explicit resume",
    )
    parser.add_argument("--smoke-gate", type=Path, help="report.json from a matching strict production smoke")
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--replay", type=Path, default=Path("data/processed-v0.2/train.jsonl"))
    parser.add_argument("--smoke-steps", type=int, help="Optional updates; must cover the deterministic smoke selection")
    args = parser.parse_args()
    if args.smoke_steps is not None and (args.smoke_steps < 1 or args.smoke_steps >= 400):
        parser.error("--smoke-steps must be in 1..399")
    if args.mode == "preflight" and args.resume:
        parser.error("--resume is only valid with smoke/train execution")
    if args.resume and args.mode != "train":
        parser.error("--resume requires --mode train")
    if args.recover_history_tail and not args.resume:
        parser.error("--recover-history-tail requires --resume")
    report = run(args)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
