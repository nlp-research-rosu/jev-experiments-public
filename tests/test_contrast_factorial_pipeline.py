"""Operational gates for the approved contrast-factorial evaluation pipeline."""

import copy
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_sha(path):
    digest = hashlib.sha256()
    for item in sorted(Path(path).rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(b"\0")
            digest.update(item.read_bytes())
    return digest.hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


@lru_cache(maxsize=1)
def _tiny_production_optimizer_fixture():
    from experiments.judgment_pipeline import optimizer_for

    class TinyCheckpointModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Module()
            self.adapter.lora_A = torch.nn.Parameter(torch.ones(8, 2))
            self.adapter.lora_B = torch.nn.Parameter(torch.ones(2, 8))
            self.compatibility = torch.nn.Linear(2, 1, bias=False)
            self.binary = torch.nn.Linear(2, 1)

    model = TinyCheckpointModel()
    optimizer = optimizer_for(model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.01)
        optimizer.step()
    readouts = {
        "compatibility.weight": model.compatibility.weight.detach().clone(),
        "binary.weight": model.binary.weight.detach().clone(),
        "binary.bias": model.binary.bias.detach().clone(),
    }
    adapters = {
        "base_model.model.layer.proj.lora_A.weight": model.adapter.lora_A.detach().clone(),
        "base_model.model.layer.proj.lora_B.weight": model.adapter.lora_B.detach().clone(),
    }
    return readouts, adapters, copy.deepcopy(optimizer.state_dict())


def _checkpoint(path, *, model_id, revision, metadata, state=None):
    from safetensors.torch import save_file

    from openjev.judgment_model import checkpoint_identity

    path.mkdir(parents=True)
    tensors, adapter_tensors, optimizer = _tiny_production_optimizer_fixture()
    save_file(tensors, path / "readouts.safetensors")
    adapter = path / "adapter"
    adapter.mkdir()
    save_file(adapter_tensors, adapter / "adapter_model.safetensors")
    _write_json(adapter / "adapter_config.json", {"r": 8, "lora_alpha": 16, "target_modules": ["proj"]})
    info = {
        "format": "openjev-judgment-v0.2",
        "model_id": model_id,
        "revision": revision,
        "kernel_backend": "fla",
        "lora": {"rank": 8, "alpha": 16, "targets": ["proj"], "gradient_checkpointing": True},
        "metadata": metadata,
    }
    info["checkpoint_id"] = checkpoint_identity(path, info)
    _write_json(path / "checkpoint.json", info)
    if state is not None:
        _write_json(path / "state.json", state)
        torch.save({"optimizer": copy.deepcopy(optimizer), "state": state, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": [torch.arange(4, dtype=torch.uint8)]},
                   path / "training.pt")
    return info["checkpoint_id"]


def _endpoint_tree(tmp_path, *, completed=400, corrupt_checkpoint=False, wrong_freeze=False, wrong_exposure=False,
                   old_checkpoint=False, empty_weights=False, empty_adam=False, missing_start=False,
                   wrong_determinism=False):
    from experiments.contrast_factorial_training import _history_hash

    repo = tmp_path / "repo"
    output = repo / "reports/contrast-factorial-v1/pipeline"
    contract = repo / "docs/contrast-factorial-contract-v1.md"
    freeze = repo / "data/contrast-factorial-v1/evaluation-freeze.json"
    assessment = repo / "data/contrast-factorial-v1/assessment/suite.json"
    calibration = repo / "data/contrast-factorial-v1/calibration/suite.json"
    legacy_manifest = repo / "data/contrast-factorial-v1/legacy/manifest.json"
    legacy_cal = repo / "data/contrast-factorial-v1/legacy/calibration.json"
    legacy_ret = repo / "data/contrast-factorial-v1/legacy/retention.json"
    legacy_source = repo / "data/processed-v0.2/validation.jsonl"
    contract.parent.mkdir(parents=True)
    contract.write_text("approved contract\n")
    for path in (assessment, calibration, legacy_cal, legacy_ret):
        _write_json(path, {"fixture": path.name})
    legacy_source.parent.mkdir(parents=True, exist_ok=True)
    legacy_source.write_text("invented validation source metadata fixture\n")
    freeze_value = {
        "study": "contrast-factorial-v1",
        "status": "frozen",
        "suites": {
            "assessment": {"path": assessment.relative_to(repo).as_posix(), "suite_sha256": _sha(assessment),
                           "families": 80, "cases": 320, "judgments": 1600},
            "calibration": {"path": calibration.relative_to(repo).as_posix(), "suite_sha256": _sha(calibration),
                            "families": 40, "cases": 160, "judgments": 800},
        },
        "no_model_predictions_used": True,
        "unresolved_review_flags": 0,
    }
    _write_json(freeze, freeze_value)
    legacy_value = {
        "source": legacy_source.relative_to(repo).as_posix(),
        "source_sha256": _sha(legacy_source),
        "selection": "Invented validation-only fixture selected without labels.",
        "splits": {
            "calibration": {"file": legacy_cal.relative_to(repo).as_posix(), "sha256": _sha(legacy_cal), "records": 150,
                            "ids": [f"cal/{i}" for i in range(150)]},
            "retention": {"file": legacy_ret.relative_to(repo).as_posix(), "sha256": _sha(legacy_ret), "records": 300,
                          "ids": [f"ret/{i}" for i in range(300)]},
        }
    }
    _write_json(legacy_manifest, legacy_value)

    base = repo / "checkpoints/base"
    base_id = _checkpoint(base, model_id="fixture/model", revision="rev", metadata={})
    base_hash = _tree_sha(base)
    common_start = {
        "checkpoint_id": base_id,
        "trainable_weights_sha256": "1" * 64,
        "cpu_rng_sha256": "2" * 64,
        "cuda_rng_sha256": ["3" * 64],
    }
    approval = repo / "reports/contrast-factorial-v1/STUDY_APPROVAL.json"
    approval_value = {
        "study": "contrast-factorial-v1", "status": "approved", "arms": list("ABCD"), "updates_per_arm": 400,
        "base_checkpoint": base.relative_to(repo).as_posix(), "base_checkpoint_sha256": base_hash,
        "base_checkpoint_id": base_id, "model_id": "fixture/model", "revision": "rev",
        "evaluation_freeze_path": freeze.relative_to(repo).as_posix(),
        "evaluation_freeze_sha256": "0" * 64 if wrong_freeze else _sha(freeze),
        "contract_path": contract.relative_to(repo).as_posix(), "contract_sha256": _sha(contract),
        "expected_frozen_body_sha256": "f" * 64,
    }
    _write_json(approval, approval_value)

    ids = [f"latent/{i:03d}" for i in range(400)]
    source = repo / "experiments/contrast_factorial_training.py"
    source.parent.mkdir(parents=True)
    source.write_text("fixture runner\n")
    train_root = repo / "data/train"
    arms = {}
    for arm in "ABCD":
        train = train_root / f"{arm}.jsonl"
        suite = train_root / f"{arm}.json"
        train.parent.mkdir(parents=True, exist_ok=True)
        train.write_text("fixture train bytes\n")
        _write_json(suite, {"ordered_family_ids": ids})
        arms[arm] = {"train_file": train.relative_to(repo).as_posix(), "sha256": _sha(train),
                     "suite_file": suite.relative_to(repo).as_posix(), "suite_sha256": _sha(suite),
                     "ordered_family_ids": ids}
    manifest = {
        "study": "contrast-factorial-v1", "arms": arms, "ordered_latent_ids": ids,
        "base_checkpoint": base.relative_to(repo).as_posix(), "base_checkpoint_sha256": base_hash,
        "evaluation_freeze_path": freeze.relative_to(repo).as_posix(), "evaluation_freeze_sha256": _sha(freeze),
        "study_approval_path": approval.relative_to(repo).as_posix(), "study_approval_sha256": _sha(approval),
        "source_fingerprints": {source.relative_to(repo).as_posix(): _sha(source)},
        "data_fingerprint": {train.relative_to(repo).as_posix(): _sha(train)},
    }
    manifest_path = repo / "data/train/manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = _sha(manifest_path)

    run = repo / "checkpoints/contrast-factorial-v1/run"
    summaries = {}
    for arm in "ABCD":
        fingerprint = {
            "study": "contrast-factorial-v1", "arm": arm, "manifest_sha256": manifest_sha,
            "base_checkpoint_sha256": base_hash, "train_sha256": arms[arm]["sha256"],
            "suite_sha256": arms[arm]["suite_sha256"], "evaluation_freeze_sha256": _sha(freeze),
            "source_fingerprints": manifest["source_fingerprints"], "data_fingerprint": manifest["data_fingerprint"],
            "production_source_sha256": {source.relative_to(repo).as_posix(): _sha(source)},
            "runtime": {"torch": torch.__version__},
            "determinism": {"seed": 42, "deterministic_algorithms": True, "cudnn_deterministic": True,
                            "cudnn_benchmark": False, "tf32": wrong_determinism,
                            "cublas_workspace_config": ":4096:8"},
            "schedule_sha256": "a" * 64, "replay_position_sha256": "b" * 64,
            "replay_sha256": "c" * 64, "prepared_prompt_fingerprint": "d" * 64,
            "backend": "fla", "max_input_tokens": 1536, "max_units": 12, "unit_batch_size": 12,
        }
        history = []
        for step in range(1, completed + 1):
            history.append({
                "step": step,
                "component_counts": {"new/noul": 12, "new/choice": 4, "new/score": 4,
                                     "old/noul": 1, "old/choice": 1, "old/score": 1},
                "exposure": {"family_id": ids[step - 1],
                             "replay_ids": {"noul": f"old/n/{step}", "choice": f"old/c/{step}", "score": f"old/s/{step}"},
                             "candidate_units": 23,
                             "unpadded_tokens": 120, "padded_tokens": 119 if wrong_exposure else 132},
                "update_time_seconds": 0.01, "total_time_seconds": step * 0.01, "peak_memory_bytes": 123,
            })
        state = {
            "version": 2, "study": "contrast-factorial-v1", "arm": arm,
            "status": "complete" if completed == 400 else "paused", "completed_updates": completed,
            "next_position": completed, "complete_boundary": True, "resumable": True, "update_phase": "boundary",
            "history": history, "history_sha256": _history_hash(history), "fingerprint": fingerprint,
            "frozen_before": "f" * 64, "frozen_after": "f" * 64, "frozen_verified_at": completed,
            "total_time_seconds": completed * 0.01, "peak_memory_bytes": 123,
        }
        if not missing_start:
            state["arm_start"] = copy.deepcopy(common_start)
        checkpoint = run / arm / ("step-0399" if old_checkpoint else "step-0400")
        _checkpoint(
            checkpoint, model_id="fixture/model", revision="rev",
            metadata={"study": "contrast-factorial-v1", "arm": arm, "completed_updates": completed,
                      "update_phase": "boundary", "fingerprint": fingerprint}, state=state,
        )
        state["checkpoint"] = str(checkpoint)
        _write_json(checkpoint / "state.json", state)
        payload = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
        payload["state"] = state
        if empty_adam:
            payload["optimizer"] = {}
        torch.save(payload, checkpoint / "training.pt")
        if empty_weights:
            (checkpoint / "readouts.safetensors").unlink()
            for child in (checkpoint / "adapter").iterdir():
                child.unlink()
            (checkpoint / "adapter").rmdir()
            from openjev.judgment_model import checkpoint_identity

            info = json.loads((checkpoint / "checkpoint.json").read_text())
            info["checkpoint_id"] = checkpoint_identity(checkpoint, info)
            _write_json(checkpoint / "checkpoint.json", info)
        if corrupt_checkpoint:
            (checkpoint / "readouts.safetensors").write_bytes(b"corrupt after identity")
        _write_json(run / arm / "progress.json", {k: state[k] for k in (
            "study", "arm", "status", "completed_updates", "next_position", "complete_boundary", "resumable",
            "update_phase", "history_sha256", "frozen_before", "frozen_after", "frozen_verified_at")})
        summaries[arm] = {"status": state["status"], "completed_updates": completed, "checkpoint": str(checkpoint)}
    status = {"study": "contrast-factorial-v1", "status": "complete" if completed == 400 else "paused",
              "completed_arms": list("ABCD") if completed == 400 else [], "summaries": summaries}
    if not missing_start:
        status["arm_starts"] = {arm: copy.deepcopy(common_start) for arm in "ABCD"}
    _write_json(run / "status.json", status)
    return {"repo": repo, "output": output, "approval": approval, "manifest": manifest_path, "run": run,
            "freeze": freeze, "legacy_manifest": legacy_manifest}


def _lock(tree, **kwargs):
    from experiments.contrast_factorial_pipeline import create_endpoint_lock

    return create_endpoint_lock(repo_root=tree["repo"], approval_path=tree["approval"], manifest_path=tree["manifest"],
                                training_root=tree["run"], output_root=tree["output"],
                                legacy_manifest_path=tree["legacy_manifest"], **kwargs)


@pytest.mark.parametrize("change", [{"completed": 399}, {"corrupt_checkpoint": True}, {"wrong_freeze": True},
                                     {"wrong_exposure": True}, {"old_checkpoint": True}, {"empty_weights": True},
                                     {"empty_adam": True}, {"wrong_determinism": True}])
def test_endpoint_gate_refuses_incomplete_corrupt_or_misbound_training_before_model_call(tmp_path, change):
    """Relaxing any endpoint/freeze/exposure gate would let inference start on an invalid endpoint."""
    from experiments.contrast_factorial_pipeline import run_evaluation

    tree = _endpoint_tree(tmp_path, **change)
    called = []
    with pytest.raises((ValueError, RuntimeError)):
        _lock(tree)
    assert not (tree["output"] / "endpoint-lock.json").exists()
    with pytest.raises((FileNotFoundError, ValueError)):
        run_evaluation(tree["output"] / "endpoint-lock.json", tree["output"], engine_loader=lambda _: called.append(1))
    assert called == []


def _question(kind):
    if kind == "noul":
        return {"type": "noul", "instructions": "Is it true?"}
    if kind == "choice":
        return {"type": "choice", "instructions": "Choose.", "criteria": {"yes": "yes", "no": "no"}}
    return {"type": "score", "instructions": "Score.", "criteria": ["low", "high"]}


def _suite(prefix, *, relations=False, flip_labels=False):
    cases = []
    for family in range(4):
        for side in range(2):
            questions = {kind: _question(kind) for kind in ("noul", "choice", "score")}
            expected = {"noul": bool(side) ^ flip_labels, "choice": ("yes" if side else "no"), "score": side}
            if flip_labels:
                expected["choice"] = "no" if expected["choice"] == "yes" else "yes"
                expected["score"] = 1 - expected["score"]
            cases.append({"id": f"{prefix}/{family}/{side}", "family_id": f"{prefix}/{family}", "domain": "fixture",
                          "category": "fixture", "variant": str(side), "layout": "flat",
                          "request": {"state": {"side": side}, "questions": questions}, "expected": expected,
                          "rationale": {kind: "fixture" for kind in questions}})
    pairs = []
    if relations:
        for family in range(4):
            for kind in ("noul", "choice", "score"):
                pairs.append({"id": f"{prefix}/{family}/{kind}", "kind": "flip", "axis": "evidence",
                              "left": {"case_id": f"{prefix}/{family}/0", "question_id": kind},
                              "right": {"case_id": f"{prefix}/{family}/1", "question_id": kind}})
    return {"name": prefix, "cases": cases, "relations": pairs}


def _legacy(prefix):
    records = []
    for family in range(4):
        for kind in ("noul", "choice", "score"):
            q = _question(kind)
            target = {"truth": family % 2 == 0} if kind == "noul" else ({"choice": "yes"} if kind == "choice" else {"level_index": family % 2})
            records.append({"id": f"{prefix}/{family}/{kind}", "group_id": f"{prefix}/{family}",
                            "provenance": {"dataset": "fixture", "assigned_split": "validation"},
                            "examples": [{"state": {"family": family}, "question": q, "target": target}]})
    return records


class FakeEngine:
    active = 0
    peak = 0

    def __init__(self, identity, events):
        self.model_id = identity
        self.model = SimpleNamespace(checkpoint_id=identity)
        self.events = events
        self.closed = False
        FakeEngine.active += 1
        FakeEngine.peak = max(FakeEngine.peak, FakeEngine.active)

    def evaluate(self, request, *, cached, details):
        assert cached is False and details is True
        self.events.append("model-call")
        answers = {}
        for qid, question in request["questions"].items():
            if question["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.8, "probabilities": {"false": 0.2, "true": 0.8},
                                "details": {"link": "sigmoid", "raw_logit": math.log(4),
                                                         "calibrated_logit": math.log(4), "temperature": 1.0,
                                                         "calibration_id": "identity-uncalibrated"}}
            else:
                labels = [str(i) for i in range(len(question["criteria"]))] if question["type"] == "score" else list(question["criteria"])
                answers[qid] = {"type": question["type"], "probabilities": {labels[0]: 0.2, labels[1]: 0.8},
                                "details": {"link": "softmax", "raw_logits": {labels[0]: 0.0, labels[1]: math.log(4)},
                                            "calibrated_logits": {labels[0]: 0.0, labels[1]: math.log(4)},
                                            "temperature": 1.0, "calibration_id": "identity-uncalibrated"}}
                if question["type"] == "score":
                    answers[qid]["score"] = 0.8
                else:
                    answers[qid]["choice"] = labels[1]
        return {"model": self.model_id, "answers": answers,
                "usage": {"max_unit_tokens": 12, "model_units": len(answers), "truncated": False}}

    def close(self):
        assert not self.closed
        self.closed = True
        FakeEngine.active -= 1
        self.events.append("engine-close")


def _evaluation_tree(tmp_path, *, assessment_flip=False, legacy_format="records"):
    from experiments.revised_evaluation import broad_suite

    tree = _endpoint_tree(tmp_path)
    lock = _lock(tree)
    partitions = lock["partitions"]
    paths = {name: tree["repo"] / spec["path"] for name, spec in partitions.items()}
    _write_json(paths["calibration"], _suite("cal"))
    _write_json(paths["assessment"], _suite("assess", relations=True, flip_labels=assessment_flip))
    legacy_values = {"legacy_calibration": _legacy("broad-cal"),
                     "legacy_retention": _legacy("broad-ret")}
    if legacy_format == "suite":
        legacy_values = {name: broad_suite(value) for name, value in legacy_values.items()}
    elif legacy_format != "records":
        raise ValueError("unknown legacy fixture format")
    for name, value in legacy_values.items():
        _write_json(paths[name], value)
    legacy_manifest = json.loads(paths["legacy_manifest"].read_text())
    for split, name in (("calibration", "legacy_calibration"), ("retention", "legacy_retention")):
        value = legacy_values[name]
        ids = [case["id"] for case in value["cases"]] if isinstance(value, dict) else [record["id"] for record in value]
        sources = [case["domain"] for case in value["cases"]] if isinstance(value, dict) else [
            record["provenance"]["dataset"] for record in value]
        legacy_manifest["splits"][split]["records"] = len(ids)
        legacy_manifest["splits"][split]["ids"] = ids
        legacy_manifest["splits"][split]["sources"] = {source: sources.count(source) for source in sorted(set(sources))}
        legacy_manifest["splits"][split]["sha256"] = _sha(paths[name])
    _write_json(paths["legacy_manifest"], legacy_manifest)
    for name, path in paths.items():
        if name == "legacy_manifest":
            partitions[name]["sha256"] = _sha(path)
            continue
        value = json.loads(path.read_text())
        partitions[name]["sha256"] = _sha(path)
        if name.startswith("legacy"):
            identities = [case["id"] for case in value["cases"]] if isinstance(value, dict) else [record["id"] for record in value]
            partitions[name]["records"] = len(identities)
            partitions[name]["record_ids_sha256"] = _canonical(identities)
        else:
            partitions[name]["families"] = len({case["family_id"] for case in value["cases"]})
            partitions[name]["cases"] = len(value["cases"])
            partitions[name]["judgments"] = sum(len(case["expected"]) for case in value["cases"])
    lock["lock_sha256"] = _canonical({k: v for k, v in lock.items() if k != "lock_sha256"})
    _write_json(tree["output"] / "endpoint-lock.json", lock)
    return tree


def test_frozen_legacy_suite_format_is_normalized_once_across_evaluation_and_reporting(tmp_path):
    """Already materialized broad suites must not be rejected or passed through broad_suite again."""
    from experiments.contrast_factorial_pipeline import build_final_report, run_evaluation

    tree = _evaluation_tree(tmp_path, legacy_format="suite")
    result = run_evaluation(tree["output"] / "endpoint-lock.json", tree["output"],
                            engine_loader=lambda endpoint: FakeEngine(endpoint["checkpoint_id"], []))
    report = build_final_report(tree["output"])
    assert result["status"] == report["status"] == "complete"
    assert report["models"]["BASE"]["variants"]["raw"]["retention"]["overall"]["questions"] == 12


def test_frozen_legacy_suite_still_enforces_manifest_order_before_model_load(tmp_path):
    """A valid suite envelope cannot bypass the frozen ordered identity population."""
    from experiments.contrast_factorial_pipeline import run_evaluation

    tree = _evaluation_tree(tmp_path, legacy_format="suite")
    lock_path = tree["output"] / "endpoint-lock.json"
    lock = json.loads(lock_path.read_text())
    suite_path = tree["repo"] / lock["partitions"]["legacy_calibration"]["path"]
    suite = json.loads(suite_path.read_text())
    suite["cases"][0], suite["cases"][1] = suite["cases"][1], suite["cases"][0]
    _write_json(suite_path, suite)
    manifest_path = tree["repo"] / lock["partitions"]["legacy_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["calibration"]["sha256"] = _sha(suite_path)
    _write_json(manifest_path, manifest)
    lock["partitions"]["legacy_calibration"]["sha256"] = _sha(suite_path)
    lock["partitions"]["legacy_manifest"]["sha256"] = _sha(manifest_path)
    lock["lock_sha256"] = _canonical({key: value for key, value in lock.items() if key != "lock_sha256"})
    _write_json(lock_path, lock)
    called = []
    with pytest.raises(ValueError, match="order/identity"):
        run_evaluation(lock_path, tree["output"], engine_loader=lambda endpoint: called.append(endpoint))
    assert called == []


def test_frozen_legacy_suite_requires_locked_validation_source_linkage(tmp_path):
    """Suite provenance comes from the frozen validation manifest/source, never the suite envelope alone."""
    from experiments.contrast_factorial_pipeline import run_evaluation

    tree = _evaluation_tree(tmp_path, legacy_format="suite")
    lock_path = tree["output"] / "endpoint-lock.json"
    lock = json.loads(lock_path.read_text())
    manifest_path = tree["repo"] / lock["partitions"]["legacy_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    test_source = tree["repo"] / "data/finaltest/legacy.jsonl"
    test_source.parent.mkdir(parents=True)
    test_source.write_text("quarantined fixture\n")
    manifest["source"] = test_source.relative_to(tree["repo"]).as_posix()
    manifest["source_sha256"] = _sha(test_source)
    _write_json(manifest_path, manifest)
    lock["partitions"]["legacy_manifest"]["sha256"] = _sha(manifest_path)
    lock["lock_sha256"] = _canonical({key: value for key, value in lock.items() if key != "lock_sha256"})
    _write_json(lock_path, lock)
    called = []
    with pytest.raises(ValueError, match="validation-only"):
        run_evaluation(lock_path, tree["output"], engine_loader=lambda endpoint: called.append(endpoint))
    assert called == []


def test_calibration_precedes_assessment_and_five_models_never_overlap(tmp_path):
    """Reading assessment early or retaining two engines violates the frozen two-stage protocol."""
    from experiments.contrast_factorial_pipeline import run_evaluation

    tree = _evaluation_tree(tmp_path)
    events = []
    FakeEngine.active = FakeEngine.peak = 0

    def loader(path):
        value = json.loads(Path(path).read_text())
        events.append("read-assessment" if "assess" in str(path) else "read-calibration")
        return value

    def engine_loader(endpoint):
        events.append("engine-open")
        return FakeEngine(endpoint["checkpoint_id"], events)

    result = run_evaluation(tree["output"] / "endpoint-lock.json", tree["output"],
                            engine_loader=engine_loader, json_loader=loader)
    assert result["status"] == "complete"
    assert FakeEngine.active == 0 and FakeEngine.peak == 1
    assert events.index("read-assessment") > events.index("engine-close")
    assert (tree["output"] / "calibration-lock.json").exists()
    assert len(result["models"]) == 5


def test_saved_noul_logits_are_fit_once_and_assessment_labels_cannot_change_temperatures(tmp_path):
    """Using probabilities as logits or assessment gold in fitting changes the locked temperatures."""
    from experiments.contrast_factorial_pipeline import run_evaluation

    temperatures = []
    for index, flip in enumerate((False, True)):
        tree = _evaluation_tree(tmp_path / str(index), assessment_flip=flip)
        events = []
        run_evaluation(tree["output"] / "endpoint-lock.json", tree["output"],
                       engine_loader=lambda endpoint: FakeEngine(endpoint["checkpoint_id"], events))
        lock = json.loads((tree["output"] / "calibration-lock.json").read_text())
        temperatures.append({model: value["temperatures"] for model, value in lock["models"].items()})
        first = json.loads((tree["output"] / "predictions" / "BASE" / "calibration-new" /
                            "calibration-records.json").read_text())[0]
        assert first["logits"] == pytest.approx([0.0, math.log(4)])
        assert first["row"]["prediction"]["probabilities"]["true"] == pytest.approx(0.8)
        assert first["row"]["prediction"]["nll"] == pytest.approx(math.log(5))
    assert temperatures[0] == temperatures[1]


def test_interrupted_stage_is_preserved_and_only_exact_complete_binding_resumes(tmp_path):
    """A crash must not erase raw responses, and a completed stage cannot be reused for another binding."""
    from experiments.contrast_factorial_pipeline import run_prediction_stage

    suite = _suite("interrupt")
    output = tmp_path / "stage"

    class Interrupt(FakeEngine):
        def evaluate(self, request, *, cached, details):
            if sum(1 for event in self.events if event == "model-call") == 1:
                raise RuntimeError("stop")
            return super().evaluate(request, cached=cached, details=details)

    with pytest.raises(RuntimeError, match="stop"):
        run_prediction_stage(Interrupt("id", []), suite, output, binding={"x": 1, "checkpoint_id": "id"}, origin="new")
    assert (output / "responses.jsonl").read_text().count("\n") == 1
    assert json.loads((output / "stage.json").read_text())["status"] == "incomplete"
    result = run_prediction_stage(FakeEngine("id", []), suite, output,
                                  binding={"x": 1, "checkpoint_id": "id"}, origin="new")
    assert result["directory"].endswith("stage.attempt-0002")
    resumed = run_prediction_stage(FakeEngine("id", []), suite, Path(result["directory"]),
                                   binding={"x": 1, "checkpoint_id": "id"}, origin="new")
    assert resumed["resumed"] is True
    with pytest.raises(ValueError, match="binding"):
        run_prediction_stage(FakeEngine("id", []), suite, Path(result["directory"]),
                             binding={"x": 2, "checkpoint_id": "id"}, origin="new")


def test_cli_runs_tiny_end_to_end_and_writes_report(tmp_path, monkeypatch):
    """The public CLI must orchestrate locks, both inference phases, calibration, and reporting."""
    from experiments import contrast_factorial_pipeline as pipeline

    tree = _evaluation_tree(tmp_path)
    events = []
    monkeypatch.setattr(pipeline, "load_factory", lambda _: lambda endpoint: FakeEngine(endpoint["checkpoint_id"], events))
    code = pipeline.main(["evaluate", "--endpoint-lock", str(tree["output"] / "endpoint-lock.json"),
                          "--output-root", str(tree["output"]), "--engine-factory", "fixture:factory"])
    assert code == 0
    code = pipeline.main(["report", "--output-root", str(tree["output"])])
    assert code == 0
    report = json.loads((tree["output"] / "final-report/report.json").read_text())
    assert report["status"] == "complete" and set(report["models"]) == {"BASE", "A", "B", "C", "D"}
    text = (tree["output"] / "final-report/RESULTS.md").read_text()
    assert "Equal updates do not imply equal compute" in text
    assert str((tree["output"] / "endpoint-lock.json").resolve()) in text


def test_training_launcher_holds_parent_lock_tracks_logs_and_stops_on_failed_smoke(tmp_path):
    """A failed smoke must prevent the four-arm child and remain durably visible."""
    import importlib.util
    import sys

    wrapper = Path(__file__).parents[1] / "reports/contrast-factorial-v1/run_pipeline.py"
    spec = importlib.util.spec_from_file_location("factorial_run_pipeline", wrapper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    program = tmp_path / "fake_training.py"
    program.write_text(
        "import argparse,json,sys\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--mode'); p.add_argument('--output',type=Path); "
        "p.add_argument('--arm'); p.add_argument('--data'); p.add_argument('--base-checkpoint'); "
        "p.add_argument('--replay'); p.add_argument('--smoke-gate',default=None); a=p.parse_args()\n"
        "a.output.mkdir(parents=True); (a.output/'report.json').write_text(json.dumps({'status':'failed' if a.mode=='smoke' else 'complete'}))\n"
        "sys.exit(7 if a.mode=='smoke' else 0)\n"
    )
    root = tmp_path / "pipeline"
    with pytest.raises(RuntimeError, match="smoke"):
        module.main(["--pipeline-root", str(root), "--training-output", str(tmp_path / "train"),
                     "--data", "fixture-data", "--base-checkpoint", "fixture-base", "--replay", "fixture-replay",
                     "--python", sys.executable, "--training-program", str(program)])
    status = json.loads((root / "pipeline-status.json").read_text())
    assert status["status"] == "failed" and status["failed_stage"] == "smoke"
    assert [row["stage"] for row in status["stages"]] == ["preflight", "smoke"]
    assert (root / "logs/preflight.log").is_file() and (root / "logs/smoke.log").is_file()
    assert not (tmp_path / "train").exists()
    with pytest.raises(FileExistsError, match="--continue-from"):
        module.main(["--pipeline-root", str(root), "--training-output", str(tmp_path / "train"),
                     "--data", "fixture-data", "--base-checkpoint", "fixture-base", "--replay", "fixture-replay",
                     "--python", sys.executable, "--training-program", str(program)])


def test_legacy_test_path_is_rejected_without_opening_it(tmp_path):
    """Changing a locked legacy path to quarantined TEST material must fail before the loader reads it."""
    from experiments.contrast_factorial_pipeline import _load_partition

    path = tmp_path / "repo/data/finaltest/records.json"
    _write_json(path, [])
    lock = {"repo_root": str(tmp_path / "repo"), "partitions": {
        "legacy_retention": {"path": "data/finaltest/records.json", "sha256": _sha(path), "records": 0}
    }}
    opened = []
    with pytest.raises(ValueError, match="quarantined"):
        _load_partition(lock, "legacy_retention", lambda value: opened.append(value))
    assert opened == []


def test_real_no_close_engine_is_destroyed_before_next_model_load(tmp_path):
    """A caller-local reference must not retain the previous production engine at the next load."""
    import gc
    import weakref

    from experiments.contrast_factorial_pipeline import run_evaluation
    from openjev.judgment_model import JudgmentEngine

    tree = _evaluation_tree(tmp_path)
    references, live_at_load = [], []

    class NoCloseEngine(JudgmentEngine):
        evaluate = FakeEngine.evaluate

    def loader(endpoint):
        gc.collect()
        live_at_load.append(sum(reference() is not None for reference in references))
        engine = NoCloseEngine.__new__(NoCloseEngine)
        engine.model_id = endpoint["checkpoint_id"]
        engine.model = SimpleNamespace(checkpoint_id=endpoint["checkpoint_id"])
        engine.events = []
        references.append(weakref.ref(engine))
        return engine

    run_evaluation(tree["output"] / "endpoint-lock.json", tree["output"], engine_loader=loader)
    assert live_at_load == [0] * 10


def test_evaluation_refuses_a_concurrent_lock_but_accepts_checked_inherited_fd(tmp_path, monkeypatch):
    """Direct evaluation must share the launcher's exclusive model-process lock."""
    import fcntl

    from experiments.contrast_factorial_pipeline import run_evaluation

    tree = _evaluation_tree(tmp_path)
    called = []
    lock = tree["output"] / "gpu.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="lock"):
            run_evaluation(tree["output"] / "endpoint-lock.json", tree["output"],
                           engine_loader=lambda endpoint: called.append(endpoint))
        assert called == []
        monkeypatch.setenv("OPENJEV_CONTRAST_FACTORIAL_LOCK_FD", str(handle.fileno()))
        result = run_evaluation(
            tree["output"] / "endpoint-lock.json", tree["output"],
            engine_loader=lambda endpoint: FakeEngine(endpoint["checkpoint_id"], []),
        )
        assert result["status"] == "complete"


def test_reporting_and_resume_reject_changed_or_missing_locked_calibration_evidence(tmp_path):
    """A calibration lock must bind the exact fit and selected prediction attempts at every later boundary."""
    from experiments.contrast_factorial_pipeline import build_final_report, run_evaluation

    tree = _evaluation_tree(tmp_path)
    run_evaluation(tree["output"] / "endpoint-lock.json", tree["output"],
                   engine_loader=lambda endpoint: FakeEngine(endpoint["checkpoint_id"], []))
    calibration = json.loads((tree["output"] / "calibration-lock.json").read_text())
    fit_path = Path(calibration["models"]["BASE"]["fit_path"])
    fit = json.loads(fit_path.read_text())
    fit["temperatures"]["global"] = 7.0
    _write_json(fit_path, fit)
    with pytest.raises(ValueError, match="fit"):
        build_final_report(tree["output"])

    # Restore the fit, then remove evidence named by the immutable lock.
    fit["temperatures"]["global"] = calibration["models"]["BASE"]["temperatures"]["global"]
    _write_json(fit_path, fit)
    records = tree["output"] / "predictions/BASE/calibration-new/calibration-records.json"
    records.rename(records.with_suffix(".json.preserved"))
    with pytest.raises(ValueError, match="calibration"):
        run_evaluation(tree["output"] / "endpoint-lock.json", tree["output"],
                       engine_loader=lambda endpoint: FakeEngine(endpoint["checkpoint_id"], []))


def test_reporting_rejects_swapped_completed_stage_binding(tmp_path):
    """A complete artifact cannot be relabeled as another model or partition in evaluation status."""
    from experiments.contrast_factorial_pipeline import build_final_report, run_evaluation

    tree = _evaluation_tree(tmp_path)
    run_evaluation(tree["output"] / "endpoint-lock.json", tree["output"],
                   engine_loader=lambda endpoint: FakeEngine(endpoint["checkpoint_id"], []))
    status_path = tree["output"] / "evaluation-status.json"
    status = json.loads(status_path.read_text())
    status["models"]["A"]["assessment"] = copy.deepcopy(status["models"]["B"]["assessment"])
    _write_json(status_path, status)
    with pytest.raises(ValueError, match="binding"):
        build_final_report(tree["output"])


def test_endpoint_lock_rejects_missing_or_disagreeing_common_start_evidence(tmp_path):
    """The independent endpoint gate must prove all arms began at the same approved state and RNG."""
    tree = _endpoint_tree(tmp_path, missing_start=True)
    with pytest.raises(ValueError, match="start"):
        _lock(tree)

    tree = _endpoint_tree(tmp_path / "mismatch")
    status = json.loads((tree["run"] / "status.json").read_text())
    checkpoint_state = json.loads((tree["run"] / "A/step-0400/state.json").read_text())
    status["arm_starts"] = {arm: copy.deepcopy(checkpoint_state["arm_start"]) for arm in "ABCD"}
    status["arm_starts"]["D"]["checkpoint_id"] = "wrong"
    _write_json(tree["run"] / "status.json", status)
    with pytest.raises(ValueError, match="start"):
        _lock(tree)


def test_endpoint_lock_rejects_nonrestorable_adamw_group_recipe(tmp_path):
    """Moment coverage without the production AdamW options is not a restorable endpoint."""
    tree = _endpoint_tree(tmp_path)
    valid = torch.load(tree["run"] / "A/step-0400/training.pt", map_location="cpu", weights_only=False)["optimizer"]
    parameters = {identity: torch.nn.Parameter(torch.ones_like(state["exp_avg"]))
                  for identity, state in valid["state"].items()}
    groups = [{"params": [parameters[identity] for identity in group["params"]], "lr": group["lr"]}
              for group in valid["param_groups"]]
    restored = torch.optim.AdamW(groups, weight_decay=0.01)
    restored.load_state_dict(valid)
    for parameter in parameters.values():
        parameter.grad = torch.zeros_like(parameter)
    restored.step()
    for arm in "ABCD":
        path = tree["run"] / arm / "step-0400/training.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for group in payload["optimizer"]["param_groups"]:
            group.pop("betas", None)
        torch.save(payload, path)
    with pytest.raises(ValueError, match="Adam"):
        _lock(tree)


def test_prediction_stage_requires_identity_diagnostics_and_uses_raw_logits_for_nll(tmp_path):
    """Declared calibrated responses and inconsistent effective logits cannot contaminate raw metrics."""
    from experiments.contrast_factorial_pipeline import run_prediction_stage

    class Miscalibrated(FakeEngine):
        def evaluate(self, *args, **kwargs):
            response = super().evaluate(*args, **kwargs)
            for answer in response["answers"].values():
                details = answer["details"]
                details.update(temperature=2.0, calibration_id="temperature-v1")
                if "raw_logit" in details:
                    details["calibrated_logit"] = 100.0
                else:
                    details["calibrated_logits"] = {key: value * 100 for key, value in details["raw_logits"].items()}
            return response

    with pytest.raises(ValueError, match="identity"):
        run_prediction_stage(Miscalibrated("id", []), _suite("diagnostic"), tmp_path / "bad",
                             binding={"checkpoint_id": "id"}, origin="new")


def test_explicit_preflight_continuation_versions_log_and_preserves_original(tmp_path):
    """An explicit safe restart must retain its prior log and use a numbered attempt path."""
    import importlib.util
    import sys

    wrapper = Path(__file__).parents[1] / "reports/contrast-factorial-v1/run_pipeline.py"
    spec = importlib.util.spec_from_file_location("factorial_continuation", wrapper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "pipeline"
    (root / "logs").mkdir(parents=True)
    (root / "logs/preflight.log").write_text("preserved first attempt\n")
    _write_json(root / "pipeline-status.json", {"study": "contrast-factorial-v1", "status": "failed"})
    program = tmp_path / "fake_training.py"
    program.write_text(
        "import argparse,json,sys\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--mode');p.add_argument('--output',type=Path);a,x=p.parse_known_args();"
        "a.output.mkdir(parents=True);(a.output/'report.json').write_text(json.dumps({'status':'complete','passed':True,'completed_requested_pass':True}));"
        "sys.exit(9 if a.mode=='smoke' else 0)\n"
    )
    with pytest.raises(RuntimeError, match="smoke"):
        module.main(["--pipeline-root", str(root), "--training-output", str(tmp_path / "train"),
                     "--data", "fixture", "--base-checkpoint", "fixture", "--replay", "fixture",
                     "--python", sys.executable, "--training-program", str(program),
                     "--continue-from", "preflight", "--continuation-note", "retry safe preflight"])
    assert (root / "logs/preflight.log").read_text() == "preserved first attempt\n"
    assert (root / "logs/preflight.attempt-0002.log").is_file()


def test_parent_termination_cannot_leave_an_active_child_without_the_study_lock(tmp_path):
    """SIGTERM to the parent must forward, wait, and keep the lock owned until the child exits."""
    import fcntl
    import os
    import signal
    import subprocess
    import sys
    import time

    root = tmp_path / "pipeline"
    child_program = tmp_path / "waiting_child.py"
    child_program.write_text(
        "import argparse,json,os,time\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a,x=p.parse_known_args();"
        "a.output.mkdir(parents=True);fd=int(os.environ['OPENJEV_CONTRAST_FACTORIAL_LOCK_FD']);"
        "s=os.fstat(fd);pstat=(a.output.parents[1]/'gpu.lock').stat();"
        "(a.output/'lock-inherited').write_text(json.dumps([s.st_dev,s.st_ino]==[pstat.st_dev,pstat.st_ino]));"
        "(a.output/'child-pid').write_text(str(os.getpid()));time.sleep(30)\n"
    )
    wrapper = Path(__file__).parents[1] / "reports/contrast-factorial-v1/run_pipeline.py"
    parent = subprocess.Popen(
        [sys.executable, str(wrapper), "--pipeline-root", str(root), "--training-output", str(tmp_path / "train"),
         "--data", "fixture", "--base-checkpoint", "fixture", "--replay", "fixture",
         "--training-program", str(child_program)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    child_pid = None
    try:
        marker = root / "training/preflight/child-pid"
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        child_pid = int(marker.read_text())
        assert json.loads((root / "training/preflight/lock-inherited").read_text()) is True
        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=10)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        with (root / "gpu.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_train_continuation_requires_manual_arm_recovery_without_mutating_status(tmp_path):
    """The wrapper must not imply that coordinator restart can recover an interrupted arm."""
    import importlib.util

    wrapper = Path(__file__).parents[1] / "reports/contrast-factorial-v1/run_pipeline.py"
    spec = importlib.util.spec_from_file_location("factorial_manual_recovery", wrapper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "pipeline"
    original = {"study": "contrast-factorial-v1", "status": "failed", "failed_stage": "train"}
    _write_json(root / "pipeline-status.json", original)
    before = (root / "pipeline-status.json").read_bytes()
    with pytest.raises(RuntimeError, match="manual --arm/--resume recovery"):
        module.main(["--pipeline-root", str(root), "--training-output", str(tmp_path / "train"),
                     "--data", "fixture", "--base-checkpoint", "fixture", "--replay", "fixture",
                     "--continue-from", "train", "--continuation-note", "recover"])
    assert (root / "pipeline-status.json").read_bytes() == before


def test_default_launcher_continues_past_training_into_endpoint_stage(tmp_path):
    """Four-arm completion alone must not mark the approved study pipeline complete."""
    import importlib.util
    import sys

    wrapper = Path(__file__).parents[1] / "reports/contrast-factorial-v1/run_pipeline.py"
    spec = importlib.util.spec_from_file_location("factorial_full_chain", wrapper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program = tmp_path / "successful_training.py"
    program.write_text(
        "import argparse,json\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--mode');p.add_argument('--output',type=Path);a,x=p.parse_known_args();"
        "a.output.mkdir(parents=True);"
        "v={'status':'complete','passed':True,'completed_requested_pass':True} if a.mode!='train' else "
        "{'status':'complete','completed_arms':list('ABCD')};"
        "(a.output/('status.json' if a.mode=='train' else 'report.json')).write_text(json.dumps(v))\n"
    )
    root = tmp_path / "pipeline"
    with pytest.raises(RuntimeError, match="endpoint"):
        module.main(["--pipeline-root", str(root), "--training-output", str(tmp_path / "train"),
                     "--data", str(tmp_path / "missing-manifest.json"), "--base-checkpoint", "fixture",
                     "--replay", "fixture", "--python", sys.executable, "--training-program", str(program)])
    status = json.loads((root / "pipeline-status.json").read_text())
    assert status["status"] == "failed" and status["failed_stage"] == "endpoint"
    assert [stage["stage"] for stage in status["stages"]] == ["preflight", "smoke", "train", "endpoint"]
