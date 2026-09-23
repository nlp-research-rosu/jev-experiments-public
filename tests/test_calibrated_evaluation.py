"""CPU-only locks, fit ordering and paired arithmetic for the development screen."""
import copy
import importlib
import json

import pytest
import torch
from test_contrast_factorial_pipeline import (
    FakeEngine,
    _canonical,
    _checkpoint,
    _evaluation_tree,
    _sha,
    _tiny_production_optimizer_fixture,
    _write_json,
)


def module():
    return importlib.import_module("experiments.calibrated_evaluation")


def fixture(tmp_path, arms=("H0", "H1"), teacher_capture=None):
    tree = _evaluation_tree(tmp_path)
    repo = tree["repo"]
    old_lock = tree["output"] / "endpoint-lock.json"
    protocol = repo / "reports/calibrated-screen-v1/PROTOCOL.md"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("Authorized calibrated-screen-v1 exploratory development protocol\n")
    source = repo / "experiments/calibrated_training.py"
    source.write_text("fixture new runner\n")
    targets_source = repo / "experiments/calibrated_targets.py"
    targets_source.write_text("fixture target binder\n")
    integrity_source = repo / "experiments/calibrated_integrity.py"
    integrity_source.write_text("fixture integrity binder\n")
    preserved = protocol.parent / "preserved-artifacts.json"
    _write_json(preserved, {"files": {old_lock.relative_to(repo).as_posix(): _sha(old_lock)}})
    from experiments.contrast_factorial_training import PRODUCTION_SOURCES
    for relative in PRODUCTION_SOURCES:
        path = repo / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture production source\n")
    manifest = json.loads(tree["manifest"].read_text())
    if teacher_capture:
        suite, target = teacher_capture
        (repo / manifest["arms"]["B"]["suite_file"]).write_bytes(suite.read_bytes())
        manifest["arms"]["B"]["suite_sha256"] = _sha(suite)
        _write_json(tree["manifest"], manifest)
        prior = json.loads(old_lock.read_text())
        prior["manifest"]["sha256"] = _sha(tree["manifest"])
        prior["lock_sha256"] = _canonical({k: v for k, v in prior.items() if k != "lock_sha256"})
        _write_json(old_lock, prior)
        _write_json(preserved, {"files": {old_lock.relative_to(repo).as_posix(): _sha(old_lock)}})
    old_state = json.loads((tree["run"] / "B/step-0400/state.json").read_text())
    common = {**old_state["arm_start"], "optimizer_state_entries": 0,
              "optimizer_initial_sha256": _canonical({"state": {}, "param_groups": _tiny_production_optimizer_fixture()[2]["param_groups"]})}
    common_path = protocol.parent / "common-start.json"
    _write_json(common_path, common)
    run = repo / "checkpoints/calibrated-screen-v1/run-v1"
    replay = repo / "data/replay.jsonl"
    replay.write_text("fixture replay\n")
    for arm in arms:
        state = copy.deepcopy(old_state)
        fp = state["fingerprint"]
        fp.update(study="calibrated-screen-v1", arm=arm, training_source_arm="B",
                  targets_sha256=_sha(teacher_capture[1]) if teacher_capture else None,
                  manifest_sha256=_sha(tree["manifest"]), suite_sha256=manifest["arms"]["B"]["suite_sha256"], loss={"alpha": 0.0 if arm[0] == "H" else 0.5, "brier_lambda": float(arm[-1]),
                        "teacher": None if arm[0] == "H" else arm[0], "teacher_temperature": 1.0,
                        "component_weight": 1 / 6, "replay": "unchanged-hard-CE"},
                  production_source_sha256={p: _sha(repo / p) for p in PRODUCTION_SOURCES},
                  calibrated_source_sha256={p.relative_to(repo).as_posix(): _sha(p)
                                            for p in (source, targets_source, integrity_source, protocol)},
                  updates=400, adapter_lr=5e-5, head_lr=2.5e-5, gradient_clip_norm=1.0,
                  replay_path=replay.relative_to(repo).as_posix(), replay_sha256=_sha(replay))
        state.update(study="calibrated-screen-v1", arm=arm, arm_start=common)
        path = run / arm / "step-0400"
        state["checkpoint"] = str(path)
        _checkpoint(path, model_id="fixture/model", revision="rev",
                    metadata={"study": state["study"], "arm": arm, "completed_updates": 400,
                              "update_phase": "boundary", "fingerprint": fp}, state=state)
        payload = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
        payload["cuda_rng"] = [torch.arange(16, dtype=torch.uint8)]
        torch.save(payload, path / "training.pt")
        _write_json(run / arm / "progress.json", {key: value for key, value in state.items() if key != "history"})
        (run / arm / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in state["history"]))
        _write_json(run / arm / "report.json", {"study": state["study"], "arm": arm,
                    "mode": "train", "status": "complete", "passed": True,
                    "completed_requested_pass": True, "fingerprint": fp, "training": state})
    return {**tree, "prior_endpoint_lock": old_lock, "protocol_path": protocol,
            "preserved_path": preserved, "common_start_path": common_path, "training_root": run,
            "output_root": protocol.parent / "evaluation", "arms": list(arms)}


def lock(tree, **kwargs):
    return module().create_endpoint_lock(repo_root=tree["repo"], manifest_path=tree["manifest"],
        training_root=tree["training_root"], output_root=tree["output_root"],
        protocol_path=tree["protocol_path"], preserved_path=tree["preserved_path"],
        prior_endpoint_lock=tree["prior_endpoint_lock"], common_start_path=tree["common_start_path"],
        arms=tree["arms"], **kwargs)


def rewrite_state(tree, arm, mutate):
    path = tree["training_root"] / arm / "step-0400"
    state = json.loads((path / "state.json").read_text())
    mutate(state)
    _write_json(path / "state.json", state)
    payload = torch.load(path / "training.pt", weights_only=False)
    payload["state"] = state
    torch.save(payload, path / "training.pt")
    from openjev.judgment_model import checkpoint_identity
    info = json.loads((path / "checkpoint.json").read_text())
    info["metadata"].update(completed_updates=state["completed_updates"], fingerprint=state["fingerprint"])
    info["checkpoint_id"] = checkpoint_identity(path, info)
    _write_json(path / "checkpoint.json", info)


def test_lock_selects_only_complete_arms_and_copies_exact_prior_partition_metadata(tmp_path):
    tree = fixture(tmp_path)
    value = lock(tree)
    assert list(value["endpoints"]) == ["H0", "H1"]
    assert value["excluded_arms"] == ["Q0", "Q1", "J0", "J1"]
    assert value["evaluation_scope"].startswith("EXPLORATORY")
    assert value["partitions"] == json.loads(tree["prior_endpoint_lock"].read_text())["partitions"]
    assert value["common_start"]["optimizer_state_entries"] == 0
    assert lock(tree) == value
    from experiments import contrast_factorial_pipeline as old
    assert old.STUDY == "contrast-factorial-v1" and old.MODELS == ("BASE", "A", "B", "C", "D")


@pytest.mark.parametrize("mutation", ["incomplete", "frozen", "loss", "source", "adam", "history", "start", "missing_source"])
def test_lock_refuses_incomplete_or_mutated_endpoint_proof(tmp_path, mutation):
    tree = fixture(tmp_path)
    path = tree["training_root"] / "H1/step-0400"
    if mutation == "incomplete":
        rewrite_state(tree, "H1", lambda s: s.update(completed_updates=399))
    elif mutation == "frozen":
        rewrite_state(tree, "H1", lambda s: s.update(frozen_after="changed"))
    elif mutation == "loss":
        rewrite_state(tree, "H1", lambda s: s["fingerprint"]["loss"].update(brier_lambda=0.0))
    elif mutation == "source":
        (tree["repo"] / "experiments/calibrated_training.py").write_text("mutated")
    elif mutation == "missing_source":
        rewrite_state(tree, "H1", lambda s: s["fingerprint"].update(calibrated_source_sha256={}))
    elif mutation == "adam":
        payload = torch.load(path / "training.pt", weights_only=False)
        payload["optimizer"] = {}
        torch.save(payload, path / "training.pt")
    elif mutation == "history":
        (tree["training_root"] / "H1/history.jsonl").write_text("")
    else:
        rewrite_state(tree, "H1", lambda s: s["arm_start"].update(cpu_rng_sha256="8" * 64))
    with pytest.raises(ValueError):
        lock(tree)
    assert not (tree["output_root"] / "endpoint-lock.json").exists()


def test_teacher_targets_cannot_be_exchanged_between_sources(tmp_path):
    tree = fixture(tmp_path, arms=("Q0",))
    rewrite_state(tree, "Q0", lambda s: s["fingerprint"].update(
        loss={"alpha": 0.5, "brier_lambda": 0.0}, targets_sha256="a" * 64))
    target = tree["repo"] / "targets.json"
    _write_json(target, {"teacher": {"model": "jev-1.13.0"}, "rows": []})
    with pytest.raises(ValueError, match="target"):
        lock(tree, targets={"Q": target})


def test_all_fits_are_locked_before_any_development_read_and_report_excludes_missing_arms(tmp_path):
    tree = fixture(tmp_path)
    endpoint = lock(tree)
    events = []
    def loader(path):
        if path.name in {"suite.json", "retention.json"} and ("assessment" in path.parts or path.name == "retention.json"):
            calibration = json.loads((tree["output_root"] / "calibration-lock.json").read_text())
            assert set(calibration["models"]) == {"H0", "H1"}
            events.append("development-read")
        return json.loads(path.read_text())
    module().run_evaluation(tree["output_root"] / "endpoint-lock.json", tree["output_root"],
                           engine_loader=lambda spec: FakeEngine(spec["checkpoint_id"], events), json_loader=loader)
    assert events.index("development-read") > events.index("engine-close")
    report = module().build_final_report(tree["output_root"])
    assert report["evaluation_scope"].startswith("EXPLORATORY")
    assert report["selected_arms"] == ["H0", "H1"]
    assert report["excluded_arms"] == ["Q0", "Q1", "J0", "J1"]
    assert "factorial_effects" not in report
    for variant in ("raw", "global", "per_primitive"):
        effects = report["pairwise_effects"][variant]
        assert set(effects["effects"]) == {"H1-H0"}
        assert effects["effects"]["H1-H0"]["estimate"] == 0
        summary = report["models"]["H0"]["variants"][variant]["assessment"]
        assert summary["primary"]["category_macro"] is not None
        assert {"unknown", "confidence", "nll", "brier"} <= set(summary["overall"])
        assert "score_mean_absolute_error" in summary["by_primitive"]["score"]
    assert endpoint["lock_sha256"] == report["endpoint_lock_sha256"]


def test_evaluation_refuses_shared_training_lock_and_changed_sources(tmp_path):
    from experiments.calibrated_training import study_lock
    tree = fixture(tmp_path)
    lock(tree)
    def forbidden(spec):
        pytest.fail("model loaded before integrity/lock validation")
    with study_lock(tree["repo"] / "reports/calibrated-screen-v1/STUDY.lock"):
        with pytest.raises(RuntimeError, match="lock"):
            module().run_evaluation(tree["output_root"] / "endpoint-lock.json", tree["output_root"], engine_loader=forbidden)
    (tree["repo"] / "experiments/calibrated_training.py").write_text("mutated")
    with pytest.raises(ValueError):
        module().run_evaluation(tree["output_root"] / "endpoint-lock.json", tree["output_root"], engine_loader=forbidden)


def test_paired_family_bootstrap_uses_category_macro_and_shared_draws():
    # Each family differs by 0.25 despite very unequal category sizes; pairing
    # therefore produces a degenerate interval, unlike independent resampling.
    categories = {"a": "small", "b": "large", "c": "large", "d": "large"}
    values = {"H0": {"a": 0.0, "b": 0.5, "c": 0.0, "d": 0.25},
              "H1": {"a": 0.25, "b": 0.75, "c": 0.25, "d": 0.5}}
    result = module().estimate_pairwise_effects(values, categories)
    assert result["means"] == {"H0": 0.125, "H1": 0.375}
    assert result["effects"]["H1-H0"]["estimate"] == 0.25
    assert result["effects"]["H1-H0"]["ci95"] == pytest.approx([0.25, 0.25], abs=1e-15)
    assert result["bootstrap_samples"] == 2000 and result["seed"] == 42
    with pytest.raises(ValueError):
        module().estimate_pairwise_effects({**values, "J0": {"a": 0.0}}, categories)


def test_lock_accepts_post_save_report_metadata_without_weakening_saved_state(tmp_path):
    tree = fixture(tmp_path)
    rewrite_state(tree, "H0", lambda state: state.pop("checkpoint"))
    report_path = tree["training_root"] / "H0/report.json"
    report = json.loads(report_path.read_text())
    report["training"]["session_time_seconds"] = 8.5
    _write_json(report_path, report)
    assert lock(tree)["selected_arms"] == ["H0", "H1"]


def test_context_restored_after_failed_stage_and_partial_logs_survive_retry(tmp_path):
    from experiments import contrast_factorial_pipeline as old
    tree = fixture(tmp_path)
    lock(tree)
    events = []
    class Interrupt(FakeEngine):
        def evaluate(self, request, **kwargs):
            if "model-call" in self.events:
                raise RuntimeError("fixture interruption")
            return super().evaluate(request, **kwargs)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        module().run_evaluation(tree["output_root"] / "endpoint-lock.json", tree["output_root"],
                               engine_loader=lambda spec: Interrupt(spec["checkpoint_id"], events))
    assert old.STUDY == "contrast-factorial-v1"
    stage = tree["output_root"] / "predictions/H0/calibration-new"
    original = (stage / "responses.jsonl").read_bytes()
    assert original and not (tree["output_root"] / "calibration-lock.json").exists()
    module().run_evaluation(tree["output_root"] / "endpoint-lock.json", tree["output_root"],
                           engine_loader=lambda spec: FakeEngine(spec["checkpoint_id"], []))
    assert (stage / "responses.jsonl").read_bytes() == original
    assert (stage.with_name("calibration-new.attempt-0002") / "stage.json").is_file()
    module().build_final_report(tree["output_root"])
    fit = tree["output_root"] / "predictions/H0/temperature-fit.json"
    fit.write_text("{}")
    with pytest.raises(ValueError, match="fit"):
        module().build_final_report(tree["output_root"])


def test_verified_teacher_capture_is_required_before_lock_and_after_lock(tmp_path, monkeypatch):
    from tests.test_calibrated_jev import run
    capture = tmp_path / "capture"
    capture.mkdir()
    _, _, suite, archive = run(capture, monkeypatch, [{}])
    target = archive / "targets.json"
    tree = fixture(tmp_path / "good", arms=("J0",), teacher_capture=(suite, target))
    lock(tree, targets={"J": target})
    exchanged = fixture(tmp_path / "exchanged", arms=("Q0",), teacher_capture=(suite, target))
    with pytest.raises(ValueError, match="source exchanged"):
        lock(exchanged, targets={"Q": target})
    from pathlib import Path
    observation = Path(json.loads(target.read_text())["rows"][0]["observation_path"])
    body = observation.with_name("response.body")
    body.write_bytes(body.read_bytes() + b"\n")
    def forbidden(spec):
        pytest.fail("model loaded despite corrupted raw teacher response")
    with pytest.raises(ValueError, match="archive"):
        module().run_evaluation(tree["output_root"] / "endpoint-lock.json", tree["output_root"], engine_loader=forbidden)


def test_optional_base_and_all_requested_pairwise_directions(tmp_path):
    tree = fixture(tmp_path)
    tree["arms"] = ["H1", "BASE", "H0"]
    value = lock(tree)
    assert list(value["endpoints"]) == ["BASE", "H0", "H1"]
    assert value["selected_arms"] == ["H0", "H1"]
    values = {arm: {"family": value} for arm, value in
              {"H0": 0.0, "H1": 0.1, "Q0": 0.2, "Q1": 0.3, "J0": 0.4, "J1": 0.5}.items()}
    result = module().estimate_pairwise_effects(values, {"family": "category"})
    expected = {"H1-H0": .1, "Q0-H0": .2, "J0-H0": .4, "Q1-Q0": .1, "J1-J0": .1,
                "Q1-H1": .2, "J1-H1": .4, "J0-Q0": .2, "J1-Q1": .2}
    assert {name: value["estimate"] for name, value in result["effects"].items()} == pytest.approx(expected)


def test_cli_lock_requires_an_explicit_population_and_preserves_existing_lock(tmp_path, monkeypatch):
    tree = fixture(tmp_path)
    api = module()
    monkeypatch.setattr(api, "REPO_ROOT", tree["repo"])
    with pytest.raises(SystemExit) as error:
        api.main(["lock"])
    assert error.value.code == 2
    args = ["lock", "--arms", "H0", "H1", "--training-root", str(tree["training_root"]),
            "--output", str(tree["output_root"]), "--manifest", str(tree["manifest"]),
            "--protocol", str(tree["protocol_path"]), "--prior-endpoint-lock", str(tree["prior_endpoint_lock"]),
            "--preserved-artifacts", str(tree["preserved_path"]), "--common-start", str(tree["common_start_path"])]
    api.main(args)
    path = tree["output_root"] / "endpoint-lock.json"
    original = path.read_bytes()
    api.main(args)
    assert path.read_bytes() == original


@pytest.mark.parametrize("change", ["negative_time", "unknown_extra", "persisted_time"])
def test_report_only_allows_valid_documented_post_save_metadata(tmp_path, change):
    tree = fixture(tmp_path)
    path = tree["training_root"] / "H0/report.json"
    report = json.loads(path.read_text())
    if change == "negative_time":
        report["training"]["session_time_seconds"] = -1
    elif change == "unknown_extra":
        report["training"]["unexpected_override"] = True
    else:
        report["training"]["total_time_seconds"] = 999
    _write_json(path, report)
    with pytest.raises(ValueError, match="report"):
        lock(tree)


def test_reuse_rechecks_approved_base_even_when_base_is_not_evaluated(tmp_path):
    tree = fixture(tmp_path)
    lock(tree)
    approval = json.loads(tree["approval"].read_text())
    path = tree["repo"] / approval["base_checkpoint"] / "readouts.safetensors"
    path.write_bytes(b"changed approved starting weights")
    def forbidden(spec):
        pytest.fail("model loaded despite changed approved starting weights")
    with pytest.raises(ValueError, match="base checkpoint"):
        module().run_evaluation(tree["output_root"] / "endpoint-lock.json", tree["output_root"], engine_loader=forbidden)


@pytest.mark.parametrize("mutation", ["cpu_unrestorable", "cpu_contents", "cpu_shape", "cuda_count", "cuda_length", "cuda_shape", "cuda_noncontiguous"])
def test_endpoint_rejects_unrestorable_or_wrong_screen_rng_layout_without_global_rng_mutation(tmp_path, mutation):
    tree = fixture(tmp_path)
    path = tree["training_root"] / "H1/step-0400/training.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if mutation == "cpu_unrestorable":
        payload["torch_rng"] = torch.tensor([1], dtype=torch.uint8)
    elif mutation == "cpu_contents":
        payload["torch_rng"] = torch.zeros_like(payload["torch_rng"])
    elif mutation == "cpu_shape":
        payload["torch_rng"] = payload["torch_rng"].reshape(2, -1)
    elif mutation == "cuda_count":
        payload["cuda_rng"] *= 2
    elif mutation == "cuda_length":
        payload["cuda_rng"] = [torch.zeros(15, dtype=torch.uint8)]
    elif mutation == "cuda_shape":
        payload["cuda_rng"] = [torch.zeros((2, 8), dtype=torch.uint8)]
    else:
        payload["cuda_rng"] = [torch.zeros(32, dtype=torch.uint8)[::2]]
    torch.save(payload, path)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="RNG"):
        lock(tree)
    assert torch.equal(torch.get_rng_state(), before)
    assert not (tree["output_root"] / "endpoint-lock.json").exists()


def test_rng_validation_accepts_coherent_cpu_only_evidence_without_touching_global_rng():
    payload = {"torch_rng": torch.Generator(device="cpu").manual_seed(123).get_state(), "cuda_rng": []}
    before = torch.get_rng_state().clone()
    module()._validate_rng_payload(payload, {"cuda_rng_sha256": []})
    assert torch.equal(torch.get_rng_state(), before)


def test_valid_screen_lock_does_not_change_global_cpu_rng(tmp_path):
    tree = fixture(tmp_path)
    before = torch.get_rng_state().clone()
    lock(tree)
    assert torch.equal(torch.get_rng_state(), before)
