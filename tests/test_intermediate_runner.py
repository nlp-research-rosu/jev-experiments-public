import fcntl
import hashlib
import importlib
import json
from pathlib import Path

import pytest


def module():
    return importlib.import_module("experiments.intermediate_runner")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
    else:
        path.write_text(value)
    return path


def fixture(tmp_path):
    repo = tmp_path / "repo"
    protocol = write(repo / "reports/intermediate-supervision-v1/PROTOCOL.md", "frozen protocol\n")
    source_files = {}
    for relative in (
        "experiments/intermediate_runner.py",
        "experiments/intermediate_training.py",
        "experiments/intermediate_study.py",
    ):
        source_files[relative] = sha(write(repo / relative, relative + "\n"))
    data = repo / "data/intermediate-supervision-v1/prepared-v2"
    data_files = {}
    for relative in ("train.jsonl", "feature-schema.json", "validation-suite.json", "confirmation-suite.json"):
        data_files[relative] = sha(write(data / relative, relative + "\n"))
    checkpoint = repo / "checkpoints/calibrated-screen-v1/run-v1/H0/step-0400"
    checkpoint_files = {}
    for relative in ("weights.bin", "config.json"):
        checkpoint_files[relative] = sha(write(checkpoint / relative, relative + "\n"))
    external = {"data/processed-v0.2/train.jsonl": sha(write(repo / "data/processed-v0.2/train.jsonl", "replay\n"))}
    approval = {
        "version": 1,
        "study": "intermediate-supervision-v1",
        "data": {"path": str(data.resolve()), "files": data_files},
        "source_files": source_files,
        "original_checkpoint": {"path": str(checkpoint.resolve()), "files": checkpoint_files},
        "external_files": external,
        "protocol_sha256": sha(protocol),
    }
    approval_path = write(repo / "reports/intermediate-supervision-v1/STUDY_APPROVAL.json", approval)
    return {
        "repo": repo,
        "data": data,
        "output": repo / "reports/intermediate-supervision-v1/pipeline-v1",
        "training": repo / "checkpoints/intermediate-supervision-v1/run-v1",
        "approval": approval_path,
    }


def _fingerprint(tree, arm):
    approval = json.loads(tree["approval"].read_text())
    return {
        "study": "intermediate-supervision-v1",
        "arm": arm,
        "settings": module().TRAINING_SETTINGS,
        "starting_checkpoint": approval["original_checkpoint"]["path"],
        "checkpoint_files": approval["original_checkpoint"]["files"],
        "data_files": approval["data"]["files"],
        "source_files": approval["source_files"],
        "prepared_sha256": "1" * 64,
        "schedule_sha256": "2" * 64,
        "schema_sha256": "3" * 64,
        "replay_sha256": next(iter(approval["external_files"].values())),
    }


class FakePopen:
    calls = []
    failure_stage = None
    invalid_smoke = None
    mutate_after = None

    def __init__(self, argv, **kwargs):
        self.argv = list(argv)
        self.kwargs = kwargs
        self.pid = 9000 + len(self.calls)
        self.returncode = None
        self.calls.append(self)
        stage = self._stage()
        if self.mutate_after == stage:
            tree = FakePopen.tree
            (tree["data"] / "train.jsonl").write_text("mutated\n")
        if argv[2] == "experiments.intermediate_training":
            self._training(stage)
        else:
            self._study(stage)

    def _stage(self):
        if self.argv[2] == "experiments.intermediate_training":
            mode = self.argv[self.argv.index("--mode") + 1]
            arm = self.argv[self.argv.index("--arm") + 1]
            return f"{mode}-{arm}"
        command = self.argv[3]
        if "--model" in self.argv:
            command += "-" + self.argv[self.argv.index("--model") + 1]
        return command

    def _training(self, stage):
        tree = FakePopen.tree
        output = Path(self.argv[self.argv.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=False)
        arm = self.argv[self.argv.index("--arm") + 1]
        fingerprint = _fingerprint(tree, arm)
        write(output / "fingerprint.json", fingerprint)
        if stage.startswith("smoke"):
            result = {
                "status": "PASSED", "arm": arm, "completed_updates": 2,
                "same_layout": {"passed": True, "max_logit_difference": 0, "tolerance": 1e-7},
                "gradient_separation": {"passed": True, "A_core_gradient_max_difference": 0,
                    "A_clipped_core_max_difference": 0, "B_adapter_aux_gradient_max": 0.01,
                    "gradient_norms": {name: {"core_gradient_norm": 0.1, "aux_gradient_norm": 0.1}
                                       for name in ("final-only", "A", "B")}},
                "resume": {"actual_disk_reload": True, "rng_equal": True, "tolerance": 1e-7,
                           "differences": {"core_parameters": 0, "aux_parameters": 0,
                                           "probabilities": 0, "reload_probabilities": 0}},
                "frozen_body_unchanged": True, "frozen_body_before": "body",
                "frozen_body_after": "body",
                "original_history": [{"update": 1, "finite_gradients": True, "loss": 1.0},
                                     {"update": 2, "finite_gradients": True, "loss": 0.9}],
                "restored_history": [{"update": 1, "finite_gradients": True, "loss": 1.0},
                                     {"update": 2, "finite_gradients": True, "loss": 0.9}],
            }
            if self.invalid_smoke == arm:
                result["completed_updates"] = 1
        else:
            exposure = [{"family_id": f"family-{index:03d}", "replay_ids": ["n", "c", "s"]}
                        for index in range(200)]
            result = {"status": "COMPLETE", "arm": arm, "completed_updates": 200,
                      "endpoint": "step-0200", "frozen_body_unchanged": True,
                      "frozen_body_before": "body", "frozen_body_after": "body",
                      "history": [{"update": index + 1, "finite_gradients": True, "loss": 1.0}
                                  for index in range(200)],
                      "exposure": exposure, "exposure_sha256": module()._canonical_hash(exposure)}
        write(output / "result.json", result)
        (output / "COMPLETE").write_text(sha(output / "result.json") + "\n")
        approval = json.loads(tree["approval"].read_text())
        write(output / "original-checkpoint-integrity.json", {
            "unchanged": True, "before": approval["original_checkpoint"]["files"],
        })

    def _study(self, stage):
        tree = FakePopen.tree
        output = tree["output"]
        approval = {"path": str(tree["approval"].resolve()), "sha256": sha(tree["approval"])}
        endpoint = output / "endpoint-lock.json"
        calibration_lock = output / "calibration-lock.json"
        if stage == "preflight":
            categories = (
                "claim_vs_completion", "permission_vs_execution", "unknown_vs_failure",
                "attribution_and_endorsement", "entity_binding", "action_binding", "temporal_scope",
                "reversal_and_current_state", "negation_and_quantifiers", "ordered_rubrics",
            )
            write(output / "preflight/report.json", {"status": "PASSED", "runtime": {
                "status": "PASSED", "frozen_body_sha256": "9" * 64,
                "rows": [{"category": category, "hook_logit_max_difference": 0,
                          "full_cached_max_probability_difference": 0.01,
                          "full_cached_argmax_changes": 0} for category in categories],
            }})
        elif stage == "lock":
            write(endpoint, {"models": ["H0", "A", "B"], "selected_updates": 200,
                             "approval": approval})
        elif stage.startswith("calibrate-"):
            model = stage.split("-", 1)[1]
            write(output / f"calibration/{model}/fit.json", {
                "model": model, "endpoint_sha256": sha(endpoint),
                "sources": {"calibration": "5" * 64}, "fits": {"global": {"temperature": 1.0}},
            })
        elif stage == "calibration-lock":
            fits = {model: {"path": str((output / f"calibration/{model}/fit.json").resolve()),
                            "sha256": sha(output / f"calibration/{model}/fit.json")}
                    for model in ("H0", "A", "B")}
            write(calibration_lock, {"endpoint_sha256": sha(endpoint), "fits": fits})
        elif stage.startswith("assess-"):
            model = stage.split("-", 1)[1]
            write(output / f"assessment/{model}/report.json", {
                "model": model, "endpoint_sha256": sha(endpoint),
                "calibration_sha256": sha(calibration_lock), "frozen_body_unchanged": True,
            })
        elif stage == "report":
            endpoint_value = json.loads(endpoint.read_text())
            endpoint_sha, calibration_sha = sha(endpoint), sha(calibration_lock)
            write(output / "final-report/report.json", {
                "status": "COMPLETE", "endpoints": endpoint_value,
                "models": {model: {"model": model, "endpoint_sha256": endpoint_sha,
                                    "calibration_sha256": calibration_sha,
                                    "frozen_body_unchanged": True} for model in ("H0", "A", "B")},
            })

    def wait(self, timeout=None):
        self.returncode = 7 if self._stage() == self.failure_stage else 0
        return self.returncode

    def poll(self):
        return self.returncode


@pytest.fixture(autouse=True)
def reset_fake():
    FakePopen.calls = []
    FakePopen.failure_stage = None
    FakePopen.invalid_smoke = None
    FakePopen.mutate_after = None


def test_builds_exact_fixed_stage_order_and_cli(tmp_path):
    tree = fixture(tmp_path)
    stages = module().build_stages(**{key: tree[key] for key in ("data", "output", "training", "approval")})
    assert [stage["id"] for stage in stages] == [
        "preflight", "smoke-A", "smoke-B", "train-A", "train-B", "lock",
        "calibrate-H0", "calibrate-A", "calibrate-B", "calibration-lock",
        "assess-H0", "assess-A", "assess-B", "report",
    ]
    commands = [part for stage in stages for part in stage["argv"]]
    assert "--resume" not in commands
    training = [stage for stage in stages if stage["module"] == "experiments.intermediate_training"]
    assert [(stage["mode"], stage["arm"]) for stage in training] == [
        ("smoke", "A"), ("smoke", "B"), ("train", "A"), ("train", "B")
    ]
    assert all("200" not in stage["argv"] for stage in stages)


def test_stage_launch_preserves_virtualenv_python_symlink(tmp_path):
    tree = fixture(tmp_path)
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path("/usr/bin/python3"))
    stages = module().build_stages(
        **{key: tree[key] for key in ("data", "output", "training", "approval")}, python=venv_python
    )
    assert all(stage["argv"][0] == str(venv_python.absolute()) for stage in stages)


def test_success_runs_once_with_inherited_lock_and_durable_receipts(tmp_path, monkeypatch):
    tree = fixture(tmp_path)
    FakePopen.tree = tree
    monkeypatch.setattr(module().subprocess, "Popen", FakePopen)
    result = module().run_pipeline(repo=tree["repo"], **{key: tree[key] for key in ("data", "output", "training", "approval")})
    assert result["status"] == "complete"
    assert len(FakePopen.calls) == 14
    assert all(call.kwargs["pass_fds"] and len(call.kwargs["pass_fds"]) == 1 for call in FakePopen.calls)
    assert all("--resume" not in call.argv for call in FakePopen.calls)
    receipts = sorted((tree["output"] / "receipts").glob("*.json"))
    assert len(receipts) == 14
    assert all(json.loads(path.read_text())["status"] == "complete" for path in receipts)
    assert len(list((tree["output"] / "logs").glob("*.log"))) == 14
    with pytest.raises(module().RecoveryRequired, match="existing pipeline output"):
        module().run_pipeline(repo=tree["repo"], **{key: tree[key] for key in ("data", "output", "training", "approval")})
    assert len(FakePopen.calls) == 14


def test_smoke_rejection_stops_before_any_full_training(tmp_path, monkeypatch):
    tree = fixture(tmp_path)
    FakePopen.tree = tree
    FakePopen.invalid_smoke = "B"
    monkeypatch.setattr(module().subprocess, "Popen", FakePopen)
    with pytest.raises(module().RecoveryRequired, match="smoke"):
        module().run_pipeline(repo=tree["repo"], **{key: tree[key] for key in ("data", "output", "training", "approval")})
    assert [call._stage() for call in FakePopen.calls] == ["preflight", "smoke-A", "smoke-B"]
    assert (tree["output"] / "failure.json").is_file()
    assert json.loads((tree["output"] / "receipts/smoke-B.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("damage", [
    "layout", "nan", "difference_keys", "b_gradient", "frozen", "history", "checkpoint_integrity",
])
def test_strict_smoke_validator_rejects_incomplete_numerical_proof(tmp_path, damage):
    tree = fixture(tmp_path)
    FakePopen.tree = tree
    argv = module().build_stages(**{key: tree[key] for key in ("data", "output", "training", "approval")})[1]["argv"]
    FakePopen(argv)
    path = tree["output"] / "smoke-A"
    result = json.loads((path / "result.json").read_text())
    if damage == "layout":
        result["same_layout"]["passed"] = False
    elif damage == "nan":
        result["resume"]["differences"]["probabilities"] = float("nan")
    elif damage == "difference_keys":
        del result["resume"]["differences"]["aux_parameters"]
    elif damage == "b_gradient":
        result["gradient_separation"]["B_adapter_aux_gradient_max"] = 0
    elif damage == "frozen":
        result["frozen_body_after"] = "changed"
    elif damage == "history":
        result["restored_history"][1]["finite_gradients"] = False
    else:
        integrity = json.loads((path / "original-checkpoint-integrity.json").read_text())
        integrity["unchanged"] = False
        write(path / "original-checkpoint-integrity.json", integrity)
    if damage != "checkpoint_integrity":
        write(path / "result.json", result)
        (path / "COMPLETE").write_text(sha(path / "result.json") + "\n")
    approval = json.loads(tree["approval"].read_text())
    with pytest.raises(ValueError, match="smoke|gradient|frozen|history|checkpoint|difference|layout|finite"):
        module()._validate_smoke(path, approval, "A")


def test_failed_child_is_preserved_and_never_relaunched(tmp_path, monkeypatch):
    tree = fixture(tmp_path)
    FakePopen.tree = tree
    FakePopen.failure_stage = "train-A"
    monkeypatch.setattr(module().subprocess, "Popen", FakePopen)
    with pytest.raises(module().RecoveryRequired, match="child exited 7"):
        module().run_pipeline(repo=tree["repo"], **{key: tree[key] for key in ("data", "output", "training", "approval")})
    launched = len(FakePopen.calls)
    with pytest.raises(module().RecoveryRequired, match="existing pipeline output"):
        module().run_pipeline(repo=tree["repo"], **{key: tree[key] for key in ("data", "output", "training", "approval")})
    assert len(FakePopen.calls) == launched
    assert (tree["output"] / "logs/train-A.log").is_file()


def test_every_pin_is_rechecked_before_each_subprocess(tmp_path, monkeypatch):
    tree = fixture(tmp_path)
    FakePopen.tree = tree
    FakePopen.mutate_after = "preflight"
    monkeypatch.setattr(module().subprocess, "Popen", FakePopen)
    with pytest.raises(module().RecoveryRequired, match="data.*changed|changed.*data"):
        module().run_pipeline(repo=tree["repo"], **{key: tree[key] for key in ("data", "output", "training", "approval")})
    assert [call._stage() for call in FakePopen.calls] == ["preflight"]


def test_unapproved_extra_data_file_is_rejected_before_output_or_launch(tmp_path, monkeypatch):
    tree = fixture(tmp_path)
    FakePopen.tree = tree
    write(tree["data"] / "unapproved.json", "extra\n")
    monkeypatch.setattr(module().subprocess, "Popen", FakePopen)
    with pytest.raises(module().RecoveryRequired, match="data tree"):
        module().run_pipeline(repo=tree["repo"], **{key: tree[key] for key in ("data", "output", "training", "approval")})
    assert not tree["output"].exists()
    assert not FakePopen.calls


def test_active_and_stale_coordinator_outputs_are_both_fail_closed(tmp_path, monkeypatch):
    tree = fixture(tmp_path)
    tree["output"].mkdir(parents=True)
    lock_path = tree["output"] / "COORDINATOR.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(module().RecoveryRequired, match="coordinator"):
            module().run_pipeline(repo=tree["repo"], **{key: tree[key] for key in ("data", "output", "training", "approval")})
    with pytest.raises(module().RecoveryRequired, match="existing pipeline output"):
        module().run_pipeline(repo=tree["repo"], **{key: tree[key] for key in ("data", "output", "training", "approval")})
