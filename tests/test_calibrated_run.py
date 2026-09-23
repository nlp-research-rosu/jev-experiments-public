"""CPU subprocess tests for durable launch, proof checking, and fail-closed recovery."""
import fcntl
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


def module():
    return importlib.import_module("experiments.calibrated_run")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fixture(tmp_path, first="complete"):
    repo = tmp_path / "repo"
    source = repo / "experiments/calibrated_training.py"
    source.parent.mkdir(parents=True)
    source.write_text('''import argparse, json, pathlib, sys
p=argparse.ArgumentParser()
p.add_argument('--arm');p.add_argument('--output');p.add_argument('--smoke-gate')
a=p.parse_args(); root=pathlib.Path.cwd()
with (root/'launched.jsonl').open('a') as f:f.write(json.dumps(a.arm)+'\\n')
payload=json.loads((root/'payloads.json').read_text())[a.arm]
if payload=='fail':sys.exit(7)
output=pathlib.Path(a.output);output.mkdir(parents=True,exist_ok=True)
(output/'report.json').write_text(json.dumps(payload))
checkpoint=output/'step-0400';checkpoint.mkdir(exist_ok=True)
(checkpoint/'state.json').write_text(json.dumps(payload['training']))
(checkpoint/'weights.bin').write_bytes(b'fixture weights')
''')
    pins = {"experiments/calibrated_training.py": sha(source)}
    for name in ("experiments/calibrated_targets.py", "experiments/calibrated_integrity.py",
                 "reports/calibrated-screen-v1/PROTOCOL.md"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixed input\n")
        pins[name] = sha(path)
    suite = write(repo / "suite.json", {"cases": []})
    manifest = write(repo / "manifest.json", {})
    replay = write(repo / "replay.json", [])
    from experiments.calibrated_training import loss_config
    stages, payloads = [], {}
    for arm in ("H0", "H1"):
        fp = {"study": "calibrated-screen-v1", "arm": arm, "updates": 400,
              "training_source_arm": "B", "loss": loss_config(arm),
              "adapter_lr": 5e-5, "head_lr": 2.5e-5, "gradient_clip_norm": 1.0,
              "targets_sha256": None, "calibrated_source_sha256": pins,
              "production_source_sha256": {"experiments/calibrated_training.py": sha(source)},
              "source_fingerprints": {"manifest.json": sha(manifest)},
              "data_fingerprint": {"suite.json": sha(suite)},
              "replay_path": "replay.json", "replay_sha256": sha(replay)}
        gate = {"study": "calibrated-screen-v1", "arm": arm, "mode": "smoke",
                "status": "complete", "passed": True, "completed_requested_pass": True,
                "fingerprint": fp,
                "training": {"gradient_check": {k: True for k in ("finite", "adapters", "heads", "frozen")},
                             "frozen_before": "body", "frozen_after": "body"},
                "resume_equivalence": {"passed": True, "tolerance": 1e-7,
                    "max_probability_difference": 0, "max_trainable_difference": 0,
                    "frozen_before": "body", "frozen_after": "body"}}
        gate_path = write(repo / "reports/calibrated-screen-v1/smoke" / arm / "report.json", gate)
        output = repo / "checkpoints/calibrated-screen-v1/run-v1" / arm
        state = {"study": "calibrated-screen-v1", "arm": arm, "status": "complete",
                 "completed_updates": 400, "next_position": 400, "fingerprint": fp,
                 "checkpoint": str(output / "step-0400"), "frozen_before": "body", "frozen_after": "body"}
        report = {"study": "calibrated-screen-v1", "arm": arm, "mode": "train", "status": "complete",
                  "passed": True, "completed_requested_pass": True, "fingerprint": fp, "training": state}
        if arm == "H0" and first == "fail":
            report = "fail"
        elif arm == "H0" and first == "paused":
            report["status"] = report["training"]["status"] = "paused"
            report["passed"] = False
        payloads[arm] = report
        smoke = {"kind": "smoke", "path": str(gate_path), "arm": arm, "sha256": sha(gate_path)}
        stages.append({"id": f"train-{arm}", "argv": [sys.executable, "-m", "experiments.calibrated_training",
                        "--arm", arm, "--output", str(output), "--smoke-gate", str(gate_path)],
                       "preconditions": [smoke], "artifact": {"kind": "train", "path": str(output / "report.json"),
                           "arm": arm, "smoke_gate": str(gate_path), "checkpoint": str(output / "step-0400")}})
    payload = write(repo / "payloads.json", payloads)
    audits = {}
    for teacher in ("Q", "J"):
        path = write(repo / f"{teacher}-audit.json", {"status": "complete", "eligible": False})
        audits[teacher] = {"path": str(path), "sha256": sha(path)}
    plan = {"version": 1, "study": "calibrated-screen-v1", "repo_root": str(repo),
            "source_pins": pins, "input_pins": {str(payload): sha(payload)},
            "selected_arms": ["H0", "H1"], "teacher_audits": audits,
            "excluded_arms": {arm: {"reason": "teacher audit failed", "audit_path": audits[arm[0]]["path"],
                                    "audit_sha256": audits[arm[0]]["sha256"]} for arm in ("Q0", "Q1", "J0", "J1")},
            "stages": stages}
    path = write(repo / "plan.json", plan)
    return {"repo": repo, "plan": plan, "path": path,
            "output": repo / "reports/calibrated-screen-v1/pipeline-v1", "payloads": payloads}


def run(tree):
    return module().run_plan(tree["path"], tree["output"])


def test_success_receipts_skip_complete_without_launching_again(tmp_path):
    tree = fixture(tmp_path)
    result = run(tree)
    assert result["status"] == "complete"
    assert (tree["repo"] / "launched.jsonl").read_text().splitlines() == ['"H0"', '"H1"']
    first_receipt = (tree["output"] / "receipts/train-H0.json").read_bytes()
    assert run(tree)["status"] == "complete"
    assert (tree["repo"] / "launched.jsonl").read_text().splitlines() == ['"H0"', '"H1"']
    assert (tree["output"] / "receipts/train-H0.json").read_bytes() == first_receipt


@pytest.mark.parametrize("behavior", ["fail", "paused"])
def test_failed_or_paused_child_stops_and_never_restarts_automatically(tmp_path, behavior):
    tree = fixture(tmp_path, behavior)
    with pytest.raises(module().RecoveryRequired):
        run(tree)
    with pytest.raises(module().RecoveryRequired):
        run(tree)
    assert (tree["repo"] / "launched.jsonl").read_text().splitlines() == ['"H0"']
    receipt = json.loads((tree["output"] / "receipts/train-H0.json").read_text())
    assert receipt["status"] == "failed"
    assert Path(receipt["log"]).exists()


@pytest.mark.parametrize("mutation", ["plan", "artifact", "source", "checkpoint"])
def test_mutated_completed_run_refuses_any_new_launch(tmp_path, mutation):
    tree = fixture(tmp_path)
    run(tree)
    if mutation == "plan":
        tree["plan"]["stages"].reverse()
        write(tree["path"], tree["plan"])
    elif mutation == "artifact":
        path = Path(tree["plan"]["stages"][0]["artifact"]["path"])
        value = json.loads(path.read_text())
        value["passed"] = False
        write(path, value)
    elif mutation == "checkpoint":
        path = Path(tree["plan"]["stages"][0]["artifact"]["checkpoint"]) / "weights.bin"
        path.write_bytes(b"mutated")
    else:
        (tree["repo"] / "experiments/calibrated_training.py").write_text("changed")
    with pytest.raises(module().RecoveryRequired):
        run(tree)
    assert len((tree["repo"] / "launched.jsonl").read_text().splitlines()) == 2


def test_coordinator_lock_collision_does_not_launch(tmp_path):
    tree = fixture(tmp_path)
    tree["output"].mkdir(parents=True)
    with (tree["output"] / "COORDINATOR.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(module().RecoveryRequired, match="coordinator"):
            run(tree)
    assert not (tree["repo"] / "launched.jsonl").exists()


def test_unknown_module_is_rejected_before_child_launch(tmp_path):
    tree = fixture(tmp_path)
    tree["plan"]["stages"][0]["argv"][2] = "http.server"
    write(tree["path"], tree["plan"])
    with pytest.raises(module().RecoveryRequired, match="module"):
        run(tree)
    assert not (tree["repo"] / "launched.jsonl").exists()


def test_interrupted_receipt_never_restarts_even_if_artifact_complete(tmp_path):
    tree = fixture(tmp_path)
    run(tree)
    path = tree["output"] / "receipts/train-H0.json"
    receipt = json.loads(path.read_text())
    receipt["status"] = "running"
    write(path, receipt)
    with pytest.raises(module().RecoveryRequired, match="recovery"):
        run(tree)
    assert len((tree["repo"] / "launched.jsonl").read_text().splitlines()) == 2


def test_preexisting_complete_artifact_requires_pin_and_then_skips(tmp_path):
    tree = fixture(tmp_path)
    stage = tree["plan"]["stages"][0]
    report = tree["payloads"]["H0"]
    path = write(Path(stage["artifact"]["path"]), report)
    checkpoint = Path(stage["artifact"]["checkpoint"])
    write(checkpoint / "state.json", report["training"])
    (checkpoint / "weights.bin").write_bytes(b"fixture weights")
    with pytest.raises(module().RecoveryRequired, match="pin"):
        run(tree)
    # Manual recovery uses a new immutable plan/output; existing logs remain.
    tree["plan"]["stages"][0]["artifact"]["sha256"] = sha(path)
    tree["path"] = write(tree["repo"] / "reviewed-plan-v2.json", tree["plan"])
    tree["output"] = tree["output"].with_name("pipeline-v2")
    assert run(tree)["status"] == "complete"
    assert (tree["repo"] / "launched.jsonl").read_text().splitlines() == ['"H1"']


def test_partial_output_refuses_launch_and_preserves_contents(tmp_path):
    tree = fixture(tmp_path)
    partial = Path(tree["plan"]["stages"][0]["artifact"]["path"]).parent / "history.jsonl"
    partial.parent.mkdir(parents=True)
    partial.write_text("partial evidence\n")
    with pytest.raises(module().RecoveryRequired, match="partial"):
        run(tree)
    assert partial.read_text() == "partial evidence\n"
    assert not (tree["repo"] / "launched.jsonl").exists()


def test_teacher_eligibility_cannot_be_overridden_by_arm_selection(tmp_path):
    tree = fixture(tmp_path)
    tree["plan"]["selected_arms"] += ["Q0", "Q1"]
    for arm in ("Q0", "Q1"):
        del tree["plan"]["excluded_arms"][arm]
    write(tree["path"], tree["plan"])
    with pytest.raises(module().RecoveryRequired, match="audit"):
        run(tree)
    assert not (tree["repo"] / "launched.jsonl").exists()


def waiting_input(tree, pid):
    suite = tree["repo"] / "suite.json"
    target = tree["repo"] / "external-targets.json"
    launch = write(tree["repo"] / "external-launch.json", {"pid": pid,
        "command": [sys.executable, "-u", "-m", "experiments.calibrated_qwen_teacher",
                    "--suite", str(suite), "--output", str(target), "--revision", "a" * 40]})
    row = {"artifact": {"kind": "targets", "path": str(target), "suite": str(suite),
                         "suite_sha256": sha(suite)},
           "producer_launch": str(launch), "producer_launch_sha256": sha(launch),
           "max_wait_seconds": 3, "poll_seconds": 1}
    tree["plan"]["wait_for_inputs"] = [row]
    write(tree["path"], tree["plan"])
    return row


@pytest.mark.parametrize("pid", [999999999, os.getpid()])
def test_dead_or_wrong_producer_identity_never_launches_students(tmp_path, pid):
    tree = fixture(tmp_path)
    waiting_input(tree, pid)
    with pytest.raises(module().RecoveryRequired, match="producer"):
        run(tree)
    assert not (tree["repo"] / "launched.jsonl").exists()


def test_complete_external_targets_are_verified_and_receipted_without_relaunch(tmp_path, monkeypatch):
    from tests.test_calibrated_jev import run as collect_fixture
    tree = fixture(tmp_path)
    row = waiting_input(tree, 999999999)
    archive = tmp_path / "archive"
    archive.mkdir()
    _, _, suite, output = collect_fixture(archive, monkeypatch, [{}])
    row["artifact"].update(path=str(output / "targets.json"), suite=str(suite), suite_sha256=sha(suite))
    launch = json.loads(Path(row["producer_launch"]).read_text())
    launch["command"][5] = str(suite)
    launch["command"][7] = row["artifact"]["path"]
    write(Path(row["producer_launch"]), launch)
    row["producer_launch_sha256"] = sha(row["producer_launch"])
    write(tree["path"], tree["plan"])
    assert run(tree)["status"] == "complete"
    receipts = list((tree["output"] / "inputs").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["artifact"]["sha256"] == sha(output / "targets.json")
    assert run(tree)["status"] == "complete"
    assert len((tree["repo"] / "launched.jsonl").read_text().splitlines()) == 2


def test_wait_for_external_producer_is_bounded_and_keeps_no_child_alive(tmp_path, monkeypatch):
    tree = fixture(tmp_path)
    waiting_input(tree, 12345)
    monkeypatch.setattr(module(), "_producer_identity", lambda launch, repo: {"pid": 12345, "start_ticks": "77"}, raising=False)
    elapsed = [0]
    monkeypatch.setattr(module().time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(module().time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    with pytest.raises(module().RecoveryRequired, match="wait.*expired"):
        run(tree)
    assert elapsed[0] == 3
    assert not (tree["repo"] / "launched.jsonl").exists()


def test_killed_coordinator_leaves_child_lock_and_no_automatic_restart(tmp_path):
    tree = fixture(tmp_path)
    source = tree["repo"] / "experiments/calibrated_training.py"
    source.write_text('''import pathlib, os, time
root=pathlib.Path.cwd()
(root/'child.pid').write_text(str(os.getpid()))
while not (root/'release-child').exists():time.sleep(.01)
''')
    tree["plan"]["source_pins"]["experiments/calibrated_training.py"] = sha(source)
    # No preconditions: the held inherited descriptor is exercised before the
    # intentionally unfinished child's artifact validation ever runs.
    tree["plan"]["stages"][0]["preconditions"] = []
    write(tree["path"], tree["plan"])
    child_pid = None
    with (tmp_path / "coordinator.log").open("wb") as log:
        coordinator = subprocess.Popen([sys.executable, "-m", "experiments.calibrated_run",
            "--plan", str(tree["path"]), "--output", str(tree["output"])], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 10
            while not (tree["repo"] / "child.pid").exists():
                assert time.monotonic() < deadline, "CPU fixture child did not launch"
                time.sleep(.01)
            child_pid = int((tree["repo"] / "child.pid").read_text())
            coordinator.kill()
            coordinator.wait(timeout=5)
            with pytest.raises(module().RecoveryRequired, match="coordinator lock"):
                run(tree)
            (tree["repo"] / "release-child").touch()
            # Child exit is observed through lock release; even then its running
            # receipt still requires recovery rather than relaunch.
            deadline = time.monotonic() + 5
            while True:
                try:
                    run(tree)
                except module().RecoveryRequired as error:
                    if "coordinator lock" not in str(error):
                        assert "running" in str(error)
                        break
                assert time.monotonic() < deadline
                time.sleep(.01)
        finally:
            if coordinator.poll() is None:
                coordinator.kill()
                coordinator.wait(timeout=5)
            if child_pid:
                try:
                    os.kill(child_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


def test_plan_cannot_claim_selected_arm_without_its_training_stage(tmp_path):
    tree = fixture(tmp_path)
    tree["plan"]["stages"].pop()
    write(tree["path"], tree["plan"])
    with pytest.raises(module().RecoveryRequired, match="selected.*training"):
        run(tree)
    assert not (tree["repo"] / "launched.jsonl").exists()
