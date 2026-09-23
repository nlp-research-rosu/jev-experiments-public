"""Durable, fail-closed coordinator for a frozen calibrated-screen-v1 plan.

Children own STUDY.lock. This process owns only COORDINATOR.lock; children
inherit that descriptor so an orphan child also prevents a duplicate coordinator.
Failed/interrupted stages require explicit manual recovery, never automatic resume.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STUDY = "calibrated-screen-v1"
ARMS = ("H0", "H1", "Q0", "Q1", "J0", "J1")
MODULES = {"experiments.calibrated_training", "experiments.calibrated_qwen_teacher",
           "experiments.calibrated_evaluation"}
KINDS = {"targets", "smoke", "train", "endpoint", "calibration", "evaluation", "report"}


class RecoveryRequired(RuntimeError):
    """The frozen run cannot safely continue without explicit recovery."""


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _read(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path(repo, path):
    path = Path(path)
    return (path if path.is_absolute() else repo / path).resolve()


def _pins(repo, mapping, label):
    if not isinstance(mapping, dict):
        raise ValueError(f"invalid {label} pins")
    for path, expected in mapping.items():
        actual = _path(repo, path)
        if not actual.is_file() or _sha(actual) != expected:
            raise ValueError(f"{label} changed or missing: {actual}")


def _required(value, fields):
    for name, expected in fields.items():
        current = value
        for part in name.split("."):
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"artifact missing required field {name}")
            current = current[part]
        if type(current) is not type(expected) or current != expected:
            raise ValueError(f"artifact requirement differs: {name}")


def _validate_plan(plan):
    if plan.get("version") != 1 or plan.get("study") != STUDY:
        raise ValueError("invalid calibrated coordinator plan")
    repo = Path(plan["repo_root"])
    if not repo.is_absolute() or not repo.is_dir():
        raise ValueError("plan requires an absolute existing repo_root")
    _pins(repo, plan["source_pins"], "source")
    _pins(repo, plan["input_pins"], "input")
    selected = plan["selected_arms"]
    if (not isinstance(selected, list) or len(selected) != len(set(selected))
            or not {"H0", "H1"} <= set(selected) or not set(selected) <= set(ARMS)):
        raise ValueError("selected arm population is invalid")
    excluded = plan["excluded_arms"]
    if set(excluded) != set(ARMS) - set(selected):
        raise ValueError("excluded arms must explicitly cover the remaining population")
    if set(plan["teacher_audits"]) != {"Q", "J"}:
        raise ValueError("both teacher audits must be frozen")
    for teacher, spec in plan["teacher_audits"].items():
        _pins(repo, {spec["path"]: spec["sha256"]}, "teacher audit")
        audit = _read(_path(repo, spec["path"]))
        if audit.get("status") != "complete" or type(audit.get("eligible")) is not bool:
            raise ValueError("teacher audit is incomplete")
        for arm in (teacher + "0", teacher + "1"):
            if (arm in selected) != audit["eligible"]:
                raise ValueError("selected arms differ from frozen teacher audit eligibility")
            if arm in excluded:
                evidence = excluded[arm]
                if (not evidence.get("reason") or evidence.get("audit_path") != spec["path"]
                        or evidence.get("audit_sha256") != spec["sha256"]):
                    raise ValueError("excluded arm lacks frozen audit evidence")
    ids = set()
    if not isinstance(plan["stages"], list) or not plan["stages"]:
        raise ValueError("plan must name explicit stages")
    for stage in plan["stages"]:
        identity, argv = stage["id"], stage["argv"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identity) or identity in ids:
            raise ValueError("stage IDs must be unique safe filenames")
        ids.add(identity)
        if (not isinstance(argv, list) or len(argv) < 3 or any(not isinstance(x, str) for x in argv)
                or not Path(argv[0]).is_absolute() or Path(argv[0]).resolve() != Path(sys.executable).resolve()
                or argv[1] != "-m" or argv[2] not in MODULES):
            raise ValueError("stage must use this Python executable and an allowlisted module")
        source = argv[2].replace(".", "/") + ".py"
        if _path(repo, source) not in {_path(repo, p) for p in plan["source_pins"]}:
            raise ValueError("stage module source is not pinned")
        if argv[2] == "experiments.calibrated_training" and any(
                x == "--resume" or x.startswith("--resume=") or x == "--recover-history-tail" for x in argv):
            raise ValueError("student resume requires manual recovery outside the coordinator")
        for spec in [stage["artifact"], *stage.get("preconditions", [])]:
            if spec["kind"] not in KINDS or not Path(spec["path"]).is_absolute():
                raise ValueError("artifact requires a known kind and absolute path")
            if "arm" in spec and spec["arm"] not in selected:
                raise ValueError("stage names an excluded arm")
        artifact = stage["artifact"]
        if argv[2] == "experiments.calibrated_training":
            if artifact["kind"] not in {"train", "smoke"} or "--arm" not in argv:
                raise ValueError("training stage requires its own arm completion artifact")
            if argv[argv.index("--arm") + 1] != artifact["arm"]:
                raise ValueError("training argv arm differs from artifact")
            if ("--smoke" in argv) != (artifact["kind"] == "smoke"):
                raise ValueError("training argv mode differs from artifact")
            if "--output" not in argv or _path(repo, argv[argv.index("--output") + 1]) != Path(artifact["path"]).parent:
                raise ValueError("training output differs from completion artifact")
            if artifact["arm"][0] in {"Q", "J"}:
                if "--targets" not in argv or not any(p["kind"] == "targets" for p in stage.get("preconditions", [])):
                    raise ValueError("teacher students require verified complete target preconditions")
    trained = [s["artifact"].get("arm") for s in plan["stages"] if s["artifact"]["kind"] == "train"]
    if len(trained) != len(selected) or set(trained) != set(selected):
        raise ValueError("selected arms must each have exactly one training stage")
    return repo


def _fingerprint(report, spec, repo):
    from experiments import calibrated_training as training
    fp = report["fingerprint"]
    arm = spec["arm"]
    _required(fp, {"study": STUDY, "arm": arm, "updates": 400, "training_source_arm": "B",
                   "loss": training.loss_config(arm), "adapter_lr": 5e-5, "head_lr": 2.5e-5,
                   "gradient_clip_norm": 1.0})
    _required(fp, spec.get("fingerprint_fields", {}))
    if spec.get("fingerprint_sha256") and _canonical(fp) != spec["fingerprint_sha256"]:
        raise ValueError("artifact fingerprint hash differs")
    for label in ("calibrated_source_sha256", "production_source_sha256", "source_fingerprints", "data_fingerprint"):
        if not fp.get(label):
            raise ValueError(f"fingerprint missing {label}")
        _pins(repo, fp[label], label)
    _pins(repo, {fp["replay_path"]: fp["replay_sha256"]}, "replay")
    if "base_checkpoint" in fp:
        if training.factorial.tree_sha256(_path(repo, fp["base_checkpoint"])) != fp["base_checkpoint_sha256"]:
            raise ValueError("base checkpoint changed")
    targets = spec.get("targets_path")
    if arm[0] in {"Q", "J"}:
        if not targets or fp.get("targets_sha256") != _sha(targets):
            raise ValueError("teacher target fingerprint differs")
        _validate_targets({"path": targets, "suite": spec["suite"], "suite_sha256": spec["suite_sha256"]})
    elif targets or fp.get("targets_sha256") is not None:
        raise ValueError("hard-label arm has unexpected teacher targets")
    return fp


def _validate_targets(spec):
    from experiments.calibrated_integrity import load_verified_targets
    if _sha(spec["suite"]) != spec["suite_sha256"]:
        raise ValueError("target source suite changed")
    return load_verified_targets(spec["path"], spec["suite"])


def _evaluation(spec):
    from experiments import calibrated_evaluation as evaluation
    pipeline = evaluation.pipeline
    root = Path(spec.get("evaluation_root", Path(spec["path"]).parent))
    raw = _read(root / "endpoint-lock.json")
    with evaluation.runner_context(tuple(raw["endpoints"])):
        endpoint = pipeline._load_lock(root / "endpoint-lock.json", kind="endpoint")
        evaluation._verify_screen_files(endpoint)
        if endpoint["selected_arms"] != spec["selected_arms"]:
            raise ValueError("evaluation selected arms differ from plan")
        if spec["kind"] == "endpoint":
            return
        calibration = pipeline._load_lock(root / "calibration-lock.json", kind="calibration")
        pipeline._validate_calibration_lock(endpoint, calibration)
        if spec["kind"] == "calibration":
            return
        status = _read(root / "evaluation-status.json")
        pipeline._validate_evaluation_status(endpoint, calibration, status)
        if spec["kind"] == "report":
            report = _read(spec["path"])
            _required(report, {"study": STUDY, "status": "complete",
                "selected_arms": endpoint["selected_arms"], "excluded_arms": endpoint["excluded_arms"],
                "baseline_included": "BASE" in endpoint["endpoints"],
                "endpoint_lock_sha256": endpoint["lock_sha256"],
                "calibration_lock_sha256": calibration["lock_sha256"]})
            if (set(report["models"]) != set(endpoint["endpoints"])
                    or not report.get("evaluation_scope", "").startswith("EXPLORATORY")
                    or set(report.get("pairwise_effects", {})) != set(evaluation.VARIANTS)):
                raise ValueError("final report population or variants differ")


def validate_artifact(spec, repo):
    """Read-only proof and transitive source checks; return durable file evidence."""
    path = Path(spec["path"])
    if not path.is_file():
        raise ValueError(f"completion artifact missing: {path}")
    digest = _sha(path)
    if spec.get("sha256") and spec["sha256"] != digest:
        raise ValueError(f"artifact hash pin differs: {path}")
    value = _read(path)
    _required(value, spec.get("requires", {}))
    evidence = {"path": str(path), "sha256": digest}
    kind = spec["kind"]
    if kind == "targets":
        _validate_targets(spec)
    elif kind in {"smoke", "train"}:
        from experiments import calibrated_training as training
        fp = _fingerprint(value, spec, repo)
        _required(value, {"study": STUDY, "arm": spec["arm"], "status": "complete", "mode": kind,
                          "passed": True, "completed_requested_pass": True})
        if kind == "smoke":
            training.validate_smoke_gate(value, fp)
        else:
            gate = _read(spec["smoke_gate"])
            training.validate_smoke_gate(gate, fp)
            state = value["training"]
            _required(state, {"study": STUDY, "arm": spec["arm"], "status": "complete",
                              "completed_updates": 400, "next_position": 400})
            checkpoint = Path(spec["checkpoint"])
            if not checkpoint.is_dir() or not any(checkpoint.iterdir()) or Path(state["checkpoint"]) != checkpoint:
                raise ValueError("completed checkpoint missing or differs")
            if state.get("fingerprint") != fp or not state.get("frozen_before") or state.get("frozen_before") != state.get("frozen_after"):
                raise ValueError("training fingerprint or frozen-body proof differs")
            evidence["checkpoint_sha256"] = training.factorial.tree_sha256(checkpoint)
            evidence["smoke_gate_sha256"] = _sha(spec["smoke_gate"])
        evidence["fingerprint_sha256"] = _canonical(fp)
    else:
        _evaluation(spec)
    if _sha(path) != digest:
        raise ValueError("artifact changed during validation")
    return evidence


@contextmanager
def coordinator_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "COORDINATOR.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RecoveryRequired("another coordinator or inherited child owns the coordinator lock") from error
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "study": STUDY, "started_utc": _utc()}))
        stream.flush()
        yield stream


def _progress(output, binding, status, **fields):
    value = {"study": STUDY, "status": status, "coordinator_pid": os.getpid(),
             "updated_utc": _utc(), "plan_sha256": binding["plan_sha256"], **fields}
    _atomic(output / "progress.json", value)
    return value


def _check_binding(plan_path, binding, plan, repo):
    if _sha(plan_path) != binding["plan_sha256"] or _sha(__file__) != binding["coordinator_sha256"]:
        raise ValueError("immutable plan or coordinator source changed")
    _pins(repo, plan["source_pins"], "source")
    _pins(repo, plan["input_pins"], "input")


def _preconditions(stage, plan, repo, output):
    evidence = []
    for spec in stage.get("preconditions", []):
        proof = validate_artifact(spec, repo)
        if not spec.get("sha256"):
            candidates = [other for other in plan["stages"] if other["artifact"]["path"] == spec["path"]]
            receipts = [_read(output / "receipts" / f"{other['id']}.json") for other in candidates]
            receipts += [_read(path) for path in (output / "inputs").glob("*.json")]
            if not any(r.get("status") == "complete" and r.get("artifact", {}).get("sha256") == proof["sha256"]
                       and r["artifact"].get("path") == proof["path"] for r in receipts):
                raise ValueError("precondition requires a frozen hash or prior completion receipt")
        evidence.append(proof)
    return evidence


def _partial(stage):
    spec = stage["artifact"]
    paths = [Path(path) for path in stage.get("partial_paths", [])]
    if spec["kind"] in {"smoke", "train"}:
        paths.append(Path(spec["path"]).parent)
    if spec["kind"] == "targets":
        archive = Path(spec["path"]).with_suffix(".observations")
        if archive.exists() and "--resume" not in stage["argv"]:
            paths.append(archive)
        elif archive.exists():
            pin = stage.get("resume_manifest")
            if not pin or _sha(archive / "manifest.json") != pin["sha256"] or Path(pin["path"]) != archive / "manifest.json":
                raise ValueError("Q collection resume requires matching pinned archive manifest")
    if any(path.exists() and (not path.is_dir() or any(path.iterdir())) for path in paths):
        raise ValueError("partial stage output exists; explicit recovery required")


def _producer_identity(launch, repo):
    """Inspect a pinned existing process; never signal it or create a replacement."""
    pid = launch["pid"]
    if type(pid) is not int or pid < 1:
        raise ValueError("producer PID must be a positive integer")
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        command = (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        cwd = (proc / "cwd").resolve(strict=True)
    except FileNotFoundError:
        return None
    if command != launch["command"] or cwd != repo.resolve():
        raise ValueError("producer PID command or working-directory identity differs")
    return {"pid": pid, "start_ticks": stat[19], "command": command, "cwd": str(cwd)}


def _wait_for_inputs(plan, plan_path, repo, output, binding):
    for index, row in enumerate(plan.get("wait_for_inputs", []), 1):
        spec = row["artifact"]
        if spec["kind"] != "targets":
            raise ValueError("external producer prerequisite must be complete teacher targets")
        duration, poll = row["max_wait_seconds"], row.get("poll_seconds", 5)
        if (type(duration) not in (int, float) or not 0 < duration <= 10800
                or type(poll) not in (int, float) or not 0 < poll <= 60):
            raise ValueError("producer wait must be positive and bounded by three hours")
        _pins(repo, {row["producer_launch"]: row["producer_launch_sha256"]}, "producer launch")
        launch = _read(row["producer_launch"])
        command = launch["command"]
        prefix = [command[0], *([] if command[1] != "-u" else ["-u"]), "-m", "experiments.calibrated_qwen_teacher"]
        if (Path(command[0]).resolve() != Path(sys.executable).resolve() or command[:len(prefix)] != prefix
                or "--suite" not in command or "--output" not in command
                or _path(repo, command[command.index("--suite") + 1]) != Path(spec["suite"])
                or _path(repo, command[command.index("--output") + 1]) != Path(spec["path"])):
            raise ValueError("producer launch is not the exact approved Qwen target collection")
        receipt_path = output / "inputs" / f"input-{index:04d}.json"
        row_hash = _canonical(row)
        if receipt_path.exists():
            previous = _read(receipt_path)
            if (previous.get("status") != "complete" or previous.get("input_sha256") != row_hash
                    or previous.get("plan_sha256") != binding["plan_sha256"]):
                raise ValueError("external producer receipt requires explicit recovery")
            if previous["artifact"] != validate_artifact(spec, repo):
                raise ValueError("external producer completed target changed")
            continue
        receipt = {"study": STUDY, "status": "waiting", "created_utc": _utc(),
                   "input_sha256": row_hash, "plan_sha256": binding["plan_sha256"],
                   "producer_launch_sha256": row["producer_launch_sha256"], "producer_pid": launch["pid"],
                   "max_wait_seconds": duration}
        _atomic(receipt_path, receipt)
        start = time.monotonic()
        identity = None
        published = None
        try:
            while True:
                _check_binding(plan_path, binding, plan, repo)
                _pins(repo, {row["producer_launch"]: row["producer_launch_sha256"]}, "producer launch")
                current = _producer_identity(launch, repo)
                if current is not None:
                    if identity is not None and current != identity:
                        raise ValueError("producer PID was reused during bounded wait")
                    identity = current
                    receipt["producer_identity"] = identity
                if Path(spec["path"]).exists():
                    if published is None:
                        published = validate_artifact(spec, repo)
                        receipt["published_artifact"] = published
                        _atomic(receipt_path, receipt)
                    elif _sha(spec["path"]) != published["sha256"]:
                        raise ValueError("producer changed its published target")
                if current is None:
                    if published is None:
                        raise ValueError("producer exited without complete validated targets")
                    receipt.update(status="complete", finished_utc=_utc(), artifact=validate_artifact(spec, repo),
                                   producer_exited=True, waited_seconds=time.monotonic() - start)
                    _atomic(receipt_path, receipt)
                    break
                elapsed = time.monotonic() - start
                if elapsed >= duration:
                    raise ValueError("bounded producer wait expired; explicit recovery required")
                _progress(output, binding, "waiting_for_input", producer_pid=launch["pid"],
                          target_path=spec["path"], waited_seconds=elapsed, max_wait_seconds=duration)
                time.sleep(min(poll, duration - elapsed))
        except BaseException as error:
            receipt.update(status="failed", error=str(error), updated_utc=_utc())
            _atomic(receipt_path, receipt)
            raise


def _run_stage(stage, plan, plan_path, repo, output, binding, lock):
    receipt_path = output / "receipts" / f"{stage['id']}.json"
    _check_binding(plan_path, binding, plan, repo)
    previous = _read(receipt_path) if receipt_path.exists() else None
    if previous and previous.get("status") != "complete":
        raise ValueError(f"stage {stage['id']} is {previous.get('status')}; explicit recovery required")
    prerequisites = _preconditions(stage, plan, repo, output)
    if previous:
        if (previous.get("plan_sha256") != binding["plan_sha256"]
                or previous.get("stage_sha256") != _canonical(stage)
                or previous.get("artifact") != validate_artifact(stage["artifact"], repo)
                or previous.get("preconditions") != prerequisites):
            raise ValueError("completed receipt, artifact, or prerequisite changed")
        return
    receipt = {"study": STUDY, "stage": stage["id"], "plan_sha256": binding["plan_sha256"],
               "stage_sha256": _canonical(stage), "argv": stage["argv"], "created_utc": _utc(),
               "coordinator_pid": os.getpid(), "preconditions": prerequisites}
    if Path(stage["artifact"]["path"]).exists():
        if not stage["artifact"].get("sha256"):
            raise ValueError("pre-existing completion artifact requires an explicit hash pin")
        receipt.update(status="complete", origin="verified-existing", artifact=validate_artifact(stage["artifact"], repo))
        _atomic(receipt_path, receipt)
        return
    _partial(stage)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    log = logs / f"{stage['id']}.attempt-0001.log"
    receipt.update(status="running", origin="launched", log=str(log), child_pid=None)
    _atomic(receipt_path, receipt)
    child = None
    try:
        with log.open("xb") as stream:
            child = subprocess.Popen(stage["argv"], cwd=repo, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, shell=False, start_new_session=True,
                pass_fds=(lock.fileno(),))
            receipt["child_pid"] = child.pid
            _atomic(receipt_path, receipt)
            while True:
                _progress(output, binding, "running", stage=stage["id"], child_pid=child.pid, log=str(log))
                try:
                    code = child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    _check_binding(plan_path, binding, plan, repo)
            stream.flush()
            os.fsync(stream.fileno())
        receipt["returncode"] = code
        if code != 0:
            raise ValueError(f"child exited {code}")
        _check_binding(plan_path, binding, plan, repo)
        if _preconditions(stage, plan, repo, output) != prerequisites:
            raise ValueError("stage prerequisites changed during child execution")
        receipt.update(status="complete", finished_utc=_utc(), artifact=validate_artifact(stage["artifact"], repo))
        _atomic(receipt_path, receipt)
    except BaseException as error:
        alive = child is not None and child.poll() is None
        receipt.update(status="interrupted" if alive else "failed", error=str(error), updated_utc=_utc())
        _atomic(receipt_path, receipt)
        raise


def run_plan(plan_path, output):
    """Execute exactly the frozen stage list once, or verify existing completions."""
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    with coordinator_lock(output) as lock:
        binding = None
        try:
            plan = _read(plan_path)
            repo = _validate_plan(plan)
            binding = {"version": 1, "study": STUDY, "plan_path": str(plan_path),
                       "plan_sha256": _sha(plan_path), "coordinator_sha256": _sha(__file__),
                       "source_pins_sha256": _canonical(plan["source_pins"]), "output": str(output),
                       "python": str(Path(sys.executable).resolve())}
            saved = output / "run-binding.json"
            if saved.exists() and _read(saved) != binding:
                raise ValueError("immutable plan or source binding differs from this coordinator output")
            if not saved.exists():
                _atomic(saved, binding)
                _atomic(output / "plan.snapshot.json", plan)
            _progress(output, binding, "running", selected_arms=plan["selected_arms"], excluded_arms=plan["excluded_arms"])
            _wait_for_inputs(plan, plan_path, repo, output, binding)
            for stage in plan["stages"]:
                _run_stage(stage, plan, plan_path, repo, output, binding, lock)
            _check_binding(plan_path, binding, plan, repo)
            return _progress(output, binding, "complete", selected_arms=plan["selected_arms"],
                             excluded_arms=plan["excluded_arms"], completed_stages=[s["id"] for s in plan["stages"]])
        except BaseException as error:
            if binding:
                _progress(output, binding, "recovery_required", error=str(error))
            raise RecoveryRequired(f"explicit recovery required: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise InterruptedError(f"coordinator received signal {signum}; child is preserved")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        result = run_plan(args.plan, args.output)
    except RecoveryRequired as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
