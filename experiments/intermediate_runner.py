"""One-shot durable coordinator for the frozen intermediate-supervision-v1 study.

The coordinator never resumes a training process or reuses a pipeline directory.
Every child inherits COORDINATOR.lock, so an orphan child continues to exclude a
duplicate coordinator after parent death.  Any failure requires review and a new
output location; the original logs, receipts, and partial artifacts stay intact.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STUDY = "intermediate-supervision-v1"
MODELS = ("H0", "A", "B")
ARMS = ("A", "B")
CATEGORIES = (
    "claim_vs_completion", "permission_vs_execution", "unknown_vs_failure",
    "attribution_and_endorsement", "entity_binding", "action_binding", "temporal_scope",
    "reversal_and_current_state", "negation_and_quantifiers", "ordered_rubrics",
)
PROTOCOL = Path("reports/intermediate-supervision-v1/PROTOCOL.md")
EXPECTED_CHECKPOINT = Path("checkpoints/calibrated-screen-v1/run-v1/H0/step-0400")
REQUIRED_SOURCES = {
    "experiments/intermediate_runner.py",
    "experiments/intermediate_training.py",
    "experiments/intermediate_study.py",
}
TRAINING_SETTINGS = {
    "seed": 42, "updates": 200, "adapter_lr": 5e-5, "head_lr": 2.5e-5,
    "weight_decay": .01, "core_clip": 1., "aux_lr": 2.5e-4, "aux_weight_decay": 0.,
    "aux_clip": 1., "aux_lambda": .25, "smooth_l1_beta": 1., "max_units": 8,
    "physical_batch_size": 4, "bucket_tokens": 128, "max_input_tokens": 1536,
    "training_policy": "canonical-full-v1-b4-k128", "aux_target_units": "raw/declared-scale-once",
    "final_objective": "six equal new/old primitive means",
    "fact_objective": "eligible feature mean, eligible candidate mean, question mean, three primitive mean",
}


class RecoveryRequired(RuntimeError):
    """A failed, stale, or ambiguous run needs an explicitly reviewed new launch."""


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _read(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value {value}")
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


def _safe_child(base, relative, label):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} requires nonempty relative paths")
    base = Path(base).resolve()
    path = (base / relative).resolve()
    try:
        path.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{label} path escapes its approved root: {relative}") from error
    return path


def _verify_files(base, mapping, label, *, exact=False):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{label} requires a nonempty file map")
    for relative, expected in mapping.items():
        path = _safe_child(base, relative, label)
        if (not isinstance(expected, str) or len(expected) != 64
                or not path.is_file() or _sha(path) != expected):
            raise ValueError(f"{label} file changed or is missing: {path}")
    if exact:
        actual = {path.relative_to(Path(base).resolve()).as_posix(): _sha(path)
                  for path in Path(base).resolve().rglob("*") if path.is_file()}
        if actual != mapping:
            raise ValueError(f"{label} tree contains unapproved, missing, or changed files")


def verify_approval(repo, data, approval_path, *, expected_sha256=None):
    """Re-read and verify every approved byte pin; safe to call before each child."""
    repo, data, approval_path = Path(repo).resolve(), Path(data).resolve(), Path(approval_path).resolve()
    if not repo.is_dir() or not approval_path.is_file():
        raise ValueError("repository or approval file is missing")
    approval_sha = _sha(approval_path)
    if expected_sha256 is not None and approval_sha != expected_sha256:
        raise ValueError("approval changed after coordinator start")
    approval = _read(approval_path)
    required = {"version", "study", "data", "source_files", "original_checkpoint",
                "external_files", "protocol_sha256"}
    if set(approval) != required or approval["version"] != 1 or approval["study"] != STUDY:
        raise ValueError("approval structure or study identity differs")
    if not isinstance(approval["data"], dict) or set(approval["data"]) != {"path", "files"}:
        raise ValueError("approval data specification is malformed")
    approved_data = Path(approval["data"]["path"])
    if not approved_data.is_absolute() or approved_data.resolve() != data:
        raise ValueError("requested data path differs from approval")
    _verify_files(data, approval["data"]["files"], "data", exact=True)
    if not isinstance(approval["original_checkpoint"], dict) or set(approval["original_checkpoint"]) != {"path", "files"}:
        raise ValueError("approval checkpoint specification is malformed")
    checkpoint = Path(approval["original_checkpoint"]["path"])
    expected_checkpoint = (repo / EXPECTED_CHECKPOINT).resolve()
    if not checkpoint.is_absolute() or checkpoint.resolve() != expected_checkpoint:
        raise ValueError("approved original checkpoint is not the fixed H0 step-0400 checkpoint")
    _verify_files(checkpoint, approval["original_checkpoint"]["files"], "original checkpoint", exact=True)
    if not REQUIRED_SOURCES <= set(approval["source_files"]):
        raise ValueError("approval does not pin every coordinator/trainer/study source")
    _verify_files(repo, approval["source_files"], "source")
    _verify_files(repo, approval["external_files"], "external")
    protocol = (repo / PROTOCOL).resolve()
    if not protocol.is_file() or _sha(protocol) != approval["protocol_sha256"]:
        raise ValueError("protocol changed or differs from approval")
    return approval, approval_sha


def _common_study_args(data, output, training, approval):
    return ["--data", str(data), "--output", str(output), "--training-root", str(training),
            "--approval", str(approval)]


def build_stages(*, data, output, training, approval, python=None):
    """Return the only authorized sequence. No caller can add an arm or update count."""
    data, output, training, approval = map(lambda value: Path(value).resolve(), (data, output, training, approval))
    python = os.path.abspath(os.fspath(python or sys.executable))

    def study(command, model=None):
        return [python, "-m", "experiments.intermediate_study", command,
                *_common_study_args(data, output, training, approval),
                *([] if model is None else ["--model", model])]

    def train(mode, arm, destination):
        return [python, "-m", "experiments.intermediate_training", "--mode", mode,
                "--arm", arm, "--data", str(data), "--output", str(destination)]
    stages = [{"id": "preflight", "module": "experiments.intermediate_study", "argv": study("preflight")}]
    stages += [{"id": f"smoke-{arm}", "module": "experiments.intermediate_training", "mode": "smoke",
                "arm": arm, "argv": train("smoke", arm, output / f"smoke-{arm}")} for arm in ARMS]
    stages += [{"id": f"train-{arm}", "module": "experiments.intermediate_training", "mode": "train",
                "arm": arm, "argv": train("train", arm, training / arm)} for arm in ARMS]
    stages.append({"id": "lock", "module": "experiments.intermediate_study", "argv": study("lock")})
    stages += [{"id": f"calibrate-{model}", "module": "experiments.intermediate_study",
                "model": model, "argv": study("calibrate", model)} for model in MODELS]
    stages.append({"id": "calibration-lock", "module": "experiments.intermediate_study",
                   "argv": study("calibration-lock")})
    stages += [{"id": f"assess-{model}", "module": "experiments.intermediate_study",
                "model": model, "argv": study("assess", model)} for model in MODELS]
    stages.append({"id": "report", "module": "experiments.intermediate_study", "argv": study("report")})
    _validate_stages(stages, data, output, training, approval, python)
    return stages


def _validate_stages(stages, data, output, training, approval, python):
    expected_ids = ["preflight", "smoke-A", "smoke-B", "train-A", "train-B", "lock",
                    "calibrate-H0", "calibrate-A", "calibrate-B", "calibration-lock",
                    "assess-H0", "assess-A", "assess-B", "report"]
    if [stage["id"] for stage in stages] != expected_ids:
        raise ValueError("coordinator stage order differs from approved sequence")
    training_stages = [stage for stage in stages if stage["module"] == "experiments.intermediate_training"]
    if [(stage["mode"], stage["arm"]) for stage in training_stages] != [
            ("smoke", "A"), ("smoke", "B"), ("train", "A"), ("train", "B")]:
        raise ValueError("training stages must contain exactly smoke/train A/B")
    for stage in stages:
        argv = stage["argv"]
        if argv[:3] != [python, "-m", stage["module"]] or any(
                value == "--resume" or value.startswith("--resume=") for value in argv):
            raise ValueError("stage command is not an exact non-resume Python module launch")
        if stage["module"] == "experiments.intermediate_training":
            if "--data" not in argv or Path(argv[argv.index("--data") + 1]) != data:
                raise ValueError("training data differs")
        else:
            common = _common_study_args(data, output, training, approval)
            if argv[4:4 + len(common)] != common:
                raise ValueError("study common arguments differ")


@contextmanager
def _new_output_lock(output):
    output = Path(output).resolve()
    if output.exists():
        lock_path = output / "COORDINATOR.lock"
        with lock_path.open("a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RecoveryRequired("another coordinator or inherited child owns COORDINATOR.lock") from error
        raise RecoveryRequired("existing pipeline output requires explicit review and a new output path")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "COORDINATOR.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RecoveryRequired("another coordinator owns COORDINATOR.lock") from error
        stream.write(json.dumps({"study": STUDY, "pid": os.getpid(), "started_utc": _utc()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        yield stream


def _complete_marker(directory):
    directory = Path(directory)
    result, marker = directory / "result.json", directory / "COMPLETE"
    if not result.is_file() or not marker.is_file() or marker.read_text().strip() != _sha(result):
        raise ValueError(f"completion marker is missing or invalid: {directory}")
    return _read(result)


def _validate_fingerprint(path, approval, arm):
    fingerprint = _read(path)
    if (fingerprint.get("study") != STUDY or fingerprint.get("arm") != arm
            or fingerprint.get("settings") != TRAINING_SETTINGS):
        raise ValueError("training fingerprint identity/settings differ")
    if Path(fingerprint.get("starting_checkpoint", "")).resolve() != Path(approval["original_checkpoint"]["path"]).resolve():
        raise ValueError("fingerprint starting checkpoint differs")
    if fingerprint.get("checkpoint_files") != approval["original_checkpoint"]["files"]:
        raise ValueError("fingerprint checkpoint files differ")
    if fingerprint.get("data_files") != approval["data"]["files"]:
        raise ValueError("fingerprint data files differ")
    approved_sources = approval["source_files"]
    if not isinstance(fingerprint.get("source_files"), dict) or any(
            approved_sources.get(path) != digest for path, digest in fingerprint["source_files"].items()):
        raise ValueError("fingerprint source files differ from approval")
    for field in ("prepared_sha256", "schedule_sha256", "schema_sha256", "replay_sha256"):
        value = fingerprint.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"fingerprint missing {field}")
    replay = approval["external_files"].get("data/processed-v0.2/train.jsonl")
    if replay is None or fingerprint["replay_sha256"] != replay:
        raise ValueError("fingerprint replay differs from approved external input")
    return fingerprint


def _validate_smoke(path, approval, arm):
    path = Path(path)
    result = _complete_marker(path)
    required = {
        "status": "PASSED", "arm": arm, "completed_updates": 2,
        "frozen_body_unchanged": True,
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError(f"smoke {arm} result is not the strict two-update pass")
    layout = result.get("same_layout", {})
    resume = result.get("resume", {})
    differences = resume.get("differences", {})
    separation = result.get("gradient_separation", {})
    expected_differences = {"core_parameters", "aux_parameters", "probabilities", "reload_probabilities"}
    gradient_norms = separation.get("gradient_norms", {})
    histories = (result.get("original_history"), result.get("restored_history"))
    if (layout.get("passed") is not True or layout.get("tolerance") != 1e-7
            or type(layout.get("max_logit_difference")) not in (int, float)
            or not math.isfinite(layout["max_logit_difference"]) or abs(layout["max_logit_difference"]) > 1e-7
            or separation.get("passed") is not True
            or separation.get("A_core_gradient_max_difference") != 0
            or separation.get("A_clipped_core_max_difference") != 0
            or type(separation.get("B_adapter_aux_gradient_max")) not in (int, float)
            or not math.isfinite(separation["B_adapter_aux_gradient_max"])
            or separation["B_adapter_aux_gradient_max"] <= 0
            or set(gradient_norms) != {"final-only", "A", "B"}
            or any((type(number) not in (bool, int, float))
                   or (type(number) in (int, float) and not math.isfinite(number))
                   for norms in gradient_norms.values() for number in norms.values())
            or resume.get("actual_disk_reload") is not True or resume.get("rng_equal") is not True
            or resume.get("tolerance") != 1e-7 or set(differences) != expected_differences
            or any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e-7
                   for value in differences.values())
            or result.get("frozen_body_before") != result.get("frozen_body_after")
            or any(not isinstance(history, list) or len(history) != 2
                   or [row.get("update") for row in history] != [1, 2]
                   or any(row.get("finite_gradients") is not True
                          or any(type(number) not in (int, float) or not math.isfinite(number)
                                 for number in row.values() if type(number) in (int, float))
                          for row in history) for history in histories)):
        raise ValueError(f"smoke {arm} lacks strict gradient/layout/reload proof")
    integrity = _read(path / "original-checkpoint-integrity.json")
    if integrity != {"unchanged": True, "before": approval["original_checkpoint"]["files"]}:
        raise ValueError(f"smoke {arm} original checkpoint integrity proof differs")
    fingerprint = _validate_fingerprint(path / "fingerprint.json", approval, arm)
    return {"path": str(path), "result_sha256": _sha(path / "result.json"),
            "fingerprint_sha256": _canonical_hash(fingerprint)}


def _validate_training(path, approval, arm, smoke_path):
    path = Path(path)
    smoke = _validate_fingerprint(Path(smoke_path) / "fingerprint.json", approval, arm)
    fingerprint = _validate_fingerprint(path / "fingerprint.json", approval, arm)
    if fingerprint != smoke:
        raise ValueError(f"train {arm} fingerprint differs from accepted smoke")
    result = _complete_marker(path)
    required = {"status": "COMPLETE", "arm": arm, "completed_updates": 200,
                "endpoint": "step-0200", "frozen_body_unchanged": True}
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError(f"train {arm} is not the exact 200-update endpoint")
    history, exposure = result.get("history"), result.get("exposure")
    if (result.get("frozen_body_before") != result.get("frozen_body_after")
            or not isinstance(history, list) or len(history) != 200
            or [row.get("update") for row in history] != list(range(1, 201))
            or any(row.get("finite_gradients") is not True
                   or any(type(number) in (int, float) and not math.isfinite(number)
                          for number in row.values()) for row in history)
            or not isinstance(exposure, list) or len(exposure) != 200
            or result.get("exposure_sha256") != _canonical_hash(exposure)):
        raise ValueError(f"train {arm} history/exposure/frozen-body proof differs")
    integrity = _read(path / "original-checkpoint-integrity.json")
    if integrity != {"unchanged": True, "before": approval["original_checkpoint"]["files"]}:
        raise ValueError(f"train {arm} original checkpoint integrity proof differs")
    return {"path": str(path), "result_sha256": _sha(path / "result.json"),
            "fingerprint_sha256": _canonical_hash(fingerprint), "completed_updates": 200}


def _approval_spec(approval_path, approval_sha):
    return {"path": str(Path(approval_path).resolve()), "sha256": approval_sha}


def _validate_study_artifact(stage, output, approval_path, approval_sha):
    output, identity = Path(output), stage["id"]
    endpoint, calibration = output / "endpoint-lock.json", output / "calibration-lock.json"
    if identity == "preflight":
        path = output / "preflight/report.json"
        value = _read(path)
        runtime = value.get("runtime", {})
        rows = runtime.get("rows", [])
        if (value.get("status") != "PASSED" or runtime.get("status") != "PASSED"
                or len(rows) != len(CATEGORIES) or {row.get("category") for row in rows} != set(CATEGORIES)
                or not isinstance(runtime.get("frozen_body_sha256"), str)
                or any(row.get("hook_logit_max_difference") != 0
                       or type(row.get("full_cached_max_probability_difference")) not in (int, float)
                       or type(row.get("full_cached_argmax_changes")) is not int for row in rows)):
            raise ValueError("study preflight did not pass")
    elif identity == "lock":
        path, value = endpoint, _read(endpoint)
        models = value.get("models")
        if set(models if isinstance(models, (list, dict)) else ()) != set(MODELS):
            raise ValueError("endpoint lock model population differs")
        if value.get("selected_updates") != 200 or value.get("approval") != _approval_spec(approval_path, approval_sha):
            raise ValueError("endpoint lock update/approval binding differs")
    elif identity.startswith("calibrate-"):
        model = stage["model"]
        path = output / f"calibration/{model}/fit.json"
        value = _read(path)
        if (value.get("model") != model or value.get("endpoint_sha256") != _sha(endpoint)
                or not value.get("sources") or not value.get("fits")):
            raise ValueError(f"calibration {model} is incomplete or endpoint-unbound")
    elif identity == "calibration-lock":
        path, value = calibration, _read(calibration)
        if value.get("endpoint_sha256") != _sha(endpoint) or set(value.get("fits", {})) != set(MODELS):
            raise ValueError("calibration lock endpoint/model binding differs")
        for model, spec in value["fits"].items():
            fit = output / f"calibration/{model}/fit.json"
            if spec != {"path": str(fit.resolve()), "sha256": _sha(fit)}:
                raise ValueError("calibration lock fit binding differs")
    elif identity.startswith("assess-"):
        model = stage["model"]
        path = output / f"assessment/{model}/report.json"
        value = _read(path)
        if (value.get("model") != model or value.get("endpoint_sha256") != _sha(endpoint)
                or value.get("calibration_sha256") != _sha(calibration)
                or value.get("frozen_body_unchanged") is not True):
            raise ValueError(f"assessment {model} is incomplete or lock-unbound")
    elif identity == "report":
        path = output / "final-report/report.json"
        value = _read(path)
        endpoint_sha, calibration_sha = _sha(endpoint), _sha(calibration)
        models = value.get("models", {})
        if (value.get("status") != "COMPLETE" or value.get("endpoints") != _read(endpoint)
                or set(models) != set(MODELS)
                or any(report.get("model") != model or report.get("endpoint_sha256") != endpoint_sha
                       or report.get("calibration_sha256") != calibration_sha
                       or report.get("frozen_body_unchanged") is not True
                       for model, report in models.items())):
            raise ValueError("final report is incomplete or lock-unbound")
    else:
        raise ValueError(f"unknown study stage {identity}")
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _stage_preconditions(stage, output, training, approval):
    if stage["id"] == "train-A":
        return [_validate_smoke(Path(output) / "smoke-A", approval, "A")]
    if stage["id"] == "train-B":
        return [_validate_smoke(Path(output) / "smoke-B", approval, "B")]
    if stage["id"] == "lock":
        return [_validate_training(Path(training) / arm, approval, arm, Path(output) / f"smoke-{arm}") for arm in ARMS]
    return []


def _progress(output, status, **fields):
    value = {"study": STUDY, "status": status, "coordinator_pid": os.getpid(),
             "updated_utc": _utc(), **fields}
    _atomic(Path(output) / "progress.json", value)
    return value


def _run_stage(stage, *, repo, output, training, approval_path, approval, approval_sha, lock):
    verify_approval(repo, approval["data"]["path"], approval_path, expected_sha256=approval_sha)
    preconditions = _stage_preconditions(stage, output, training, approval)
    receipt_path = Path(output) / "receipts" / f"{stage['id']}.json"
    if receipt_path.exists():
        raise ValueError("pre-existing stage receipt requires explicit recovery")
    receipt = {"study": STUDY, "stage": stage["id"], "status": "running",
               "argv": stage["argv"], "argv_sha256": _canonical_hash(stage["argv"]),
               "approval_sha256": approval_sha, "preconditions": preconditions,
               "created_utc": _utc(), "coordinator_pid": os.getpid()}
    log = Path(output) / "logs" / f"{stage['id']}.log"
    receipt["log"] = str(log)
    _atomic(receipt_path, receipt)
    child = None
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("xb") as stream:
            child = subprocess.Popen(stage["argv"], cwd=repo, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, shell=False, start_new_session=True,
                pass_fds=(lock.fileno(),))
            receipt["child_pid"] = child.pid
            _atomic(receipt_path, receipt)
            while True:
                _progress(output, "running", stage=stage["id"], child_pid=child.pid, log=str(log))
                try:
                    code = child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    # Avoid re-hashing the large immutable checkpoint every five
                    # seconds. Full source/data/checkpoint verification occurs
                    # immediately before and after every subprocess.
                    if _sha(approval_path) != approval_sha:
                        raise ValueError("approval changed while subprocess was running")
            stream.flush()
            os.fsync(stream.fileno())
        receipt["returncode"] = code
        if code != 0:
            raise ValueError(f"child exited {code}")
        verify_approval(repo, approval["data"]["path"], approval_path, expected_sha256=approval_sha)
        if _stage_preconditions(stage, output, training, approval) != preconditions:
            raise ValueError("stage preconditions changed during subprocess")
        if stage["module"] == "experiments.intermediate_training":
            destination = Path(stage["argv"][stage["argv"].index("--output") + 1])
            artifact = (_validate_smoke(destination, approval, stage["arm"])
                        if stage["mode"] == "smoke" else
                        _validate_training(destination, approval, stage["arm"], Path(output) / f"smoke-{stage['arm']}"))
        else:
            artifact = _validate_study_artifact(stage, output, approval_path, approval_sha)
        receipt.update(status="complete", finished_utc=_utc(), artifact=artifact)
        _atomic(receipt_path, receipt)
        return artifact
    except BaseException as error:
        alive = child is not None and child.poll() is None
        receipt.update(status="interrupted" if alive else "failed", error={
            "type": type(error).__name__, "message": str(error)}, updated_utc=_utc())
        _atomic(receipt_path, receipt)
        raise


def run_pipeline(*, repo, data, output, training, approval):
    """Run the fixed sequence once; existing output is never resumed automatically."""
    repo, data, output, training, approval_path = map(
        lambda value: Path(value).resolve(), (repo, data, output, training, approval))
    try:
        approved, approval_sha = verify_approval(repo, data, approval_path)
        stages = build_stages(data=data, output=output, training=training, approval=approval_path)
        with _new_output_lock(output) as lock:
            if any((training / arm).exists() for arm in ARMS):
                raise ValueError("training arm output already exists; explicit recovery requires a new training root")
            binding = {"version": 1, "study": STUDY, "repo": str(repo), "data": str(data),
                       "output": str(output), "training_root": str(training),
                       "approval": _approval_spec(approval_path, approval_sha),
                       "coordinator_sha256": _sha(__file__), "python": os.path.abspath(sys.executable),
                       "stage_plan_sha256": _canonical_hash(stages)}
            _atomic(output / "run-binding.json", binding)
            _atomic(output / "approval.snapshot.json", approved)
            _atomic(output / "stage-plan.json", {"stages": stages})
            _progress(output, "running", next_stage=stages[0]["id"], completed_stages=[])
            complete = []
            for stage in stages:
                _run_stage(stage, repo=repo, output=output, training=training,
                           approval_path=approval_path, approval=approved, approval_sha=approval_sha, lock=lock)
                complete.append(stage["id"])
                _progress(output, "running", next_stage=(stages[len(complete)]["id"]
                          if len(complete) < len(stages) else None), completed_stages=complete)
            return _progress(output, "complete", completed_stages=complete,
                             endpoint_lock_sha256=_sha(output / "endpoint-lock.json"),
                             calibration_lock_sha256=_sha(output / "calibration-lock.json"),
                             report_sha256=_sha(output / "final-report/report.json"))
    except RecoveryRequired:
        raise
    except BaseException as error:
        if output.exists():
            _atomic(output / "failure.json", {"study": STUDY, "status": "recovery_required",
                    "updated_utc": _utc(), "error": {"type": type(error).__name__, "message": str(error)}})
            _progress(output, "recovery_required", error=str(error))
        raise RecoveryRequired(f"explicit recovery required: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    try:
        result = run_pipeline(repo=repo, data=args.data, output=args.output,
                              training=args.training_root, approval=args.approval)
    except RecoveryRequired as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
