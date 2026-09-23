"""Six-arm calibrated screen with the unchanged factorial checkpoint machinery.

Only the scheduling process may launch the CUDA CLI. All outputs are new; the
old modules are adapted in memory, never edited. ``runner_context`` must span
checkpoint validation/restoration because the study and arm identities differ.
"""

import argparse
import fcntl
import gc
import json
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from experiments import contrast_factorial_training as factorial
from experiments import contrast_scaling_training as scaling
from experiments.judgment_pipeline import optimizer_for
from experiments.calibrated_integrity import load_verified_targets
from openjev.judgment_training import PreparedBundle, TrainingGroup, prepare_bundle, target_vector

STUDY = "calibrated-screen-v1"
ARMS = ("H0", "H1", "Q0", "Q1", "J0", "J1")
REPO_ROOT = factorial.REPO_ROOT
MANIFEST = REPO_ROOT / "data/contrast-factorial-v1/train/v1/manifest.json"
REPLAY = REPO_ROOT / "data/processed-v0.2/train.jsonl"
BASE = REPO_ROOT / "checkpoints/judgment-full-v0.2/final"
PROTOCOL = REPO_ROOT / "reports/calibrated-screen-v1/PROTOCOL.md"
STUDY_LOCK = REPO_ROOT / "reports/calibrated-screen-v1/STUDY.lock"
_CONTEXT_LOCK = threading.Lock()
_ORIGINAL_BACKWARD = scaling.backward_family_loss


def loss_config(arm):
    if arm not in ARMS:
        raise ValueError("unknown calibrated arm")
    return {"alpha": 0.0 if arm[0] == "H" else 0.5, "brier_lambda": float(arm[1]),
            "teacher": None if arm[0] == "H" else arm[0], "teacher_temperature": 1.0,
            "component_weight": 1 / 6, "replay": "unchanged-hard-CE"}


def _teacher_vector(teacher, width):
    if (not isinstance(teacher, (tuple, list)) or len(teacher) != width
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in teacher)
            or not math.isclose(sum(teacher), 1.0, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("teacher must be a finite normalized complete distribution")
    return teacher


def group_loss(logits, group, *, arm, teacher=None):
    """New-data loss; the binary vector includes both false and true outcomes."""
    config = loss_config(arm)
    hard = scaling._group_loss(logits, group)
    if arm == "H0":
        return hard
    target = logits.new_tensor(target_vector(group))
    alpha, weight = config["alpha"], config["brier_lambda"]
    loss = hard
    if alpha:
        soft = logits.new_tensor(_teacher_vector(teacher, len(target)))
        teacher_ce = (F.binary_cross_entropy_with_logits(logits[0], soft[1]) if group.primitive == "noul"
                      else -(soft * logits.log_softmax(-1)).sum())
        loss = (1 - alpha) * hard + alpha * teacher_ce
    if weight:
        p = logits[0].sigmoid() if group.primitive == "noul" else None
        probabilities = torch.stack((1 - p, p)) if p is not None else logits.softmax(-1)
        loss = loss + weight * (probabilities - target).square().sum()
    return loss


def backward_family_loss(model, family, replay, *, arm, max_units=12, unit_batch_size=12):
    """Release activations per complete group, retaining six equal component means."""
    loss_config(arm)
    if arm == "H0":
        return _ORIGINAL_BACKWARD(model, family, replay, max_units=max_units, unit_batch_size=unit_batch_size)
    components = scaling._component_groups(family, replay)
    totals, counts = {}, {}
    for (origin, kind), values in components.items():
        bundle, wanted = values[0][0], [group for _, group in values]
        numeric = []
        for prompts, kinds, chosen in scaling._complete_microbatches(bundle, max_units, selected=wanted):
            logits = model.score_prompts(prompts, kinds, unit_batch_size=unit_batch_size)
            losses = [scaling._group_loss(logits[list(local)], group) if origin == "old"
                      else group_loss(logits[list(local)], group, arm=arm,
                                      teacher=getattr(group, "teacher", None)) for group, local in chosen]
            micro = torch.stack(losses).sum() / (6 * len(wanted))
            if not torch.isfinite(micro):
                raise RuntimeError("non-finite complete-candidate calibrated microbatch loss")
            micro.backward()
            numeric.extend(loss.detach().item() for loss in losses)
        if len(numeric) != len(wanted):
            raise RuntimeError("missing complete candidate group during calibrated backward")
        key = f"{origin}/{kind}"
        totals[key], counts[key] = sum(numeric) / len(wanted), len(wanted)
    return {"loss": sum(totals.values()) / 6, "components": totals, "counts": counts}


@contextmanager
def runner_context(arm):
    """Process-local adapter; fail concurrent/nested use and restore on all exits.

    Validation, serialization, scheduling, Adam checks, and failure preservation
    remain the existing functions. Their globals receive the new study/arm IDs.
    The scaling loss is also replaced so its save/reload proof uses this loss.
    """
    loss_config(arm)
    if not _CONTEXT_LOCK.acquire(blocking=False):
        raise RuntimeError("a calibrated runner context is already active")
    original = factorial.STUDY, factorial.ARMS, factorial.backward_family_loss, scaling.backward_family_loss
    try:
        def backward(model, family, replay, **kwargs):
            return backward_family_loss(model, family, replay, arm=arm, **kwargs)

        factorial.STUDY, factorial.ARMS = STUDY, ARMS
        factorial.backward_family_loss = scaling.backward_family_loss = backward
        yield
    finally:
        factorial.STUDY, factorial.ARMS, factorial.backward_family_loss, scaling.backward_family_loss = original
        _CONTEXT_LOCK.release()


@contextmanager
def study_lock(path=STUDY_LOCK):
    """One shared advisory lock for every calibrated GPU process."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"calibrated study lock is held: {path}") from error
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({"pid": os.getpid(), "study": STUDY}) + "\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def production_resume_equivalence(engine, optimizer, checkpoint, load_engine, family, replay, *,
                                  output, arm, fingerprint):
    """Use the real disk reload and additionally run the stricter Adam/RNG verifier."""
    factorial.validate_resume_checkpoint(checkpoint, arm=arm, fingerprint=fingerprint, output=output)

    def verified_loader(path):
        reloaded = load_engine(Path(path))
        verification_optimizer = optimizer_for(reloaded.model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
        factorial.restore_arm(path, reloaded, verification_optimizer, arm=arm, fingerprint=fingerprint, output=output)
        del verification_optimizer
        return reloaded

    proof = scaling.production_resume_equivalence(
        engine, optimizer, checkpoint, verified_loader, family, replay,
        max_units=12, unit_batch_size=12, tolerance=1e-7,
    )
    for key in ("max_probability_difference", "max_trainable_difference"):
        if not math.isfinite(proof[key]) or proof[key] > 1e-7:
            raise RuntimeError("strict calibrated resume proof failed")
    return proof


def validate_smoke_gate(gate, fingerprint):
    factorial.validate_smoke_gate_static(gate)
    if gate.get("study") != STUDY or gate.get("arm") != fingerprint.get("arm"):
        raise ValueError("smoke study/arm mismatch")
    if gate.get("fingerprint") != fingerprint:
        raise ValueError("smoke fingerprint differs from this calibrated run")
    training = gate.get("training", {})
    gradient = training.get("gradient_check", {})
    if any(gradient.get(key) is not True for key in ("finite", "adapters", "heads", "frozen")):
        raise ValueError("smoke requires a finite adapter/head/frozen gradient proof")
    if not training.get("frozen_before") or training["frozen_before"] != training.get("frozen_after"):
        raise ValueError("smoke frozen body differs")
    proof = gate["resume_equivalence"]
    if any(not math.isfinite(proof[key]) or proof[key] < 0 for key in
           ("max_probability_difference", "max_trainable_difference", "tolerance")):
        raise ValueError("smoke requires finite nonnegative resume differences")


@dataclass(frozen=True)
class BoundGroup(TrainingGroup):
    case_id: str
    question_id: str
    request_sha256: str
    teacher: tuple | None


def prepare_bound_families(engine, records, suite_path, *, arm, targets_path=None,
                           target_loader=load_verified_targets):
    """Compile each identified example, attaching distributions by semantic key.

    No positional join is used: even reversed teacher rows and training examples
    keep their own case/question/request/answer-space identities. Each group's
    metadata is attached while its own single-question request is compiled.
    """
    from experiments.calibrated_targets import labels_for, request_sha256

    config = loss_config(arm)
    if bool(targets_path) != bool(config["alpha"]):
        raise ValueError("teacher arms require targets; hard arms must not receive targets")
    scaling.validate_family_coverage(records, [record["id"] for record in records])
    suite = json.loads(Path(suite_path).read_text())
    cases = {case["id"]: case for case in suite["cases"]}
    if len(cases) != len(suite["cases"]):
        raise ValueError("duplicate suite case identity")
    expected = {(case["id"], qid) for case in cases.values() for qid in case["request"]["questions"]}
    targets = target_loader(targets_path, suite_path) if targets_path else None
    if targets:
        wanted_model = {"J": "jev-1.13.0", "Q": "Qwen/Qwen3.5-4B"}[arm[0]]
        if targets["teacher"]["model"] != wanted_model:
            raise ValueError("teacher model does not match the requested arm")
    rows = {(row["case_id"], row["question_id"]): row for row in targets["rows"]} if targets else {}
    prepared, seen = {}, set()
    for record in records:
        prompts, kinds, groups = [], [], []
        for example in record["examples"]:
            key = example["case_id"], example["question_id"]
            if key in seen or key not in expected:
                raise ValueError("duplicate or unknown training case/question identity")
            seen.add(key)
            case = cases[key[0]]
            request, question = case["request"], case["request"]["questions"][key[1]]
            if (case.get("family_id") != record["id"] or example["state"] != request["state"]
                    or example["question"] != question):
                raise ValueError("training example differs from suite family/request definition")
            single = prepare_bundle(engine, {**record, "examples": [example], "relations": []})
            if len(single.groups) != 1:
                raise ValueError("compiled example must have exactly one candidate group")
            original = single.groups[0]
            if original.primitive != question["type"]:
                raise ValueError("compiled primitive differs from request")
            labels = (["false", "true"] if original.primitive == "noul" else list(original.criteria)
                      if original.primitive == "choice" else [str(i) for i in range(len(original.criteria))])
            if set(labels) != set(labels_for(question)):
                raise ValueError("compiled answer space differs from request")
            digest = request_sha256(request)
            teacher = None
            if targets:
                row = rows[key]
                if row["request_sha256"] != digest:
                    raise ValueError("teacher request identity differs from training request")
                teacher = tuple(row["probabilities"][label] for label in labels)
                _teacher_vector(teacher, len(target_vector(original)))
            group = BoundGroup(original.primitive, tuple(i + len(prompts) for i in original.indices),
                               original.criteria, original.target, key[0], key[1], digest, teacher)
            groups.append(group)
            prompts.extend(single.prompts)
            kinds.extend(single.kinds)
        if max(map(len, prompts)) > factorial.MAX_INPUT_TOKENS:
            raise ValueError("compiled training prompt exceeds fixed token budget")
        prepared[record["id"]] = PreparedBundle(record["id"], single.source, prompts, kinds, groups,
                                                 record.get("relations", []))
    if seen != expected:
        raise ValueError("training population does not exactly cover the target suite")
    scaling.validate_prepared_group_widths(prepared, max_units=12)
    return prepared


def record_common_start(engine, optimizer, path):
    """Freeze matched fresh weights, optimizer and all RNG states across arms."""
    if optimizer.state:
        raise ValueError("common start requires fresh Adam state")
    value = {**factorial._arm_start_identity(engine), "optimizer_state_entries": 0,
             "optimizer_initial_sha256": factorial._canonical_hash(optimizer.state_dict())}
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("common start weights/optimizer/RNG differ")
    else:
        factorial._write(path, value)
    return value


def build_fingerprint(manifest_path, manifest, arm, schedule, replay_path, prepared, *,
                      targets_path=None, repo_root=REPO_ROOT):
    value = factorial.build_fingerprint(manifest_path, manifest, "B", schedule, replay_path, prepared, repo_root)
    sources = ("experiments/calibrated_training.py", "experiments/calibrated_targets.py",
               "experiments/calibrated_integrity.py",
               "reports/calibrated-screen-v1/PROTOCOL.md")
    value.update(study=STUDY, arm=arm, loss=loss_config(arm), training_source_arm="B",
                 targets_sha256=factorial._sha(targets_path) if targets_path else None,
                 calibrated_source_sha256={path: factorial._sha(Path(repo_root) / path) for path in sources},
                 updates=400, adapter_lr=5e-5, head_lr=2.5e-5, gradient_clip_norm=1.0)
    return value


def execute_arm(arm, output, *, fingerprint, schedule, prepared, base, smoke=False,
                smoke_selection=None, smoke_gate=None, resume=None, pause_file=None,
                common_start_path=REPO_ROOT / "reports/calibrated-screen-v1/common-start.json",
                engine_loader=None, recover_history_tail=False):
    """Execute one fresh or resumed arm; caller owns the shared study lock."""
    loss_config(arm)
    if smoke and resume:
        raise ValueError("smoke must start fresh")
    if recover_history_tail and not resume:
        raise ValueError("history recovery requires a resume checkpoint")
    if not smoke:
        if smoke_gate is None:
            raise ValueError("full training requires a strict smoke gate")
        validate_smoke_gate(smoke_gate, fingerprint)
    output = Path(output)
    engine_loader = engine_loader or factorial._new_engine
    actual_schedule, actual_fingerprint = schedule, fingerprint
    if smoke:
        if not smoke_selection or len(smoke_selection) >= 400:
            raise ValueError("smoke selection must leave one next-update proof")
        chosen = [row["family_id"] for row in smoke_selection]
        actual_schedule = [{**row, "family_id": chosen[i % len(chosen)]} for i, row in enumerate(schedule)]
        actual_fingerprint = {**fingerprint, "schedule_sha256": factorial._canonical_hash(actual_schedule),
                              "purpose": "smoke", "smoke_selection": smoke_selection}
    training_output = output / arm if smoke else output
    if not resume and output.exists():
        raise FileExistsError(output)
    engine = optimizer = None
    try:
        with runner_context(arm):
            if resume:
                if recover_history_tail:
                    factorial.recover_history_tail(resume, arm=arm, fingerprint=fingerprint, output=output)
                factorial.validate_resume_checkpoint(resume, arm=arm, fingerprint=fingerprint, output=output)
            scaling.apply_determinism(factorial.SEED)
            engine = engine_loader(Path(resume) if resume else Path(base))
            optimizer = optimizer_for(engine.model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
            if resume:
                start_state = factorial.restore_arm(resume, engine, optimizer, arm=arm,
                                                     fingerprint=fingerprint, output=output)
                start_state["checkpoint"] = str(resume)
                arm_start = start_state.get("arm_start")
            else:
                start_state = None
                arm_start = record_common_start(engine, optimizer, common_start_path)
            state = factorial.run_arm(
                arm, engine, actual_schedule, prepared, training_output,
                fingerprint=actual_fingerprint, optimizer=optimizer, start_state=start_state,
                pause_file=pause_file, stop_after=len(smoke_selection) if smoke else 400, arm_start=arm_start,
            )
            report = {"study": STUDY, "arm": arm, "mode": "smoke" if smoke else "train",
                      "status": state["status"], "fingerprint": fingerprint, "training": state,
                      "passed": False, "completed_requested_pass": state["status"] == "complete"}
            if smoke and state["status"] == "complete":
                next_row = actual_schedule[len(smoke_selection)]
                replay = {item["primitive"]: prepared[item["id"]] for item in next_row["replay"]}
                proof = production_resume_equivalence(
                    engine, optimizer, state["checkpoint"], engine_loader, prepared[next_row["family_id"]], replay,
                    output=training_output, arm=arm, fingerprint=actual_fingerprint,
                )
                report.update(resume_equivalence=proof, smoke_selection=smoke_selection,
                              selection_is_outcome_independent=True, passed=proof["passed"])
                validate_smoke_gate(report, fingerprint)
            elif not smoke:
                report["passed"] = state["status"] == "complete" and state["completed_updates"] == 400
            factorial._write(output / "report.json", report)
            return report
    except BaseException as error:
        if output.exists():
            factorial._write(output / "run-failure.json", {"study": STUDY, "arm": arm,
                             "status": "failed", "error": repr(error), "fingerprint": fingerprint})
        raise
    finally:
        del optimizer, engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _validate_output(output):
    output = Path(output).resolve()
    allowed = [REPO_ROOT / name / STUDY for name in ("reports", "checkpoints", "data")]
    if not any(root in output.parents for root in allowed):
        raise ValueError("output must be a new subdirectory under reports/checkpoints/data/calibrated-screen-v1")
    return output


def run(args):
    """Freeze read-only training provenance before loading the root-owned GPU."""
    output = _validate_output(args.output)
    if args.smoke and args.resume:
        raise ValueError("smoke cannot resume")
    if bool(args.targets) != bool(loss_config(args.arm)["alpha"]):
        raise ValueError("teacher arms require --targets; hard arms must omit --targets")
    gate = None
    if not args.smoke:
        if not args.smoke_gate:
            raise ValueError("full training requires --smoke-gate")
        gate = json.loads(Path(args.smoke_gate).read_text())
        factorial.validate_smoke_gate_static(gate)
    with study_lock():
        pause_file = Path(args.pause_file) if args.pause_file else REPO_ROOT / "reports/calibrated-screen-v1/PAUSE"
        if pause_file.exists():
            # No model or output directory is created for a pristine pause.
            return {"study": STUDY, "arm": args.arm, "status": "paused", "model_loaded": False,
                    "pause_reason": "request-file-present-before-preparation"}
        manifest = json.loads(MANIFEST.read_text())
        records = factorial.validate_manifest(manifest, REPO_ROOT)["B"]
        if (REPO_ROOT / manifest["base_checkpoint"]).resolve() != BASE.resolve():
            raise ValueError("manifest does not declare the fixed v0.2 starting checkpoint")
        suite_path = REPO_ROOT / manifest["arms"]["B"]["suite_file"]
        scaling.apply_determinism(factorial.SEED)
        preparation = factorial._new_preparation_engine(BASE)
        try:
            prepared = prepare_bound_families(preparation, records, suite_path,
                                              arm=args.arm, targets_path=args.targets)
            replay_prepared, replay_pools = scaling.prepare_canonical_replay(preparation, REPLAY)
            prepared.update(replay_prepared)
        finally:
            del preparation
            gc.collect()
        schedule = factorial.make_schedule(manifest["ordered_latent_ids"], replay_pools)
        fingerprint = build_fingerprint(MANIFEST, manifest, args.arm, schedule, REPLAY, prepared,
                                         targets_path=args.targets)
        if gate is not None:
            validate_smoke_gate(gate, fingerprint)
        selection = factorial.select_smoke_families({"B": records}, {"B": prepared}) if args.smoke else None
        factorial._require_cuda()
        return execute_arm(args.arm, output, fingerprint=fingerprint, schedule=schedule, prepared=prepared,
                            base=BASE, smoke=args.smoke, smoke_selection=selection, smoke_gate=gate,
                            resume=args.resume, pause_file=pause_file,
                            recover_history_tail=args.recover_history_tail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-gate", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--recover-history-tail", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report.get(key) for key in ("study", "arm", "mode", "status", "passed")},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
