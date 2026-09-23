from experiments.calibrated_targets import load_targets as load_unverified_fixture_targets
"""CPU contracts for the six-arm calibrated screen; no model downloads."""

import copy
import importlib
import json
import math
from dataclasses import replace

import pytest
import torch
from test_contrast_factorial_training import (
    TinyEngine,
    _family,
    _family_record,
    _fingerprint,
    _load_tiny,
    _manifest_tree,
    _optimizer,
    _prepared,
    _replay,
    _schedule,
)

from openjev.judgment_training import TrainingGroup


def module():
    return importlib.import_module("experiments.calibrated_training")


@pytest.mark.parametrize("arm,expected,gradient", [
    ("H0", math.log(2), -0.5),
    ("H1", math.log(2) + 0.5, -1.0),
    ("Q0", math.log(2), -0.125),
    ("Q1", math.log(2) + 0.5, -0.625),
    ("J0", math.log(2), -0.125),
    ("J1", math.log(2) + 0.5, -0.625),
])
def test_binary_mixture_and_brier_use_both_outcomes(arm, expected, gradient):
    """Omitting false from Brier or doubling CE changes these literal gradients."""
    logits = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    group = TrainingGroup("noul", (0,), None, {"truth": True})
    loss = module().group_loss(logits, group, arm=arm, teacher=(0.75, 0.25) if arm[0] != "H" else None)
    loss.backward()
    assert loss.item() == pytest.approx(expected, abs=1e-12)
    assert logits.grad.item() == pytest.approx(gradient, abs=1e-12)


@pytest.mark.parametrize("kind,target", [("choice", {"choice": "a"}), ("score", {"level_index": 0})])
def test_categorical_mixture_gradient_matches_hand_computation(kind, target):
    logits = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    group = TrainingGroup(kind, (0, 1, 2), ("a", "b", "c"), target)
    loss = module().group_loss(logits, group, arm="J1", teacher=(0.0, 0.25, 0.75))
    loss.backward()
    assert loss.item() == pytest.approx(math.log(3) + 2 / 3)
    assert logits.grad.tolist() == pytest.approx([-11 / 18, 31 / 72, 13 / 72])


@pytest.mark.parametrize("teacher", [None, (0.2,), (0.3, 0.3), (float("nan"), 0.0), (-0.1, 1.1)])
def test_teacher_loss_rejects_missing_or_invalid_distribution(teacher):
    with pytest.raises(ValueError, match="teacher"):
        module().group_loss(torch.zeros(1), TrainingGroup("noul", (0,), None, {"truth": True}),
                            arm="Q0", teacher=teacher)


def test_hard_off_is_bit_exact_old_path_and_old_replay_remains_hard():
    from experiments.contrast_scaling_training import backward_family_loss as original

    torch.manual_seed(42)
    old = TinyEngine().model
    new = copy.deepcopy(old)
    family, replay = _family(), _replay()
    old_result = original(old, family, replay, max_units=12, unit_batch_size=12)
    new_result = module().backward_family_loss(new, family, replay, arm="H0")
    assert old_result == new_result
    for a, b in zip(old.parameters(), new.parameters(), strict=True):
        assert (a.grad is None and b.grad is None) or torch.equal(a.grad, b.grad)
    changed = copy.deepcopy(old)
    changed.zero_grad()
    result = module().backward_family_loss(changed, family, replay, arm="H1")
    for primitive in ("noul", "choice", "score"):
        assert result["components"]["old/" + primitive] == old_result["components"]["old/" + primitive]
    assert result["counts"] == {"new/noul": 12, "new/choice": 4, "new/score": 4,
                                "old/noul": 1, "old/choice": 1, "old/score": 1}
    assert result["loss"] == pytest.approx(sum(result["components"].values()) / 6)


def test_six_primitive_means_do_not_reweight_when_noul_examples_repeat():
    family, replay = _family(), _replay()
    repeated = replace(family, groups=family.groups + [replace(g) for g in family.groups if g.primitive == "noul"])
    torch.manual_seed(42)
    first = TinyEngine().model
    second = copy.deepcopy(first)
    one = module().backward_family_loss(first, family, replay, arm="H1")
    two = module().backward_family_loss(second, repeated, replay, arm="H1")
    assert one["loss"] == pytest.approx(two["loss"], abs=1e-7)
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        if a.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=1e-7, rtol=0)


def new_fingerprint(arm="H1"):
    return {**_fingerprint(), "study": "calibrated-screen-v1", "arm": arm,
            "loss": {"alpha": 0.0, "brier_lambda": 1.0}, "targets_sha256": None}


def test_adapter_restores_original_modules_even_after_exception():
    from experiments import contrast_factorial_training as old
    from experiments import contrast_scaling_training as scaling

    before = old.STUDY, old.ARMS, old.backward_family_loss, scaling.backward_family_loss
    with pytest.raises(RuntimeError, match="deliberate"):
        with module().runner_context("H1"):
            assert old.STUDY == "calibrated-screen-v1"
            with pytest.raises(RuntimeError, match="active"):
                with module().runner_context("J1"):
                    pass
            raise RuntimeError("deliberate")
    assert (old.STUDY, old.ARMS, old.backward_family_loss, scaling.backward_family_loss) == before


def test_calibrated_boundary_resume_matches_next_update_and_rejects_source_drift(tmp_path):
    from experiments import contrast_factorial_training as old

    torch.manual_seed(42)
    uninterrupted = TinyEngine()
    optimizer = _optimizer(uninterrupted.model)
    fp, schedule, prepared = new_fingerprint(), _schedule(), _prepared()
    with module().runner_context("H1"):
        state = old.run_arm("H1", uninterrupted, schedule, prepared, tmp_path / "arm", fingerprint=fp,
                            optimizer=optimizer, stop_after=1)
        assert state["study"] == "calibrated-screen-v1"
        payload = old.validate_resume_checkpoint(state["checkpoint"], arm="H1", fingerprint=fp,
                                                  output=tmp_path / "arm")
        assert payload["torch_rng"].dtype == torch.uint8 and payload["optimizer"]["state"]
        bad = copy.deepcopy(fp)
        bad["source_fingerprints"]["runner.py"] = "changed"
        with pytest.raises(ValueError, match="fingerprint"):
            old.validate_resume_checkpoint(state["checkpoint"], arm="H1", fingerprint=bad,
                                           output=tmp_path / "arm")
        proof = module().production_resume_equivalence(uninterrupted, optimizer,
                                                       state["checkpoint"], _load_tiny, prepared["family/0"],
                                                       _replay(), output=tmp_path / "arm", arm="H1", fingerprint=fp)
    assert proof["passed"] and proof["max_trainable_difference"] <= 1e-7


def test_smoke_gate_rejects_nonfinite_and_missing_gradient_proof():
    fp = new_fingerprint()
    gate = {"study": "calibrated-screen-v1", "arm": "H1", "mode": "smoke", "status": "complete",
            "passed": True, "completed_requested_pass": True, "fingerprint": fp,
            "training": {"gradient_check": {"finite": True, "adapters": True, "heads": True, "frozen": True},
                         "frozen_before": "body", "frozen_after": "body"},
            "resume_equivalence": {"passed": True, "tolerance": 1e-7, "max_probability_difference": 0.0,
                                   "max_trainable_difference": 0.0, "frozen_before": "body", "frozen_after": "body"}}
    module().validate_smoke_gate(gate, fp)
    for mutation in (
        lambda g: g["resume_equivalence"].__setitem__("max_trainable_difference", float("nan")),
        lambda g: g["resume_equivalence"].__setitem__("max_trainable_difference", 1e-6),
        lambda g: g["training"].pop("gradient_check"),
        lambda g: g["training"]["gradient_check"].__setitem__("frozen", False),
        lambda g: g["fingerprint"].__setitem__("targets_sha256", "changed"),
    ):
        bad = copy.deepcopy(gate)
        mutation(bad)
        with pytest.raises(ValueError):
            module().validate_smoke_gate(bad, fp)


def test_shared_study_lock_prevents_second_runner(tmp_path):
    lock = tmp_path / "study.lock"
    with module().study_lock(lock):
        with pytest.raises(RuntimeError, match="lock"):
            with module().study_lock(lock):
                pass
    with module().study_lock(lock):
        pass


def binding_fixture(tmp_path):
    from experiments.calibrated_targets import labels_for, request_sha256, write_targets
    from openjev.judgment_model import JudgmentEngine

    class Tokenizer:
        def encode(self, value, **kwargs):
            return [32 if value == "A" else 33]

        def apply_chat_template(self, messages, **kwargs):
            # A deterministic local tokenizer double; real request compiler stays active.
            return list(json.dumps(messages, sort_keys=True).encode())

    record = _family_record("latent/000", "B")
    cases = {}
    for example in record["examples"]:
        example["question"]["instructions"] = example["question_id"]
        case = cases.setdefault(example["case_id"], {"id": example["case_id"], "family_id": record["id"],
                                                     "request": {"state": {}, "questions": {}}})
        case["request"]["questions"][example["question_id"]] = example["question"]
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"cases": list(cases.values())}))
    observation = tmp_path / "observation.json"
    observation.write_text("{}")
    rows = []
    for case in cases.values():
        for qid, question in case["request"]["questions"].items():
            labels = labels_for(question)
            p = 0.8 if qid == "n1" else 0.2
            probabilities = {label: 1 / len(labels) for label in reversed(labels)}
            if question["type"] == "noul":
                probabilities = {"true": p, "false": 1 - p}
            rows.append({"case_id": case["id"], "question_id": qid, "probabilities": probabilities,
                         "request_sha256": request_sha256(case["request"]), "observation_path": str(observation)})
    targets = tmp_path / "targets.json"
    write_targets(suite, list(reversed(rows)), {"model": "jev-1.13.0"}, targets)
    engine = JudgmentEngine(None, Tokenizer(), max_input_tokens=1536, unit_batch_size=12)
    return engine, record, suite, targets


def test_target_binding_survives_reordering_and_matches_actual_compiled_groups(tmp_path):
    from experiments.contrast_scaling_training import prepare_families

    engine, record, suite, targets = binding_fixture(tmp_path)
    old = prepare_families(engine, [record], [record["id"]])[record["id"]]
    prepared = module().prepare_bound_families(engine, [record], suite, arm="J0", targets_path=targets, target_loader=load_unverified_fixture_targets)
    new = prepared[record["id"]]
    assert new.prompts == old.prompts and new.kinds == old.kinds
    for group in new.groups:
        if group.primitive == "noul":
            assert group.teacher == pytest.approx((0.2, 0.8) if group.question_id == "n1" else (0.8, 0.2))
    reordered = copy.deepcopy(record)
    reordered["examples"].reverse()
    other = module().prepare_bound_families(engine, [reordered], suite, arm="J0", targets_path=targets, target_loader=load_unverified_fixture_targets)
    assert {(g.case_id, g.question_id): g.teacher for g in other[record["id"]].groups} == {
        (g.case_id, g.question_id): g.teacher for g in new.groups}


@pytest.mark.parametrize("mutation", ["question", "state", "case", "family", "teacher"])
def test_binding_rejects_training_request_identity_or_teacher_mismatch(tmp_path, mutation):
    engine, record, suite, targets = binding_fixture(tmp_path)
    if mutation == "question":
        record["examples"][0]["question"] = {"type": "noul", "instructions": "wrong definition"}
    elif mutation == "state":
        record["examples"][0]["state"] = {"changed": True}
    elif mutation == "case":
        record["examples"][0]["case_id"] = "unknown"
    elif mutation == "family":
        value = json.loads(suite.read_text())
        value["cases"][0]["family_id"] = "wrong-family"
        suite.write_text(json.dumps(value))
    else:
        value = json.loads(targets.read_text())
        value["teacher"]["model"] = "Qwen/Qwen3.5-4B"
        targets.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module().prepare_bound_families(engine, [record], suite, arm="J0", targets_path=targets, target_loader=load_unverified_fixture_targets)


def test_common_start_requires_fresh_adam_and_identical_weights_rng(tmp_path):
    torch.manual_seed(42)
    engine = TinyEngine()
    optimizer = _optimizer(engine.model)
    start = module().record_common_start(engine, optimizer, tmp_path / "common-start.json")
    assert start["optimizer_state_entries"] == 0
    torch.rand(1)
    with pytest.raises(ValueError, match="common start"):
        module().record_common_start(engine, optimizer, tmp_path / "common-start.json")


def test_smoke_failure_preserves_nonresumable_partial_and_restores_old_globals(tmp_path):
    from experiments import contrast_factorial_training as old

    engine = TinyEngine(fail_sync_at=3)
    with pytest.raises(RuntimeError, match="synchronization"):
        with module().runner_context("H1"):
            old.run_arm("H1", engine, _schedule(), _prepared(), tmp_path / "partial",
                        fingerprint=new_fingerprint(), stop_after=1)
    failure = json.loads((tmp_path / "partial/failure.json").read_text())
    assert not failure["resumable"] and failure["update_phase"] == "post-step"
    assert old.STUDY == "contrast-factorial-v1"
    assert list((tmp_path / "partial").glob("partial-*/training.pt"))


def test_fingerprint_binds_loss_targets_sources_and_full_training_exposure(tmp_path):
    from experiments import contrast_factorial_training as old

    repo, manifest_path, manifest = _manifest_tree(tmp_path)
    for name in (*old.PRODUCTION_SOURCES, "experiments/calibrated_training.py",
                 "experiments/calibrated_targets.py", "experiments/calibrated_integrity.py", "reports/calibrated-screen-v1/PROTOCOL.md"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source")
    replay = repo / "replay.jsonl"
    replay.write_text("replay")
    targets = repo / "targets.json"
    targets.write_text("teacher")
    fp = module().build_fingerprint(manifest_path, manifest, "Q1", _schedule(), replay,
                                    _prepared(), targets_path=targets, repo_root=repo)
    assert fp["arm"] == "Q1" and fp["study"] == "calibrated-screen-v1"
    assert fp["loss"]["alpha"] == 0.5 and fp["loss"]["brier_lambda"] == 1.0
    assert fp["train_sha256"] == manifest["arms"]["B"]["sha256"]
    assert fp["schedule_sha256"] and fp["replay_position_sha256"] and fp["targets_sha256"]
    source = repo / "experiments/calibrated_training.py"
    source.write_text("changed source")
    changed = module().build_fingerprint(manifest_path, manifest, "Q1", _schedule(), replay,
                                         _prepared(), targets_path=targets, repo_root=repo)
    assert changed != fp


def test_execute_smoke_writes_strict_gate_and_requires_it_for_training(tmp_path):
    fp, schedule, prepared = new_fingerprint(), _schedule(), _prepared()
    result = module().execute_arm(
        "H1", tmp_path / "smoke", fingerprint=fp, schedule=schedule, prepared=prepared,
        base=tmp_path / "base", smoke=True, smoke_selection=[{"family_id": "family/0"}],
        common_start_path=tmp_path / "common.json", engine_loader=lambda path: (
            _load_tiny(path) if (path / "tiny-model.pt").exists() else TinyEngine()),
    )
    module().validate_smoke_gate(result, fp)
    assert result["training"]["completed_updates"] == 1
    assert result["resume_equivalence"]["max_trainable_difference"] <= 1e-7
    assert (tmp_path / "smoke/report.json").exists()
    with pytest.raises(ValueError, match="smoke"):
        module().execute_arm("H1", tmp_path / "full", fingerprint=fp, schedule=schedule,
                             prepared=prepared, base=tmp_path / "base", smoke=False)


def test_default_training_reader_requires_valid_raw_teacher_capture(tmp_path):
    # This synthetic normalized target has an empty raw observation: it must never
    # become trainable merely because the target vector and observation hash parse.
    engine,record,suite,targets=binding_fixture(tmp_path)
    with pytest.raises(ValueError,match='archive'):
        module().prepare_bound_families(engine,[record],suite,arm='J0',targets_path=targets)
