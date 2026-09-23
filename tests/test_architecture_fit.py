"""CPU checks for the bounded, fit-only architecture diagnostic."""

import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from openjev.judgment_training import PreparedBundle, TrainingGroup


class Scores(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.linspace(-.6, .8, 14, dtype=torch.float64))
        self.calls = []

    def score_prompts(self, prompts, kinds, *, unit_batch_size):
        self.calls.append(([p[0] for p in prompts], unit_batch_size))
        return self.values[[p[0] for p in prompts]]


def family():
    groups = [
        TrainingGroup("noul", (0,), None, {"truth": True}),
        TrainingGroup("noul", (1,), None, {"truth": False}),
        TrainingGroup("noul", (2,), None, {"truth": True}),
        TrainingGroup("choice", (3, 4, 5), ("a", "b", "c"), {"choice": "c"}),
        TrainingGroup("choice", (6, 7), ("a", "b"), {"choice": "a"}),
        TrainingGroup("score", (8, 9, 10, 11, 12, 13), tuple(range(6)), {"level_index": 4}),
    ]
    return PreparedBundle("f0", "new", [[i] for i in range(14)], [1] * 3 + [0] * 11, groups, [])


def test_three_equal_means_match_unchunked_gradient_without_replay():
    """Counting Noul examples equally or dividing by six changes the gradient."""
    from experiments.architecture_fit import backward_fit_family

    bounded, reference = Scores(), Scores()
    expected = (
        F.binary_cross_entropy_with_logits(reference.values[:3], torch.tensor([1., 0., 1.], dtype=torch.float64))
        + (F.cross_entropy(reference.values[3:6], torch.tensor(2))
           + F.cross_entropy(reference.values[6:8], torch.tensor(0))) / 2
        + F.cross_entropy(reference.values[8:14], torch.tensor(4))
    ) / 3
    expected.backward()
    result = backward_fit_family(bounded, family(), max_units=8, unit_batch_size=4)
    assert result["loss"] == pytest.approx(expected.item())
    torch.testing.assert_close(bounded.values.grad, reference.values.grad)
    assert result["counts"] == {"noul": 3, "choice": 2, "score": 1}
    assert all(len(indices) <= 8 and batch == 4 for indices, batch in bounded.calls)
    assert sorted(i for indices, _ in bounded.calls for i in indices) == list(range(14))
    for candidate_set in ({3, 4, 5}, {6, 7}, {8, 9, 10, 11, 12, 13}):
        assert sum(candidate_set <= set(indices) for indices, _ in bounded.calls) == 1


@pytest.mark.parametrize("mutation", ["missing_primitive", "soft_target", "partial_group", "oversize"])
def test_invalid_fit_family_rejected_before_any_backward(mutation):
    """Malformed supervision must not leave a partially accumulated gradient."""
    from experiments.architecture_fit import backward_fit_family

    bundle = family()
    if mutation == "missing_primitive":
        bundle.groups.pop()
    elif mutation == "soft_target":
        bundle.groups[0] = TrainingGroup("noul", (0,), None, {"probability_true": .7})
    elif mutation == "partial_group":
        bundle.groups[-1] = TrainingGroup("score", (8, 9), tuple(range(6)), {"level_index": 4})
    model = Scores()
    with pytest.raises(ValueError):
        backward_fit_family(model, bundle, max_units=4 if mutation == "oversize" else 8)
    assert model.values.grad is None
    assert model.calls == []


def test_saved_boundary_reproduces_next_adam_update_rng_and_rejects_changed_fingerprint(tmp_path):
    """Omitting actual weights, Adam moments, RNG or fingerprint makes continuation unsafe."""
    from experiments.architecture_fit import restore_boundary, save_boundary

    random.seed(42)
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)

    def update():
        optimizer.zero_grad(set_to_none=True)
        loss = (model(torch.rand(3, 2)) - random.random()).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()

    update()
    checkpoint = tmp_path / "step-0001"
    state = {"completed_updates": 1, "training_seconds": 1.5, "history": [{"loss": .5}], "update_phase": "boundary"}
    save_boundary(checkpoint, model, optimizer, state, {"data": "fixed"})
    expected_loss = update()
    expected = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="fingerprint"):
        restore_boundary(checkpoint, model, optimizer, {"data": "changed"})
    recovered = restore_boundary(checkpoint, model, optimizer, {"data": "fixed"})
    assert recovered == state
    assert torch.equal(update(), expected_loss)
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])
    with pytest.raises(FileExistsError):
        save_boundary(checkpoint, model, optimizer, state, {"data": "fixed"})


def test_nonboundary_checkpoint_is_rejected(tmp_path):
    from experiments.architecture_fit import save_boundary

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="boundary"):
        save_boundary(tmp_path / "partial", model, optimizer, {"update_phase": "backward"}, {})


def test_fit_caps_reject_extension_and_allow_only_scheduled_success():
    """A cap cannot be silently extended and endpoint accuracy is not the early-stop gate."""
    from experiments.architecture_fit import fit_stop_reason, validate_limits

    for updates, seconds in ((81, 900), (80, 901), (0, 900), (80, -1), (80, float("nan"))):
        with pytest.raises(ValueError):
            validate_limits(updates, seconds)
    assert fit_stop_reason(9, 100, max_updates=80, max_seconds=900, perfect_fit=True) is None
    assert fit_stop_reason(10, 100, max_updates=80, max_seconds=900, perfect_fit=True) == "fit_success"
    assert fit_stop_reason(1, 901, max_updates=80, max_seconds=900, perfect_fit=False) == "time_cap_inconclusive"
    assert fit_stop_reason(80, 100, max_updates=80, max_seconds=900, perfect_fit=False) == "update_cap_inconclusive"


def test_success_gate_requires_complete_individual_and_nonempty_both_correct_pairs():
    from experiments.architecture_fit import perfect_fit

    complete = {"overall": {"correct": 200, "questions": 200}, "primary": {"both_correct": 60, "pairs": 60}}
    assert perfect_fit(complete)
    for field, key, value in (("overall", "correct", 199), ("primary", "both_correct", 59),
                              ("primary", "pairs", 0), ("overall", "questions", 0)):
        broken = copy.deepcopy(complete)
        broken[field][key] = value
        assert not perfect_fit(broken)


class TinyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1)
        self.backbone.requires_grad_(False)
        self.backbone.lora_A = torch.nn.Parameter(torch.tensor(.1, dtype=torch.float64))
        self.compatibility = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.binary = torch.nn.Linear(1, 1, dtype=torch.float64)
        self.lora_settings = {"rank": 8}

    def score_prompts(self, prompts, kinds, *, unit_batch_size):
        feature = torch.tensor([p[0] / 14 for p in prompts], dtype=torch.float64).reshape(-1, 1)
        feature = feature + self.backbone.lora_A
        return torch.where(torch.tensor(kinds).bool(), self.binary(feature).squeeze(-1),
                           self.compatibility(feature).squeeze(-1))


def test_trainability_and_optimizer_contract_reject_body_training_or_wrong_current_lr():
    from experiments.architecture_fit import check_training_contract
    from experiments.judgment_pipeline import optimizer_for

    model = TinyTrainModel()
    optimizer = optimizer_for(model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
    check_training_contract(model, optimizer)
    optimizer.param_groups[0]["lr"] = .1
    with pytest.raises(ValueError, match="learning rate"):
        check_training_contract(model, optimizer)
    optimizer.param_groups[0]["lr"] = 5e-5
    model.backbone.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable"):
        check_training_contract(model, optimizer)


def test_loop_cycles_families_stops_at_cap_and_keeps_real_boundary_state(tmp_path, monkeypatch):
    """This exercises real optimization/checkpointing; only costly model evaluation is substituted."""
    from experiments import architecture_fit as fit

    def metrics(split_names, engine, suites, directory):
        return {split: {"overall": {"correct": 1, "questions": 200},
                        "primary": {"both_correct": 0, "pairs": 60}} for split in split_names}

    monkeypatch.setattr(fit, "evaluate_suites", metrics)
    model = TinyTrainModel()
    engine = SimpleNamespace(model=model, model_id="original-H0", synchronize=lambda: None)
    first, second = family(), family()
    second = PreparedBundle("f1", second.source, second.prompts, second.kinds, second.groups, [])
    result = fit.run_fit(engine, [first, second], {}, tmp_path, {"test": True},
                         max_updates=3, max_training_seconds=900)
    assert result["reason"] == "update_cap_inconclusive"
    assert result["completed_updates"] == 3
    assert [row["family_id"] for row in result["history"]] == ["f0", "f1", "f0"]
    assert result["frozen_body_unchanged"]
    assert engine.model_id.startswith("openjev-architecture-fit-v1/sha256-")
    assert result["endpoint_model_id"] == engine.model_id
    assert set(result["evaluations"]["0"]) == {"fit", "calibration", "development", "retention"}
    assert set(result["evaluations"]["3"]) == {"fit", "calibration", "development", "retention"}
    final = torch.load(tmp_path / "endpoint-0003" / "training.pt", weights_only=False)
    assert final["state"]["completed_updates"] == 3
    assert len(final["optimizer"]["state"]) > 0


def test_time_cap_abandons_partial_gradients_before_optimizer_step(tmp_path, monkeypatch):
    from experiments import architecture_fit as fit

    def metrics(split_names, engine, suites, directory):
        return {split: {"overall": {"correct": 1, "questions": 200},
                        "primary": {"both_correct": 0, "pairs": 60}} for split in split_names}

    def interrupted(model, *args, **kwargs):
        model.binary.weight.sum().backward()
        raise fit.FitTimeLimit("simulated cap between complete candidate batches")

    monkeypatch.setattr(fit, "evaluate_suites", metrics)
    monkeypatch.setattr(fit, "backward_fit_family", interrupted)
    model = TinyTrainModel()
    original = copy.deepcopy(model.state_dict())
    result = fit.run_fit(SimpleNamespace(model=model, synchronize=lambda: None), [family()], {}, tmp_path, {},
                         max_updates=80, max_training_seconds=900)
    assert result["reason"] == "time_cap_inconclusive"
    assert result["completed_updates"] == 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())


def test_independent_evaluation_preserves_completed_response_when_next_case_fails(tmp_path):
    from experiments.architecture_fit import evaluate_suites
    from openjev.judgments import assemble_response, compile_request

    request = {"state": "done", "questions": {"n": {"type": "noul", "instructions": "Done?"}}}

    class Engine:
        count = 0

        def evaluate(self, request, *, details, cached):
            assert details is True and cached is False
            self.count += 1
            if self.count == 2:
                raise RuntimeError("evaluation interrupted")
            return {**assemble_response(compile_request(request), [1.], details=True),
                    "usage": {"input_tokens": 2}}

    suites = {"fit": {"cases": [{"id": "a", "request": request}, {"id": "b", "request": request}]}}
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_suites(("fit",), Engine(), suites, tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "fit-responses.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["case_id"] == "a"
    assert rows[0]["response"]["answers"]["n"]["noul"] == pytest.approx(.7310585786)


def test_output_cannot_overwrite_original_or_any_existing_artifact(tmp_path):
    from experiments.architecture_fit import validate_output

    original = tmp_path / "original"
    original.mkdir()
    for candidate in (original, original / "nested", tmp_path):
        with pytest.raises(ValueError):
            validate_output(candidate, original)
    validate_output(tmp_path / "fresh", original)


def test_fingerprint_detects_data_source_and_prepared_prompt_changes(tmp_path):
    from experiments.architecture_fit import make_fingerprint

    data, source = tmp_path / "data", tmp_path / "checkpoint"
    data.mkdir()
    source.mkdir()
    for name in ("fit.json", "calibration.json", "development.json", "retention.json", "fit-bundles.jsonl", "manifest.json"):
        (data / name).write_text("{}")
    (source / "weights.bin").write_bytes(b"original weights")
    bundle = family()
    baseline = make_fingerprint(data, source, [bundle], max_updates=80, max_training_seconds=900)
    bundle.prompts[0].append(4)
    changed = make_fingerprint(data, source, [bundle], max_updates=80, max_training_seconds=900)
    assert changed["prepared_sha256"] != baseline["prepared_sha256"]
    (data / "fit.json").write_text('{"changed":true}')
    assert make_fingerprint(data, source, [family()], max_updates=80, max_training_seconds=900) != baseline
    (source / "weights.bin").write_bytes(b"other weights")
    assert make_fingerprint(data, source, [family()], max_updates=80, max_training_seconds=900)["checkpoint_files"] != baseline["checkpoint_files"]


def test_cli_help_does_not_load_model_and_rejects_extended_caps(tmp_path):
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "experiments.architecture_fit", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--max-training-seconds" in result.stdout
    result = subprocess.run([sys.executable, "-m", "experiments.architecture_fit", "--max-updates", "81",
                             "--output", str(tmp_path / "never-created")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "between 1 and 80" in result.stderr
    assert not (tmp_path / "never-created").exists()


def test_population_gate_excludes_partial_or_sentinel_fit_sets():
    from experiments.architecture_fit import validate_population

    with pytest.raises(ValueError, match="ten complete"):
        validate_population([{"id": "numeric-sentinel"}], {})


def _imperfect_metrics(split_names, engine, suites, directory):
    return {split: {"overall": {"correct": 1, "questions": 200},
                    "primary": {"both_correct": 0, "pairs": 60}} for split in split_names}


def test_nonfinite_gradient_preserves_failure_boundary_and_body_integrity(tmp_path, monkeypatch):
    from experiments import architecture_fit as fit

    monkeypatch.setattr(fit, "evaluate_suites", _imperfect_metrics)
    model = TinyTrainModel()
    model.binary.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    original = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="non-finite trainable gradient"):
        fit.run_fit(SimpleNamespace(model=model, synchronize=lambda: None), [family()], {}, tmp_path, {})
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["completed_updates"] == 0
    assert failure["frozen_body_unchanged"]
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())
    assert (tmp_path / "failure-0000" / "training.pt").exists()


def test_deadline_rechecked_after_gradient_validation_before_adam(tmp_path, monkeypatch):
    from experiments import architecture_fit as fit

    monkeypatch.setattr(fit, "evaluate_suites", _imperfect_metrics)
    real_clock, real_clip = fit.time.monotonic, torch.nn.utils.clip_grad_norm_
    offset = [0]
    monkeypatch.setattr(fit.time, "monotonic", lambda: real_clock() + offset[0])

    def slow_clip(*args, **kwargs):
        value = real_clip(*args, **kwargs)
        offset[0] = 1000
        return value

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", slow_clip)
    model = TinyTrainModel()
    original = copy.deepcopy(model.state_dict())
    result = fit.run_fit(SimpleNamespace(model=model, synchronize=lambda: None), [family()], {}, tmp_path, {})
    assert result["completed_updates"] == 0
    assert result["reason"] == "time_cap_inconclusive"
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())


@pytest.mark.parametrize("progress", [{"completed_updates": 2, "reason": None},
                                      {"completed_updates": 1, "reason": "time_cap_inconclusive"}])
def test_resume_rejects_stale_boundary_or_already_capped_run(tmp_path, progress):
    from experiments.architecture_fit import restore_boundary, save_boundary

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    save_boundary(tmp_path / "step-0001", model, optimizer,
                  {"completed_updates": 1, "update_phase": "boundary"}, {})
    (tmp_path / "progress.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="stale|capped"):
        restore_boundary(tmp_path / "step-0001", model, optimizer, {})
