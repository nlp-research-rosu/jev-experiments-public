import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from openjev.judgment_training import PreparedBundle, TrainingGroup


def _bundle(identity, groups):
    prompts, kinds, indexed = [], [], []
    for primitive, target, width in groups:
        start = len(prompts)
        prompts.extend([[index] for index in range(start, start + width)])
        kinds.extend([1] * width)
        criteria = ("a", "b") if primitive == "choice" else tuple(range(width)) if primitive == "score" else None
        indexed.append(TrainingGroup(primitive, tuple(range(start, start + width)), criteria, target))
    return PreparedBundle(identity, "new", prompts, kinds, indexed, [])


def _family():
    # Four cases × (three Noul, Choice, Score), using literal labels and candidate widths.
    groups = []
    for case in range(4):
        groups.extend(
            [
                ("noul", {"truth": bool((case + index) % 2)}, 1)
                for index in range(3)
            ]
            + [("choice", {"choice": "a" if case % 2 else "b"}, 2), ("score", {"level_index": case % 3}, 3)]
        )
    return _bundle("family/0", groups)


def _replay():
    return {
        "noul": _bundle("old/noul", [("noul", {"truth": True}, 1)]),
        "choice": _bundle("old/choice", [("choice", {"choice": "a"}, 2)]),
        "score": _bundle("old/score", [("score", {"level_index": 1}, 3)]),
    }


class TinyScores(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.linspace(-.4, .7, 64))
        self.calls = []

    def score_prompts(self, prompts, kinds, *, unit_batch_size):
        self.calls.append((tuple(tuple(p) for p in prompts), unit_batch_size))
        return self.weights[torch.tensor([p[0] for p in prompts])]


def _primitive_loss(logits, group):
    if group.primitive == "noul":
        return F.binary_cross_entropy_with_logits(logits[0], logits.new_tensor(float(group.target["truth"])))
    target = logits.new_zeros(len(group.indices))
    if group.primitive == "choice":
        target[list(group.criteria).index(group.target["choice"])] = 1
    else:
        target[group.target["level_index"]] = 1
    return -(target * logits.log_softmax(-1)).sum()


def _unchunked_reference(model, family, replay):
    totals = {}
    for origin, bundle in (("new", family), *[("old", replay[k]) for k in ("noul", "choice", "score")]):
        logits = model.score_prompts(bundle.prompts, bundle.kinds, unit_batch_size=100)
        for group in bundle.groups:
            totals.setdefault((origin, group.primitive), []).append(_primitive_loss(logits[list(group.indices)], group))
    return sum(torch.stack(value).mean() for value in totals.values()) / 6


def test_family_loss_keeps_candidate_groups_whole_and_matches_unchunked_gradient():
    """A split candidate normalization or count-weighted Noul term changes this gradient."""
    from experiments.contrast_scaling_training import backward_family_loss, family_loss_components

    family, replay = _family(), _replay()
    bounded, reference = TinyScores(), TinyScores()
    reference.load_state_dict(copy.deepcopy(bounded.state_dict()))
    result = family_loss_components(bounded, family, replay, max_units=4, unit_batch_size=2)
    # The runner's bounded path releases every complete candidate microbatch
    # after backward; it must still reproduce the unchunked update exactly.
    bounded.calls.clear()
    bounded_result = backward_family_loss(bounded, family, replay, max_units=4, unit_batch_size=2)
    expected = _unchunked_reference(reference, family, replay)
    expected.backward()

    assert result["counts"] == {"new/noul": 12, "new/choice": 4, "new/score": 4, "old/noul": 1, "old/choice": 1, "old/score": 1}
    assert torch.allclose(result["total"], expected)
    assert torch.allclose(bounded.weights.grad, reference.weights.grad)
    assert bounded_result["counts"] == result["counts"]
    # Calls may hold several groups, but a two/three-option group must never be split across them.
    assert all(len(call[0]) <= 4 for call in bounded.calls)
    assert any(len(call[0]) == 3 for call in bounded.calls)
    assert sum(len(call[0]) for call in bounded.calls) == len(family.prompts) + sum(
        len(bundle.prompts) for bundle in replay.values()
    )


def test_schedule_uses_ordered_prefixes_and_matches_replay_at_equal_positions():
    """Changing replay seeding or cycling a non-prefix family would break the data-scale comparison."""
    from experiments.contrast_scaling_training import make_family_schedule

    families = [f"family/{index}" for index in range(6)]
    pools = {
        kind: {source: [f"old/{kind}/{source}/{index}" for index in range(3)] for source in ("a", "b")}
        for kind in ("noul", "choice", "score")
    }
    primary = make_family_schedule(families, pools, total_steps=6, seed=42)
    repeat = make_family_schedule(families, pools, total_steps=6, repeat_size=2, seed=42)

    assert [row["family_id"] for row in primary] == families
    assert [row["family_id"] for row in repeat] == ["family/0", "family/1"] * 3
    assert [[item["id"] for item in row["replay"]] for row in repeat] == [
        [item["id"] for item in row["replay"]] for row in primary
    ]
    assert all([item["primitive"] for item in row["replay"]] == ["noul", "choice", "score"] for row in primary)
    assert {item["source"] for row in primary for item in row["replay"]} == {"a", "b"}
    with pytest.raises(ValueError, match="ordered family IDs"):
        make_family_schedule(families[:2], pools, total_steps=3)


def test_coverage_requires_all_twenty_members_and_relation_endpoints():
    """A partial family must never become an endpoint checkpoint."""
    from experiments.contrast_scaling_training import validate_family_coverage

    complete = _complete_family_record()
    validate_family_coverage([complete], ["family/0"])
    incomplete = copy.deepcopy(complete)
    incomplete["examples"].pop()
    with pytest.raises(ValueError, match="20 examples"):
        validate_family_coverage([incomplete], ["family/0"])
    broken_relation = copy.deepcopy(complete)
    broken_relation["relations"] = [{"left": 0, "right": 20}]
    with pytest.raises(ValueError, match="relation endpoint"):
        validate_family_coverage([broken_relation], ["family/0"])


def test_resume_restores_tiny_model_optimizer_and_torch_rng_exactly(tmp_path):
    """Dropping optimizer moments or RNG makes the repeat branch a different run."""
    from experiments.contrast_scaling_training import restore_resume_state, save_resume_state

    def update(model, optimizer):
        optimizer.zero_grad()
        scale = torch.rand(())
        loss = ((model.weight - scale) ** 2).sum()
        loss.backward()
        optimizer.step()
        return loss.detach()

    torch.manual_seed(9)
    uninterrupted = torch.nn.Linear(2, 1, bias=False)
    baseline_state = copy.deepcopy(uninterrupted.state_dict())
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=.01)
    update(uninterrupted, uninterrupted_optimizer)
    expected_loss = update(uninterrupted, uninterrupted_optimizer)
    expected_weights = copy.deepcopy(uninterrupted.state_dict())

    torch.manual_seed(9)
    resumed = torch.nn.Linear(2, 1, bias=False)
    resumed.load_state_dict(baseline_state)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.01)
    update(resumed, resumed_optimizer)
    save_resume_state(tmp_path, resumed, resumed_optimizer, {"next_step": 1})
    torch.rand(7)
    restored = torch.nn.Linear(2, 1, bias=False)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.01)
    assert restore_resume_state(tmp_path, restored, restored_optimizer) == {"next_step": 1}
    actual_loss = update(restored, restored_optimizer)

    assert torch.equal(actual_loss, expected_loss)
    assert all(torch.equal(restored.state_dict()[name], value) for name, value in expected_weights.items())


def _complete_family_record():
    examples = []
    for case in range(4):
        for question_id in ("n1", "n2", "n3"):
            examples.append(
                {
                    "case_id": f"case-{case}", "question_id": question_id,
                    "question": {"type": "noul", "criteria": {"true": "yes", "false": "no"}},
                    "target": {"truth": question_id != "n2"},
                }
            )
        examples.extend(
            [
                {"case_id": f"case-{case}", "question_id": "c1",
                 "question": {"type": "choice", "criteria": {"yes": "yes", "no": "no"}},
                 "target": {"choice": "yes"}},
                {"case_id": f"case-{case}", "question_id": "s1",
                 "question": {"type": "score", "criteria": ["low", "high"]}, "target": {"level_index": 1}},
            ]
        )
    return {"id": "family/0", "group_id": "family/0", "examples": examples, "relations": []}


def test_coverage_requires_four_complete_cases_hard_targets_and_valid_relations():
    """Aggregate primitive totals must not disguise malformed case-level supervision."""
    from experiments.contrast_scaling_training import validate_family_coverage

    complete = _complete_family_record()
    validate_family_coverage([complete], ["family/0"])
    cases = copy.deepcopy(complete)
    for index, example in enumerate(cases["examples"]):
        example["case_id"] = f"case-{index}"
    with pytest.raises(ValueError, match="four cases"):
        validate_family_coverage([cases], ["family/0"])
    soft = copy.deepcopy(complete)
    soft["examples"][0]["target"] = {"probability_true": .5}
    with pytest.raises(ValueError, match="hard Noul"):
        validate_family_coverage([soft], ["family/0"])
    for relation in ({"left": "0", "right": 1}, {"left": 0, "right": 0}, {"left": 0, "right": 19, "kind": "invariant"}):
        broken = copy.deepcopy(complete)
        broken["relations"] = [relation]
        with pytest.raises(ValueError, match="relation"):
            validate_family_coverage([broken], ["family/0"])


def test_prefix_balance_requires_every_category_at_each_declared_size():
    """A valid full 5000 population cannot hide category blocks in the prefixes."""
    from experiments.contrast_scaling_training import validate_prefix_category_balance

    categories = [f"category-{index}" for index in range(10)]
    balanced = [categories[index % 10] for index in range(5000)]
    validate_prefix_category_balance(balanced, categories)
    grouped = [category for category in categories for _ in range(500)]
    with pytest.raises(ValueError, match="prefix 200"):
        validate_prefix_category_balance(grouped, categories)


def test_fingerprint_binds_starting_weights_compiler_and_prepared_prompts(tmp_path):
    """A smoke gate must fail if the initial weights or rendered training units change."""
    from experiments.contrast_scaling_training import fingerprint

    data, checkpoint = tmp_path / "data", tmp_path / "checkpoint"
    (data / "train").mkdir(parents=True)
    (data / "train.jsonl").write_text('{"id":"family/0"}\n')
    (data / "train" / "manifest.json").write_text("{}")
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_text(json.dumps({"checkpoint_id": "first"}))
    (checkpoint / "readouts.safetensors").write_bytes(b"weights-a")
    args = SimpleNamespace(data=data, checkpoint=checkpoint, unit_batch_size=4, max_units=8, seed=42)
    first = fingerprint(args, prompt_fingerprint="prompts-a")
    (checkpoint / "readouts.safetensors").write_bytes(b"weights-b")
    second = fingerprint(args, prompt_fingerprint="prompts-a")
    changed_prompts = fingerprint(args, prompt_fingerprint="prompts-b")

    assert first["starting_checkpoint"] != second["starting_checkpoint"]
    assert second["prepared_prompt_fingerprint"] != changed_prompts["prepared_prompt_fingerprint"]
    assert "src/openjev/judgments.py" in second["source_sha256"]
    assert second["precision"] == {
        "tf32": False, "seed": 42, "deterministic_algorithms": True,
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
    }


def test_contract_precision_policy_applies_fixed_seed_and_deterministic_backward(monkeypatch):
    """Matching reload predictions cannot detect nondeterministic GPU gradients."""
    import os

    from experiments.contrast_scaling_training import apply_determinism

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    torch.use_deterministic_algorithms(False)
    with pytest.raises(ValueError, match="seed 42"):
        apply_determinism(7)
    policy = apply_determinism(42)
    assert policy["deterministic_algorithms"] is True
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    first = torch.rand(3)
    apply_determinism(42)
    assert torch.equal(first, torch.rand(3))
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_production_checkpoint_reload_matches_next_real_family_update(tmp_path):
    """A readable training.pt is insufficient; restored production heads must continue identically."""
    from experiments.contrast_scaling_training import production_resume_equivalence, save_checkpoint
    from openjev.judgment_model import JudgmentModel

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(pad_token_id=0)
            self.embedding = torch.nn.Embedding(64, 4)
            with torch.no_grad():
                self.embedding.weight.copy_(torch.arange(256, dtype=torch.float32).reshape(64, 4) / 100)

        def forward(self, input_ids, **_):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    class Causal:
        def __init__(self):
            self.model = Body()
            self.output = torch.nn.Embedding(64, 4)
            with torch.no_grad():
                self.output.weight.copy_(torch.arange(256, dtype=torch.float32).reshape(64, 4) / 80)

        def get_output_embeddings(self):
            return self.output

    def engine_from_checkpoint(path=None):
        if path is None:
            model = JudgmentModel(Causal(), model_id="tiny", revision="test")
        else:
            model, _ = JudgmentModel.load_checkpoint(
                path, base_factory=Causal, device="cpu", trainable=True, kernel_backend="reference"
            )
        return SimpleNamespace(model=model, unit_batch_size=2)

    family, replay = _family(), _replay()
    engine = engine_from_checkpoint()
    from experiments.judgment_pipeline import optimizer_for

    optimizer = optimizer_for(engine.model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(engine, checkpoint, optimizer, step=0, state={"max_units": 4, "next_step": 0}, stream="smoke")
    proof = production_resume_equivalence(
        engine, optimizer, checkpoint, engine_from_checkpoint, family, replay, max_units=4, unit_batch_size=2
    )

    assert proof["passed"] is True
    assert proof["max_logit_difference"] <= 1e-7
    assert proof["max_trainable_difference"] <= 1e-7
    assert proof["frozen_before"] == proof["frozen_after"]
    assert proof["probability_probe"] == {"requires_grad": False, "device": "cpu"}


def test_rejected_rerun_preserves_existing_report_bytes(tmp_path):
    """A failed rerun must not erase the completed run record it refused to replace."""
    from experiments.contrast_scaling_training import run

    output = tmp_path / "run"
    output.mkdir()
    report = output / "report.json"
    report.write_bytes(b'{"status":"complete","completed_updates":{"primary":5000}}\n')
    args = SimpleNamespace(mode="preflight", output=output, data=tmp_path / "missing", checkpoint=tmp_path / "checkpoint",
                           unit_batch_size=4, max_units=8, seed=42, max_seconds=1, smoke_steps=1, smoke_gate=None)
    with pytest.raises(FileExistsError, match="output already exists"):
        run(args)
    assert report.read_bytes() == b'{"status":"complete","completed_updates":{"primary":5000}}\n'


def test_preflight_rejects_any_prepared_group_wider_than_unit_budget():
    """A smoke subset cannot approve a budget that a later replay candidate cannot fit."""
    from experiments.contrast_scaling_training import validate_prepared_group_widths

    oversized = _bundle("old/choice/wide", [("choice", {"choice": "a"}, 5)])
    with pytest.raises(ValueError, match="requires 5 units"):
        validate_prepared_group_widths({oversized.id: oversized}, max_units=4)


def test_post_step_sync_failure_marks_partial_checkpoint_unresumable(tmp_path):
    """Weights changed after optimizer.step cannot claim the preceding update boundary."""
    from experiments.contrast_scaling_training import run_stream

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.binary = torch.nn.Linear(64, 1, bias=False)
            self.adapter = torch.nn.Module()
            self.adapter.lora_weight = torch.nn.Parameter(torch.linspace(-.2, .3, 64))
            self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)

        def score_prompts(self, prompts, kinds, *, unit_batch_size):
            indices = torch.tensor([prompt[0] for prompt in prompts])
            return self.binary.weight[0, indices] + self.adapter.lora_weight[indices]

        def save_checkpoint(self, path, *, metadata, optimizer, training_state):
            path.mkdir(parents=True)
            torch.save({"state": training_state}, path / "training.pt")
            return "tiny-checkpoint"

    class Engine:
        def __init__(self):
            self.model = Model()
            self.unit_batch_size = 2

        def synchronize(self):
            raise RuntimeError("post-step synchronization failed")

    family, replay = _family(), _replay()
    prepared = {family.id: family, **{bundle.id: bundle for bundle in replay.values()}}
    schedule = [{"step": 0, "family_id": family.id,
                 "replay": [{"primitive": kind, "id": bundle.id} for kind, bundle in replay.items()]}]
    engine = Engine()
    before = engine.model.binary.weight.detach().clone()
    with pytest.raises(RuntimeError, match="post-step"):
        run_stream(engine, schedule, prepared, output=tmp_path / "stream", max_seconds=1, max_units=4)
    state = torch.load(tmp_path / "stream" / "partial-0000" / "training.pt", weights_only=False)["state"]

    assert not torch.equal(before, engine.model.binary.weight.detach())
    assert state["completed_updates"] == 0
    assert state["attempted_step"] == 1
    assert state["resumable"] is False
