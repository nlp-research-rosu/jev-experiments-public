import copy
import hashlib
import json
import os
import signal
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from openjev.judgment_training import PreparedBundle, TrainingGroup


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


def _family_record(identity, arm):
    index = int(identity.rsplit("/", 1)[-1])
    categories = (
        "claim_vs_completion", "permission_vs_execution", "unknown_vs_failure",
        "attribution_and_endorsement", "entity_binding", "action_binding", "temporal_scope",
        "reversal_and_current_state", "negation_and_quantifiers", "ordered_rubrics",
    )
    score_k = 2 + (index % (2 if arm in "AC" else 4))
    examples = []
    for case in range(4):
        for qid in ("n1", "n2", "n3"):
            examples.append({
                "case_id": f"{identity}/v{case}", "question_id": qid, "state": {},
                "question": {"type": "noul", "criteria": {"true": "yes", "false": "no"}},
                "target": {"truth": qid != "n2"},
            })
        examples.extend([
            {"case_id": f"{identity}/v{case}", "question_id": "c1", "state": {},
             "question": {"type": "choice", "criteria": {"a": "a", "b": "b"}}, "target": {"choice": "a"}},
            {"case_id": f"{identity}/v{case}", "question_id": "s1", "state": {},
             "question": {"type": "score", "criteria": [f"level-{item}" for item in range(score_k)]},
             "target": {"level_index": 1}},
        ])
    return {"id": identity, "group_id": identity, "examples": examples, "relations": [],
            "provenance": {"arm": arm, "assigned_split": "train", "category": categories[index % 10]}}


def _manifest_tree(tmp_path):
    repo = tmp_path
    ids = [f"latent/{index:03d}" for index in range(400)]
    freeze, source, data_source = repo / "data/freeze.json", repo / "experiments/factory.py", repo / "data/blueprints.json"
    base = repo / "checkpoints/base"
    for path, contents in ((freeze, "freeze\n"), (source, "source\n"), (data_source, "data\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    base.mkdir(parents=True)
    (base / "checkpoint.json").write_text('{"checkpoint_id":"base"}\n')
    arms = {}
    for arm in "ABCD":
        train, suite = repo / f"data/train/{arm}.jsonl", repo / f"data/train/{arm}-suite.json"
        train.parent.mkdir(parents=True, exist_ok=True)
        train.write_text("".join(json.dumps(_family_record(identity, arm)) + "\n" for identity in ids))
        suite.write_text(json.dumps({"ordered_family_ids": ids, "families": [{"id": item} for item in ids]}))
        arms[arm] = {
            "train_file": train.relative_to(repo).as_posix(), "suite_file": suite.relative_to(repo).as_posix(),
            "ordered_family_ids": ids, "sha256": _sha(train), "suite_sha256": _sha(suite),
        }
    manifest = {
        "study": "contrast-factorial-v1", "arms": arms, "ordered_latent_ids": ids,
        "evaluation_freeze_path": freeze.relative_to(repo).as_posix(), "evaluation_freeze_sha256": _sha(freeze),
        "base_checkpoint": base.relative_to(repo).as_posix(), "base_checkpoint_sha256": _tree_sha(base),
        "source_fingerprints": {source.relative_to(repo).as_posix(): _sha(source)},
        "data_fingerprint": {data_source.relative_to(repo).as_posix(): _sha(data_source)},
    }
    manifest_path = repo / "data/manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return repo, manifest_path, manifest


def test_manifest_checks_repo_relative_train_suite_freeze_source_data_and_base_hashes(tmp_path):
    """Missing or changed declared inputs must fail before any model is loaded."""
    from experiments.contrast_factorial_training import validate_manifest

    repo, _, manifest = _manifest_tree(tmp_path)
    loaded = validate_manifest(manifest, repo)
    assert set(loaded) == set("ABCD") and all(len(rows) == 400 for rows in loaded.values())
    for mutate, message in (
        (lambda value: value["arms"]["D"].__setitem__("suite_sha256", "0" * 64), "suite"),
        (lambda value: value.__setitem__("evaluation_freeze_sha256", "0" * 64), "freeze"),
        (lambda value: value.__setitem__("base_checkpoint_sha256", "0" * 64), "base"),
        (lambda value: value.__setitem__("source_fingerprints", {}), "source"),
        (lambda value: value.__setitem__("data_fingerprint", {}), "data"),
    ):
        bad = copy.deepcopy(manifest)
        mutate(bad)
        with pytest.raises(ValueError, match=message):
            validate_manifest(bad, repo)


def _bundle(identity, groups):
    prompts, kinds, indexed = [], [], []
    for primitive, target, width in groups:
        start = len(prompts)
        prompts.extend([[index + 1, index + 2] for index in range(start, start + width)])
        kinds.extend([1] * width)
        criteria = ("a", "b") if primitive == "choice" else tuple(range(width)) if primitive == "score" else None
        indexed.append(TrainingGroup(primitive, tuple(range(start, start + width)), criteria, target))
    return PreparedBundle(identity, "fixture", prompts, kinds, indexed, [])


def _family(identity="family/0"):
    groups = []
    for case in range(4):
        groups.extend([("noul", {"truth": bool((case + index) % 2)}, 1) for index in range(3)])
        groups.extend([("choice", {"choice": "a" if case % 2 else "b"}, 2),
                       ("score", {"level_index": case % 3}, 3)])
    return _bundle(identity, groups)


def _replay():
    return {
        "noul": _bundle("old/noul", [("noul", {"truth": True}, 1)]),
        "choice": _bundle("old/choice", [("choice", {"choice": "a"}, 2)]),
        "score": _bundle("old/score", [("score", {"level_index": 1}, 3)]),
    }


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Module()
        self.adapter.lora_weight = torch.nn.Parameter(torch.linspace(-0.4, 0.7, 96))
        self.binary = torch.nn.Linear(1, 1)
        self.frozen_body = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
        self.checkpoint_id = "tiny/base"

    def score_prompts(self, prompts, kinds, *, unit_batch_size):
        indices = torch.tensor([prompt[0] for prompt in prompts])
        values = self.adapter.lora_weight[indices]
        return values + self.binary(values[:, None]).squeeze(-1)

    def save_checkpoint(self, path, *, metadata, optimizer, training_state):
        path = Path(path)
        path.mkdir(parents=True)
        torch.save(self.state_dict(), path / "tiny-model.pt")
        identity = f"tiny/{metadata['arm']}/{metadata['completed_updates']}"
        (path / "checkpoint.json").write_text(json.dumps({"checkpoint_id": identity, "metadata": metadata}))
        torch.save({"optimizer": optimizer.state_dict(), "state": copy.deepcopy(training_state),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": []}, path / "training.pt")
        return identity


class TinyEngine:
    def __init__(self, model=None, fail_sync_at=None):
        self.model = model or TinyModel()
        self.unit_batch_size = 12
        self.model_id = self.model.checkpoint_id
        self.syncs, self.fail_sync_at = 0, fail_sync_at

    def synchronize(self):
        self.syncs += 1
        if self.syncs == self.fail_sync_at:
            raise RuntimeError("injected post-step synchronization failure")


def _load_tiny(path):
    model = TinyModel()
    model.load_state_dict(torch.load(Path(path) / "tiny-model.pt", weights_only=True))
    return TinyEngine(model)


def _schedule(length=400):
    return [{"step": step, "family_id": "family/0",
             "replay": [{"primitive": kind, "source": "fixture", "id": bundle.id}
                        for kind, bundle in _replay().items()]} for step in range(length)]


def _prepared():
    family, replay = _family(), _replay()
    return {family.id: family, **{bundle.id: bundle for bundle in replay.values()}}


def _fingerprint():
    schedule_hash = hashlib.sha256(json.dumps(_schedule(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "study": "contrast-factorial-v1", "arm": "A", "schedule_sha256": schedule_hash,
        "replay_sha256": "b" * 64, "manifest_sha256": "c" * 64, "base_checkpoint_sha256": "d" * 64,
        "source_fingerprints": {"runner.py": "e" * 64}, "data_fingerprint": {"train.jsonl": "f" * 64},
        "runtime": {"torch": torch.__version__}, "determinism": {"seed": 42},
        "prepared_prompt_fingerprint": "1" * 64,
    }


def _optimizer(model):
    from experiments.judgment_pipeline import optimizer_for
    return optimizer_for(model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))


def test_run_arm_uses_all_twenty_labels_equal_six_means_and_records_exact_exposure(tmp_path):
    """Dropping a family member or reweighting a primitive changes history and exposure."""
    from experiments.contrast_factorial_training import run_arm

    torch.manual_seed(42)
    engine = TinyEngine()
    state = run_arm("A", engine, _schedule(), _prepared(), tmp_path / "arm", fingerprint=_fingerprint(),
                    optimizer=_optimizer(engine.model), stop_after=1)
    row = state["history"][0]
    assert row["component_counts"] == {
        "new/noul": 12, "new/choice": 4, "new/score": 4,
        "old/noul": 1, "old/choice": 1, "old/score": 1,
    }
    assert set(row["component_losses"]) == {
        "new/noul", "new/choice", "new/score", "old/noul", "old/choice", "old/score"
    }
    assert row["exposure"] == {
        "family_id": "family/0", "replay_ids": {"choice": "old/choice", "noul": "old/noul", "score": "old/score"},
        "candidate_units": 38, "unpadded_tokens": 76, "padded_tokens": 76,
    }
    assert row["update_phase"] == "boundary"
    assert state["gradient_check"] == {"adapters": True, "heads": True, "frozen": True, "finite": True}


def test_pause_resume_restores_real_adam_and_rng_for_bit_exact_next_update(tmp_path):
    """A complete-boundary pause must reproduce the uninterrupted second update bit-for-bit."""
    from experiments.contrast_factorial_training import resume_arm, run_arm

    prepared, schedule, fingerprint = _prepared(), _schedule(), _fingerprint()
    torch.manual_seed(42)
    uninterrupted = TinyEngine()
    run_arm("A", uninterrupted, schedule, prepared, tmp_path / "uninterrupted", fingerprint=fingerprint,
            optimizer=_optimizer(uninterrupted.model), stop_after=2)
    expected = copy.deepcopy(uninterrupted.model.state_dict())

    pause = tmp_path / "PAUSE"
    torch.manual_seed(42)
    paused = TinyEngine()
    def request_pause(_arm, completed, _entry):
        if completed == 1:
            pause.write_text("pause\n")

    first = run_arm("A", paused, schedule, prepared, tmp_path / "resumed", fingerprint=fingerprint,
                    optimizer=_optimizer(paused.model), pause_file=pause, progress=request_pause)
    assert first["status"] == "paused" and first["next_position"] == 1
    pause.unlink()
    resumed = resume_arm("A", Path(first["checkpoint"]), schedule, prepared, tmp_path / "resumed",
                         fingerprint=fingerprint, engine_loader=_load_tiny, optimizer_factory=_optimizer, stop_after=2)
    assert resumed["state"]["completed_updates"] == 2
    assert resumed["state"]["restore_verification"]["cpu_rng_states"] == 1
    assert resumed["state"]["restore_verification"]["cuda_rng_states"] == 0
    assert resumed["state"]["restore_verification"]["optimizer_state_entries"] > 0
    assert all(torch.equal(resumed["engine"].model.state_dict()[name], value) for name, value in expected.items())
    history = [json.loads(line) for line in (tmp_path / "resumed/history.jsonl").read_text().splitlines()]
    assert [row["step"] for row in history] == [1, 2]


def test_existing_pause_file_blocks_first_update_and_resume_rejects_wrong_fingerprint(tmp_path):
    """An already-requested pause cannot advance weights, and changed inputs cannot restore state."""
    from experiments.contrast_factorial_training import resume_arm, run_arm

    pause = tmp_path / "PAUSE"
    pause.write_text("pause\n")
    engine = TinyEngine()
    before = copy.deepcopy(engine.model.state_dict())
    state = run_arm("A", engine, _schedule(), _prepared(), tmp_path / "blocked", fingerprint=_fingerprint(),
                    optimizer=_optimizer(engine.model), pause_file=pause)
    assert state["completed_updates"] == 0
    assert all(torch.equal(engine.model.state_dict()[name], value) for name, value in before.items())

    pause.unlink()
    completed = run_arm("A", engine, _schedule(), _prepared(), tmp_path / "checkpointed", fingerprint=_fingerprint(),
                        optimizer=_optimizer(engine.model), stop_after=1)
    wrong = copy.deepcopy(_fingerprint())
    wrong["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        resume_arm("A", Path(completed["checkpoint"]), _schedule(), _prepared(), tmp_path / "checkpointed",
                   fingerprint=wrong, engine_loader=_load_tiny, optimizer_factory=_optimizer, stop_after=2)


def test_post_optimizer_failure_is_preserved_but_never_resumable(tmp_path):
    """A synchronization failure after optimizer.step must not claim the previous boundary."""
    from experiments.contrast_factorial_training import resume_arm, run_arm

    engine, output = TinyEngine(fail_sync_at=3), tmp_path / "failed"
    with pytest.raises(RuntimeError, match="post-step"):
        run_arm("A", engine, _schedule(), _prepared(), output, fingerprint=_fingerprint(),
                optimizer=_optimizer(engine.model))
    failure = json.loads((output / "failure.json").read_text())
    assert failure["update_phase"] == "post-step" and failure["resumable"] is False
    with pytest.raises(ValueError, match="complete boundary"):
        resume_arm("A", Path(failure["checkpoint"]), _schedule(), _prepared(), output,
                   fingerprint=_fingerprint(), engine_loader=_load_tiny, optimizer_factory=_optimizer)


def test_same_schedule_replay_and_four_sequential_fresh_starts_are_independent(tmp_path):
    """Every arm must use the same seeded load path after the prior engine dies."""
    from experiments.contrast_factorial_training import make_schedule, run_all_arms

    ids = [f"family/{index}" for index in range(400)]
    pools = {kind: {"old": [f"old/{kind}/{index}" for index in range(3)]} for kind in ("noul", "choice", "score")}
    assert make_schedule(ids, pools) == make_schedule(ids, pools)
    starts, references, prior_dead = [], [], []
    def factory(_checkpoint):
        if references:
            prior_dead.append(references[-1]() is None)
        engine = TinyEngine()
        torch.rand(17)
        references.append(weakref.ref(engine))
        return engine

    def optimizer_factory(model):
        starts.append(torch.get_rng_state().clone())
        return _optimizer(model)

    prepared = _prepared()
    summaries = run_all_arms(
        engine_factory=factory, base_checkpoint=tmp_path / "base", arms=("A", "B", "C", "D"),
        schedule=_schedule(), prepared_by_arm={arm: prepared for arm in "ABCD"}, output=tmp_path / "four",
        fingerprints={arm: {**_fingerprint(), "arm": arm} for arm in "ABCD"},
        optimizer_factory=optimizer_factory, stop_after=1,
    )
    status = json.loads((tmp_path / "four/status.json").read_text())
    assert set(summaries) == set("ABCD") and prior_dead == [True, True, True]
    assert all(torch.equal(starts[0], start) for start in starts[1:])
    assert len({row["trainable_weights_sha256"] for row in status["arm_starts"].values()}) == 1
    assert len({row["cpu_rng_sha256"] for row in status["arm_starts"].values()}) == 1


def test_full_training_requires_strict_matching_production_smoke_gate():
    """A completed-looking smoke cannot authorize changed code/data/runtime or a loose resume proof."""
    from experiments.contrast_factorial_training import common_fingerprint, validate_smoke_gate

    fingerprints = {arm: {**_fingerprint(), "arm": arm} for arm in "ABCD"}
    common = common_fingerprint(fingerprints["A"])
    gate = {
        "mode": "smoke", "status": "complete", "passed": True, "completed_requested_pass": True,
        "common_fingerprint": common,
        "resume_equivalence": {
            "passed": True, "tolerance": 1e-7, "max_probability_difference": 0.0,
            "max_trainable_difference": 0.0, "frozen_before": "same", "frozen_after": "same",
        },
    }
    validate_smoke_gate(gate, fingerprints)
    changed = copy.deepcopy(fingerprints)
    changed["D"]["runtime"] = {"torch": "different"}
    with pytest.raises(ValueError, match="fingerprint"):
        validate_smoke_gate(gate, changed)
    loose = copy.deepcopy(gate)
    loose["resume_equivalence"]["max_trainable_difference"] = 2e-7
    with pytest.raises(ValueError, match="equivalence"):
        validate_smoke_gate(loose, fingerprints)


def test_smoke_selection_covers_categories_broad_score_k_and_global_compiled_workloads():
    """Smoke selection must not depend on labels or the first-N family ordering."""
    from experiments.contrast_factorial_training import CONTRACT_CATEGORIES, select_smoke_families

    records, prepared = {arm: [] for arm in "ABCD"}, {arm: {} for arm in "ABCD"}
    for index, category in enumerate(CONTRACT_CATEGORIES):
        for score_k in (2, 3, 4, 5):
            identity = f"broad/{index}/{score_k}"
            record = _family_record("latent/000", "B")
            record["id"] = record["group_id"] = identity
            record["provenance"]["category"] = category
            for example in record["examples"]:
                if example["question"]["type"] == "score":
                    example["question"]["criteria"] = [f"level-{item}" for item in range(score_k)]
                    example["target"] = {"level_index": 0}
            records["B"].append(record)
            prepared["B"][identity] = _family(identity)

    longest = _family_record("latent/000", "A")
    longest["id"] = longest["group_id"] = "global/longest"
    longest["provenance"]["category"] = CONTRACT_CATEGORIES[0]
    records["A"].append(longest)
    longest_bundle = _family("global/longest")
    longest_bundle.prompts[0].extend(range(100))
    prepared["A"][longest_bundle.id] = longest_bundle

    selection = select_smoke_families(records, prepared)
    selected = {(row["arm"], row["family_id"]) for row in selection}
    assert {row["category"] for row in selection} == set(CONTRACT_CATEGORIES)
    assert {row["score_k"] for row in selection if row["arm"] in ("B", "D")} == {2, 3, 4, 5}
    assert ("A", "global/longest") in selected
    assert "global_max_prompt_tokens" in next(
        row["reasons"] for row in selection if row["family_id"] == "global/longest"
    )


@pytest.mark.parametrize(("pause_update", "stop_after"), [(1, 2), (400, 400)])
def test_sigterm_pause_stops_coordinator_before_next_arm_even_at_endpoint(tmp_path, pause_update, stop_after):
    """A signal pause is never counted complete and must not fall through to arm B."""
    from experiments.contrast_factorial_training import run_all_arms

    loaded = []
    sent = False
    def factory(_checkpoint):
        loaded.append(len(loaded))
        return TinyEngine()

    def progress(arm, completed, _entry):
        nonlocal sent
        if arm == "A" and completed == pause_update and not sent:
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)

    result = run_all_arms(
        engine_factory=factory, base_checkpoint=tmp_path / "base", arms=("A", "B"),
        schedule=_schedule(), prepared_by_arm={arm: _prepared() for arm in "AB"}, output=tmp_path / "run",
        fingerprints={arm: {**_fingerprint(), "arm": arm} for arm in "AB"}, optimizer_factory=_optimizer,
        stop_after=stop_after, progress=progress,
    )
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert loaded == [0]
    assert result["A"]["status"] == "paused"
    assert result["A"]["completed_updates"] == pause_update
    assert status["status"] == "paused" and status["completed_arms"] == []


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing-adapter", "coverage"),
        ("missing-head", "coverage"),
        ("mixed-counter", "counter"),
        ("zero-counter", "counter"),
        ("bad-shape", "shape"),
        ("bad-dtype", "dtype"),
        ("nonfinite", "finite"),
    ],
)
def test_restore_rejects_corrupt_adam_before_rng_mutation(tmp_path, corruption, message):
    """Every trainable parameter needs exact finite Adam moments at the completed counter."""
    from experiments.contrast_factorial_training import restore_arm, run_arm

    output = tmp_path / "arm"
    engine = TinyEngine()
    complete = run_arm("A", engine, _schedule(), _prepared(), output, fingerprint=_fingerprint(),
                       optimizer=_optimizer(engine.model), stop_after=1)
    checkpoint = Path(complete["checkpoint"])
    payload = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
    groups, states = payload["optimizer"]["param_groups"], payload["optimizer"]["state"]
    adapter_id, head_id = groups[0]["params"][0], groups[1]["params"][0]
    if corruption == "missing-adapter":
        states.pop(adapter_id)
    elif corruption == "missing-head":
        states.pop(head_id)
    elif corruption == "mixed-counter":
        states[adapter_id]["step"] = torch.tensor(0.0)
    elif corruption == "zero-counter":
        for value in states.values():
            value["step"] = torch.tensor(0.0)
    elif corruption == "bad-shape":
        states[adapter_id]["exp_avg"] = states[adapter_id]["exp_avg"][:1]
    elif corruption == "bad-dtype":
        states[adapter_id]["exp_avg"] = states[adapter_id]["exp_avg"].double()
    else:
        states[adapter_id]["exp_avg"].flatten()[0] = float("nan")
    torch.save(payload, checkpoint / "training.pt")

    restored = TinyEngine(copy.deepcopy(engine.model))
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match=message):
        restore_arm(checkpoint, restored, _optimizer(restored.model), arm="A",
                    fingerprint=_fingerprint(), output=output)
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_explicit_history_tail_recovery_archives_bytes_and_reconstructs_prefix(tmp_path):
    """Power-loss rollback is explicit, prefix-verified, preservation-first, and one-shot."""
    from experiments.contrast_factorial_training import recover_history_tail, run_arm, validate_resume_checkpoint

    output = tmp_path / "arm"
    engine = TinyEngine()
    complete = run_arm("A", engine, _schedule(), _prepared(), output, fingerprint=_fingerprint(),
                       optimizer=_optimizer(engine.model), stop_after=1)
    checkpoint = Path(complete["checkpoint"])
    original_prefix = (output / "history.jsonl").read_bytes()
    extra = copy.deepcopy(complete["history"][-1])
    extra["step"] = 2
    with (output / "history.jsonl").open("ab") as stream:
        stream.write((json.dumps(extra, sort_keys=True) + "\n").encode())
        stream.write(b'{"step":3')
    original = (output / "history.jsonl").read_bytes()

    with pytest.raises(ValueError, match="invalid JSON|history on disk differs"):
        validate_resume_checkpoint(checkpoint, arm="A", fingerprint=_fingerprint(), output=output)
    archive = recover_history_tail(checkpoint, arm="A", fingerprint=_fingerprint(), output=output)
    assert (output / "history.jsonl").read_bytes() == original_prefix
    assert (archive / "history.original.jsonl").read_bytes() == original
    assert (archive / "progress.json").is_file()
    assert archive.stat().st_mode & 0o777 == 0o555
    manifest = json.loads((archive / "recovery.json").read_text())
    assert manifest["accepted_prefix_updates"] == 1
    assert manifest["archived_complete_tail"] == [2, 2]
    assert manifest["torn_final_line"] is True
    assert manifest["recompute_updates"] == [2, 3]
    for name, digest in manifest["archived_sha256"].items():
        assert _sha(archive / name) == digest
    with pytest.raises(ValueError, match="no history tail"):
        recover_history_tail(checkpoint, arm="A", fingerprint=_fingerprint(), output=output)


def test_history_recovery_refuses_mismatched_prefix_and_fingerprint(tmp_path):
    """Recovery cannot turn an unrelated or edited history into an accepted prefix."""
    from experiments.contrast_factorial_training import recover_history_tail, run_arm

    output = tmp_path / "arm"
    engine = TinyEngine()
    complete = run_arm("A", engine, _schedule(), _prepared(), output, fingerprint=_fingerprint(),
                       optimizer=_optimizer(engine.model), stop_after=1)
    checkpoint = Path(complete["checkpoint"])
    row = copy.deepcopy(complete["history"][0])
    row["loss"] += 1
    (output / "history.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="prefix"):
        recover_history_tail(checkpoint, arm="A", fingerprint=_fingerprint(), output=output)
    wrong = copy.deepcopy(_fingerprint())
    wrong["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        recover_history_tail(checkpoint, arm="A", fingerprint=wrong, output=output)
    assert not (output / "recovery").exists()


@pytest.mark.parametrize("missing", ["identity", "fingerprint", "both"])
def test_history_recovery_rejects_present_unbound_snapshot_without_mutation(tmp_path, missing):
    """A present snapshot must bind exact study/arm and at least one complete fingerprint representation."""
    from experiments.contrast_factorial_training import recover_history_tail, run_arm

    output = tmp_path / "arm"
    engine = TinyEngine()
    complete = run_arm("A", engine, _schedule(), _prepared(), output, fingerprint=_fingerprint(),
                       optimizer=_optimizer(engine.model), stop_after=1)
    checkpoint = Path(complete["checkpoint"])
    extra = copy.deepcopy(complete["history"][-1])
    extra["step"] = 2
    with (output / "history.jsonl").open("a") as stream:
        stream.write(json.dumps(extra, sort_keys=True) + "\n")
    progress_path = output / "progress.json"
    snapshot = json.loads(progress_path.read_text())
    if missing in ("identity", "both"):
        snapshot.pop("study")
        snapshot.pop("arm")
    if missing in ("fingerprint", "both"):
        snapshot.pop("fingerprint_sha256")
    progress_path.write_text(json.dumps(snapshot))
    history_before, snapshot_before = (output / "history.jsonl").read_bytes(), progress_path.read_bytes()

    with pytest.raises(ValueError, match="identity|fingerprint"):
        recover_history_tail(checkpoint, arm="A", fingerprint=_fingerprint(), output=output)
    assert (output / "history.jsonl").read_bytes() == history_before
    assert progress_path.read_bytes() == snapshot_before
    assert not (output / "recovery").exists()


@pytest.mark.parametrize("snapshot_format", ["full", "both", "absent"])
def test_history_recovery_accepts_supported_or_absent_snapshot_formats(tmp_path, snapshot_format):
    """Legacy full fingerprints, compact hashes, both together, and absent snapshots remain supported."""
    from experiments.contrast_factorial_training import recover_history_tail, run_arm

    output = tmp_path / "arm"
    engine = TinyEngine()
    complete = run_arm("A", engine, _schedule(), _prepared(), output, fingerprint=_fingerprint(),
                       optimizer=_optimizer(engine.model), stop_after=1)
    checkpoint = Path(complete["checkpoint"])
    extra = copy.deepcopy(complete["history"][-1])
    extra["step"] = 2
    with (output / "history.jsonl").open("a") as stream:
        stream.write(json.dumps(extra, sort_keys=True) + "\n")
    progress_path = output / "progress.json"
    if snapshot_format == "absent":
        progress_path.unlink()
    else:
        snapshot = json.loads(progress_path.read_text())
        snapshot["fingerprint"] = _fingerprint()
        if snapshot_format == "full":
            snapshot.pop("fingerprint_sha256")
        progress_path.write_text(json.dumps(snapshot))

    archive = recover_history_tail(checkpoint, arm="A", fingerprint=_fingerprint(), output=output)
    assert archive.is_dir()
    assert len((output / "history.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("damage", ["failure-missing-binding", "contradictory-progress"])
def test_history_recovery_checks_failure_snapshot_and_both_fingerprint_representations(tmp_path, damage):
    """Failure snapshots use the same gate, and one correct representation cannot excuse a contradictory one."""
    from experiments.contrast_factorial_training import recover_history_tail, run_arm

    output = tmp_path / "arm"
    engine = TinyEngine()
    complete = run_arm("A", engine, _schedule(), _prepared(), output, fingerprint=_fingerprint(),
                       optimizer=_optimizer(engine.model), stop_after=1)
    checkpoint = Path(complete["checkpoint"])
    extra = copy.deepcopy(complete["history"][-1])
    extra["step"] = 2
    with (output / "history.jsonl").open("a") as stream:
        stream.write(json.dumps(extra, sort_keys=True) + "\n")
    if damage == "failure-missing-binding":
        (output / "failure.json").write_text(json.dumps({
            "study": "contrast-factorial-v1", "arm": "A", "completed_updates": 1,
        }))
    else:
        progress_path = output / "progress.json"
        snapshot = json.loads(progress_path.read_text())
        snapshot["fingerprint"] = {**_fingerprint(), "manifest_sha256": "0" * 64}
        progress_path.write_text(json.dumps(snapshot))
    history_before = (output / "history.jsonl").read_bytes()

    with pytest.raises(ValueError, match="fingerprint"):
        recover_history_tail(checkpoint, arm="A", fingerprint=_fingerprint(), output=output)
    assert (output / "history.jsonl").read_bytes() == history_before
    assert not (output / "recovery").exists()


def test_zero_update_pause_checkpoint_can_resume_after_request_removed(tmp_path):
    """A pause before update one produces a usable production-shaped step-0000."""
    from experiments.contrast_factorial_training import resume_arm, run_arm

    output, pause = tmp_path / "arm", tmp_path / "PAUSE"
    pause.write_text("pause\n")
    engine = TinyEngine()
    paused = run_arm("A", engine, _schedule(), _prepared(), output, fingerprint=_fingerprint(),
                     optimizer=_optimizer(engine.model), pause_file=pause)
    assert paused["completed_updates"] == 0
    assert Path(paused["checkpoint"]).name == "step-0000"
    pause.unlink()
    resumed = resume_arm("A", paused["checkpoint"], _schedule(), _prepared(), output,
                         fingerprint=_fingerprint(), engine_loader=_load_tiny,
                         optimizer_factory=_optimizer, stop_after=1)
    assert resumed["state"]["status"] == "complete" and resumed["state"]["completed_updates"] == 1


def test_before_arm_pause_can_continue_same_coordinator_output(tmp_path):
    """Removing a pristine coordinator pause must not strand its chosen output directory."""
    from experiments.contrast_factorial_training import run_all_arms

    pause, output = tmp_path / "PAUSE", tmp_path / "run"
    pause.write_text("pause\n")
    kwargs = {
        "engine_factory": lambda _: TinyEngine(), "base_checkpoint": tmp_path / "base", "arms": ("A",),
        "schedule": _schedule(), "prepared_by_arm": {"A": _prepared()}, "output": output,
        "fingerprints": {"A": _fingerprint()}, "optimizer_factory": _optimizer,
        "pause_file": pause, "stop_after": 1,
    }
    assert run_all_arms(**kwargs) == {}
    pause.unlink()
    result = run_all_arms(**kwargs)
    assert result["A"]["status"] == "complete"


def test_initialization_failure_records_original_error_and_restores_sigterm_handler(tmp_path):
    """Initial sync failures still leave a durable failure record and clean handler state."""
    from experiments.contrast_factorial_training import run_arm

    output = tmp_path / "arm"
    previous = signal.getsignal(signal.SIGTERM)
    engine = TinyEngine(fail_sync_at=1)
    with pytest.raises(RuntimeError, match="injected"):
        run_arm("A", engine, _schedule(), _prepared(), output, fingerprint=_fingerprint(),
                optimizer=_optimizer(engine.model), stop_after=1)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["update_phase"] == "initialization"
    assert "injected post-step" in failure["error"]
    assert signal.getsignal(signal.SIGTERM) is previous


def test_progress_is_compact_while_checkpoint_keeps_full_history(tmp_path):
    """Per-phase progress stays fixed-size; durable history and checkpoints keep full state."""
    from experiments.contrast_factorial_training import run_arm

    output = tmp_path / "arm"
    engine = TinyEngine()
    complete = run_arm("A", engine, _schedule(), _prepared(), output, fingerprint=_fingerprint(),
                       optimizer=_optimizer(engine.model), stop_after=2)
    progress = json.loads((output / "progress.json").read_text())
    assert "history" not in progress and "fingerprint" not in progress
    assert progress["history_updates"] == 2
    assert progress["history_sha256"] == complete["history_sha256"]
    assert len((output / "progress.json").read_bytes()) < len((output / "history.jsonl").read_bytes()) * 3
    checkpoint_state = torch.load(Path(complete["checkpoint"]) / "training.pt", weights_only=False)["state"]
    assert len(checkpoint_state["history"]) == 2


def test_missing_gate_and_preexisting_pause_stop_before_cuda_or_engine_setup(tmp_path, monkeypatch):
    """Cheap static lifecycle gates must run before any production model allocation."""
    import experiments.contrast_factorial_training as runner

    repo, manifest_path, manifest = _manifest_tree(tmp_path)
    base = repo / manifest["base_checkpoint"]
    calls = []
    monkeypatch.setattr(runner, "_new_engine", lambda path: calls.append(path))
    args = SimpleNamespace(
        repo_root=repo, data=manifest_path, base_checkpoint=base, output=tmp_path / "missing-gate",
        mode="train", arm="all", resume=None, smoke_gate=None, pause_file=tmp_path / "PAUSE",
        replay=repo / "data/replay.jsonl", smoke_steps=None, recover_history_tail=False,
    )
    with pytest.raises(ValueError, match="smoke-gate"):
        runner.run(args)
    assert calls == []

    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "mode": "smoke", "status": "complete", "passed": True, "completed_requested_pass": True,
        "resume_equivalence": {
            "passed": True, "tolerance": 1e-7, "max_probability_difference": 0.0,
            "max_trainable_difference": 0.0, "frozen_before": "same", "frozen_after": "same",
        },
    }))
    args.smoke_gate = gate
    args.output = tmp_path / "paused"
    args.pause_file.write_text("pause\n")
    result = runner.run(args)
    assert result["status"] == "paused" and result["before_arm"] == "A"
    assert calls == []
    assert (args.output / "pristine-pause.json").is_file()


def test_run_recovery_flag_archives_tail_and_resumes_real_tiny_cli_path(tmp_path, monkeypatch):
    """The CLI-facing run path performs explicit recovery before restoring Adam/RNG and continuing."""
    import experiments.contrast_factorial_training as runner

    output = tmp_path / "arm"
    engine = TinyEngine()
    complete = runner.run_arm(
        "A", engine, _schedule(), _prepared(), output, fingerprint=_fingerprint(),
        optimizer=_optimizer(engine.model), stop_after=1,
    )
    checkpoint = Path(complete["checkpoint"])
    extra = copy.deepcopy(complete["history"][-1])
    extra["step"] = 2
    with (output / "history.jsonl").open("a") as stream:
        stream.write(json.dumps(extra, sort_keys=True) + "\n")

    repo, manifest_path, manifest = _manifest_tree(tmp_path / "repo")
    replay_path = repo / "data/replay.jsonl"
    replay_path.write_text("fixture\n")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({
        "mode": "smoke", "status": "complete", "passed": True, "completed_requested_pass": True,
        "common_fingerprint": runner.common_fingerprint(_fingerprint()),
        "resume_equivalence": {
            "passed": True, "tolerance": 1e-7, "max_probability_difference": 0.0,
            "max_trainable_difference": 0.0, "frozen_before": "same", "frozen_after": "same",
        },
    }))
    replay = _replay()
    pools = {kind: {"fixture": [bundle.id]} for kind, bundle in replay.items()}
    monkeypatch.setattr(runner, "_require_cuda", lambda: None)
    monkeypatch.setattr(runner, "_new_preparation_engine", lambda _: object())
    monkeypatch.setattr(runner, "prepare_families", lambda *_: _prepared())
    monkeypatch.setattr(runner, "prepare_canonical_replay", lambda *_: ({b.id: b for b in replay.values()}, pools))
    monkeypatch.setattr(runner, "make_schedule", lambda *_: _schedule())
    monkeypatch.setattr(runner, "build_fingerprint", lambda *_, **__: _fingerprint())
    monkeypatch.setattr(runner, "_new_engine", _load_tiny)
    args = SimpleNamespace(
        repo_root=repo, data=manifest_path, base_checkpoint=repo / manifest["base_checkpoint"],
        output=output, mode="train", arm="A", resume=checkpoint, smoke_gate=gate_path,
        pause_file=tmp_path / "PAUSE", replay=replay_path, smoke_steps=None, recover_history_tail=True,
    )
    result = runner.run(args)
    assert result["status"] == "complete" and result["completed_updates"] == 400
    archives = list((output / "recovery").iterdir())
    assert len(archives) == 1
    assert json.loads((archives[0] / "recovery.json").read_text())["recompute_updates"] == [2, 2]
