import copy

import pytest


def test_matched_schedules_share_replay_slots_and_keep_fixed_update_budget():
    from experiments.contrast_pilot import build_schedule

    pool = {"a": ["a1", "a2"], "b": ["b1", "b2"]}
    control = build_schedule(["c1", "c2", "c3"], pool, arm="control", steps=9, seed=42)
    contrast = build_schedule(["c1", "c2", "c3"], pool, arm="contrast", steps=9, seed=42)
    assert len(control) == len(contrast) == 9
    assert all(len(x) == len(y) == 4 for x, y in zip(control, contrast, strict=True))
    for x, y in zip(control, contrast, strict=True):
        assert x[:2] == y[:2]
        assert all(item["kind"] == "replay" for item in x)
        assert [item["kind"] for item in y] == ["replay", "replay", "contrast", "contrast"]
    assert {r["id"] for step in contrast for r in step[2:]} == {"c1", "c2", "c3"}
    assert control == build_schedule(["c1", "c2", "c3"], pool, arm="control", steps=9, seed=42)


def test_replay_retains_only_canonical_train_examples_and_rejects_other_partitions():
    from experiments.contrast_pilot import canonical_replay

    record = {
        "id": "source/a", "group_id": "one", "provenance": {"assigned_split": "train", "dataset": "source"},
        "examples": [{"state": "one", "question": {}, "target": {"truth": True}},
                     {"state": "two", "question": {}, "target": {"truth": False}}],
        "relations": [{"kind": "complement", "left": 0, "right": 1}],
    }
    original = copy.deepcopy(record)
    selected = canonical_replay(record)
    assert selected["examples"] == [record["examples"][0]] and selected["relations"] == []
    assert record == original
    for split in ["test", "validation", "calibration", None]:
        record["provenance"]["assigned_split"] = split
        with pytest.raises(ValueError):
            canonical_replay(record)


def test_schedule_rejects_invalid_or_overlapping_sources():
    from experiments.contrast_pilot import build_schedule

    for pool, arm, steps in [({"a": []}, "control", 1), ({"a": ["c1"]}, "contrast", 1),
                              ({"a": ["a1"]}, "unknown", 1), ({"a": ["a1"]}, "contrast", 0)]:
        with pytest.raises(ValueError):
            build_schedule(["c1"], pool, arm=arm, steps=steps, seed=42)


def test_scaling_gate_requires_semantic_gain_and_broad_retention():
    from experiments.contrast_pilot import scaling_gate

    def evaluation(accuracy, balanced, paired, broad):
        return {"test": {"summary": {
            "overall": {"accuracy": accuracy},
            "by_question": {str(i): {"balanced_binary_accuracy": balanced} for i in range(6)},
            "relations": {k: {"both_correct_rate": paired} for k in ("flip", "question_contrast", "invariant")},
        }}, "broad": {"canonical": {"accuracy": broad}}}

    evaluations = {"starting": evaluation(.60, .55, .30, .87), "control": evaluation(.65, .56, .35, .87),
                   "contrast": evaluation(.80, .70, .60, .86)}
    assert scaling_gate(evaluations)["passed"] is True
    evaluations["contrast"]["broad"]["canonical"]["accuracy"] = .84
    result = scaling_gate(evaluations)
    assert result["passed"] is False and result["checks"]["broad_retention_within_2pp"] is False


def test_evaluation_timeout_interrupts_and_restores_signal_handler():
    import signal
    import time

    from experiments.contrast_pilot import evaluation_deadline

    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError):
        with evaluation_deadline(.02):
            time.sleep(.1)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_run_cannot_masquerade_as_a_smoke_gate():
    from experiments.contrast_pilot import validate_smoke

    fingerprint = {"sources": {"a": "hash"}}
    validate_smoke({"mode": "smoke", "status": "complete", "passed": True, "fingerprint": fingerprint}, fingerprint)
    for patch in [{"mode": "run"}, {"status": "failed"}, {"passed": False}, {"fingerprint": {}}]:
        with pytest.raises(ValueError):
            validate_smoke({"mode": "smoke", "status": "complete", "passed": True, "fingerprint": fingerprint, **patch}, fingerprint)


def test_broad_evaluation_preserves_finished_logits_when_later_scoring_fails(tmp_path):
    import json
    from types import SimpleNamespace

    import torch

    from experiments.contrast_pilot import evaluate_broad
    from openjev.judgment_training import PreparedBundle, TrainingGroup

    class Scorer:
        calls = 0

        def eval(self):
            pass

        def score_prompts(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("interrupted scoring")
            return torch.tensor([2.0])

    prepared = [PreparedBundle(name, "test", [[1]], [1],
                              [TrainingGroup("noul", (0,), {}, {"truth": True})], []) for name in ("first", "second")]
    path = tmp_path / "raw.jsonl"
    with pytest.raises(RuntimeError):
        evaluate_broad(SimpleNamespace(model=Scorer(), unit_batch_size=4), prepared, path)
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"id": "first", "logits": [2.0]}]
