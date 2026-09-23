import pytest


def record(
    identity="train/evidence/01",
    group="train/evidence/01",
    *,
    state="Template state",
    instructions="Template question",
    target=True,
    criteria=None,
):
    return {
        "id": identity,
        "group_id": group,
        "examples": [
            {
                "state": state,
                "question": {
                    "type": "noul",
                    "instructions": instructions,
                    "criteria": criteria if criteria is not None else {"true": "yes", "false": "no"},
                },
                "target": {"truth": target},
            }
        ],
        "relations": [],
        "provenance": {"dataset": "paired-language-v1", "assigned_split": "train", "family": group},
    }


def test_pair_alignment_allows_only_state_and_instruction_rendering_differences():
    from experiments.paired_language_training import compare_paired_records

    template = [record()]
    natural = [record(state="Natural narrative", instructions="Natural predicate")]
    result = compare_paired_records(template, natural)
    assert result == {"records": 1, "rendering_differences": ["train/evidence/01/0"]}


@pytest.mark.parametrize(
    "edit",
    [
        lambda item: item.__setitem__("id", "different-id"),
        lambda item: item.__setitem__("group_id", "other-family"),
        lambda item: item["examples"][0].__setitem__("target", {"truth": False}),
        lambda item: item["examples"][0]["question"].__setitem__("criteria", {"true": "affirm", "false": "deny"}),
    ],
)
def test_pair_alignment_rejects_ids_targets_group_and_criteria(edit):
    from experiments.paired_language_training import compare_paired_records

    template, natural = [record()], [record()]
    edit(natural[0])
    with pytest.raises(ValueError):
        compare_paired_records(template, natural)


def test_pair_alignment_rejects_question_field_order_changes():
    from experiments.paired_language_training import compare_paired_records

    template, natural = [record()], [record()]
    q = natural[0]["examples"][0]["question"]
    natural[0]["examples"][0]["question"] = {
        "instructions": q["instructions"],
        "type": q["type"],
        "criteria": q["criteria"],
    }
    with pytest.raises(ValueError, match="field order"):
        compare_paired_records(template, natural)


def test_schedule_is_identical_for_arms_and_has_the_fixed_six_slots():
    from experiments.paired_language_training import make_schedule

    pools = {
        origin: {
            kind: {"source-a": [f"{origin}/{kind}/a"], "source-b": [f"{origin}/{kind}/b"]}
            for kind in ("noul", "choice", "score")
        }
        for origin in ("original", "new")
    }
    first = make_schedule(pools, steps=4, seed=42)
    assert first == make_schedule(pools, steps=4, seed=42)
    assert all(
        [(item["primitive"], item["origin"]) for item in row]
        == [
            ("noul", "original"),
            ("noul", "new"),
            ("choice", "original"),
            ("choice", "new"),
            ("score", "original"),
            ("score", "new"),
        ]
        for row in first
    )


def test_checkpoint_metadata_names_the_study_and_common_initial_checkpoint(tmp_path):
    from experiments.paired_language_training import STUDY, save

    class Model:
        def save_checkpoint(self, path, *, metadata, optimizer, training_state):
            self.path, self.metadata, self.training_state = path, metadata, training_state
            return "new-checkpoint"

    model = Model()
    engine = type("Engine", (), {"model": model})()
    assert save(engine, tmp_path / "checkpoint", "template", 250, "original-v0.2") == "new-checkpoint"
    assert model.metadata == {"study": STUDY, "arm": "template", "step": 250, "initial_checkpoint_id": "original-v0.2"}
    assert engine.model_id == "new-checkpoint"


def test_leakage_rejects_evaluation_case_and_family_names(monkeypatch):
    import experiments.paired_language_training as runner

    monkeypatch.setattr(runner, "validate_suite", lambda suite: None)
    template, natural = [record()], [record()]
    validation = {"cases": [{"id": "validation-case", "family_id": "train/evidence/01"}]}
    with pytest.raises(ValueError, match="leaks"):
        runner.validate_population_boundaries(template, natural, validation, {"cases": []})


def test_oversize_paired_example_is_fatal_in_each_arm(monkeypatch):
    import experiments.paired_language_training as runner

    monkeypatch.setattr(
        runner,
        "prepare_bundle",
        lambda engine, bundle: (_ for _ in ()).throw(ValueError("rendered unit exceeds 1536 tokens; no truncation")),
    )
    for arm in ("template", "natural"):
        with pytest.raises(ValueError, match=rf"{arm}/train/evidence/01/0; no dropping"):
            runner.prepare_paired_example(object(), arm, "train/evidence/01", 0, record()["examples"][0])


def test_full_run_requires_exact_completed_smoke_fingerprint():
    from experiments.paired_language_training import validate_smoke_gate

    fingerprint = {"source": "abc", "data": "def"}
    validate_smoke_gate(
        {
            "mode": "smoke",
            "status": "complete",
            "passed": True,
            "completed_requested_pass": True,
            "fingerprint": fingerprint,
        },
        fingerprint,
    )
    for patch in ({"fingerprint": {"source": "changed"}}, {"completed_requested_pass": False}, {"status": "failed"}):
        gate = {
            "mode": "smoke",
            "status": "complete",
            "passed": True,
            "completed_requested_pass": True,
            "fingerprint": fingerprint,
            **patch,
        }
        with pytest.raises(ValueError):
            validate_smoke_gate(gate, fingerprint)


def test_pair_alignment_rejects_inner_choice_order_changes():
    from experiments.paired_language_training import compare_paired_records

    template, natural = [record()], [record()]
    for r in (template[0], natural[0]):
        r["examples"][0]["question"]["type"] = "choice"
        r["examples"][0]["target"] = {"choice": "a"}
    template[0]["examples"][0]["question"]["criteria"] = {"a": "First meaning", "b": "Second meaning"}
    natural[0]["examples"][0]["question"]["criteria"] = {"b": "Second meaning", "a": "First meaning"}
    with pytest.raises(ValueError, match="candidate order"):
        compare_paired_records(template, natural)


def test_validation_and_test_cannot_share_a_family(monkeypatch):
    import experiments.paired_language_training as runner

    monkeypatch.setattr(runner, "validate_suite", lambda suite: None)
    validation = {"cases": [{"id": "v", "family_id": "shared"}]}
    test = {"cases": [{"id": "t", "family_id": "shared"}]}
    with pytest.raises(ValueError, match="validation/test overlap"):
        runner.validate_population_boundaries([record()], [record()], validation, test)
