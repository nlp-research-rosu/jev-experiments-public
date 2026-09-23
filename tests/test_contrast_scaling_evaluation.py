import copy

import pytest


def point(stream, step, objective):
    return {
        "status": "complete",
        "stream": stream,
        "step": step,
        "objective": objective,
        "checkpoint": f"/checkpoints/{stream}/{step}",
        "checkpoint_id": f"{stream}-{step}",
    }


def curves():
    main = [
        point("primary", s, v)
        for s, v in [
            (0, 0.9),
            (200, 0.6),
            (400, 0.4),
            (600, 0.5),
            (800, 0.45),
            (1000, 0.55),
            (2000, 0.3),
            (3000, 0.35),
            (4000, 0.25),
            (5000, 0.28),
        ]
    ]
    repeat = [copy.deepcopy(main[0]), copy.deepcopy(main[1])] + [
        point("repeat", s, v) for s, v in [(400, 0.65), (600, 0.5), (800, 0.48), (1000, 0.52)]
    ]
    return main, repeat


def test_selections_respect_each_prefix_and_keep_complete_pass_endpoints():
    from experiments.contrast_scaling_evaluation import choose_checkpoints

    primary, repeat = curves()
    s = choose_checkpoints(primary, repeat)
    assert s["endpoints"]["data_200"]["step"] == 200
    assert s["endpoints"]["data_1000"]["step"] == 1000
    assert s["endpoints"]["data_5000"]["step"] == 5000
    assert s["endpoints"]["repeat_200"]["step"] == 1000
    assert s["selected"]["data_200"]["step"] == 200
    assert s["selected"]["data_1000"]["step"] == 400
    assert s["selected"]["data_5000"]["step"] == 4000
    assert s["selected"]["repeat_200"]["step"] == 800
    assert s["selected"]["repeat_200"]["stream"] == "repeat"


def test_incomplete_or_nonfinite_validation_cannot_unlock_final_testing():
    from experiments.contrast_scaling_evaluation import choose_checkpoints

    primary, repeat = curves()
    with pytest.raises(ValueError):
        choose_checkpoints(primary[:-1], repeat)
    bad = copy.deepcopy(primary)
    bad[3]["status"] = "failed"
    with pytest.raises(ValueError):
        choose_checkpoints(bad, repeat)
    bad = copy.deepcopy(primary)
    bad[3]["objective"] = float("nan")
    with pytest.raises(ValueError):
        choose_checkpoints(bad, repeat)


def test_checkpoint_reuse_groups_roles_by_exact_weight_identity():
    from experiments.contrast_scaling_evaluation import choose_checkpoints, unique_test_checkpoints

    primary, repeat = curves()
    s = choose_checkpoints(primary, repeat)
    jobs = unique_test_checkpoints(s)
    ids = [p["checkpoint_id"] for p in jobs]
    assert len(ids) == len(set(ids))
    p = next(p for p in jobs if p["checkpoint_id"] == "primary-200")
    assert "endpoint/data_200" in p["roles"] and "selected/data_200" in p["roles"]
    assert any("baseline" in p["roles"] for p in jobs)


def test_validation_reuse_preserves_distinct_endpoint_coordinates():
    from experiments.contrast_scaling_evaluation import materialize_curve

    cached = point("primary", 200, 0.4)
    cached["evaluation"] = {"marker": "same predictions"}
    metadata = [
        {"stream": "primary", "step": 200, "checkpoint": "/p/200", "checkpoint_id": cached["checkpoint_id"]},
        {"stream": "primary", "step": 400, "checkpoint": "/p/400", "checkpoint_id": cached["checkpoint_id"]},
        {"stream": "repeat-200", "step": 600, "checkpoint": "/r/600", "checkpoint_id": cached["checkpoint_id"]},
    ]
    curve = materialize_curve(metadata, {cached["checkpoint_id"]: cached})
    assert [(p["stream"], p["step"], p["checkpoint"]) for p in curve] == [
        ("primary", 200, "/p/200"),
        ("primary", 400, "/p/400"),
        ("repeat-200", 600, "/r/600"),
    ]
    assert all(p["objective"] == 0.4 and p["evaluation"] == cached["evaluation"] for p in curve)
    assert cached["step"] == 200
