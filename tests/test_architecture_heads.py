"""Frozen features must bind targets, preserve initial logits and stay immutable."""

import copy
import hashlib
import json
import math

import pytest
import torch


def fixture_cache(tmp_path):
    suite = {"cases": [{
        "id": "case-a", "family_id": "family-a", "category": "category-a", "domain": "category-a",
        "request": {"state": "Three unrelated judgments.", "questions": {
            "pick": {"type": "choice", "criteria": {"red": "first", "blue": "second", "green": "third"}},
            "rank": {"type": "score", "criteria": ["low", "high"]},
            "yes": {"type": "noul"},
        }}, "expected": {"pick": "blue", "rank": 0, "yes": True},
        "rationale": {"pick": "given", "rank": "given", "yes": "given"},
    }], "relations": []}
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    request_raw = json.dumps(suite["cases"][0]["request"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    request_sha = hashlib.sha256(request_raw.encode()).hexdigest()
    groups = [
        {"case_id": "case-a", "question_id": "pick", "primitive": "choice", "labels": ["red", "blue", "green"],
         "target_index": 1, "indices": [0, 1, 2], "request_sha256": request_sha},
        {"case_id": "case-a", "question_id": "rank", "primitive": "score", "labels": ["0", "1"],
         "target_index": 0, "indices": [3, 4], "request_sha256": request_sha},
        {"case_id": "case-a", "question_id": "yes", "primitive": "noul", "labels": ["false", "true"],
         "target_index": 1, "indices": [5], "request_sha256": request_sha},
    ]
    hidden = torch.zeros(6, 2048)
    hidden[:, 0] = torch.tensor([1., 2., 3., 4., 5., 6.])
    compatibility = torch.zeros(1, 2048)
    compatibility[0, 0] = 2.
    binary = torch.zeros(1, 2048)
    binary[0, 0] = -1.
    cache = {"version": 1, "checkpoint": "fixture/H0/step-0400", "checkpoint_id": "fixture-H0-sha",
             "source_suite_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "hidden": hidden,
             "groups": groups, "heads": {"compatibility.weight": compatibility, "binary.weight": binary,
                                            "binary.bias": torch.tensor([.5])},
             "sourcepins": {"body": "fixture-frozen-body", "renderer": "fixture-renderer"},
             "original_logits": torch.tensor([2., 4., 6., 8., 10., -5.5])}
    cache["cpu_reference_logits"] = cache["original_logits"].clone()
    return path, suite, cache


def test_feature_cache_validates_all_bindings_and_returns_detached_cpu_features(tmp_path):
    from experiments.architecture_heads import validate_cache

    path, _, cache = fixture_cache(tmp_path)
    cache["hidden"].requires_grad_(True)
    checked = validate_cache(cache, path)
    assert checked["hidden"].device.type == "cpu"
    assert not checked["hidden"].requires_grad
    assert checked["hidden"].shape == (6, 2048)


@pytest.mark.parametrize("damage", ["suite", "request", "duplicate", "missing", "labels", "index", "target",
                                    "original_logits", "nonfinite", "width"])
def test_feature_cache_rejects_misaligned_or_unreproduced_features(tmp_path, damage):
    from experiments.architecture_heads import validate_cache

    path, _, cache = fixture_cache(tmp_path)
    if damage == "suite":
        cache["source_suite_sha256"] = "bad"
    elif damage == "request":
        cache["groups"][0]["request_sha256"] = "bad"
    elif damage == "duplicate":
        cache["groups"][1] = copy.deepcopy(cache["groups"][0])
    elif damage == "missing":
        cache["groups"].pop()
    elif damage == "labels":
        cache["groups"][0]["labels"].reverse()
    elif damage == "index":
        cache["groups"][0]["indices"] = [1, 0, 2]
    elif damage == "target":
        cache["groups"][0]["target_index"] = 0
    elif damage == "original_logits":
        cache["original_logits"][0] += .01
    elif damage == "nonfinite":
        cache["hidden"][0, 0] = float("nan")
    else:
        cache["hidden"] = cache["hidden"][:, :100]
    with pytest.raises(ValueError):
        validate_cache(cache, path)


@pytest.mark.parametrize("variant", ["frozen-original", "linear-refit", "split-choice-score", "residual-mlp128"])
def test_every_variant_starts_at_original_dynamic_candidate_logits(tmp_path, variant):
    from experiments.architecture_heads import HeadProbe, unit_primitives

    _, _, cache = fixture_cache(tmp_path)
    model = HeadProbe(cache["heads"], variant)
    actual = model(cache["hidden"], unit_primitives(cache))
    torch.testing.assert_close(actual, torch.tensor([2., 4., 6., 8., 10., -5.5]), rtol=0, atol=1e-6)
    assert actual.shape == (6,)
    if variant == "frozen-original":
        assert not any(p.requires_grad for p in model.parameters())


def test_primitive_balancing_gives_each_kind_equal_weight_despite_group_count():
    from experiments.architecture_heads import balanced_loss

    # Two binary groups with loss log(2), one choice loss log(4).
    groups = [{"primitive": "noul", "indices": [0], "target_index": 1, "labels": ["false", "true"]},
              {"primitive": "noul", "indices": [1], "target_index": 0, "labels": ["false", "true"]},
              {"primitive": "choice", "indices": [2, 3, 4, 5], "target_index": 2, "labels": list("abcd")}]
    logits = torch.zeros(6, requires_grad=True)
    loss = balanced_loss(logits, groups)
    assert loss.item() == pytest.approx((math.log(2) + math.log(4)) / 2)
    loss.backward()
    assert logits.grad[0].item() == pytest.approx(-.125)
    assert logits.grad[4].item() == pytest.approx(-.375)


@pytest.mark.parametrize("variant", ["linear-refit", "split-choice-score", "residual-mlp128"])
def test_cpu_fit_improves_fit_loss_and_leaves_features_and_original_heads_untouched(tmp_path, variant):
    from experiments.architecture_heads import HeadProbe, train_probe

    _, _, cache = fixture_cache(tmp_path)
    cache["hidden"].requires_grad_(True)
    before = copy.deepcopy(cache)
    model = HeadProbe(cache["heads"], variant)
    history = train_probe(model, cache, updates=20, max_seconds=10.)
    assert history["updates_completed"] == 20
    assert history["endpoint_policy"] == "fixed-update-or-walltime-cap"
    assert history["history"][-1]["loss"] < history["history"][0]["loss"]
    assert cache["hidden"].grad is None
    torch.testing.assert_close(cache["hidden"], before["hidden"], rtol=0, atol=0)
    for key in cache["heads"]:
        torch.testing.assert_close(cache["heads"][key], before["heads"][key], rtol=0, atol=0)


def test_temperature_fit_uses_only_passed_calibration_and_balances_primitives():
    from experiments.architecture_heads import calibrate

    groups = []
    values = []
    # Noul 3/4 correct; Choice 1/2 correct, duplicated to vary population size.
    for kind, labels, targets, repeats in [("noul", ["false", "true"], [1, 1, 1, 0], 1),
                                           ("choice", ["a", "b"], [1, 0], 7)]:
        for target in targets * repeats:
            indices = [len(values)] if kind == "noul" else [len(values), len(values) + 1]
            values.extend([math.log(4)] if kind == "noul" else [0., math.log(4)])
            groups.append({"primitive": kind, "indices": indices, "labels": labels, "target_index": target})
    fitted = calibrate(torch.tensor(values), groups)
    # Equal primitive mean is 5/8 correct, optimum inverse-link log(5/3).
    assert fitted["global"]["temperature"] == pytest.approx(math.log(4) / math.log(5 / 3), rel=1e-6)
    assert fitted["per_primitive"]["noul"]["temperature"] == pytest.approx(math.log(4) / math.log(3), rel=1e-6)
    assert fitted["per_primitive"]["choice"]["temperature"] == pytest.approx(100.)


def test_responses_preserve_dynamic_label_and_score_space_under_temperature(tmp_path):
    from experiments.architecture_heads import responses_for

    _, suite, cache = fixture_cache(tmp_path)
    responses = responses_for(suite, cache["original_logits"], temperatures={"choice": 2., "score": 2., "noul": 2.})
    answers = responses["case-a"]["answers"]
    assert list(answers["pick"]["probabilities"]) == ["red", "blue", "green"]
    assert answers["pick"]["choice"] == "green"
    assert answers["rank"]["score"] == pytest.approx(1 / (1 + math.exp(-1)))
    assert answers["yes"]["noul"] == pytest.approx(1 / (1 + math.exp(2.75)))


def test_split_caches_cannot_mix_checkpoint_or_readout_sources(tmp_path):
    from experiments.architecture_heads import check_common_source

    _, _, cache = fixture_cache(tmp_path)
    other = copy.deepcopy(cache)
    check_common_source({"fit": cache, "calibration": other})
    for field in ("checkpoint", "checkpoint_id", "sourcepins", "heads"):
        changed = copy.deepcopy(other)
        if field == "heads":
            changed[field]["binary.bias"] += .1
        else:
            changed[field] = "wrong source"
        with pytest.raises(ValueError):
            check_common_source({"fit": cache, "calibration": changed})


def fixture_splits(tmp_path):
    from experiments.architecture_heads import SPLITS

    data, features = tmp_path / "data", tmp_path / "features"
    data.mkdir()
    features.mkdir()
    path, suite, cache = fixture_cache(tmp_path)
    for split in SPLITS:
        chosen, chosen_cache = copy.deepcopy(suite), copy.deepcopy(cache)
        chosen["cases"][0].update(id=f"case-{split}", family_id=f"family-{split}")
        suite_path = data / f"{split}.json"
        suite_path.write_text(json.dumps(chosen))
        chosen_cache["source_suite_sha256"] = hashlib.sha256(suite_path.read_bytes()).hexdigest()
        for group in chosen_cache["groups"]:
            group["case_id"] = f"case-{split}"
        torch.save(chosen_cache, features / f"{split}.pt")
    return data, features


def test_run_saves_reconstructable_states_logits_responses_and_coverage_without_overwriting(tmp_path):
    from experiments.architecture_heads import HeadProbe, run

    data, features = fixture_splits(tmp_path)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in features.iterdir()}
    output = tmp_path / "output"
    report = run(data, features, output, updates=2, max_seconds=10.)
    assert report["input_files_unchanged"] is True
    for variant, result in report["variants"].items():
        directory = output / variant
        saved = torch.load(directory / "head-state.pt", weights_only=True)
        raw = torch.load(directory / "raw-logits.pt", weights_only=True)
        cache = torch.load(features / "fit.pt", weights_only=True)
        model = HeadProbe(cache["heads"], variant)
        model.load_state_dict(saved["state_dict"], strict=True)
        torch.testing.assert_close(model(cache["hidden"], torch.tensor([0, 0, 0, 1, 1, 2])), raw["logits"]["fit"])
        assert raw["groups"]["calibration"][0]["case_id"] == "case-calibration"
        assert result["training"]["updates_completed"] == (0 if variant == "frozen-original" else 2)
        for split in ("fit", "development", "retention"):
            for mode in ("raw", "GT", "PT"):
                metric = result["splits"][split][mode]
                assert metric["overall"]["questions"] == 3
                assert metric["overall"]["accuracy"] == 0.
                assert metric["overall"]["confidence"]["0.95"]["covered"] == (1 if mode == "raw" else 0)
                assert "category_macro" in metric["primary"]
                assert (directory / f"{split}-{mode}-responses.json").is_file()
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes} == hashes
    previous = (output / "results.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(data, features, output, updates=2, max_seconds=10.)
    assert (output / "results.json").read_bytes() == previous


def test_changing_development_and_retention_targets_cannot_change_trained_heads_or_calibration(tmp_path):
    from experiments.architecture_heads import run

    data, features = fixture_splits(tmp_path)
    first = run(data, features, tmp_path / "first", updates=2, max_seconds=10.)
    for split in ("development", "retention"):
        path = data / f"{split}.json"
        suite = json.loads(path.read_text())
        suite["cases"][0]["expected"]["pick"] = "green"
        path.write_text(json.dumps(suite))
        cache_path = features / f"{split}.pt"
        cache = torch.load(cache_path, weights_only=True)
        cache["source_suite_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        cache["groups"][0]["target_index"] = 2
        torch.save(cache, cache_path)
    second = run(data, features, tmp_path / "second", updates=2, max_seconds=10.)
    for variant in first["variants"]:
        a = torch.load(tmp_path / "first" / variant / "head-state.pt", weights_only=True)
        b = torch.load(tmp_path / "second" / variant / "head-state.pt", weights_only=True)
        assert a["calibration"] == b["calibration"]
        for key in a["state_dict"]:
            torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], atol=0, rtol=0)
        assert first["variants"][variant]["splits"]["development"]["raw"]["overall"]["correct"] == 0
        assert second["variants"][variant]["splits"]["development"]["raw"]["overall"]["correct"] == 1


def test_training_walltime_cap_keeps_initial_state_and_records_zero_updates(tmp_path):
    from experiments.architecture_heads import HeadProbe, train_probe

    _, _, cache = fixture_cache(tmp_path)
    model = HeadProbe(cache["heads"], "linear-refit")
    original = copy.deepcopy(model.state_dict())
    history = train_probe(model, cache, max_seconds=1e-12)
    assert history["updates_completed"] == 0
    assert history["stop_reason"] == "walltime-cap"
    for key in original:
        torch.testing.assert_close(original[key], model.state_dict()[key], atol=0, rtol=0)


def test_cross_hardware_rounding_passes_only_separate_gpu_gate_and_keeps_originals(tmp_path):
    from experiments.architecture_heads import head_reproduction, validate_cache

    path, _, cache = fixture_cache(tmp_path)
    cache["original_logits"] += 6e-6
    original = cache["original_logits"].clone()
    checked = validate_cache(cache, path)
    evidence = head_reproduction(checked)
    assert evidence["cpu_vs_gpu"]["max_absolute_logit_error"] > 1e-6
    assert evidence["cpu_vs_gpu"]["max_absolute_probability_error"] <= 1e-6
    assert evidence["cpu_vs_gpu"]["decision_changes"] == 0
    assert evidence["cpu_reconstruction"]["max_absolute_logit_error"] == 0.
    assert evidence["fp64_reference"]["cpu_max_absolute_error"] == 0.
    assert evidence["fp64_reference"]["gpu_max_absolute_error"] > 1e-6
    torch.testing.assert_close(checked["original_logits"], original, rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["missing_cpu_reference", "wrong_cpu_reference", "changed_head", "gpu_logits",
                                    "probability_shift", "argmax_flip", "tie_change"])
def test_numerical_repair_rejects_changes_even_when_scalar_tolerance_would_pass(tmp_path, damage):
    from experiments.architecture_heads import validate_cache

    path, _, cache = fixture_cache(tmp_path)
    if damage == "missing_cpu_reference":
        del cache["cpu_reference_logits"]
    elif damage == "wrong_cpu_reference":
        cache["cpu_reference_logits"][0] += 3e-6
    elif damage == "changed_head":
        cache["heads"]["compatibility.weight"][0, 0] += .01
    elif damage == "gpu_logits":
        cache["original_logits"][0] += .001
    elif damage == "probability_shift":
        cache["hidden"][5, 0] = -.5  # CPU binary logit 1.0, with a stable positive decision.
        cache["cpu_reference_logits"][5] = 1.
        cache["original_logits"][5] = 1.00001
    else:
        cache["heads"]["binary.weight"].zero_()
        cache["heads"]["binary.bias"][0] = -1e-8 if damage == "argmax_flip" else 0.
        cache["cpu_reference_logits"][5] = -1e-8 if damage == "argmax_flip" else 0.
        cache["original_logits"][5] = 1e-8
    with pytest.raises(ValueError):
        validate_cache(cache, path)


def test_rounding_cache_run_initializes_variants_against_cpu_not_gpu_reference(tmp_path):
    from experiments.architecture_heads import run

    data, features = fixture_splits(tmp_path)
    for cache_path in features.iterdir():
        cache = torch.load(cache_path, weights_only=True)
        cache["original_logits"] += 6e-6
        torch.save(cache, cache_path)
    report = run(data, features, tmp_path / "output", updates=1, max_seconds=10.)
    assert report["head_reproduction"]["fit"]["cpu_vs_gpu"]["max_absolute_logit_error"] > 1e-6
    for variant in report["variants"].values():
        assert variant["initial_reference"] == "cpu_reference_logits"
        assert all(error == 0. for error in variant["initial_max_absolute_logit_error"].values())
