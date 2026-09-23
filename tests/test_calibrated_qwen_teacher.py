"""CPU checks for semantic mapping, immutable capture, and teacher isolation."""

import hashlib
import importlib
import json
import math
from pathlib import Path

import pytest


def collector():
    return importlib.import_module("experiments.calibrated_qwen_teacher")


class CharacterTokenizer:
    chat_template = "fixture: no-thinking chat"
    all_special_ids = [0]

    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def encode(self, value, *, add_special_tokens=False):
        return [ord(c) for c in value]

    def decode(self, values, **kwargs):
        return "".join(chr(i) for i in values)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return "\n".join(item["content"] for item in messages) + "\nASSISTANT:\n"


def sample_case():
    return {
        "id": "case/one",
        "expected": {"decision": "GOLD_SENTINEL"},
        "rationale": "RATIONALE_SENTINEL",
        "request": {
            "state": {"evidence": ["literal evidence"], "rules": "explicit rules"},
            "questions": {"decision": {
                "type": "choice", "instructions": "Choose the supported outcome.",
                "criteria": {"zeta": "Third exact definition", "alpha": "First exact definition",
                             "middle": {"nested": ["Second exact definition"]}},
            }},
        },
    }


def test_reverse_probabilities_are_remapped_before_arithmetic_average():
    module = collector()
    result = module.average_orders(
        ["alpha", "middle", "zeta"],
        [math.log(2), 0.0, 0.0],
        [math.log(6), math.log(3), 0.0],
    )
    assert result == pytest.approx({"alpha": 0.3, "middle": 0.275, "zeta": 0.425})


def test_prompt_uses_complete_question_and_exact_state_without_gold():
    module = collector()
    case = sample_case()
    canonical = module.prepare_question(CharacterTokenizer(), case["request"], "decision")
    assert canonical[0]["labels"] == ["alpha", "middle", "zeta"]
    assert canonical[1]["labels"] == ["zeta", "middle", "alpha"]
    for prepared in canonical:
        payload = json.loads(prepared["messages"][1]["content"])
        assert payload["state"] == case["request"]["state"]
        assert payload["question"] == case["request"]["questions"]["decision"]
        assert [entry["label"] for entry in payload["answers"]] == prepared["labels"]
        assert [entry["code"] for entry in payload["answers"]] == ["A", "B", "C"]
        assert "GOLD_SENTINEL" not in prepared["prompt"]
        assert "RATIONALE_SENTINEL" not in prepared["prompt"]
        assert prepared["code_token_ids"] == [65, 66, 67]
    changed_gold = {**case, "expected": {"decision": "alpha"}}
    assert module.prepare_question(CharacterTokenizer(), changed_gold["request"], "decision") == canonical


@pytest.mark.parametrize("criteria", [None, {"true": "Custom positive", "false": "Custom negative"},
                                       {"false": {"rule": ["Custom negative only"]}}])
def test_noul_has_explicit_boolean_meanings_and_retains_optional_criteria(criteria):
    request = {"state": "evidence", "questions": {"n": {"type": "noul", "instructions": "Is it supported?"}}}
    if criteria is not None:
        request["questions"]["n"]["criteria"] = criteria
    prepared = collector().prepare_question(CharacterTokenizer(), request, "n")[0]
    payload = json.loads(prepared["messages"][1]["content"])
    assert payload["question"] == request["questions"]["n"]
    answers = {entry["label"]: entry for entry in payload["answers"]}
    assert "no" in answers["false"]["meaning"].lower()
    assert "yes" in answers["true"]["meaning"].lower()
    assert answers["true"]["criterion"] == (criteria or {}).get("true")
    assert answers["false"]["criterion"] == (criteria or {}).get("false")


def test_score_uses_ordinal_labels_and_all_exact_definitions():
    criteria = ["Never", {"when": ["one", "two"]}, "Always"]
    request = {"state": {}, "questions": {"score": {"type": "score", "criteria": criteria}}}
    prepared = collector().prepare_question(CharacterTokenizer(), request, "score")[0]
    answers = json.loads(prepared["messages"][1]["content"])["answers"]
    assert [item["label"] for item in answers] == ["0", "1", "2"]
    assert [item["criterion"] for item in answers] == criteria


@pytest.mark.parametrize("mode", ["multiple", "collision", "boundary", "special"])
def test_ambiguous_or_non_single_token_codes_are_rejected(mode):
    class InvalidTokenizer(CharacterTokenizer):
        all_special_ids = [65] if mode == "special" else [0]

        def encode(self, value, *, add_special_tokens=False):
            if mode == "multiple" and value == "A":
                return [65, 65]
            if mode == "collision" and value == "B":
                return [65]
            if mode == "boundary" and value.endswith("\nA"):
                return [1]
            return super().encode(value, add_special_tokens=add_special_tokens)

    with pytest.raises(ValueError, match="token|code"):
        collector().prepare_question(InvalidTokenizer(), sample_case()["request"], "decision")


def test_overlong_prompt_is_rejected_without_truncation():
    with pytest.raises(ValueError, match="truncat|exceeds"):
        collector().prepare_question(CharacterTokenizer(), sample_case()["request"], "decision", max_input_tokens=50)


def setup_collection(tmp_path):
    module = collector()
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"cases": [sample_case()]}))
    protocol = tmp_path / "PROTOCOL.md"
    protocol.write_text("Frozen protocol fixture")
    teacher = module.teacher_metadata(
        CharacterTokenizer(), model="Qwen/Qwen3.5-4B", revision="a" * 40,
        device="cpu", protocol_path=protocol,
    )
    return module, suite, tmp_path / "targets.json", teacher


def test_failed_second_forward_retains_first_and_resume_skips_it(tmp_path):
    module, suite, output, teacher = setup_collection(tmp_path)
    calls = []

    def interrupted(prompt, ids):
        calls.append((prompt, ids))
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return [math.log(2), 0.0, 0.0]

    with pytest.raises(RuntimeError, match="simulated"):
        module.collect(suite, output, CharacterTokenizer(), interrupted, teacher)
    records = sorted(output.with_suffix(".observations").rglob("*.json"))
    before = {path: path.read_bytes() for path in records}
    assert any(path.name == "canonical.json" for path in records)
    assert not output.exists()
    result = module.collect(
        suite, output, CharacterTokenizer(), lambda prompt, ids: [math.log(6), math.log(3), 0.0],
        teacher, resume=True,
    )
    assert result["complete"] is True
    targets = json.loads(output.read_text())
    assert targets["rows"][0]["probabilities"] == pytest.approx({"alpha": 0.3, "middle": 0.275, "zeta": 0.425})
    assert all(path.read_bytes() == data for path, data in before.items())
    observation = json.loads(Path(targets["rows"][0]["observation_path"]).read_text())
    assert len(observation["orders"]) == 2
    assert observation["request"] == sample_case()["request"]
    assert "GOLD_SENTINEL" not in json.dumps(observation)
    assert observation["teacher"]["revision"] == "a" * 40
    output_before = output.read_bytes()
    result = module.collect(
        suite, output, CharacterTokenizer(), lambda *args: pytest.fail("completed unit repeated"), teacher, resume=True,
    )
    assert result["complete"] is True
    assert output.read_bytes() == output_before


@pytest.mark.parametrize("change", ["request", "revision", "protocol", "source"])
def test_resume_rejects_changed_identity_without_touching_partial_capture(tmp_path, change):
    module, suite, output, teacher = setup_collection(tmp_path)
    module.collect(suite, output, CharacterTokenizer(), lambda *args: [0.0, 0.0, 0.0], teacher)
    files = [p for p in tmp_path.rglob("*") if p.is_file() and p != suite]
    before = {path: path.read_bytes() for path in files}
    if change == "request":
        case = sample_case()
        case["request"]["state"] = "changed evidence"
        suite.write_text(json.dumps({"cases": [case]}))
    else:
        teacher = {**teacher, {"revision": "revision", "protocol": "protocol_sha256", "source": "source_sha256"}[change]:
                   "b" * (40 if change == "revision" else 64)}
    with pytest.raises(ValueError, match="identity|binding"):
        module.collect(suite, output, CharacterTokenizer(), lambda *args: pytest.fail("identity ignored"),
                       teacher, resume=True)
    assert all(path.read_bytes() == data for path, data in before.items())


def test_max_cases_preserves_partial_archive_without_publishing_incomplete_targets(tmp_path):
    module, suite, output, teacher = setup_collection(tmp_path)
    other = sample_case()
    other["id"] = "second"
    suite.write_text(json.dumps({"cases": [sample_case(), other]}))
    result = module.collect(suite, output, CharacterTokenizer(), lambda *args: [0.0] * 3, teacher, max_cases=1)
    assert result["complete"] is False and result["completed_questions"] == 1
    assert not output.exists()
    result = module.collect(suite, output, CharacterTokenizer(), lambda *args: [0.0] * 3, teacher, resume=True)
    assert result["complete"] is True and result["completed_questions"] == 2


@pytest.mark.parametrize("revision", [None, "main", "", "a" * 39])
def test_unpinned_revision_is_rejected_before_inference(revision, tmp_path):
    with pytest.raises(ValueError, match="revision"):
        collector().teacher_metadata(CharacterTokenizer(), model="Qwen/Qwen3.5-4B", revision=revision,
                                     device="cpu", protocol_path=tmp_path / "unused")


def test_selected_projection_matches_original_head_and_rejects_mutation():
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, layer_types=["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
    )
    model = Qwen3_5ForCausalLM(config).to(torch.bfloat16).eval().requires_grad_(False)
    scorer = collector().ModelScorer(model)
    prompt, code_ids = [1, 2, 3, 4], [32, 34, 33]
    with torch.inference_mode():
        hidden = model.model(torch.tensor([prompt]), use_cache=False).last_hidden_state[:, -1].float()
        expected = torch.nn.functional.linear(hidden, model.lm_head.weight.float())[0, code_ids].tolist()
    assert scorer(prompt, code_ids) == pytest.approx(expected, abs=1e-6)
    with torch.no_grad():
        model.lm_head.weight.add_(1)
    with pytest.raises(ValueError, match="head.*chang|changed.*head"):
        scorer(prompt, code_ids)


@pytest.mark.parametrize("field", ["missing_keys", "mismatched_keys", "error_msgs", "unexpected_keys"])
def test_loading_rejects_incomplete_or_changed_text_weights(field):
    with pytest.raises(ValueError, match="load|weight"):
        collector().validate_loading({field: ["lm_head.weight"]})


def test_study_lock_rejects_overlapping_gpu_jobs_and_releases_after_failure(tmp_path):
    module = collector()
    path = tmp_path / "STUDY.lock"
    with pytest.raises(RuntimeError, match="interrupted"):
        with module.study_lock(path):
            with pytest.raises(RuntimeError, match="lock|owns|running"):
                with module.study_lock(path):
                    pytest.fail("overlapping model job entered shared lock")
            raise RuntimeError("interrupted")
    with module.study_lock(path):
        pass


@pytest.mark.parametrize("scores", [[0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, float("inf"), 0.0]])
def test_invalid_model_logits_never_publish_targets(tmp_path, scores):
    module, suite, output, teacher = setup_collection(tmp_path)
    with pytest.raises(ValueError, match="logit"):
        module.collect(suite, output, CharacterTokenizer(), lambda *args: scores, teacher)
    assert not output.exists()
    assert not list(output.with_suffix(".observations").rglob("canonical.json"))


def test_resume_rejects_tampered_saved_prompt_before_more_inference(tmp_path):
    module, suite, output, teacher = setup_collection(tmp_path)
    module.collect(suite, output, CharacterTokenizer(), lambda *args: [0.0] * 3, teacher)
    path = next(output.with_suffix(".observations").rglob("canonical.json"))
    stored = json.loads(path.read_text())
    stored["prepared"]["prompt"] += "tampered"
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="binding"):
        module.collect(suite, output, CharacterTokenizer(), lambda *args: pytest.fail("tamper ignored"),
                       teacher, resume=True)


def cached_checkpoint_fixture(tmp_path, monkeypatch):
    """Real Qwen outer/text configs and content-addressed HF-cache layout, tiny weights."""
    import torch
    import transformers.utils.hub
    from transformers import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    revision = "a" * 40
    cache = tmp_path / "models--Qwen--Qwen3.5-4B"
    snapshot = cache / "snapshots" / revision
    blobs = cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    outer = Qwen3_5Config(text_config={
        "vocab_size": 96, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
        "linear_num_key_heads": 2, "linear_num_value_heads": 2, "linear_key_head_dim": 16,
        "linear_value_head_dim": 16, "layer_types": ["full_attention"], "dtype": "bfloat16",
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                            "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
    })
    outer.save_pretrained(snapshot)
    text_config = Qwen3_5TextConfig.from_pretrained(snapshot, local_files_only=True)
    assert text_config._commit_hash is None  # Actual extraction drops the snapshot commit.
    model = Qwen3_5ForCausalLM(text_config).to(torch.bfloat16)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"},
    }))
    weight_bytes = b"small fixture representing pinned checkpoint bytes"
    digest = hashlib.sha256(weight_bytes).hexdigest()
    (blobs / digest).write_bytes(weight_bytes)
    (snapshot / "model-00001-of-00001.safetensors").symlink_to(blobs / digest)

    def cached_file(model_id, filename, *, revision, local_files_only, **kwargs):
        assert model_id == "Qwen/Qwen3.5-4B" and local_files_only is True
        return str(snapshot / filename)

    monkeypatch.setattr(transformers.utils.hub, "cached_file", cached_file)
    # The 4B weight allocation is the slow/external boundary; config extraction,
    # cache provenance checks, loader validation and the model scorer stay real.
    monkeypatch.setattr(Qwen3_5ForCausalLM, "from_pretrained", lambda *args, **kwargs: (model, {}))
    monkeypatch.setattr("openjev.judgment_model.configure_kernels", lambda backend: None)
    return snapshot, model, revision


def test_pinned_snapshot_allows_real_text_subconfig_without_inherited_commit(tmp_path, monkeypatch):
    snapshot, model, revision = cached_checkpoint_fixture(tmp_path, monkeypatch)
    scorer = collector()._load_model("Qwen/Qwen3.5-4B", revision, "cpu")
    assert scorer.model is model
    assert scorer.model.config._commit_hash is None
    assert scorer.checkpoint_provenance["snapshot"] == str(snapshot)
    assert scorer.checkpoint_provenance["revision"] == revision
    assert scorer.checkpoint_provenance["files"]["model-00001-of-00001.safetensors"]["sha256"] == (
        "dffd4856a76fd2c011ed5e2b5d6f1f21b5fde37a1117d16ba558aa792f6dd63c"
    )


def test_explicit_wrong_loaded_revision_still_rejects(tmp_path, monkeypatch):
    _, model, revision = cached_checkpoint_fixture(tmp_path, monkeypatch)
    model.config._commit_hash = "b" * 40
    with pytest.raises(ValueError, match="revision"):
        collector()._load_model("Qwen/Qwen3.5-4B", revision, "cpu")


def test_cached_snapshot_must_resolve_to_requested_revision(tmp_path, monkeypatch):
    _, _, revision = cached_checkpoint_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="revision|snapshot"):
        collector()._checkpoint_provenance("Qwen/Qwen3.5-4B", "b" * 40)


def test_tampered_cached_weight_is_rejected_before_model_loading(tmp_path, monkeypatch):
    snapshot, _, revision = cached_checkpoint_fixture(tmp_path, monkeypatch)
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="weight|digest|hash"):
        collector()._load_model("Qwen/Qwen3.5-4B", revision, "cpu")


def test_loaded_text_configuration_must_match_pinned_snapshot(tmp_path, monkeypatch):
    _, model, revision = cached_checkpoint_fixture(tmp_path, monkeypatch)
    model.config.max_position_embeddings += 1
    with pytest.raises(ValueError, match="configuration"):
        collector()._load_model("Qwen/Qwen3.5-4B", revision, "cpu")


def test_real_tiny_checkpoint_loads_from_verified_snapshot_on_cpu(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM

    original_loader = Qwen3_5ForCausalLM.from_pretrained
    snapshot, original, revision = cached_checkpoint_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(Qwen3_5ForCausalLM, "from_pretrained", original_loader)
    weights = {key: value.clone() for key, value in original.state_dict().items()}
    temporary = tmp_path / "tiny.safetensors"
    save_file(weights, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    blob = snapshot.parent.parent / "blobs" / digest
    temporary.rename(blob)
    shard_name = "model-00001-of-00001.safetensors"
    (snapshot / shard_name).unlink()
    (snapshot / shard_name).symlink_to(blob)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(value.numel() * value.element_size() for value in weights.values())},
        "weight_map": {key: shard_name for key in weights},
    }))
    scorer = collector()._load_model("Qwen/Qwen3.5-4B", revision, "cpu")
    assert scorer.model.config._commit_hash is None
    assert scorer.checkpoint_provenance["files"][shard_name]["sha256"] == digest
    for key, value in scorer.model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_shared_xet_blob_uses_repository_sha256_link_for_content_identity(tmp_path, monkeypatch):
    snapshot, _, revision = cached_checkpoint_fixture(tmp_path, monkeypatch)
    shard = snapshot / "model-00001-of-00001.safetensors"
    repo_blob = shard.resolve()
    shared_blob = snapshot.parent.parent.parent / "blobs" / "ee" / ("e" * 64)
    shared_blob.parent.mkdir(parents=True)
    repo_blob.rename(shared_blob)
    repo_blob.symlink_to(shared_blob)
    provenance = collector()._checkpoint_provenance("Qwen/Qwen3.5-4B", revision)
    assert provenance["files"][shard.name]["sha256"] == (
        "dffd4856a76fd2c011ed5e2b5d6f1f21b5fde37a1117d16ba558aa792f6dd63c"
    )
