"""CPU contracts for the bounded native-interface diagnostic."""

import hashlib
import importlib
import json
import math
from types import SimpleNamespace

import pytest


def native():
    return importlib.import_module("experiments.architecture_native")


class CharacterTokenizer:
    chat_template = "fixture native thinking template"
    all_special_ids = [0]
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()

    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True
        suffix = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        return "\n".join(m["content"] for m in messages) + "\nASSISTANT:\n" + suffix


def case():
    return {
        "id": "example", "expected": {"c": "DO_NOT_FEED_GOLD"}, "rationale": "DO_NOT_FEED_RATIONALE",
        "request": {"state": {"rules": ["Select by all definitions"], "facts": ["x"]},
                    "questions": {"c": {"type": "choice", "instructions": {"rule": "apply"},
                                          "criteria": {"z": "last", "a": {"nested": ["first"]}}}}},
    }


def test_native_prompt_retains_complete_criteria_and_remaps_orders_without_gold():
    n = native()
    prepared = n.prepare_question(CharacterTokenizer(), case()["request"], "c", mode="direct")
    assert [p["labels"] for p in prepared] == [["a", "z"], ["z", "a"]]
    for p in prepared:
        payload = json.loads(p["messages"][1]["content"])
        assert payload["state"] == case()["request"]["state"]
        assert payload["question"] == case()["request"]["questions"]["c"]
        assert "DO_NOT_FEED" not in p["prompt"]
        assert p["input_ids"][-1] == 10
        assert p["code_token_ids"] == [65, 66]
    result = n.average_orders({"a": .8, "z": .2}, {"z": .6, "a": .4})
    assert result == pytest.approx({"a": .6, "z": .4})


@pytest.mark.parametrize("primitive,criteria,labels", [
    ("noul", {"false": {"rule": ["absent"]}}, ["false", "true"]),
    ("score", ["none", {"when": "some"}, "all"], ["0", "1", "2"]),
])
def test_native_preserves_boolean_definitions_and_ordered_score_levels(primitive, criteria, labels):
    request = {"state": [], "questions": {"q": {"type": primitive, "criteria": criteria}}}
    p = native().prepare_question(CharacterTokenizer(), request, "q", mode="reason")[0]
    payload = json.loads(p["messages"][1]["content"])
    assert p["labels"] == labels
    assert payload["question"]["criteria"] == criteria
    assert p["prompt"].endswith("<think>\n")
    if primitive == "noul":
        assert payload["answers"][0]["criterion"] == {"rule": ["absent"]}
        assert payload["answers"][1]["meaning"] == "yes / true"


@pytest.mark.parametrize("failure", ["boundary", "special", "collision", "multiple"])
def test_native_rejects_ambiguous_codes(failure):
    class BadTokenizer(CharacterTokenizer):
        all_special_ids = [65] if failure == "special" else [0]

        def encode(self, text, **kwargs):
            if failure == "boundary" and text.endswith("\nA"):
                return [1]
            if failure == "collision" and text == "B":
                return [65]
            if failure == "multiple" and text == "A":
                return [65, 65]
            return super().encode(text, **kwargs)

    with pytest.raises(ValueError, match="code|token"):
        native().prepare_question(BadTokenizer(), case()["request"], "c", mode="direct")


def test_native_no_truncation_or_more_than_128_scratchpad_tokens():
    with pytest.raises(ValueError, match="truncat|exceed"):
        native().prepare_question(CharacterTokenizer(), case()["request"], "c", mode="direct", max_input_tokens=20)
    with pytest.raises(ValueError, match="128|budget"):
        native().validate_run("reason", "original", max_seconds=1200, max_scratchpad_tokens=129)
    with pytest.raises(ValueError, match="original"):
        native().validate_run("reason", "H0", max_seconds=1200)
    with pytest.raises(ValueError, match="1200|budget"):
        native().validate_run("reason", "original", max_seconds=1201)


def test_full_vocabulary_mass_and_greedy_validity_survive_closed_set_normalization():
    import torch

    result = native().summarize_logits(torch.tensor([math.log(1), math.log(2), math.log(7)]), [0, 1], ["a", "z"])
    assert result["semantic_probabilities"] == pytest.approx({"a": 1 / 3, "z": 2 / 3})
    assert result["valid_code_mass"] == pytest.approx(.3)
    assert result["greedy_token_id"] == 2
    assert result["greedy_code_valid"] is False
    assert result["raw_code_logits"] == pytest.approx([0, math.log(2)])


def test_projection_uses_only_final_hidden_vector_and_keeps_bfloat16_head():
    import torch

    class Body(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.tensor([[[99., 0.], [1., 2.]]], dtype=torch.bfloat16))

    class Head(torch.nn.Linear):
        def forward(self, hidden):
            assert hidden.shape == (1, 2)
            assert hidden.dtype == self.weight.dtype == torch.bfloat16
            return super().forward(hidden)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()
            self.lm_head = Head(2, 3, bias=False, dtype=torch.bfloat16)
            self.config = SimpleNamespace(vocab_size=3, hidden_size=2)
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 1.]]))

        def get_output_embeddings(self):
            return self.lm_head

    model = Model()
    scorer = native().NativeScorer(model)
    scored = scorer.score([1, 2], [0, 1], ["a", "z"])
    assert scored["raw_code_logits"] == [1., 2.]
    assert scored["greedy_token_id"] == 2
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert all(not p.requires_grad for p in model.parameters())


def test_probability_response_reconstructs_all_public_primitive_shapes():
    request = {"state": "x", "questions": {"n": {"type": "noul"},
        "c": {"type": "choice", "criteria": {"z": "z", "a": "a"}},
        "s": {"type": "score", "criteria": ["low", "high"]}}}
    response = native().response_from_probabilities(request, {
        "n": {"false": .7, "true": .3}, "c": {"a": .6, "z": .4}, "s": {"0": .25, "1": .75}})
    assert response["values"] == pytest.approx({"n": .3, "c": "a", "s": .75})
    assert response["answers"]["c"]["probabilities"] == pytest.approx({"z": .4, "a": .6})
    assert response["answers"]["s"]["legend"] == {"0": "low", "1": "high"}


def test_reason_forcing_preserves_original_tokens_and_explicitly_closes_truncated_thought():
    tokenizer = CharacterTokenizer()
    p = native().prepare_question(tokenizer, case()["request"], "c", mode="reason")[0]
    generated = tokenizer.encode("unfinished reasoning")
    final = native().prepare_final_readout(tokenizer, p, generated, stop_reason="token_cap")
    assert final["generated_ids"] == generated
    assert final["generated_text"] == "unfinished reasoning"
    assert final["truncated"] is True
    assert final["forced_prefix"] == "\n</think>\n\nAnswer code:\n"
    assert final["prompt"].endswith("unfinished reasoning\n</think>\n\nAnswer code:\n")


def test_reason_eos_and_already_closed_thought_record_unmodified_original_generation():
    tokenizer = CharacterTokenizer()
    p = native().prepare_question(tokenizer, case()["request"], "c", mode="reason")[0]
    final = native().prepare_final_readout(tokenizer, p, tokenizer.encode("done</think>"), stop_reason="closing_think")
    assert final["forced_prefix"] == "\n\nAnswer code:\n"
    assert final["truncated"] is False
    final = native().prepare_final_readout(tokenizer, p, [100, 0], stop_reason="eos")
    assert final["generated_ids"] == [100, 0]
    assert final["generated_text"] == "d\u0000"
    assert "\u0000" not in final["prompt"]


def test_reason_generation_stops_on_closing_tag_and_uses_incremental_cache():
    import torch

    output_ids = CharacterTokenizer().encode("ok</think>IGNORED")

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = 0

        def forward(self, *, input_ids, attention_mask, use_cache, past_key_values=None):
            assert use_cache is True
            if self.offset:
                assert input_ids.shape[1] == 1 and past_key_values == "saved-cache"
            hidden = torch.zeros((1, 1, 128), dtype=torch.bfloat16)
            hidden[0, 0, output_ids[self.offset]] = 10
            self.offset += 1
            return SimpleNamespace(last_hidden_state=hidden, past_key_values="saved-cache")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()
            self.lm_head = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.eye(128))
            self.config = SimpleNamespace(vocab_size=128, hidden_size=128)

        def get_output_embeddings(self):
            return self.lm_head

    scorer = native().NativeScorer(Model())
    result = scorer.generate([1, 2], CharacterTokenizer(), deadline=100, clock=lambda: 0)
    assert result["generated_ids"] == list(b"ok</think>")
    assert result["stop_reason"] == "closing_think"
    assert result["generated_tokens"] == 10


def test_reason_deadline_prevents_even_initial_generation_forward():
    # A head is sufficient: expired budget must avoid touching the absent body.
    import torch

    model = torch.nn.Module()
    model.head = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
    model.config = SimpleNamespace(vocab_size=3, hidden_size=2)
    model.get_output_embeddings = lambda: model.head
    result = native().NativeScorer(model).generate([1], CharacterTokenizer(), deadline=1, clock=lambda: 1)
    assert result["stop_reason"] == "walltime"
    assert result["generated_ids"] == []


def _fixture_collection(tmp_path):
    source = tmp_path / "suite.json"
    source.write_text(json.dumps({"cases": [case()], "relations": []}))
    return source, tmp_path / "native.json", {"model": "Qwen/Qwen3.5-2B", "revision": native().REVISION,
                                            "fixture_source_hash": "fixed"}


class FixtureScorer:
    def score(self, input_ids, code_ids, labels):
        return {"raw_code_logits": [math.log(4), 0],
                "semantic_probabilities": {labels[0]: .8, labels[1]: .2},
                "valid_code_mass": .5, "greedy_token_id": 127, "greedy_code_valid": False,
                "full_vocab_logsumexp": math.log(10), "vocabulary_size": 128}


def test_archive_preserves_canonical_on_failed_reverse_and_resume_rejects_source_drift(tmp_path):
    source, output, metadata = _fixture_collection(tmp_path)

    class FailsSecond(FixtureScorer):
        def __init__(self):
            self.count = 0

        def score(self, *args):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("interrupted before reverse score")
            return super().score(*args)

    with pytest.raises(RuntimeError, match="interrupted"):
        native().collect(source, output, CharacterTokenizer(), FailsSecond(), metadata,
                         mode="direct", body="original", metrics_fn=lambda *args: {})
    records = list(output.with_suffix(".observations").glob("question-*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["prepared"]["order"] == "canonical"
    with pytest.raises(ValueError, match="manifest|binding"):
        native().collect(source, output, CharacterTokenizer(), FixtureScorer(), {**metadata, "fixture_source_hash": "changed"},
                         mode="direct", body="original", resume=True, metrics_fn=lambda *args: {})
    result = native().collect(source, output, CharacterTokenizer(), FixtureScorer(), metadata,
                             mode="direct", body="original", resume=True, metrics_fn=lambda *args: {})
    assert result["complete"] is True
    assert result["responses"]["example"]["answers"]["c"]["probabilities"] == pytest.approx({"a": .5, "z": .5})
    assert result["canonical_responses"]["example"]["answers"]["c"]["probabilities"] == pytest.approx({"a": .8, "z": .2})
    assert len(list(output.with_suffix(".observations").glob("question-*.json"))) == 2


def test_reason_partial_capture_persists_generated_tokens_and_never_invents_final_probability(tmp_path):
    source, output, metadata = _fixture_collection(tmp_path)

    class PartialScorer:
        def generate(self, *args, **kwargs):
            return {"generated_ids": [111, 107], "generated_tokens": 2, "stop_reason": "walltime", "elapsed_seconds": .1}

        def score(self, *args):
            raise AssertionError("no final forward after a walltime stop")

    result = native().collect(source, output, CharacterTokenizer(), PartialScorer(), metadata,
                             mode="reason", body="original", metrics_fn=lambda *args: {})
    assert result["complete"] is False
    assert result["completed_questions"] == 0
    assert result["generated_tokens"] == 2
    record = json.loads(next(output.with_suffix(".observations").glob("question-*.json")).read_text())
    assert record["generation"]["generated_text"] == "ok"
    assert record["generation"]["stop_reason"] == "walltime"
    assert record["score"] is None
    assert result["metrics"] is None


def test_resume_cannot_extend_budget_or_rewrite_captured_records(tmp_path):
    source, output, metadata = _fixture_collection(tmp_path)
    native().collect(source, output, CharacterTokenizer(), FixtureScorer(), metadata,
                     mode="direct", body="original", max_seconds=30, metrics_fn=lambda *args: {})
    with pytest.raises(ValueError, match="manifest|binding"):
        native().collect(source, output, CharacterTokenizer(), FixtureScorer(), metadata,
                         mode="direct", body="original", max_seconds=31, resume=True, metrics_fn=lambda *args: {})
    record_path = next(output.with_suffix(".observations").glob("question-*.json"))
    record = json.loads(record_path.read_text())
    record["prepared"]["input_ids"].append(999)
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="record|binding|digest"):
        native().collect(source, output, CharacterTokenizer(), FixtureScorer(), metadata,
                         mode="direct", body="original", max_seconds=30, resume=True, metrics_fn=lambda *args: {})


def test_h0_metadata_rejects_wrong_base_before_loading_adapter(tmp_path):
    checkpoint = tmp_path / "H0"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_text(json.dumps({
        "format": "openjev-judgment-v0.2", "model_id": "wrong-base", "revision": native().REVISION}))
    with pytest.raises(ValueError, match="base|model"):
        native().adapter_metadata(checkpoint)


def test_native_preserves_fp32_saved_lora_but_rejects_fp32_base_or_head():
    import torch

    model = torch.nn.Module()
    model.lora_A = torch.nn.Linear(2, 2, bias=False, dtype=torch.float32)
    model.head = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
    model.config = SimpleNamespace(vocab_size=3, hidden_size=2)
    model.get_output_embeddings = lambda: model.head
    native().NativeScorer(model)
    assert model.lora_A.weight.dtype == torch.float32
    model.base = torch.nn.Linear(2, 2, bias=False, dtype=torch.float32)
    with pytest.raises(ValueError, match="BF16|bfloat16"):
        native().NativeScorer(model)


def test_pinned_cache_provenance_checks_bytes_and_revision_without_model_load(tmp_path):
    repo = tmp_path / "models--Qwen--Qwen3.5-2B"
    snapshot = repo / "snapshots" / native().REVISION
    blobs = repo / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {"hidden_size": 2}}))
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.language_model.embed_tokens.weight": "weights.safetensors"}}))
    data = b"immutable-test-weight-bytes"
    blob = blobs / hashlib.sha256(data).hexdigest()
    blob.write_bytes(data)
    (snapshot / "weights.safetensors").symlink_to(blob)
    provenance = native().checkpoint_provenance(snapshot)
    assert provenance["files"]["weights.safetensors"]["sha256"] == hashlib.sha256(data).hexdigest()
    blob.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash|SHA256"):
        native().checkpoint_provenance(snapshot)


def test_cli_rejects_reason_on_sentinels_before_any_metadata_or_model_load(monkeypatch, tmp_path):
    monkeypatch.setattr(native(), "cached_metadata", lambda **kwargs: pytest.fail("must validate scope first"))
    with pytest.raises(ValueError, match="development|reason"):
        native().main(["--mode", "reason", "--body", "original", "--suite",
                       str(native().ROOT / "data/architecture-diagnostics-v1/sentinels.json"),
                       "--output", str(tmp_path / "out.json")])


def test_reason_final_readout_failure_retains_scratchpad_and_refuses_regeneration(tmp_path):
    source, output, metadata = _fixture_collection(tmp_path)

    class FinalFails(FixtureScorer):
        def generate(self, *args, **kwargs):
            return {"generated_ids": list(b"done</think>"), "generated_tokens": 12,
                    "stop_reason": "closing_think", "elapsed_seconds": .1}

        def score(self, *args):
            raise RuntimeError("final projection failed")

    result = native().collect(source, output, CharacterTokenizer(), FinalFails(), metadata,
                             mode="reason", body="original", metrics_fn=lambda *args: {})
    assert result["complete"] is False
    record = json.loads(next(output.with_suffix(".observations").glob("question-*.json")).read_text())
    assert record["generation"]["generated_text"] == "done</think>"
    assert "final projection failed" in record["generation"]["final_readout_error"]
    assert record["score"] is None
    result2 = native().collect(source, output, CharacterTokenizer(), None, metadata,
                              mode="reason", body="original", resume=True, metrics_fn=lambda *args: {})
    assert result2["generated_tokens"] == 12
    assert result2["stop_reason"] == "saved-partial-question"


def test_cpu_archive_preflight_creates_nothing_and_refuses_drift_before_model_load(tmp_path):
    source, output, metadata = _fixture_collection(tmp_path)
    checked = native().collect(source, output, CharacterTokenizer(), None, metadata, mode="direct", body="original",
                               validate_only=True)
    assert checked["ready"] is True
    assert not output.with_suffix(".observations").exists()
    native().collect(source, output, CharacterTokenizer(), FixtureScorer(), metadata,
                     mode="direct", body="original", metrics_fn=lambda *args: {})
    record_path = next(output.with_suffix(".observations").glob("question-*.json"))
    record = json.loads(record_path.read_text())
    record["score"]["valid_code_mass"] = .1
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="record|digest|binding"):
        native().collect(source, output, CharacterTokenizer(), None, metadata, mode="direct", body="original",
                         validate_only=True, resume=True)


def test_h0_frozen_digest_matches_training_names_excluding_adapter_and_head():
    import torch

    backbone = torch.nn.Module()
    backbone.layer = torch.nn.Module()
    backbone.layer.register_parameter("weight", torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.bfloat16)))
    backbone.layer.lora_A = torch.nn.Linear(2, 2, dtype=torch.float32)
    expected = hashlib.sha256(b"backbone.layer.weight" + bytes.fromhex("803f0040")).hexdigest()
    assert native()._h0_frozen_digest(backbone) == expected


def test_walltime_between_generation_and_final_readout_is_reported_explicitly(tmp_path, monkeypatch):
    source, output, metadata = _fixture_collection(tmp_path)
    now = [0.]
    monkeypatch.setattr(native().time, "monotonic", lambda: now[0])

    class AtLimit:
        def generate(self, *args, **kwargs):
            now[0] = 30.
            return {"generated_ids": list(b"ok</think>"), "generated_tokens": 10,
                    "stop_reason": "closing_think", "elapsed_seconds": 30.}

        def score(self, *args):
            raise AssertionError("must not start final readout at the time limit")

    result = native().collect(source, output, CharacterTokenizer(), AtLimit(), metadata,
                             mode="reason", body="original", max_seconds=30, metrics_fn=lambda *args: {})
    assert result["complete"] is False
    assert result["stop_reason"] == "walltime-before-final-readout"
    record = json.loads(next(output.with_suffix(".observations").glob("question-*.json")).read_text())
    assert record["generation"]["stop_reason"] == "closing_think"
    assert record["generation"]["final_readout_skipped"] == "walltime"
