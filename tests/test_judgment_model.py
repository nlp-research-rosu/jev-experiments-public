import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from test_engine import tiny_model


class JudgmentModelTests(unittest.TestCase):
    def test_engine_rejects_unknown_model_instead_of_silently_replacing_it(self):
        from openjev.judgment_model import JudgmentEngine, JudgmentModel

        class Tokenizer:
            def encode(self, value, **kwargs):
                return [32 if value == "A" else 33]

        engine = JudgmentEngine(JudgmentModel(tiny_model(True)), Tokenizer())
        with self.assertRaisesRegex(ValueError, "model"):
            engine.prepare({"model": "some-other-model", "state": {}, "questions": {"x": {"type": "noul"}}})

    def test_readout_padding_and_kind_match_independent_backbone(self):
        from openjev.judgment_model import JudgmentModel

        causal = tiny_model(True)
        model = JudgmentModel(causal)
        model.binary.bias.data.fill_(0.37)
        prompts = [[1, 2, 3, 4], [1, 2, 3, 8, 9, 10], [1, 2, 3, 7]]
        kinds = [0, 1, 1]
        with torch.inference_mode():
            expected = []
            w = causal.get_output_embeddings().weight[32].float() - causal.get_output_embeddings().weight[33].float()
            for p, k in zip(prompts, kinds):
                h = causal.model(torch.tensor([p]), use_cache=False).last_hidden_state[0, -1].float()
                expected.append(h @ w + (0.37 if k else 0))
            actual = model.score_prompts(prompts, kinds, unit_batch_size=2)
        torch.testing.assert_close(actual, torch.stack(expected), atol=2e-6, rtol=1e-4)

    def test_cached_scoring_preserves_hybrid_state_and_endpoints(self):
        from openjev.judgment_model import JudgmentModel

        model = JudgmentModel(tiny_model(True))
        prompts = [[1, 2, 3, 4], [1, 2, 3, 8, 9, 10], [1, 2, 3, 7]]
        with torch.inference_mode():
            full = model.score_prompts(prompts, [0, 1, 0], unit_batch_size=2)
            cached, stats = model.score_cached(prompts, [0, 1, 0], unit_batch_size=2)
            again, _ = model.score_cached(prompts[::-1], [0, 1, 0], unit_batch_size=2)
        torch.testing.assert_close(full, cached, atol=2e-6, rtol=1e-4)
        torch.testing.assert_close(full, again.flip(0), atol=2e-6, rtol=1e-4)
        self.assertEqual(stats["forward_calls"], 3)

    def test_differentiable_scores_train_heads_with_base_frozen(self):
        from openjev.judgment_model import JudgmentModel

        model = JudgmentModel(tiny_model(True))
        logits = model.score_prompts([[1, 2, 3], [1, 4, 5]], [0, 1])
        logits.square().sum().backward()
        self.assertGreater(model.compatibility.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.binary.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.backbone.parameters()))

    def test_checkpoint_roundtrip_preserves_heads_and_metadata(self):
        from openjev.judgment_model import JudgmentModel

        model = JudgmentModel(tiny_model(True))
        model.binary.bias.data.fill_(0.37)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint"
            model.save_checkpoint(path, metadata={"step": 7})
            loaded, metadata = JudgmentModel.load_checkpoint(path, base_factory=lambda: tiny_model(True))
            with torch.inference_mode():
                a = model.score_prompts([[1, 2, 3]], [1])
                b = loaded.score_prompts([[1, 2, 3]], [1])
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(metadata["step"], 7)
            with self.assertRaises(FileExistsError):
                model.save_checkpoint(path)

    def test_checkpoint_identity_distinguishes_weights_and_detects_corruption(self):
        from openjev.judgment_model import JudgmentModel

        model = JudgmentModel(tiny_model(True))
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a/final", Path(td) / "b/final"
            first = model.save_checkpoint(a)
            model.binary.bias.data.add_(0.2)
            second = model.save_checkpoint(b)
            self.assertIsInstance(first, str)
            self.assertNotEqual(first, second)
            (b / "readouts.safetensors").write_bytes((a / "readouts.safetensors").read_bytes())
            with self.assertRaisesRegex(ValueError, "identity"):
                JudgmentModel.load_checkpoint(b, base_factory=lambda: tiny_model(True))

    @unittest.skipUnless(importlib.util.find_spec("peft"), "training overlay required")
    def test_lora_updates_only_adapters_and_survives_reload(self):
        from openjev.judgment_model import JudgmentModel, frozen_digest

        model = JudgmentModel(tiny_model(True))
        inventory = model.add_lora(rank=2, alpha=4, gradient_checkpointing=False)
        self.assertGreater(inventory["adapter_parameters"], 0)
        before = frozen_digest(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        loss = model.score_prompts([[1, 2, 3], [1, 4, 5]], [0, 1]).square().sum()
        loss.backward()
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_B" in n)
        )
        optimizer.step()
        self.assertEqual(before, frozen_digest(model))
        model.eval()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint"
            model.save_checkpoint(path)
            loaded, _ = JudgmentModel.load_checkpoint(path, base_factory=lambda: tiny_model(True))
            with torch.inference_mode():
                a = model.score_prompts([[1, 2, 3]], [1])
                b = loaded.score_prompts([[1, 2, 3]], [1])
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_invalid_batch_inputs_fail_explicitly(self):
        from openjev.judgment_model import JudgmentModel

        model = JudgmentModel(tiny_model(True))
        for prompts, kinds in (([], []), ([[1]], []), ([[]], [0]), ([[1]], [2])):
            with self.subTest(prompts=prompts, kinds=kinds), self.assertRaises(ValueError):
                model.score_prompts(prompts, kinds)
