"""CPU reproducers for the GPU diagnostic's snapshot/alignment contract."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "reports/post-pause-audit-v1/gpu_kernel_probe_v2.py"
spec = importlib.util.spec_from_file_location("post_pause_probe_v2", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class GradModeSensitiveModel(torch.nn.Module):
    """A deliberate mode difference detects re-scoring the wrong forward."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.125))
        self.frozen = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)

    def cuda(self):
        # This fixture intentionally cannot launch GPU work.
        return self

    def score_prompts(self, prompts, kinds, *, unit_batch_size):
        values = torch.tensor([p[0]/10 + kind for p, kind in zip(prompts, kinds)])
        return self.weight*values + (0. if torch.is_grad_enabled() else .75)


class GPUHarnessAuditTests(unittest.TestCase):
    def test_v1_mix_of_training_loss_and_nograd_logits_is_reproducible_on_cpu(self):
        inputs = probe.base.fixture()
        with patch.object(torch.cuda, "reset_peak_memory_stats"), patch.object(torch.cuda, "synchronize"), \
             patch.object(torch.cuda, "empty_cache"), patch.object(torch.cuda, "max_memory_allocated", return_value=0):
            snapshot, stats = probe.base.run_variant(GradModeSensitiveModel(), inputs, backend="reference",
                                                    dtype=torch.float32, batch_size=12, production_backward=True)
        reconstructed = probe.base.grouped(snapshot["logits"], inputs[2])[0].item()
        # This test proves the archived v1 comparison can pair a correct training
        # loss with unrelated diagnostic logits when modes genuinely differ.
        self.assertGreater(abs(reconstructed-stats["loss"]), .01)

    def test_v2_captures_actual_loss_logits_despite_duplicate_text_and_reordering(self):
        inputs = probe.base.fixture()
        prompts, kinds, groups, _, _ = inputs
        for prompt in prompts:
            prompt[:] = [7, 8]
        self.assertEqual(len({tuple(p) for p in prompts}), 1)
        self.assertEqual(set(kinds), {0, 1})
        self.assertEqual(len({id(p) for p in prompts}), 48)
        model = GradModeSensitiveModel()
        expected = model.score_prompts(prompts, kinds, unit_batch_size=12).detach()
        actual, _, result, observed_order = probe.capture_production_backward(model, inputs)
        self.assertNotEqual(observed_order, list(range(48)))
        self.assertEqual(set(observed_order), set(range(48)))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertAlmostEqual(probe.base.grouped(actual, groups)[0].item(), result["loss"], places=6)
        with torch.no_grad():
            unrelated = model.score_prompts(prompts, kinds, unit_batch_size=12)
        self.assertGreater((actual-unrelated).abs().max().item(), .7)

    def test_real_v1_fixture_has_no_duplicate_token_tuples(self):
        prompts, _, _, _, _ = probe.base.fixture()
        self.assertEqual(len(prompts), len({tuple(p) for p in prompts}))


if __name__ == "__main__":
    unittest.main()
