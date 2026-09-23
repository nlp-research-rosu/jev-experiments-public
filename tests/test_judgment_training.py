import math
import unittest

import torch


class JudgmentLossTests(unittest.TestCase):
    def test_evaluation_nll_remains_correct_after_probability_saturation(self):
        from types import SimpleNamespace

        from openjev.judgment_training import PreparedBundle, TrainingGroup, evaluate_bundles

        class Scorer:
            def eval(self):
                pass

            def score_prompts(self, *args, **kwargs):
                return torch.tensor([17.0, 40.0])

        bundle = PreparedBundle(
            "test",
            "test",
            [[1], [2]],
            [1, 1],
            [
                TrainingGroup("noul", (0,), {}, {"truth": False}),
                TrainingGroup("noul", (1,), {}, {"truth": False}),
            ],
            [],
        )
        result = evaluate_bundles(SimpleNamespace(model=Scorer(), unit_batch_size=2), [bundle])
        self.assertAlmostEqual(result["nll"], 28.5, places=5)
        self.assertAlmostEqual(result["canonical"]["nll"], 17.0, places=5)

    def test_grouped_choice_loss_not_independent_binary_losses(self):
        from openjev.judgment_training import TrainingGroup, bundle_loss

        z = torch.tensor([math.log(0.2), math.log(0.8)], dtype=torch.float64, requires_grad=True)
        groups = [TrainingGroup("choice", (0, 1), {"a": None, "b": None}, {"choice": "b"})]
        loss = bundle_loss(z, groups, [])
        self.assertAlmostEqual(loss["total"].item(), -math.log(0.8), places=6)
        loss["total"].backward()
        torch.testing.assert_close(z.grad, torch.tensor([0.2, -0.2], dtype=torch.float64))

    def test_complement_loss_does_not_replace_supervision(self):
        from openjev.judgment_training import TrainingGroup, bundle_loss

        groups = [TrainingGroup("noul", (0,), {}, {"truth": True}), TrainingGroup("noul", (1,), {}, {"truth": False})]
        relation = [{"kind": "complement", "left": 0, "right": 1}]
        flat = bundle_loss(torch.zeros(2), groups, relation)
        self.assertAlmostEqual(flat["consistency"].item(), 0)
        self.assertAlmostEqual(flat["total"].item(), math.log(2), places=6)
        wrong = bundle_loss(torch.tensor([-math.log(99), math.log(99)]), groups, relation)
        self.assertGreater(wrong["total"].item(), 4)
        conflicting = bundle_loss(torch.tensor([math.log(4), math.log(4)]), groups, relation)
        self.assertAlmostEqual(conflicting["consistency"].item(), 0.36, places=6)

    def test_invariant_choice_aligns_names_before_probability_distance(self):
        from openjev.judgment_training import TrainingGroup, bundle_loss

        groups = [
            TrainingGroup("choice", (0, 1), {"a": None, "b": None}, {"choice": "b"}),
            TrainingGroup("choice", (2, 3), {"b": None, "a": None}, {"choice": "b"}),
        ]
        result = bundle_loss(torch.tensor([0.0, 1.0, 1.0, 0.0]), groups, [{"kind": "invariant", "left": 0, "right": 1}])
        self.assertAlmostEqual(result["consistency"].item(), 0)

    def test_score_loss_uses_distribution_not_only_its_mean(self):
        from openjev.judgment_training import TrainingGroup, bundle_loss

        group = [TrainingGroup("score", (0, 1, 2), ["low", "mid", "high"], {"level_index": 1})]
        good = bundle_loss(torch.log(torch.tensor([0.05, 0.9, 0.05])), group, [])
        bad = bundle_loss(torch.log(torch.tensor([0.49, 0.02, 0.49])), group, [])
        self.assertLess(good["total"].item(), bad["total"].item())

    def test_invalid_targets_relations_and_partial_groups_rejected(self):
        from openjev.judgment_training import TrainingGroup, bundle_loss

        for group in (
            TrainingGroup("noul", (0,), {}, {"truth": 1}),
            TrainingGroup("choice", (0, 1), {"a": None, "b": None}, {"choice": "c"}),
            TrainingGroup("score", (0, 1), ["a", "b"], {"level_index": True}),
            TrainingGroup("score", (0, 1), ["a", "b"], {"probabilities": [0.2, 0.2]}),
        ):
            with self.subTest(group=group), self.assertRaises(ValueError):
                bundle_loss(torch.zeros(len(group.indices)), [group], [])
        with self.assertRaises(ValueError):
            bundle_loss(torch.zeros(2), [TrainingGroup("noul", (0,), {}, {"truth": True})], [])
        with self.assertRaises(ValueError):
            bundle_loss(torch.tensor([float("nan")]), [TrainingGroup("noul", (0,), {}, {"truth": True})], [])
