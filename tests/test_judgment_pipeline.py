import unittest


class PipelineGateTests(unittest.TestCase):
    def test_gate_rejects_partial_smoke_and_mismatched_runtime(self):
        from experiments.judgment_pipeline import validate_gate

        good = {"stage": "smoke", "passed": True, "completed_requested_pass": True, "fingerprint": {"a": 1}}
        validate_gate(good, {"a": 1})
        for change in ({"stage": "full"}, {"passed": False}, {"completed_requested_pass": False}, {"fingerprint": {}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_gate({**good, **change}, {"a": 1})

    def test_nested_probability_comparison_checks_paths_and_values(self):
        from experiments.judgment_pipeline import compare_probabilities

        a = {"x": [{"type": "noul", "probabilities": {"false": 0.2, "true": 0.8}}, {}]}
        b = {"x": [{"type": "noul", "probabilities": {"false": 0.21, "true": 0.79}}, {}]}
        self.assertAlmostEqual(compare_probabilities(a, b), 0.01)
        with self.assertRaises(ValueError):
            compare_probabilities(a, {"wrong": []})
        with self.assertRaises(ValueError):
            compare_probabilities(a, {"x": [{"type": "noul", "probabilities": {"false": 0.8, "true": 0.2}}, {}]})
