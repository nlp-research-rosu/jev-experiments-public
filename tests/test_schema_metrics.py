import math
import unittest


class SchemaTests(unittest.TestCase):
    def test_preserves_boolean_types_and_shared_prefix_labels(self):
        from openjev.schema import parse_schema

        fields = parse_schema(
            {
                "action": {"type": "enum", "description": "Action", "choices": ["ALLOW", "ALLOW_WITH_REVIEW"]},
                "urgent": {"type": "boolean", "description": "Urgent?"},
            }
        )
        self.assertEqual(fields[0].choices, ("ALLOW", "ALLOW_WITH_REVIEW"))
        self.assertEqual(fields[1].choices, (False, True))
        self.assertIs(fields[1].choices[1], True)

    def test_rejects_empty_duplicate_and_unsupported_schemas(self):
        from openjev.schema import parse_schema

        for schema in (
            {},
            {"x": {"type": "enum", "choices": []}},
            {"x": {"type": "enum", "choices": ["a", "a"]}},
            {"x": {"type": "string"}},
            {"x": {"type": "enum", "choices": [1, 2]}},
            {"": {"type": "boolean"}},
        ):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                parse_schema(schema)

    def test_format_validity_checks_values_types_and_exact_keys(self):
        from openjev.schema import parse_schema, valid_answer

        fields = parse_schema({"ok": {"type": "boolean"}, "route": {"type": "enum", "choices": ["a", "b"]}})
        self.assertTrue(valid_answer({"ok": True, "route": "a"}, fields))
        for answer in (
            {"ok": "true", "route": "a"},
            {"ok": 1, "route": "a"},
            {"ok": True},
            {"ok": True, "route": "c"},
            {"ok": True, "route": "a", "extra": 0},
            [],
        ):
            self.assertFalse(valid_answer(answer, fields))


class MetricTests(unittest.TestCase):
    def test_hand_computed_metrics_and_invalid_answer_penalty(self):
        from openjev.metrics import summarize

        cases = [
            {"id": "a", "schema": {"x": {"type": "enum", "choices": ["yes", "no"]}}, "expected": {"x": "yes"}},
            {"id": "b", "schema": {"x": {"type": "enum", "choices": ["yes", "no"]}}, "expected": {"x": "no"}},
        ]
        results = [
            {"id": "a", "values": {"x": "yes"}, "probabilities": {"x": [0.8, 0.2]}, "elapsed_ms": 10},
            {"id": "b", "values": {}, "probabilities": {}, "elapsed_ms": 30},
        ]
        summary = summarize(cases, results)
        self.assertEqual(summary["field_accuracy"], 0.5)
        self.assertEqual(summary["exact_case_accuracy"], 0.5)
        self.assertEqual(summary["schema_validity"], 0.5)
        self.assertAlmostEqual(summary["brier"], 0.08)
        self.assertAlmostEqual(summary["log_loss"], -math.log(0.8))
        self.assertEqual(summary["probability_coverage"], 0.5)
        self.assertAlmostEqual(summary["ece_10_bins"], 0.2)
        self.assertEqual(summary["median_ms"], 20)

    def test_unlabeled_cases_do_not_invent_accuracy(self):
        from openjev.metrics import summarize

        summary = summarize(
            [{"id": "a", "schema": {"ok": {"type": "boolean"}}}], [{"id": "a", "values": {"ok": True}, "elapsed_ms": 1}]
        )
        self.assertIsNone(summary["field_accuracy"])
        self.assertIsNone(summary["brier"])

    def test_incomplete_gold_and_missing_results_rejected(self):
        from openjev.metrics import summarize

        with self.assertRaises(ValueError):
            summarize(
                [{"id": "a", "schema": {"ok": {"type": "boolean"}}, "expected": {}}],
                [{"id": "a", "values": {"ok": True}, "elapsed_ms": 1}],
            )
        with self.assertRaises(ValueError):
            summarize([{"id": "a", "schema": {"ok": {"type": "boolean"}}}], [])

    def test_probability_validation_rejects_nan_and_wrong_length(self):
        from openjev.metrics import summarize

        case = {"id": "a", "schema": {"ok": {"type": "boolean"}}, "expected": {"ok": True}}
        for probs in ([float("nan"), 1], [1], [-0.1, 1.1], [0.2, 0.2]):
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                summarize(
                    [case], [{"id": "a", "values": {"ok": True}, "probabilities": {"ok": probs}, "elapsed_ms": 1}]
                )


if __name__ == "__main__":
    unittest.main()
