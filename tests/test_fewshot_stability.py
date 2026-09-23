import json

import pytest

from experiments.fewshot_stability import layout_fields, prepare_messages, stability_summary, validate_split
from openjev.prompts import render_prompts
from openjev.schema import Field


class EchoTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages


FIELDS = (
    Field("department", "Route", ("billing", "technical", "account_access", "sales")),
    Field("ok", "Is it true?", (False, True)),
)


@pytest.mark.parametrize("json_output", [False, True])
def test_zero_shot_identity_is_exact_existing_prompt(json_output):
    messages, _, _ = prepare_messages("New request", FIELDS, [], shift=0, reverse=False, json_output=json_output)
    expected = render_prompts(EchoTokenizer(), "New request", FIELDS, list("ABCD"), json_output=json_output)
    assert messages == expected


def test_display_reversal_preserves_letter_to_value_mapping():
    normal, codes = layout_fields(FIELDS, shift=1, reverse=False)
    reversed_fields, reversed_codes = layout_fields(FIELDS, shift=1, reverse=True)
    for a, ca, b, cb in zip(normal, codes, reversed_fields, reversed_codes, strict=True):
        assert dict(zip(ca, a.choices)) == dict(zip(cb, b.choices))
        assert a.choices == b.choices[::-1]
    assert dict(zip(codes[0], normal[0].choices)) == {
        "B": "billing",
        "C": "technical",
        "D": "account_access",
        "A": "sales",
    }


def test_worked_example_codes_follow_current_mapping():
    examples = [{"context": "An example", "expected": {"department": "sales", "ok": True}}]
    original, _, _ = prepare_messages("Actual request", FIELDS, examples, shift=0, reverse=False)
    shifted, _, _ = prepare_messages("Actual request", FIELDS, examples, shift=1, reverse=True)
    assert '"answer_codes": {"department": "D", "ok": "B"}' in original[0][1]["content"]
    assert '"answer_codes": {"department": "A", "ok": "A"}' in shifted[0][1]["content"]
    assert "Actual request" in shifted[0][1]["content"]


def test_json_demonstrations_keep_boolean_types_and_semantic_values():
    examples = [{"context": "An example", "expected": {"department": "sales", "ok": True}}]
    messages, _, _ = prepare_messages("Actual request", FIELDS, examples, json_output=True)
    assert '"answer": {"department": "sales", "ok": true}' in messages[0][1]["content"]
    assert "answer_codes" not in messages[0][1]["content"]


def test_examples_and_evaluation_contexts_cannot_overlap():
    schema = {"ok": {"type": "boolean"}}
    example = {"id": "example", "context": "Same context", "expected": {"ok": True}, "schema": schema}
    copied = {**example, "id": "different_id", "context": "  SAME   CONTEXT "}
    with pytest.raises(ValueError, match="overlap"):
        validate_split([copied], [example])


def test_stability_counts_unique_cases_not_repeated_timings():
    schema = {"ok": {"type": "boolean"}, "priority": {"type": "enum", "choices": ["normal", "urgent"]}}
    cases = [
        {"id": "a", "schema": schema, "expected": {"ok": True, "priority": "normal"}},
        {"id": "b", "schema": schema, "expected": {"ok": False, "priority": "urgent"}},
    ]
    rows = []
    for layout in ["s0_forward", "s0_reverse"]:
        for c in cases:
            values = dict(c["expected"])
            if layout == "s0_reverse" and c["id"] == "a":
                values["ok"] = False
            rows.append({"id": c["id"], "layout": layout, "repeat": 0, "values": values, "elapsed_ms": 1})
    rows.append({**rows[0], "repeat": 1, "values": {"ok": False, "priority": "normal"}, "elapsed_ms": 99})
    result = stability_summary(cases, rows, ["s0_forward", "s0_reverse"])
    assert result["mean_field_accuracy"] == pytest.approx(0.875)
    assert result["worst_layout_accuracy"] == pytest.approx(0.75)
    assert result["stable_field_fraction"] == pytest.approx(0.75)
    assert result["pairwise_field_agreement"] == pytest.approx(0.75)
    assert result["unique_labeled_fields"] == 4
    assert result["per_layout"]["s0_forward"]["repeat_value_disagreements"] == 1
    with pytest.raises(ValueError, match="missing|duplicate"):
        stability_summary(cases, rows[:-2], ["s0_forward", "s0_reverse"])
    json.dumps(result, allow_nan=False)
