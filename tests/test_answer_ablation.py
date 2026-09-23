import pytest

from experiments.answer_ablation import decode_scores, prepare_prompt_text
from openjev.prompts import render_prompts
from openjev.schema import Field


class EchoTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages


def test_control_prompt_matches_existing_implementation_exactly():
    fields = (Field("x", "Is it present?", (False, True)), Field("priority", "Choose urgency", ("normal", "urgent")))
    messages, codes, ordered = prepare_prompt_text("A request", fields, "codes_original")
    assert messages == render_prompts(EchoTokenizer(), "A request", fields, ["A", "B", "C", "D"])
    assert codes == [["A", "B"], ["A", "B"]]
    assert ordered == fields


def test_reversed_choices_map_probabilities_back_to_canonical_labels():
    original = (Field("ok", "Question", (False, True)),)
    _, _, reordered = prepare_prompt_text("Context", original, "codes_reversed")
    values, probabilities = decode_scores(original, reordered, [[0.9, 0.1]])
    assert values == {"ok": True}
    assert probabilities["ok"] == pytest.approx([0.1, 0.9])


def test_single_field_scope_excludes_unrelated_schema():
    fields = (Field("first", "First question", (False, True)), Field("second", "Second question", (False, True)))
    messages, _, _ = prepare_prompt_text("Shared context", fields, "single_field")
    assert '"field": "first"' in messages[0][1]["content"]
    assert '"field": "second"' not in messages[0][1]["content"]
    assert "Shared context" in messages[1][1]["content"]


def test_semantic_codes_retain_original_typed_values():
    fields = (
        Field("department", "Route", ("billing", "technical", "account_access", "sales")),
        Field("ok", "Question", (False, True)),
    )
    _, codes, ordered = prepare_prompt_text("Context", fields, "semantic")
    assert codes == [["billing", "technical", "account", "sales"], ["false", "true"]]
    values, _ = decode_scores(fields, ordered, [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0]])
    assert values == {"department": "account_access", "ok": True}
