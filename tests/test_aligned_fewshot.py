import pytest

from experiments.aligned_fewshot import compare_execution, prepare_aligned_messages
from experiments.fewshot_stability import LAYOUTS, prepare_messages
from openjev.schema import Field

FIELDS = (
    Field("department", "Route", ("billing", "technical", "account_access", "sales")),
    Field("ok", "Is it true?", (False, True)),
)
EXAMPLES = [
    {"context": "First example", "expected": {"department": "sales", "ok": True}},
    {"context": "Second example", "expected": {"department": "billing", "ok": False}},
]


@pytest.mark.parametrize("shift,reverse", LAYOUTS)
def test_aligned_exchange_answers_only_queried_field_and_remaps_codes(shift, reverse):
    messages, fields, codes = prepare_aligned_messages("Target text", FIELDS, EXAMPLES, shift=shift, reverse=reverse)
    original, _, _ = prepare_messages("Target text", FIELDS, [], shift=shift, reverse=reverse)
    for i, (history, field, cs) in enumerate(zip(messages, fields, codes, strict=True)):
        assert [m["role"] for m in history] == ["system", "user", "assistant", "user", "assistant", "user"]
        assert history[0] == original[i][0]
        assert history[-1] == original[i][-1]
        assert sum("Target text" in m["content"] for m in history) == 1
        for j, example in enumerate(EXAMPLES):
            question, answer = history[1 + 2 * j : 3 + 2 * j]
            expected_question, _, _ = prepare_messages(example["context"], FIELDS, [], shift=shift, reverse=reverse)
            assert question == expected_question[i][-1]
            assert answer["content"] in cs
            assert len(answer["content"]) == 1
            assert field.choices[cs.index(answer["content"])] == example["expected"][field.name]


def test_zero_aligned_is_exact_original():
    assert prepare_aligned_messages("Target", FIELDS, []) == prepare_messages("Target", FIELDS, [])


def test_invalid_demonstration_labels_rejected_before_scoring():
    with pytest.raises(ValueError, match="labels"):
        prepare_aligned_messages("Target", FIELDS, [{"context": "Example", "expected": {"ok": 1}}])


def test_execution_comparison_uses_matching_identity_prompts_and_ignores_repeats():
    cases = [{"id": "a", "schema": {"ok": {"type": "boolean"}}}]
    cached = {
        "id": "a",
        "layout": "s0_forward",
        "repeat": 0,
        "prompt_sha256": "same",
        "values": {"ok": True},
        "probabilities": {"ok": [0.49, 0.51]},
    }
    serial = {**cached, "values": {"ok": False}, "probabilities": {"ok": [0.52, 0.48]}}
    result = compare_execution(cases, [cached, {**serial, "repeat": 1}], [serial])
    assert result["compared_fields"] == 1
    assert result["choice_disagreements"] == 1
    assert result["max_abs_probability_difference"] == pytest.approx(0.03)
    with pytest.raises(ValueError, match="prompts differ"):
        compare_execution(cases, [cached], [{**serial, "prompt_sha256": "different"}])
