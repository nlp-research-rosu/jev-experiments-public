import json

import pytest

from experiments.compact_fewshot import prepare_compact_messages, select_candidate
from experiments.fewshot_stability import LAYOUTS, prepare_messages
from openjev.schema import Field

FIELDS = (
    Field("route", "Where does it go?", ("billing", "technical", "account", "sales")),
    Field("call", "Call?", (False, True)),
)
EXAMPLES = [
    {"context": "Past request one", "expected": {"route": "sales", "call": True}},
    {"context": "Past request two", "expected": {"route": "billing", "call": False}},
]


@pytest.mark.parametrize("style", ["field", "shared", "field_questions"])
@pytest.mark.parametrize("shift,reverse", LAYOUTS)
def test_compact_examples_preserve_answers_and_final_question(style, shift, reverse):
    prompts, fields, codes = prepare_compact_messages(
        "New request", FIELDS, EXAMPLES, style=style, shift=shift, reverse=reverse
    )
    original, _, _ = prepare_messages("New request", FIELDS, [], shift=shift, reverse=reverse)
    for index, history in enumerate(prompts):
        assert history[-1] == original[index][-1]
        assert sum("New request" in m["content"] for m in history) == 1
        schema = json.loads(history[0]["content"].split("SCHEMA FOR EVERY REQUEST:\n")[1])
        assert {f["field"] for f in schema} == {f.name for f in fields}
        demonstrated = []
        for question, answer in zip(history[1:-1:2], history[2:-1:2], strict=True):
            assert question["role"] == "user" and answer["role"] == "assistant"
            assert len(answer["content"]) == 1
            request = question["content"].split("CONTEXT (data):\n")[1].split("\nQuestion for ")[0]
            field_name = json.loads(question["content"].split("\nQuestion for ")[1].split(":")[0])
            field_index = next(i for i, f in enumerate(fields) if f.name == field_name)
            value = fields[field_index].choices[codes[field_index].index(answer["content"])]
            expected = next(e for e in EXAMPLES if e["context"] == request)["expected"][field_name]
            assert type(value) is type(expected) and value == expected
            demonstrated.append((request, field_name))
        expected_fields = fields if style == "shared" else fields[index : index + 1]
        assert demonstrated == [(e["context"], f.name) for e in EXAMPLES for f in expected_fields]
    if style == "shared":
        assert all(p[:-1] == prompts[0][:-1] for p in prompts)


def test_empty_examples_keep_exact_existing_prompt():
    for style in ["field", "shared", "field_questions"]:
        assert prepare_compact_messages("Target", FIELDS, [], style=style) == prepare_messages("Target", FIELDS, [])


def test_full_question_variant_retains_example_question_and_options():
    prompts, _, _ = prepare_compact_messages("Target", FIELDS, EXAMPLES, style="field_questions", shift=1, reverse=True)
    for i, history in enumerate(prompts):
        for j, example in enumerate(EXAMPLES):
            original, _, _ = prepare_messages(example["context"], FIELDS, [], shift=1, reverse=True)
            suffix = original[i][-1]["content"].split("\nQuestion for ")[1]
            question = history[1 + 2 * j]["content"]
            assert question.split("\nQuestion for ")[1] == suffix
            assert "SCHEMA:" not in question


def test_bad_demo_labels_rejected():
    with pytest.raises(ValueError, match="labels"):
        prepare_compact_messages("Target", FIELDS, [{"context": "bad", "expected": {"route": "sales", "call": 1}}])


def test_selection_requires_quality_and_speed_before_choosing_fastest():
    baseline = {
        "mean_field_accuracy": 0.9,
        "stable_field_fraction": 0.9,
        "worst_layout_accuracy": 0.86,
        "per_layout": {"s0_forward": {"median_ms": 700}},
    }

    def candidate(accuracy, stable, worst, ms):
        return {
            **baseline,
            "mean_field_accuracy": accuracy,
            "stable_field_fraction": stable,
            "worst_layout_accuracy": worst,
            "per_layout": {"s0_forward": {"median_ms": ms}},
        }

    candidates = {"field_4": candidate(0.90, 0.90, 0.86, 400), "shared_4": candidate(0.80, 0.90, 0.86, 150)}
    result = select_candidate(candidates, baseline)
    assert result["selected"] == "field_4"
    assert result["eligible"] == ["field_4"]
    candidates["shared_4"] = candidate(0.895, 0.89, 0.85, 200)
    assert select_candidate(candidates, baseline)["selected"] == "shared_4"
    candidates = {"field_4": candidate(0.87, 0.9, 0.86, 400), "shared_4": candidate(0.85, 0.9, 0.86, 100)}
    result = select_candidate(candidates, baseline)
    assert result["eligible"] == []
    assert result["selected"] == "field_4"
    assert result["status"] == "exploratory_fallback"
