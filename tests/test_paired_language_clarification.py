import copy

from openjev.judgments import compile_request, render_unit_messages


def fixture():
    return {
        "contract": "self contained leaves",
        "cases": [
            {
                "id": "test/rubric/01/v0",
                "family_id": "test/rubric/01",
                "category": "ordered_rubrics",
                "layout": "prose",
                "request": {
                    "state": "Gap is 6.2 mm.",
                    "questions": {
                        "n1": {
                            "type": "noul",
                            "instructions": "Is the gap Urgent?",
                            "criteria": {"true": "Supported.", "false": "Unsupported."},
                        },
                        "c1": {
                            "type": "choice",
                            "instructions": "Choose gap band.",
                            "criteria": {"normal": "Under 5 mm.", "urgent": "At least 5 mm."},
                        },
                        "s1": {
                            "type": "score",
                            "instructions": "Rate gap severity.",
                            "criteria": ["Normal: under 5 mm.", "Urgent: at least 5 mm."],
                        },
                    },
                },
                "expected": {"n1": True, "c1": "urgent", "s1": 1},
            }
        ],
        "relations": [],
    }


def test_missing_sibling_definition_is_exposed_without_gold_or_question_changes():
    from experiments.paired_language_clarification import clarify_rubrics

    original = fixture()
    before = copy.deepcopy(original)
    rendered = str(render_unit_messages(compile_request(original["cases"][0]["request"]).units[0]))
    assert "at least 5 mm" not in rendered.lower()
    corrected = clarify_rubrics(original)
    rendered = str(render_unit_messages(compile_request(corrected["cases"][0]["request"]).units[0]))
    assert "at least 5 mm" in rendered.lower()
    assert original == before
    assert corrected["cases"][0]["expected"] == original["cases"][0]["expected"]
    assert corrected["cases"][0]["request"]["questions"] == original["cases"][0]["request"]["questions"]
    assert "expected" not in corrected["cases"][0]["request"]["state"]
    assert corrected["cases"][0]["request"]["state"]["evidence"] == "Gap is 6.2 mm."


def test_unaffected_categories_and_relations_stay_identical():
    from experiments.paired_language_clarification import clarify_rubrics

    original = fixture()
    other = copy.deepcopy(original["cases"][0])
    other.update(id="test/other/01/v0", family_id="test/other/01", category="entity_binding")
    original["cases"].append(other)
    corrected = clarify_rubrics(original)
    assert corrected["cases"][1] == original["cases"][1]
    assert corrected["relations"] == original["relations"]
