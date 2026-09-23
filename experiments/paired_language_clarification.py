"""Post-selection diagnostic variants exposing already-public sibling definitions.

This does not change the inference compiler, training data, labels, or the frozen
primary study. Corrected inputs must be reported as a post-hoc sensitivity study.
"""

import copy


def clarify_rubrics(suite, *, extra_definitions=None):
    """Copy ordered rubric definitions into shared evidence without reading gold.

    Additional audited family scope definitions may be supplied separately. The
    caller must record their provenance in the original public request text.
    """
    result = copy.deepcopy(suite)
    for case in result["cases"]:
        definitions = {}
        if case.get("category") == "ordered_rubrics":
            questions = case["request"]["questions"]
            definitions = {
                "choice_bands": copy.deepcopy(questions["c1"]["criteria"]),
                "ordered_levels_low_to_high": copy.deepcopy(questions["s1"]["criteria"]),
            }
        definitions.update(copy.deepcopy((extra_definitions or {}).get(case["family_id"], {})))
        if definitions:
            case["request"]["state"] = {
                "evidence": case["request"]["state"],
                "shared_definitions": definitions,
            }
            case["layout"] = "shared_definitions_clarified"
    return result
