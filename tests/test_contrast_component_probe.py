import copy
import importlib.util

import pytest
import torch
from test_engine import tiny_model


def action_case():
    from experiments.contrast_action_cases import build_action_suite

    return next(c for c in build_action_suite()["cases"] if c["id"] == "action/support/live_success")


def test_context_groups_preserve_all_questions_and_remove_only_declared_sources():
    from experiments.contrast_component_probe import context_groups

    case = action_case()
    before = copy.deepcopy(case)
    groups = context_groups(case["request"], filtered=True)
    by_question = {qid: group for group in groups for qid in group["questions"]}
    assert set(by_question) == set(case["request"]["questions"])
    assert sum(len(g["questions"]) for g in groups) == len(by_question)
    assert set(by_question["completion_confirmed"]["state"]) == {"request", "audit_scope", "audit_events"}
    assert by_question["completion_confirmed"]["state"]["audit_events"] == case["request"]["state"]["audit_events"]
    assert set(by_question["claimed_done"]["state"]) == {"request", "assistant_messages"}
    assert set(by_question["permitted"]["state"]) == {"request", "policy"}
    assert case == before
    controls = context_groups(case["request"], filtered=False)
    assert [set(x["questions"]) for x in controls] == [set(x["questions"]) for x in groups]
    assert all(x["state"] == case["request"]["state"] for x in controls)


def test_registry_filter_keeps_authentication_conflicts_and_effective_time():
    from experiments.contrast_bank_cases import build_bank_suite
    from experiments.contrast_component_probe import context_groups

    case = next(c for c in build_bank_suite()["cases"] if c["id"] == "payment.conflicting_records")
    groups = context_groups(case["request"], filtered=True)
    state = next(g["state"] for g in groups if "record_status" in g["questions"])
    assert set(state) == {"evaluated_at", "authority", "baseline", "current_record_snapshots"}
    assert len(state["current_record_snapshots"]) == 2
    assert state["current_record_snapshots"] == case["request"]["state"]["current_record_snapshots"]
    with pytest.raises(ValueError):
        context_groups({"state": {}, "questions": {"new_unreviewed_predicate": {"type": "noul"}}}, filtered=True)


@pytest.mark.skipif(importlib.util.find_spec("peft") is None, reason="training overlay required")
def test_component_swap_recovers_base_outputs_and_restores_heads_and_adapters_on_error():
    from experiments.contrast_component_probe import model_components, snapshot_heads
    from openjev.judgment_model import JudgmentModel

    model = JudgmentModel(tiny_model(True))
    initial = snapshot_heads(model)
    prompts, kinds = [[1, 2, 3], [1, 4, 5]], [0, 1]
    with torch.inference_mode():
        baseline = model.score_prompts(prompts, kinds)
    model.add_lora(rank=2, alpha=4, gradient_checkpointing=False)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.03)
        model.binary.bias.add_(0.7)
    model.eval()
    trained = snapshot_heads(model)
    assert initial["binary.bias"].item() == 0
    with torch.inference_mode():
        adapted = model.score_prompts(prompts, kinds)
        with pytest.raises(RuntimeError, match="probe stopped"):
            with model_components(model, initial, adapters=False):
                torch.testing.assert_close(model.score_prompts(prompts, kinds), baseline, atol=1e-6, rtol=1e-5)
                raise RuntimeError("probe stopped")
        torch.testing.assert_close(model.score_prompts(prompts, kinds), adapted, atol=0, rtol=0)
    assert model.backbone.get_model_status().enabled is True
    for name, value in snapshot_heads(model).items():
        torch.testing.assert_close(value, trained[name], atol=0, rtol=0)
