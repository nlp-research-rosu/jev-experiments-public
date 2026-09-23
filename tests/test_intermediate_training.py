"""CPU regression checks for the intermediate pilot's scientific and resume contracts."""
import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from openjev.judgment_training import PreparedBundle, TrainingGroup

SCHEMA = {"version": 1, "features": [
    {"name": "known", "kind": "binary", "scale": 1},
    {"name": "amount", "kind": "regression", "scale": 10},
]}


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(200, 3)
        self.embed.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.tensor([.1, .2, .3]))
        self.config = SimpleNamespace(pad_token_id=0)
        self.layouts = []

    def forward(self, input_ids, attention_mask, use_cache=False):
        self.layouts.append((input_ids.detach().clone(), attention_mask.detach().clone()))
        assert use_cache is False
        return SimpleNamespace(last_hidden_state=self.embed(input_ids) + self.lora_A)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.binary = torch.nn.Linear(3, 1)
        self.compatibility = torch.nn.Linear(3, 1, bias=False)
        self.lora_settings = {"rank": 8}

    @property
    def device(self):
        return self.binary.weight.device

    def _read(self, hidden, kinds):
        return torch.where(kinds.bool(), self.binary(hidden.float()).squeeze(-1),
                           self.compatibility(hidden.float()).squeeze(-1))

    def forward(self, input_ids, attention_mask, score_positions, readout_kind):
        h = self.backbone(input_ids, attention_mask, False).last_hidden_state
        return self._read(h[torch.arange(len(input_ids)), score_positions], readout_kind)


def family():
    groups = [TrainingGroup("noul", (0,), None, {"truth": True}),
              TrainingGroup("noul", (1,), None, {"truth": False}),
              TrainingGroup("choice", (2, 3), ("a", "b"), {"choice": "b"}),
              TrainingGroup("score", (4, 5, 6), ("x", "y", "z"), {"level_index": 1})]
    return PreparedBundle("family", "new", [[i + 1] for i in range(7)], [1, 1, 0, 0, 0, 0, 0], groups, [])


def replay():
    return {kind: PreparedBundle("old/" + kind, "old", [[10 + i] for i in range(len(indices))],
                                [int(kind == "noul")] * len(indices),
                                [TrainingGroup(kind, indices, criteria, target)], [])
            for kind, indices, criteria, target in [("noul", (0,), None, {"truth": False}),
                                                    ("choice", (0, 1), ("a", "b"), {"choice": "a"}),
                                                    ("score", (0, 1), ("x", "y"), {"level_index": 1})]}


def annotations():
    return [{"targets": [1., 20.], "mask": [True, True]},
            {"targets": [0., 0.], "mask": [True, False]},
            {"targets": [1., 10.], "mask": [True, True]},
            {"targets": [0., 30.], "mask": [True, True]}]


def optimizers(model, aux):
    from experiments.judgment_pipeline import optimizer_for
    return (optimizer_for(model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5)),
            torch.optim.AdamW(aux.parameters(), lr=2.5e-4, weight_decay=0))


def test_fact_mask_scale_and_candidate_normalization():
    """Unmasked padding labels, double scaling or sum-over-candidates changes this value/gradient."""
    from experiments.intermediate_training import fact_loss
    predictions = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
    targets = torch.tensor([[1., 20.], [0., float('nan')]])
    masks = torch.tensor([[True, True], [True, False]])
    loss = fact_loss(predictions, targets, masks, SCHEMA)
    assert loss.item() == pytest.approx((1.5 + 3 * .6931471805599453) / 4)
    loss.backward()
    torch.testing.assert_close(predictions.grad, torch.tensor([[-.125, -.25], [.25, 0.]]))
    one = fact_loss(torch.zeros(1, 2), torch.tensor([[1., 20.]]), torch.ones(1, 2, dtype=torch.bool), SCHEMA)
    many = fact_loss(torch.zeros(6, 2), torch.tensor([[1., 20.]]).expand(6, -1),
                     torch.ones(6, 2, dtype=torch.bool), SCHEMA)
    assert one.item() == pytest.approx(many.item(), abs=2e-7)


def test_full_layout_hidden_matches_canonical_and_dummy_rows_add_no_gradient():
    """Selecting padding instead of final real tokens or keeping dummy outputs changes results."""
    from experiments.intermediate_training import score_full_hidden
    from openjev.consistent_inference import canonical_score_prompts
    torch.manual_seed(2)
    model = TinyModel().eval()
    prompts, kinds = [[1], [2] * 128, [3, 4]], [1, 0, 0]
    logits, hidden, stats = score_full_hidden(model, prompts, kinds)
    assert hidden.shape == (3, 3)
    assert [(x.shape[0], x.shape[1]) for x, _ in model.backbone.layouts] == [(4, 128), (4, 256)]
    assert all((~mask[:, -1]).all() for _, mask in model.backbone.layouts)
    reference, _ = canonical_score_prompts(model, prompts, kinds, max_input_tokens=1536)
    assert torch.equal(logits, reference)
    assert stats["dummy_rows"] == 5
    hidden.sum().backward()
    assert torch.equal(model.backbone.lora_A.grad, torch.full((3,), 3.))


def test_six_final_means_and_three_fact_means_match_independent_reference():
    from experiments.intermediate_training import AuxiliaryHead, backward_objective, score_full_hidden
    torch.manual_seed(3)
    bounded, direct = TinyModel(), None
    direct = copy.deepcopy(bounded)
    aux = AuxiliaryHead(3, SCHEMA)
    with torch.no_grad():
        aux.linear.weight.fill_(.13)
    reference_aux = copy.deepcopy(aux)
    f, old, labels = family(), replay(), annotations()
    logits, h, _ = score_full_hidden(direct, f.prompts, f.kinds)
    final = (F.binary_cross_entropy_with_logits(logits[:2], torch.tensor([1., 0.]))
             + F.cross_entropy(logits[2:4], torch.tensor(1))
             + F.cross_entropy(logits[4:7], torch.tensor(1))) / 6
    for kind, bundle in old.items():
        scores, _, _ = score_full_hidden(direct, bundle.prompts, bundle.kinds)
        final = final + (F.binary_cross_entropy_with_logits(scores[0], torch.tensor(0.)) if kind == 'noul'
                         else F.cross_entropy(scores, torch.tensor(0 if kind == 'choice' else 1))) / 6
    p = reference_aux(h)
    fact_q = []
    for group, label in zip(f.groups, labels, strict=True):
        binary = F.binary_cross_entropy_with_logits(p[list(group.indices), 0],
                                                   torch.full((len(group.indices),), label['targets'][0]))
        if label['mask'][1]:
            numeric = F.smooth_l1_loss(p[list(group.indices), 1],
                                      torch.full((len(group.indices),), label['targets'][1] / 10))
            fact_q.append((binary + numeric) / 2)
        else:
            fact_q.append(binary)
    expected = final + .25 * ((fact_q[0] + fact_q[1]) / 2 + fact_q[2] + fact_q[3]) / 3
    expected.backward()
    result = backward_objective(bounded, aux, f, old, labels, SCHEMA, 'B')
    assert result['loss'] == pytest.approx(expected.item())
    for actual, wanted in zip(bounded.parameters(), direct.parameters(), strict=True):
        if actual.requires_grad:
            torch.testing.assert_close(actual.grad, wanted.grad, atol=1e-7, rtol=1e-6)
    for actual, wanted in zip(aux.parameters(), reference_aux.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad, atol=1e-7, rtol=1e-6)


def test_detached_auxiliary_never_changes_core_gradient_or_core_clipping():
    from experiments.intermediate_training import AuxiliaryHead, backward_objective, separate_clip
    torch.manual_seed(4)
    original = TinyModel()
    values = []
    for arm in ('final-only', 'A', 'B'):
        model, aux = copy.deepcopy(original), AuxiliaryHead(3, SCHEMA)
        with torch.no_grad():
            aux.linear.weight.fill_(3.)
        backward_objective(model, aux, family(), replay(), annotations(), SCHEMA, arm)
        norms = separate_clip(model, aux, include_aux=arm != 'final-only')
        values.append(({n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}, norms))
    assert values[0][1]['core_gradient_norm'] == values[1][1]['core_gradient_norm']
    assert all(torch.equal(values[0][0][n], values[1][0][n]) for n in values[0][0])
    assert not torch.equal(values[0][0]['backbone.lora_A'], values[2][0]['backbone.lora_A'])


def test_invalid_late_group_rejected_before_any_backward():
    from experiments.intermediate_training import AuxiliaryHead, backward_objective
    model, aux = TinyModel(), AuxiliaryHead(3, SCHEMA)
    bad = annotations()
    bad[-1]['mask'] = [True]
    with pytest.raises(ValueError):
        backward_objective(model, aux, family(), replay(), bad, SCHEMA, 'A')
    assert all(p.grad is None for p in model.parameters())
    assert not model.backbone.layouts


def test_complete_disk_restore_reproduces_both_adam_and_rng_next_update(tmp_path):
    from experiments.intermediate_training import AuxiliaryHead, restore_boundary, save_boundary, separate_clip
    torch.manual_seed(42)
    random.seed(42)
    model, aux = TinyModel(), AuxiliaryHead(3, SCHEMA)
    core_opt, aux_opt = optimizers(model, aux)

    def step():
        core_opt.zero_grad(set_to_none=True)
        aux_opt.zero_grad(set_to_none=True)
        h = model.backbone.lora_A + torch.rand(3)
        loss = model.binary(h).square().mean() + aux(h).sub(random.random()).square().mean()
        # Both final heads participate even in this tiny stochastic test.
        loss = loss + model.compatibility(h).square().mean()
        loss.backward()
        separate_clip(model, aux)
        core_opt.step()
        aux_opt.step()
        return model.binary(h).sigmoid().detach().clone()

    step()
    state = {'completed_updates': 1, 'update_phase': 'boundary', 'history': [{'update': 1}],
             'exposure': [{'family_id': 'family', 'replay_ids': ['a', 'b', 'c']}]}
    path = tmp_path / 'step-0001'
    save_boundary(path, model, aux, core_opt, aux_opt, state, {'pin': 'fixed'})
    assert (path / 'COMPLETE').is_file()
    expected = step()
    weights = copy.deepcopy(model.state_dict()), copy.deepcopy(aux.state_dict())
    assert restore_boundary(path, model, aux, core_opt, aux_opt, {'pin': 'fixed'}) == state
    assert torch.equal(step(), expected)
    for module, saved in zip((model, aux), weights, strict=True):
        assert all(torch.equal(value, saved[name]) for name, value in module.state_dict().items())
    with pytest.raises(FileExistsError):
        save_boundary(path, model, aux, core_opt, aux_opt, state, {'pin': 'fixed'})
    with pytest.raises(ValueError, match='fingerprint'):
        restore_boundary(path, model, aux, core_opt, aux_opt, {'pin': 'changed'})
    (path / 'COMPLETE').unlink()
    with pytest.raises(ValueError, match='complete|COMPLETE'):
        restore_boundary(path, model, aux, core_opt, aux_opt, {'pin': 'fixed'})


def test_failed_checkpoint_write_preserves_evidence_and_blocks_restart(tmp_path, monkeypatch):
    from experiments.intermediate_training import AuxiliaryHead, save_boundary
    model, aux = TinyModel(), AuxiliaryHead(3, SCHEMA)
    a, b = optimizers(model, aux)
    path = tmp_path / 'step-0000'
    def interrupted(payload, destination):
        with open(destination, 'wb') as stream:
            stream.write(b'partial interrupted write')
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', interrupted)
    with pytest.raises(OSError, match='disk full'):
        save_boundary(path, model, aux, a, b, {'update_phase': 'boundary'}, {})
    pending = list(tmp_path.glob('.step-0000.pending-*'))
    assert len(pending) == 1
    assert not (pending[0] / 'COMPLETE').exists()
    assert (pending[0] / 'training.pt').read_bytes() == b'partial interrupted write'
    with pytest.raises(FileExistsError):
        save_boundary(path, model, aux, a, b, {'update_phase': 'boundary'}, {})


def tiny_population():
    bundles = {}
    for i in range(200):
        b = family()
        bundles[f'f{i}'] = PreparedBundle(f'f{i}', b.source, b.prompts, b.kinds, b.groups, [])
    old = replay()
    rows = [{'step': i, 'family_id': f'f{i}', 'replay': [
        {'primitive': k, 'id': old[k].id, 'origin': 'old', 'source': 'old'} for k in ('noul', 'choice', 'score')
    ]} for i in range(200)]
    return {'families': bundles, 'replay': {b.id: b for b in old.values()},
            'annotations': {i: annotations() for i in bundles}, 'schema': SCHEMA, 'schedule': rows}


def test_fixed_run_records_every_update_and_failure_prevents_partial_restart(tmp_path, monkeypatch):
    """An injected optimizer failure must preserve uncertain mutation and reject stale resume."""
    from experiments import intermediate_training as training
    model = TinyModel()
    engine = SimpleNamespace(model=model, synchronize=lambda: None)
    context = tiny_population()
    actual_backward = training.backward_objective
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('interrupted second family')
        return actual_backward(*args, **kwargs)
    monkeypatch.setattr(training, 'backward_objective', fail_second)
    out = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='second family'):
        training.run_training(engine, context, out, {'fixed': True}, 'A')
    state = json.loads((out / 'failure.json').read_text())
    assert state['completed_updates'] == 1
    assert [row['family_id'] for row in state['exposure']] == ['f0']
    assert len(state['history']) == 1
    assert (out / 'step-0000' / 'COMPLETE').is_file()
    assert not (out / 'COMPLETE').exists()
    with pytest.raises(ValueError, match='new|exists'):
        training.run_training(engine, context, out, {'fixed': True}, 'A')
    aux = training.AuxiliaryHead(3, SCHEMA)
    a, b = optimizers(model, aux)
    with pytest.raises(ValueError, match='stale'):
        training.restore_boundary(out / 'step-0000', model, aux, a, b, {'fixed': True})


def test_training_schedule_rejects_repeat_exposure_or_missing_replay_before_forward():
    from experiments.intermediate_training import validate_context
    context = tiny_population()
    validate_context(context)
    context['schedule'][199]['family_id'] = 'f0'
    with pytest.raises(ValueError, match='once|unique|schedule'):
        validate_context(context)


def test_cli_help_and_invalid_arm_have_no_model_side_effects(tmp_path):
    import subprocess
    import sys
    args = [sys.executable, '-m', 'experiments.intermediate_training']
    help_run = subprocess.run([*args, '--help'], capture_output=True, text=True)
    assert help_run.returncode == 0
    assert '--resume' in help_run.stdout
    invalid = subprocess.run([*args, '--mode', 'train', '--arm', 'C', '--output', str(tmp_path / 'out')],
                             capture_output=True, text=True)
    assert invalid.returncode != 0
    assert not (tmp_path / 'out').exists()


def test_both_smoke_modes_use_actual_portable_disk_reload_next_step(tmp_path, monkeypatch):
    """Reload must restore both Adam states, learned probes and core weights at the next update."""
    from experiments import intermediate_training as training
    def save_tiny(self, destination, **kwargs):
        destination.mkdir()
        torch.save(self.state_dict(), destination / 'tiny.pt')
    monkeypatch.setattr(TinyModel, 'save_checkpoint', save_tiny, raising=False)
    monkeypatch.setattr(training, 'prepare_inputs', lambda *args: tiny_population())
    monkeypatch.setattr(training, 'make_fingerprint', lambda data, context, arm: {'arm': arm})
    loaded = []
    def factory(path):
        model = TinyModel()
        if path.name == 'core':
            model.load_state_dict(torch.load(path / 'tiny.pt', weights_only=True))
            loaded.append(str(path))
        return SimpleNamespace(model=model, synchronize=lambda: None)
    for arm in ('A', 'B'):
        torch.manual_seed(42)
        random.seed(42)
        result = training.run_smoke(factory, tmp_path, tmp_path / arm, arm)
        assert result['completed_updates'] == 2
        assert result['resume']['actual_disk_reload']
        assert result['resume']['rng_equal']
        assert max(result['resume']['differences'].values()) == 0
        assert result['gradient_separation']['A_core_gradient_max_difference'] == 0
        assert result['gradient_separation']['A_clipped_core_max_difference'] == 0
        assert result['gradient_separation']['B_adapter_aux_gradient_max'] > 0
        assert result['frozen_body_unchanged']
    assert len(loaded) == 2


def test_full_200_update_endpoint_keeps_all_families_and_fixed_milestones(tmp_path):
    from experiments import intermediate_training as training
    model = TinyModel()
    output = tmp_path / 'full-run'
    result = training.run_training(SimpleNamespace(model=model, synchronize=lambda: None), tiny_population(),
                                   output, {'fixed': True}, 'A')
    assert result['completed_updates'] == 200
    assert result['status'] == 'COMPLETE'
    assert len(result['history']) == len(result['exposure']) == 200
    assert [r['family_id'] for r in result['exposure']] == [f'f{i}' for i in range(200)]
    assert sum(len(r['replay_ids']) for r in result['exposure']) == 600
    assert sorted(p.name for p in output.glob('step-*')) == ['step-0000', 'step-0050', 'step-0100', 'step-0200']
    assert (output / 'COMPLETE').is_file()
    assert result['frozen_body_unchanged']


def test_evaluation_auxiliary_load_rejects_same_width_different_feature_order(tmp_path):
    from experiments import intermediate_training as training
    model, aux = TinyModel(), training.AuxiliaryHead(3, SCHEMA)
    core, probe = optimizers(model, aux)
    path = tmp_path / 'step'
    training.save_boundary(path, model, aux, core, probe, {'update_phase': 'boundary'},
                           {'schema_sha256': training.canonical_hash(SCHEMA)})
    loaded = training.load_auxiliary(path, SCHEMA)
    assert torch.equal(loaded(torch.ones(1, 3)), aux(torch.ones(1, 3)))
    changed = copy.deepcopy(SCHEMA)
    changed['features'].reverse()
    with pytest.raises(ValueError, match='schema'):
        training.load_auxiliary(path, changed)


def test_invalid_auxiliary_optimizer_cannot_silently_change_separate_recipe():
    from experiments import intermediate_training as training
    model, aux = TinyModel(), training.AuxiliaryHead(3, SCHEMA)
    core, probe = optimizers(model, aux)
    training.check_optimizers(model, aux, core, probe)
    probe.param_groups[0]['weight_decay'] = .01
    with pytest.raises(ValueError, match='auxiliary'):
        training.check_optimizers(model, aux, core, probe)


def test_preflight_exercises_gradient_enabled_training_path_and_preserves_weights():
    from experiments.intermediate_training import same_layout_preflight
    model = TinyModel()
    original = copy.deepcopy(model.state_dict())
    observed = []
    def capture(module, args, result):
        observed.append((module.training, torch.is_grad_enabled(), result.last_hidden_state.requires_grad))
    handle = model.backbone.register_forward_hook(capture)
    result = same_layout_preflight(model, family())
    handle.remove()
    assert result['passed']
    assert (True, True, True) in observed
    assert (False, False, False) in observed
    assert all(torch.equal(p, original[n]) for n, p in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
