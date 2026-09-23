import json

import pytest
import torch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_approval_verification_rejects_mutated_inputs(tmp_path):
    from experiments.intermediate_study import sha, tree_hashes, verify_approval
    for folder in ('data', 'checkpoint'):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'file').write_text('immutable')
    (tmp_path / 'source.py').write_text('source')
    (tmp_path / 'PROTOCOL.md').write_text('protocol')
    approval = {'version': 1, 'study': 'intermediate-supervision-v1',
                'data': {'path': str(tmp_path / 'data'), 'files': tree_hashes(tmp_path / 'data')},
                'original_checkpoint': {'path': str(tmp_path / 'checkpoint'),
                                        'files': tree_hashes(tmp_path / 'checkpoint')},
                'source_files': {'source.py': sha(tmp_path / 'source.py')},
                'external_files': {}, 'protocol_sha256': sha(tmp_path / 'PROTOCOL.md')}
    verify_approval(approval, repo=tmp_path, protocol=tmp_path / 'PROTOCOL.md')
    (tmp_path / 'data' / 'extra').write_text('silent drift')
    with pytest.raises(ValueError, match='data'):
        verify_approval(approval, repo=tmp_path, protocol=tmp_path / 'PROTOCOL.md')


def test_assessment_requires_all_model_fits_and_endpoint_binding(tmp_path):
    from experiments.intermediate_study import assessment_gate, sha
    write(tmp_path / 'endpoint-lock.json', {'models': {'H0': {}, 'A': {}, 'B': {}}})
    with pytest.raises((FileNotFoundError, ValueError)):
        assessment_gate(tmp_path)
    calibration = {'endpoint_sha256': sha(tmp_path / 'endpoint-lock.json'), 'fits': {}}
    for name in ('H0', 'A', 'B'):
        p = tmp_path / 'calibration' / name / 'fit.json'
        write(p, {'model': name})
        calibration['fits'][name] = {'path': str(p), 'sha256': sha(p)}
    write(tmp_path / 'calibration-lock.json', calibration)
    assert assessment_gate(tmp_path)['endpoint_sha256'] == calibration['endpoint_sha256']
    write(tmp_path / 'calibration' / 'B' / 'fit.json', {'model': 'B', 'changed': True})
    with pytest.raises(ValueError, match='fit'):
        assessment_gate(tmp_path)


def test_paired_intervals_keep_family_pairing_and_category_weights():
    from experiments.intermediate_study import paired_effect
    categories = {'a1': 'a', 'a2': 'a', 'b1': 'b'}
    a = {'a1': 0., 'a2': 1., 'b1': .1}
    b = {key: value + .1 for key, value in a.items()}
    out = paired_effect(a, b, categories, draws=100)
    assert out['estimate'] == pytest.approx(.1)
    assert out['ci95'] == pytest.approx([.1, .1])
    with pytest.raises(ValueError, match='population'):
        paired_effect(a, {'a1': 0.}, categories)


def test_boundary_gate_rejects_incomplete_or_mutated_checkpoint(tmp_path):
    from experiments.intermediate_study import sha, tree_hashes, verify_boundary
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_boundary(tmp_path)
    (tmp_path / 'training.pt').write_text('placeholder')
    write(tmp_path / 'checkpoint.json', {'format': 'openjev-intermediate-v1',
          'files': tree_hashes(tmp_path), 'state': {'completed_updates': 200, 'update_phase': 'boundary'}})
    (tmp_path / 'COMPLETE').write_text(sha(tmp_path / 'checkpoint.json') + '\n')
    (tmp_path / 'training.pt').write_text('corrupted')
    with pytest.raises(ValueError, match='file'):
        verify_boundary(tmp_path)


def test_evaluation_keeps_training_frozen_body_membership(monkeypatch):
    from experiments import intermediate_study as study
    calls = []
    class Model:
        def eval(self):
            calls.append('eval')
    class Raw:
        model = Model()
    def loader(**kwargs):
        assert kwargs['trainable'] is True  # LoRA excluded from frozen-body hash as during training
        return Raw()
    monkeypatch.setattr(study, 'load_judgment_engine', loader)
    monkeypatch.setattr(study, 'ConsistentCachedJudgmentEngine', lambda x: x)
    assert isinstance(study.load_engine_for({'checkpoint': 'test'}), Raw)
    assert calls == ['eval']


def test_optimizer_verifier_requires_every_parameter_and_matching_moment_shapes():
    from experiments.intermediate_study import validate_optimizer
    tensors = {'weight': torch.zeros(2, 3), 'bias': torch.zeros(2)}
    opt = {'param_groups': [{'params': [0, 1], 'lr': .1, 'weight_decay': 0.}], 'state': {
        0: {'step': torch.tensor(200), 'exp_avg': torch.zeros(2, 3), 'exp_avg_sq': torch.zeros(2, 3)},
        1: {'step': torch.tensor(200), 'exp_avg': torch.zeros(2), 'exp_avg_sq': torch.zeros(2)}}}
    validate_optimizer(opt, [list(tensors.values())], [(.1, 0.)], 200)
    del opt['state'][1]
    with pytest.raises(ValueError, match='population'):
        validate_optimizer(opt, [list(tensors.values())], [(.1, 0.)], 200)
    opt['state'][1] = {'step': torch.tensor(200), 'exp_avg': torch.zeros(3), 'exp_avg_sq': torch.zeros(3)}
    with pytest.raises(ValueError, match='moment'):
        validate_optimizer(opt, [list(tensors.values())], [(.1, 0.)], 200)


def test_portable_export_must_match_training_core_exactly(tmp_path):
    from safetensors.torch import save_file

    from experiments.intermediate_study import verify_portable_core
    adapter = torch.zeros(2, 2)
    core = {'backbone.base_model.model.layer.lora_A.default.weight': adapter,
            'binary.weight': torch.ones(1, 2), 'binary.bias': torch.zeros(1),
            'compatibility.weight': torch.zeros(1, 2)}
    (tmp_path / 'adapter').mkdir()
    save_file({'base_model.model.layer.lora_A.weight': adapter}, str(tmp_path / 'adapter/adapter_model.safetensors'))
    heads = {k: v for k, v in core.items() if not k.startswith('backbone.')}
    save_file(heads, str(tmp_path / 'readouts.safetensors'))
    verify_portable_core(tmp_path, core)
    heads['binary.bias'] = torch.ones(1)
    save_file(heads, str(tmp_path / 'readouts.safetensors'))
    with pytest.raises(ValueError, match='portable'):
        verify_portable_core(tmp_path, core)


def test_endpoint_verification_checks_locked_aux_boundary_and_result(tmp_path):
    from experiments.intermediate_study import spec, tree_hashes, verify_endpoint_evidence
    models = {}
    for name in ('H0', 'A', 'B'):
        core = tmp_path / name / 'core'
        core.mkdir(parents=True)
        (core / 'file').write_text('core')
        models[name] = {'checkpoint': str(core), 'files': tree_hashes(core)}
    approval = tmp_path / 'approval.json'
    write(approval, {})
    for name in ('A', 'B'):
        boundary = tmp_path / name / 'boundary'
        write(boundary / 'checkpoint.json', {})
        write(tmp_path / name / 'result.json', {})
        models[name].update(boundary=str(boundary), boundary_manifest=spec(boundary / 'checkpoint.json'),
                            result=spec(tmp_path / name / 'result.json'))
    write(tmp_path / 'endpoint-lock.json', {'models': models, 'approval': spec(approval)})
    write(tmp_path / 'A/result.json', {'changed': True})
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_endpoint_evidence(tmp_path)


def test_unknown_semantics_come_from_bound_predicate_not_option_name():
    from experiments.intermediate_study import unknown_predicate_metrics
    suite = {'cases': [{'id': 'one', 'request': {'questions': {'n1': {'type': 'noul'}}}, 'expected': {'n1': True}}]}
    raw = {'cases': {'one': {'program': {'noul_rules': [{'feature': 'unknown'}]}}}}
    responses = {'one': {'answers': {'n1': {'noul': .8, 'details': {'raw_logit': 1.3862943611198906}}}}}
    result = unknown_predicate_metrics(suite, responses, raw)
    assert result['questions'] == 1 and result['recall'] == 1 and result['precision'] == 1


def test_endpoint_lock_accepts_complete_matched_runs_and_rejects_exposure_drift(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from experiments import intermediate_study as s
    base = tmp_path / 'H0'
    pipeline, training, data = tmp_path / 'pipeline', tmp_path / 'training', tmp_path / 'data'
    data.mkdir()
    core = {'backbone.base_model.model.layer.lora_A.default.weight': torch.zeros(2, 2),
            'binary.weight': torch.zeros(1, 2), 'binary.bias': torch.zeros(1),
            'compatibility.weight': torch.zeros(1, 2)}

    def export(path, tensors):
        (path / 'adapter').mkdir(parents=True)
        save_file({k: v for k, v in tensors.items() if not k.startswith('backbone.')}, str(path / 'readouts.safetensors'))
        save_file({'base_model.model.layer.lora_A.weight': tensors[next(iter(core))]}, str(path / 'adapter/adapter_model.safetensors'))

    export(base, core)
    approval = {'data': {'files': {}}, 'original_checkpoint': {'files': s.tree_hashes(base)}}
    approval_path = tmp_path / 'approval.json'
    write(approval_path, approval)
    monkeypatch.setattr(s, 'CHECKPOINT', base)
    monkeypatch.setattr(s, 'ROOT', tmp_path)
    monkeypatch.setattr(s, 'verify_approval', lambda _: None)
    schedule = [{'family_id': f'f{i}', 'step': i, 'replay': [{'id': f'{k}{i}'} for k in 'ncs']} for i in range(200)]
    exposure = [{'family_id': r['family_id'], 'replay_ids': [v['id'] for v in r['replay']]} for r in schedule]
    history = [{'update': i + 1, 'family_id': r['family_id'], 'finite_gradients': True} for i, r in enumerate(schedule)]
    common = {'settings': s.SETTINGS, 'data_files': {}, 'checkpoint_files': s.tree_hashes(base), 'source_files': {}}
    write(pipeline / 'preflight/schedule.json', schedule)
    write(pipeline / 'preflight/training-fingerprint-A.json', {**common, 'arm': 'A'})
    write(pipeline / 'preflight/report.json', {'schedule': s.spec(pipeline / 'preflight/schedule.json'),
          'training_fingerprint_A': s.spec(pipeline / 'preflight/training-fingerprint-A.json')})

    def optimizer(groups, recipes, step):
        param_groups, states, identifier = [], {}, 0
        for tensors, (lr, wd) in zip(groups, recipes):
            ids = []
            for tensor in tensors:
                ids.append(identifier)
                states[identifier] = {'step': torch.tensor(step), 'exp_avg': torch.zeros_like(tensor),
                                      'exp_avg_sq': torch.zeros_like(tensor)}
                identifier += 1
            param_groups.append({'params': ids, 'lr': lr, 'weight_decay': wd})
        return {'param_groups': param_groups, 'state': states}

    for arm in ('A', 'B'):
        fp = {**common, 'arm': arm}
        for step in (0, 50, 100, 200):
            boundary = training / arm / f'step-{step:04d}'
            tensors = {k: v + step / 100 for k, v in core.items()}
            export(boundary / 'core', tensors)
            aux = {'linear.weight': torch.zeros(2, 2), 'linear.bias': torch.zeros(2)}
            state = {'arm': arm, 'completed_updates': step, 'update_phase': 'boundary',
                     'history': history[:step], 'exposure': exposure[:step],
                     'frozen_body_before': 'body', 'frozen_body_after': 'body'}
            payload = {'fingerprint': fp, 'state': state, 'core': tensors, 'aux': aux,
                'core_optimizer': optimizer([[tensors[next(iter(core))]], list(tensors.values())[1:]], [(5e-5, .01), (2.5e-5, .01)], step),
                'aux_optimizer': optimizer([list(aux.values())], [(2.5e-4, 0.)], step),
                'rng': {'cpu': torch.zeros(2), 'cuda': [torch.zeros(2)], 'python': []}}
            torch.save(payload, boundary / 'training.pt')
            write(boundary / 'checkpoint.json', {'format': 'openjev-intermediate-v1', 'state': state,
                                                'fingerprint': fp, 'files': s.tree_hashes(boundary)})
            (boundary / 'COMPLETE').write_text(s.sha(boundary / 'checkpoint.json'))
        result = {**state, 'status': 'COMPLETE', 'frozen_body_unchanged': True, 'training_seconds': 1.,
                  'exposure_sha256': s.canonical_hash(exposure)}
        write(training / arm / 'result.json', result)
        (training / arm / 'COMPLETE').write_text(s.sha(training / arm / 'result.json'))
    lock = s.lock_endpoints(data, pipeline, training, approval_path)
    assert lock['selected_updates'] == 200 and set(lock['models']) == {'H0', 'A', 'B'}
    changed = s.read(training / 'B/result.json')
    changed['exposure'][0]['replay_ids'][0] = 'different-replay'
    write(training / 'B/result.json', changed)
    (training / 'B/COMPLETE').write_text(s.sha(training / 'B/result.json'))
    with pytest.raises(ValueError, match='exposure'):
        s.lock_endpoints(data, pipeline, training, approval_path)
