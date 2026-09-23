"""Fixed A/B intermediate-supervision pilot, with differentiable full-prompt scoring.

No confirmation inference, teacher access or historical-file mutation is performed.
Portable serving weights live in each complete checkpoint's ``core/`` directory.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import time
import uuid
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from experiments.architecture_fit import check_training_contract, tree_hashes, validate_output
from experiments.contrast_scaling_training import (
    _complete_microbatches,
    _component_groups,
    _group_loss,
    _package_versions,
    apply_determinism,
    make_family_schedule,
    prepare_canonical_replay,
    sha,
    validate_family_coverage,
    write,
)
from experiments.judgment_pipeline import optimizer_for
from openjev.judgment_model import frozen_digest
from openjev.judgment_training import load_bundles, prepare_bundle, target_vector

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'checkpoints/calibrated-screen-v1/run-v1/H0/step-0400'
REPLAY = ROOT / 'data/processed-v0.2/train.jsonl'
KINDS = ('noul', 'choice', 'score')
MILESTONES = (0, 50, 100, 200)
SETTINGS = {
    'seed': 42, 'updates': 200, 'adapter_lr': 5e-5, 'head_lr': 2.5e-5,
    'weight_decay': .01, 'core_clip': 1., 'aux_lr': 2.5e-4, 'aux_weight_decay': 0.,
    'aux_clip': 1., 'aux_lambda': .25, 'smooth_l1_beta': 1., 'max_units': 8,
    'physical_batch_size': 4, 'bucket_tokens': 128, 'max_input_tokens': 1536,
    'training_policy': 'canonical-full-v1-b4-k128', 'aux_target_units': 'raw/declared-scale-once',
    'final_objective': 'six equal new/old primitive means',
    'fact_objective': 'eligible feature mean, eligible candidate mean, question mean, three primitive mean',
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_schema(schema):
    features = schema.get('features', [])
    if not features or len({row.get('name') for row in features}) != len(features):
        raise ValueError('feature schema needs nonempty unique feature names')
    for row in features:
        if not isinstance(row.get('name'), str) or not row['name'] or row.get('kind') not in ('binary', 'regression'):
            raise ValueError('invalid auxiliary feature name or kind')
        scale = row.get('scale')
        if type(scale) not in (float, int) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('feature scales must be finite positive numbers')
        if row['kind'] == 'binary' and scale != 1:
            raise ValueError('binary features require scale one')
    return features


class AuxiliaryHead(nn.Module):
    """The trainable probe is separate from the deployable core and both core heads."""
    def __init__(self, hidden_size, schema, *, device=None):
        super().__init__()
        self.linear = nn.Linear(hidden_size, len(validate_schema(schema)), device=device, dtype=torch.float32)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.linear(hidden.float())


def score_full_hidden(model, prompts, kinds):
    """Return (scalar logits, same final-real-token hidden states, physical costs).

    This is differentiable and uses exactly the canonical full-inference layout.
    Each prompt owns its bucket. Duplicated physical rows never enter a loss.
    """
    if not prompts or len(prompts) != len(kinds) or any(type(k) is not int or k not in (0, 1) for k in kinds):
        raise ValueError('nonempty prompts and matching integer readout kinds required')
    if any(not p or len(p) > 1536 or any(type(t) is not int or t < 0 for t in p) for p in prompts):
        raise ValueError('invalid prompt tokens or input over 1536; no truncation')
    pad = getattr(model.backbone.config, 'pad_token_id', None)
    pad = 0 if pad is None else pad
    if type(pad) is not int or pad < 0:
        raise ValueError('invalid pad token')
    buckets = defaultdict(list)
    for index, prompt in enumerate(prompts):
        buckets[((len(prompt) + 128) // 128) * 128].append(index)
    scores, states = [None] * len(prompts), [None] * len(prompts)
    stats = {'input_tokens': sum(map(len, prompts)), 'model_units': len(prompts),
             'forward_calls': 0, 'physical_rows': 0, 'dummy_rows': 0, 'physical_tokens': 0}
    for bucket, indices in sorted(buckets.items()):
        for start in range(0, len(indices), 4):
            real = indices[start:start + 4]
            tile = real + [real[0]] * (4 - len(real))
            lengths = torch.tensor([len(prompts[i]) for i in tile], device=model.device)
            inputs = torch.tensor([list(prompts[i]) + [pad] * (bucket - len(prompts[i])) for i in tile],
                                  device=model.device, dtype=torch.long)
            mask = torch.arange(bucket, device=model.device)[None] < lengths[:, None]
            outputs = model.backbone(input_ids=inputs, attention_mask=mask, use_cache=False)
            hidden = outputs.last_hidden_state[torch.arange(4, device=model.device), lengths - 1]
            logits = model._read(hidden, torch.tensor([kinds[i] for i in tile], device=model.device))
            if logits.shape != (4,) or not torch.isfinite(logits).all() or not torch.isfinite(hidden).all():
                raise RuntimeError('non-finite or malformed training forward')
            for row, index in enumerate(real):
                scores[index], states[index] = logits[row], hidden[row]
            stats['forward_calls'] += 1
            stats['physical_rows'] += 4
            stats['dummy_rows'] += 4 - len(real)
            stats['physical_tokens'] += 4 * bucket
    return torch.stack(scores), torch.stack(states), stats


def fact_loss(predictions, targets, mask, schema):
    """One logical question's mean loss, independent of candidate cardinality."""
    features = validate_schema(schema)
    if predictions.ndim != 2 or predictions.shape[-1] != len(features):
        raise ValueError('auxiliary prediction shape differs from schema')
    targets = torch.as_tensor(targets, dtype=torch.float32, device=predictions.device)
    mask = torch.as_tensor(mask, device=predictions.device)
    if targets.shape != predictions.shape or mask.shape != predictions.shape or mask.dtype != torch.bool:
        raise ValueError('auxiliary target/mask shape or type mismatch')
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets[mask]).all() or not mask.any():
        raise ValueError('finite predictions/eligible targets and eligible supervision required')
    binary = torch.tensor([f['kind'] == 'binary' for f in features], device=predictions.device)
    binary_values = targets[:, binary][mask[:, binary]]
    if ((binary_values != 0) & (binary_values != 1)).any():
        raise ValueError('eligible binary targets must be zero or one')
    # Replace unknown placeholders before computing losses, avoiding NaN * zero.
    targets = torch.where(mask, targets, torch.zeros_like(targets))
    scales = predictions.new_tensor([f['scale'] for f in features])
    targets = targets / scales
    losses = torch.where(binary[None], F.binary_cross_entropy_with_logits(predictions, targets, reduction='none'),
                         F.smooth_l1_loss(predictions, targets, beta=1., reduction='none'))
    counts = mask.sum(-1)
    means = (losses * mask).sum(-1) / counts.clamp_min(1)
    return means[counts > 0].mean()


def _validate_objective(family, replay, annotations, schema, arm):
    if arm not in ('A', 'B', 'final-only'):
        raise ValueError('arm must be A or B (final-only is a smoke control)')
    features = validate_schema(schema)
    components = _component_groups(family, replay)
    if len(annotations) != len(family.groups):
        raise ValueError('one annotation is required per logical question')
    for bundle in (family, *replay.values()):
        if len(bundle.kinds) != len(bundle.prompts) or sorted(i for g in bundle.groups for i in g.indices) != list(range(len(bundle.prompts))):
            raise ValueError('complete groups must cover each prompt once')
        for group in bundle.groups:
            target_vector(group)
        list(_complete_microbatches(bundle, SETTINGS['max_units']))
        if any(not p or len(p) > 1536 for p in bundle.prompts):
            raise ValueError('prompt must fit without truncation')
    for annotation in annotations:
        targets, mask = annotation.get('targets'), annotation.get('mask')
        if not isinstance(targets, list) or not isinstance(mask, list) or len(targets) != len(features) or len(mask) != len(features):
            raise ValueError('annotation target/mask size differs from schema')
        if any(type(m) is not bool for m in mask) or not any(mask):
            raise ValueError('annotation masks must contain eligible boolean entries')
        for feature, target, enabled in zip(features, targets, mask, strict=True):
            if enabled and (type(target) not in (float, int) or not math.isfinite(target)
                            or (feature['kind'] == 'binary' and target not in (0, 1))):
                raise ValueError('invalid eligible auxiliary target')
    return components


def backward_objective(model, aux, family, replay, annotations, schema, arm):
    """Bound graph lifetime to complete question batches while preserving all means."""
    components = _validate_objective(family, replay, annotations, schema, arm)
    labels = {id(group): annotation for group, annotation in zip(family.groups, annotations, strict=True)}
    final_means, fact_means, counts = {}, {}, {}
    costs = defaultdict(int)
    for (origin, kind), members in components.items():
        bundle, count = members[0][0], len(members)
        final_values, fact_values = [], []
        for prompts, kinds, chosen in _complete_microbatches(bundle, 8, selected=[g for _, g in members]):
            logits, hidden, stats = score_full_hidden(model, prompts, kinds)
            for key, value in stats.items():
                costs[key] += value
            finals = [_group_loss(logits[list(local)], group) for group, local in chosen]
            micro = torch.stack(finals).sum() / (6 * count)
            facts = []
            if origin == 'new' and arm != 'final-only':
                predictions = aux(hidden.detach() if arm == 'A' else hidden)
                for group, local in chosen:
                    label = labels[id(group)]
                    targets = predictions.new_tensor(label['targets']).expand(len(local), -1)
                    mask = torch.tensor(label['mask'], dtype=torch.bool, device=predictions.device).expand(len(local), -1)
                    facts.append(fact_loss(predictions[list(local)], targets, mask, schema))
                micro = micro + .25 * torch.stack(facts).sum() / (3 * count)
            if not torch.isfinite(micro):
                raise RuntimeError('non-finite objective')
            micro.backward()
            final_values.extend(v.detach().item() for v in finals)
            fact_values.extend(v.detach().item() for v in facts)
            del micro, finals, facts, logits, hidden
        final_means[f'{origin}/{kind}'] = sum(final_values) / count
        counts[f'{origin}/{kind}'] = count
        if fact_values:
            fact_means[kind] = sum(fact_values) / count
    final_loss = sum(final_means.values()) / 6
    auxiliary_loss = sum(fact_means.values()) / 3
    return {'loss': final_loss + .25 * auxiliary_loss, 'final_loss': final_loss,
            'auxiliary_loss': auxiliary_loss, 'final_components': final_means,
            'auxiliary_components': fact_means, 'counts': counts, 'cost': dict(costs)}


def separate_clip(model, aux, *, include_aux=True):
    core = [p for p in model.parameters() if p.requires_grad]
    probe = list(aux.parameters()) if include_aux else []
    if set(map(id, core)) & set(map(id, probe)):
        raise ValueError('core and probe parameters overlap')
    for parameters in (core, probe):
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError('missing or non-finite gradient')
    core_norm = torch.nn.utils.clip_grad_norm_(core, 1., error_if_nonfinite=True).item()
    aux_norm = torch.nn.utils.clip_grad_norm_(probe, 1., error_if_nonfinite=True).item() if probe else 0.
    return {'core_gradient_norm': core_norm, 'aux_gradient_norm': aux_norm,
            'core_clipped': core_norm > 1., 'aux_clipped': aux_norm > 1., 'finite_gradients': True}


def check_optimizers(model, aux, core, probe):
    check_training_contract(model, core)
    if (len(probe.param_groups) != 1 or probe.param_groups[0]['lr'] != 2.5e-4
            or probe.param_groups[0]['weight_decay'] != 0
            or [id(p) for p in probe.param_groups[0]['params']] != [id(p) for p in aux.parameters()]):
        raise ValueError('auxiliary optimizer recipe or parameter ownership differs')
    if not isinstance(core, torch.optim.AdamW) or not isinstance(probe, torch.optim.AdamW):
        raise ValueError('both optimizers must be AdamW')
    for group in [*core.param_groups, *probe.param_groups]:
        if group['betas'] != (.9, .999) or group['eps'] != 1e-8 or group['amsgrad'] or group['maximize']:
            raise ValueError('AdamW recipe differs from original defaults')


def make_optimizers(model, aux):
    core = optimizer_for(model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
    check_training_contract(model, core)
    if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise ValueError('trainable core tensors must be FP32')
    probe = torch.optim.AdamW(aux.parameters(), lr=2.5e-4, weight_decay=0.)
    check_optimizers(model, aux, core, probe)
    return core, probe


def _trainable(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def _rng():
    return {'python': random.getstate(), 'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(rng):
    if rng['cuda'] and (not torch.cuda.is_available() or len(rng['cuda']) != torch.cuda.device_count()):
        raise ValueError('CUDA RNG device population differs')
    random.setstate(rng['python'])
    torch.set_rng_state(rng['cpu'])
    if rng['cuda']:
        torch.cuda.set_rng_state_all(rng['cuda'])


def save_boundary(path, model, aux, core_optimizer, aux_optimizer, state, fingerprint):
    """Atomic complete boundary; partial writes remain for inspection and block retry."""
    if state.get('update_phase') != 'boundary':
        raise ValueError('only a complete update boundary may be saved')
    path = Path(path)
    if path.exists() or list(path.parent.glob(f'.{path.name}.pending-*')):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.parent / f'.{path.name}.pending-{uuid.uuid4().hex}'
    pending.mkdir()
    payload = {'version': 1, 'fingerprint': fingerprint, 'state': copy.deepcopy(state),
               'core': _trainable(model), 'aux': {n: p.detach().cpu().clone() for n, p in aux.state_dict().items()},
               'core_optimizer': core_optimizer.state_dict(), 'aux_optimizer': aux_optimizer.state_dict(), 'rng': _rng()}
    torch.save(payload, pending / 'training.pt')
    portable = hasattr(model, 'save_checkpoint')
    if portable:
        model.save_checkpoint(pending / 'core', metadata={'study': 'intermediate-supervision-v1',
                              'arm': state.get('arm'), 'completed_updates': state.get('completed_updates'),
                              'fingerprint_sha256': canonical_hash(fingerprint)})
    manifest = {'format': 'openjev-intermediate-v1', 'fingerprint': fingerprint,
                'state': state, 'portable_core': 'core' if portable else None, 'files': tree_hashes(pending)}
    write(pending / 'checkpoint.json', manifest)
    # Marker is the final file. Failed serialization/export leaves an unresumable pending directory.
    (pending / 'COMPLETE').write_text(sha(pending / 'checkpoint.json') + '\n')
    pending.rename(path)
    return manifest


def restore_boundary(path, model, aux, core_optimizer, aux_optimizer, fingerprint):
    path = Path(path)
    if not (path / 'COMPLETE').is_file():
        raise ValueError('resume requires COMPLETE checkpoint')
    if (path / 'COMPLETE').read_text().strip() != sha(path / 'checkpoint.json'):
        raise ValueError('complete checkpoint manifest hash mismatch')
    manifest = json.loads((path / 'checkpoint.json').read_text())
    if manifest.get('format') != 'openjev-intermediate-v1' or manifest.get('fingerprint') != fingerprint:
        raise ValueError('resume fingerprint or format mismatch')
    actual = tree_hashes(path)
    actual.pop('checkpoint.json')
    actual.pop('COMPLETE')
    if actual != manifest['files']:
        raise ValueError('complete checkpoint file set/hash mismatch')
    payload = torch.load(path / 'training.pt', map_location='cpu', weights_only=False)
    if payload['fingerprint'] != fingerprint or payload['state'] != manifest['state']:
        raise ValueError('resume fingerprint/state mismatch')
    if payload['state'].get('update_phase') != 'boundary':
        raise ValueError('resume state is not at a complete boundary')
    completed = payload['state'].get('completed_updates', 0)
    for name in ('progress.json', 'failure.json', 'result.json'):
        if (path.parent / name).exists():
            latest = json.loads((path.parent / name).read_text())
            if latest.get('completed_updates', 0) > completed or latest.get('update_phase') == 'optimizer':
                raise ValueError('stale checkpoint or uncertain optimizer mutation cannot resume')
            if latest.get('status') == 'COMPLETE':
                raise ValueError('completed endpoint cannot be extended')
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(named) != set(payload['core']):
        raise ValueError('core trainable parameter names differ')
    for n, p in named.items():
        if p.shape != payload['core'][n].shape or p.dtype != payload['core'][n].dtype:
            raise ValueError('core parameter shape/dtype differs')
    if any(n not in payload['aux'] or p.shape != payload['aux'][n].shape or p.dtype != payload['aux'][n].dtype
           for n, p in aux.state_dict().items()) or set(aux.state_dict()) != set(payload['aux']):
        raise ValueError('auxiliary parameter names/shape/dtype differ')
    if payload['rng']['cuda'] and (not torch.cuda.is_available() or len(payload['rng']['cuda']) != torch.cuda.device_count()):
        raise ValueError('CUDA RNG device population differs')
    with torch.no_grad():
        for n, p in named.items():
            p.copy_(payload['core'][n])
    aux.load_state_dict(payload['aux'], strict=True)
    core_optimizer.load_state_dict(payload['core_optimizer'])
    aux_optimizer.load_state_dict(payload['aux_optimizer'])
    _restore_rng(payload['rng'])
    return payload['state']


def validate_context(context):
    families, replay, schedule = context['families'], context['replay'], context['schedule']
    if len(families) != 200 or len(schedule) != 200 or [r.get('step') for r in schedule] != list(range(200)):
        raise ValueError('exactly 200 ordered updates/families required')
    if len({r['family_id'] for r in schedule}) != 200 or set(r['family_id'] for r in schedule) != set(families):
        raise ValueError('schedule must expose every family exactly once')
    if set(context['annotations']) != set(families):
        raise ValueError('annotation family coverage differs')
    for row in schedule:
        by_kind = {r['primitive']: replay[r['id']] for r in row['replay']}
        if len(row['replay']) != 3 or set(by_kind) != set(KINDS):
            raise ValueError('schedule requires one old replay per primitive')
        _validate_objective(families[row['family_id']], by_kind, context['annotations'][row['family_id']],
                            context['schema'], 'A')
    return True


def prepare_inputs(engine, data):
    """Prepare public inputs and private targets separately; old replay never receives facts."""
    data = Path(data)
    records = load_bundles(data / 'train.jsonl')
    if len(records) != 200:
        raise ValueError('training requires exactly 200 complete families')
    validate_family_coverage(records, [r['id'] for r in records])
    categories = [r['provenance']['category'] for r in records]
    if len(set(categories)) != 10 or any(categories.count(c) != 20 for c in set(categories)):
        raise ValueError('training requires 20 families in each of ten categories')
    schema = json.loads((data / 'feature-schema.json').read_text())
    validate_schema(schema)
    labels = json.loads((data / 'train-aux.json').read_text())
    if labels['schema_sha256'] != sha(data / 'feature-schema.json'):
        raise ValueError('auxiliary schema file hash mismatch')
    suite = json.loads((data / 'train-suite.json').read_text())
    by_case = {c['id']: c for c in suite['cases']}
    if len(by_case) != 800 or set(by_case) != set(labels['cases']):
        raise ValueError('train suite and auxiliary case populations differ')
    families, annotations, members = {}, {}, set()
    for record in records:
        facts = []
        for example in record['examples']:
            case_id, qid = example['case_id'], example['question_id']
            case = by_case[case_id]
            primitive = example['question']['type']
            target_key = {'noul': 'truth', 'choice': 'choice', 'score': 'level_index'}[primitive]
            if (case['family_id'] != record['id'] or example['state'] != case['request']['state']
                    or example['question'] != case['request']['questions'][qid]
                    or example['target'] != {target_key: case['expected'][qid]}):
                raise ValueError('training raw request/target and public suite alignment differs')
            fact = labels['cases'][case_id]
            if fact.get('family_id', record['id']) != record['id']:
                raise ValueError('auxiliary family alignment differs')
            facts.append(fact)
            members.add((case_id, qid))
        families[record['id']] = prepare_bundle(engine, record)
        annotations[record['id']] = facts
    if members != {(c['id'], q) for c in by_case.values() for q in c['request']['questions']}:
        raise ValueError('training questions do not exactly cover suite')
    old, pools = prepare_canonical_replay(engine, REPLAY)
    context = {'families': families, 'annotations': annotations, 'replay': old, 'schema': schema,
               'schedule': make_family_schedule(list(families), pools, 200, seed=42)}
    validate_context(context)
    return context


def make_fingerprint(data, context, arm):
    sources = (
        'experiments/intermediate_training.py', 'experiments/intermediate_data.py',
        'experiments/architecture_fit.py', 'experiments/contrast_scaling_training.py',
        'experiments/judgment_pipeline.py', 'src/openjev/judgment_model.py',
        'src/openjev/judgment_training.py', 'src/openjev/judgment_cli.py',
        'src/openjev/judgments.py', 'src/openjev/runtime.py',
        'src/openjev/consistent_inference.py', 'src/openjev/consistent_cached_inference.py',
        'reports/intermediate-supervision-v1/PROTOCOL.md',
    )
    selected_old = {r['id'] for row in context['schedule'] for r in row['replay']}
    prepared = [*context['families'].values(), *(context['replay'][i] for i in sorted(selected_old))]
    serial = [{'id': b.id, 'prompts': b.prompts, 'kinds': b.kinds, 'groups': [vars(g) for g in b.groups]}
              for b in prepared]
    return {'study': 'intermediate-supervision-v1', 'arm': arm, 'settings': SETTINGS,
            'starting_checkpoint': str(CHECKPOINT), 'checkpoint_files': tree_hashes(CHECKPOINT),
            'data_files': tree_hashes(data), 'replay_sha256': sha(REPLAY),
            'source_files': {p: sha(ROOT / p) for p in sources},
            'prepared_sha256': canonical_hash(serial), 'schedule_sha256': canonical_hash(context['schedule']),
            'schema_sha256': canonical_hash(context['schema']),
            'runtime': _package_versions(), 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda}


def _row_inputs(context, position):
    row = context['schedule'][position]
    return (context['families'][row['family_id']],
            {r['primitive']: context['replay'][r['id']] for r in row['replay']},
            context['annotations'][row['family_id']])


def _new_state(arm, model, context):
    return {'arm': arm, 'completed_updates': 0, 'update_phase': 'boundary', 'history': [], 'exposure': [],
            'schedule_sha256': canonical_hash(context['schedule']), 'settings': SETTINGS,
            'frozen_body_before': frozen_digest(model), 'training_seconds': 0., 'status': 'RUNNING'}


def _validate_state(state, context, arm):
    n = state['completed_updates']
    if state.get('arm') != arm or state.get('settings') != SETTINGS or not 0 <= n < 200:
        raise ValueError('resume arm/settings or endpoint differs')
    if len(state['history']) != n or len(state['exposure']) != n or state['schedule_sha256'] != canonical_hash(context['schedule']):
        raise ValueError('resume history/exposure/schedule mismatch')
    for index, (history, exposure) in enumerate(zip(state['history'], state['exposure'], strict=True)):
        row = context['schedule'][index]
        if history['update'] != index + 1 or history['family_id'] != row['family_id'] or exposure != {
            'family_id': row['family_id'], 'replay_ids': [r['id'] for r in row['replay']]
        }:
            raise ValueError('resume exposure history differs from fixed schedule')


def _update(engine, aux, core_optimizer, aux_optimizer, context, state, arm, output=None):
    row = context['schedule'][state['completed_updates']]
    family, replay, annotations = _row_inputs(context, state['completed_updates'])
    core_optimizer.zero_grad(set_to_none=True)
    aux_optimizer.zero_grad(set_to_none=True)
    engine.model.train()
    aux.train()
    engine.synchronize()
    started = time.monotonic()
    state['update_phase'] = 'backward'
    if output is not None:
        write(Path(output) / 'progress.json', state)
    metrics = backward_objective(engine.model, aux, family, replay, annotations, context['schema'], arm)
    metrics.update(separate_clip(engine.model, aux))
    state['update_phase'] = 'optimizer'
    if output is not None:
        write(Path(output) / 'progress.json', state)
    core_optimizer.step()
    aux_optimizer.step()
    engine.synchronize()
    elapsed = time.monotonic() - started
    state['completed_updates'] += 1
    state['update_phase'] = 'boundary'
    state['training_seconds'] += elapsed
    state['history'].append({'update': state['completed_updates'], 'family_id': row['family_id'],
                             'seconds': elapsed, **metrics})
    state['exposure'].append({'family_id': row['family_id'], 'replay_ids': [r['id'] for r in row['replay']]})
    core_optimizer.zero_grad(set_to_none=True)
    aux_optimizer.zero_grad(set_to_none=True)
    if output is not None:
        write(Path(output) / 'progress.json', state)
    return metrics


def run_training(engine, context, output, fingerprint, arm, *, resume=None):
    """Exactly 200 successful updates; resume uses a fresh output and immutable boundary."""
    validate_output(output, CHECKPOINT)
    validate_context(context)
    if arm not in ('A', 'B'):
        raise ValueError('training arm must be A or B')
    output = Path(output)
    output.mkdir(parents=True)
    write(output / 'fingerprint.json', fingerprint)
    model = engine.model
    aux = AuxiliaryHead(model.binary.in_features, context['schema'], device=model.device)
    core, probe = make_optimizers(model, aux)
    state = _new_state(arm, model, context)
    try:
        if resume:
            before = state['frozen_body_before']
            state = restore_boundary(resume, model, aux, core, probe, fingerprint)
            _validate_state(state, context, arm)
            if state['frozen_body_before'] != before:
                raise ValueError('resume frozen body differs')
        check_optimizers(model, aux, core, probe)
        save_boundary(output / f"step-{state['completed_updates']:04d}", model, aux, core, probe, state, fingerprint)
        while state['completed_updates'] < 200:
            _update(engine, aux, core, probe, context, state, arm, output)
            completed = state['completed_updates']
            if completed in MILESTONES:
                after = frozen_digest(model)
                if after != state['frozen_body_before']:
                    raise RuntimeError('frozen body changed')
                state['frozen_body_after'] = after
                save_boundary(output / f'step-{completed:04d}', model, aux, core, probe, state, fingerprint)
        state['status'] = 'COMPLETE'
        state['frozen_body_unchanged'] = state['frozen_body_before'] == state['frozen_body_after']
        state['endpoint'] = 'step-0200'
        state['exposure_sha256'] = canonical_hash(state['exposure'])
        write(output / 'result.json', state)
        write(output / 'progress.json', state)
        (output / 'COMPLETE').write_text(sha(output / 'result.json') + '\n')
        return state
    except BaseException as error:
        state['status'] = 'FAILED'
        state['error'] = {'type': type(error).__name__, 'message': str(error), 'phase': state['update_phase']}
        write(output / 'failure.json', state)
        raise


def same_layout_preflight(model, family):
    """Compare the new forward with canonical full inference at identical physical shapes."""
    from openjev.consistent_inference import canonical_score_prompts
    maximum = 0.
    for prompts, kinds, _ in _complete_microbatches(family, 8):
        model.train()
        with torch.enable_grad():
            ours, hidden, _ = score_full_hidden(model, prompts, kinds)
        if not ours.requires_grad or not hidden.requires_grad:
            raise RuntimeError('training forward did not retain a differentiable graph')
        expected = ours.detach().cpu().clone()
        del ours, hidden
        model.eval()
        canonical, _ = canonical_score_prompts(model, prompts, kinds, max_input_tokens=1536)
        maximum = max(maximum, (expected - canonical.cpu()).abs().max().item())
    if maximum > 1e-7:
        raise RuntimeError(f'identical-layout full scoring differs: {maximum}')
    return {'passed': True, 'max_logit_difference': maximum, 'tolerance': 1e-7,
            'comparison': 'gradient-enabled training versus canonical full inference at identical B4/128 shapes; cached/full policy difference is separate'}


def _probabilities(model, bundle):
    model.eval()
    with torch.no_grad():
        logits, _, _ = score_full_hidden(model, bundle.prompts, bundle.kinds)
        return torch.cat([torch.stack((1 - logits[g.indices[0]].sigmoid(), logits[g.indices[0]].sigmoid()))
                          if g.primitive == 'noul' else logits[list(g.indices)].softmax(-1)
                          for g in bundle.groups]).cpu()


def _max_parameter_difference(first, second):
    if set(first) != set(second):
        raise ValueError('parameter identities differ during resume proof')
    return max((first[n] - second[n]).abs().max().item() for n in first)


def gradient_separation_proof(model, aux, context, position=1):
    """Check the trained head: A has exact final-only core gradients before and after clipping."""
    if not torch.count_nonzero(aux.linear.weight):
        raise ValueError('gradient probe requires a learned nonzero auxiliary head')
    family, old, labels = _row_inputs(context, position)
    rng = _rng()
    gradients, clipped, norms = {}, {}, {}
    for arm in ('final-only', 'A', 'B'):
        _restore_rng(rng)
        model.zero_grad(set_to_none=True)
        aux.zero_grad(set_to_none=True)
        model.train()
        backward_objective(model, aux, family, old, labels, context['schema'], arm)
        gradients[arm] = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        norms[arm] = separate_clip(model, aux, include_aux=arm != 'final-only')
        clipped[arm] = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    a_delta = _max_parameter_difference(gradients['A'], gradients['final-only'])
    a_clipped_delta = _max_parameter_difference(clipped['A'], clipped['final-only'])
    b_delta = max((gradients['B'][n] - gradients['final-only'][n]).abs().max().item()
                  for n in gradients['B'] if '.lora_' in n)
    model.zero_grad(set_to_none=True)
    aux.zero_grad(set_to_none=True)
    _restore_rng(rng)
    if a_delta != 0 or a_clipped_delta != 0 or norms['A']['core_gradient_norm'] != norms['final-only']['core_gradient_norm']:
        raise RuntimeError('A detached probe changed core gradients/clipping')
    if b_delta <= 0:
        raise RuntimeError('B learned auxiliary head did not affect adapter gradients')
    return {'passed': True, 'A_core_gradient_max_difference': a_delta,
            'A_clipped_core_max_difference': a_clipped_delta, 'B_adapter_aux_gradient_max': b_delta,
            'gradient_norms': norms}


def run_smoke(engine_factory, data, output, arm):
    """Two updates and actual disk reload proof, with one model resident on GPU at a time."""
    validate_output(output, CHECKPOINT)
    output = Path(output)
    output.mkdir(parents=True)
    stage = 'load'
    try:
        engine = engine_factory(CHECKPOINT)
        context = prepare_inputs(engine, data)
        fingerprint = make_fingerprint(data, context, arm)
        write(output / 'fingerprint.json', fingerprint)
        aux = AuxiliaryHead(engine.model.binary.in_features, context['schema'], device=engine.model.device)
        core, probe = make_optimizers(engine.model, aux)
        state = _new_state(arm, engine.model, context)
        first = context['families'][context['schedule'][0]['family_id']]
        layout = same_layout_preflight(engine.model, first)
        stage = 'update-one'
        _update(engine, aux, core, probe, context, state, arm)
        stage = 'gradient-separation'
        separation = gradient_separation_proof(engine.model, aux, context)
        boundary = output / 'step-0001'
        save_boundary(boundary, engine.model, aux, core, probe, state, fingerprint)
        start_probability = _probabilities(engine.model, first)
        stage = 'original-next-update'
        _update(engine, aux, core, probe, context, state, arm)
        expected_core = _trainable(engine.model)
        expected_aux = {n: p.detach().cpu().clone() for n, p in aux.state_dict().items()}
        expected_probability = _probabilities(engine.model, first)
        expected_rng = _rng()
        frozen_before = state['frozen_body_before']
        frozen_after = frozen_digest(engine.model)
        if frozen_before != frozen_after:
            raise RuntimeError('frozen body changed in original smoke updates')
        original_history = copy.deepcopy(state['history'])
        del engine, aux, core, probe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stage = 'disk-reload'
        engine = engine_factory(boundary / 'core')
        aux = AuxiliaryHead(engine.model.binary.in_features, context['schema'], device=engine.model.device)
        core, probe = make_optimizers(engine.model, aux)
        state = restore_boundary(boundary, engine.model, aux, core, probe, fingerprint)
        check_optimizers(engine.model, aux, core, probe)
        if frozen_digest(engine.model) != frozen_before:
            raise RuntimeError('disk reload frozen body differs')
        reload_probability_difference = (start_probability - _probabilities(engine.model, first)).abs().max().item()
        stage = 'restored-next-update'
        _update(engine, aux, core, probe, context, state, arm)
        differences = {'core_parameters': _max_parameter_difference(expected_core, _trainable(engine.model)),
                       'aux_parameters': _max_parameter_difference(expected_aux,
                           {n: p.detach().cpu() for n, p in aux.state_dict().items()}),
                       'probabilities': (expected_probability - _probabilities(engine.model, first)).abs().max().item(),
                       'reload_probabilities': reload_probability_difference}
        restored_rng = _rng()
        rng_equal = (expected_rng['python'] == restored_rng['python']
                     and torch.equal(expected_rng['cpu'], restored_rng['cpu'])
                     and len(expected_rng['cuda']) == len(restored_rng['cuda'])
                     and all(torch.equal(a, b) for a, b in zip(expected_rng['cuda'], restored_rng['cuda'], strict=True)))
        if max(differences.values()) > 1e-7 or not rng_equal or frozen_digest(engine.model) != frozen_before:
            raise RuntimeError(f'actual disk next-update equivalence failed: {differences}; RNG={rng_equal}')
        result = {'status': 'PASSED', 'arm': arm, 'completed_updates': 2, 'same_layout': layout,
                  'gradient_separation': separation, 'resume': {'differences': differences, 'tolerance': 1e-7,
                  'rng_equal': rng_equal, 'actual_disk_reload': True}, 'frozen_body_unchanged': True,
                  'frozen_body_before': frozen_before, 'frozen_body_after': frozen_after,
                  'original_history': original_history, 'restored_history': state['history']}
        write(output / 'result.json', result)
        (output / 'COMPLETE').write_text(sha(output / 'result.json') + '\n')
        return result
    except BaseException as error:
        write(output / 'failure.json', {'status': 'FAILED', 'stage': stage,
              'error': {'type': type(error).__name__, 'message': str(error)}})
        raise


def load_auxiliary(checkpoint, schema, *, device='cpu'):
    """Load only the trained probe for evaluation; serving core remains independent."""
    checkpoint = Path(checkpoint)
    if not (checkpoint / 'COMPLETE').is_file():
        raise ValueError('auxiliary load requires COMPLETE checkpoint')
    if (checkpoint / 'COMPLETE').read_text().strip() != sha(checkpoint / 'checkpoint.json'):
        raise ValueError('auxiliary checkpoint manifest hash mismatch')
    manifest = json.loads((checkpoint / 'checkpoint.json').read_text())
    if sha(checkpoint / 'training.pt') != manifest['files']['training.pt']:
        raise ValueError('auxiliary training state hash mismatch')
    if manifest['fingerprint'].get('schema_sha256') != canonical_hash(schema):
        raise ValueError('auxiliary feature schema identity differs')
    payload = torch.load(checkpoint / 'training.pt', map_location='cpu', weights_only=False)
    head = AuxiliaryHead(payload['aux']['linear.weight'].shape[1], schema, device=device)
    head.load_state_dict(payload['aux'], strict=True)
    return head.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'train', 'resume'), required=True)
    parser.add_argument('--arm', choices=('A', 'B'), required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/intermediate-supervision-v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if bool(args.resume) != (args.mode == 'resume'):
        parser.error('--resume is required only for mode resume')
    try:
        validate_output(args.output, CHECKPOINT)
    except ValueError as error:
        parser.error(str(error))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from experiments.architecture_diagnostics import GPU_lock
    from openjev.judgment_cli import load_judgment_engine
    def factory(checkpoint):
        return load_judgment_engine(checkpoint=checkpoint, device='cuda', backend='fla',
                                    unit_batch_size=4, max_input_tokens=1536, trainable=True)
    with GPU_lock():
        apply_determinism(42)
        original = tree_hashes(CHECKPOINT)
        try:
            if args.mode == 'smoke':
                result = run_smoke(factory, args.data, args.output, args.arm)
            else:
                engine = factory(CHECKPOINT)
                context = prepare_inputs(engine, args.data)
                fingerprint = make_fingerprint(args.data, context, args.arm)
                if args.mode == 'preflight':
                    args.output.mkdir(parents=True)
                    write(args.output / 'fingerprint.json', fingerprint)
                    first = context['families'][context['schedule'][0]['family_id']]
                    result = {'status': 'PASSED', 'families': 200, 'replay_judgments': 600,
                              'same_layout': same_layout_preflight(engine.model, first),
                              'max_input_tokens': max(len(p) for b in context['families'].values() for p in b.prompts)}
                    write(args.output / 'result.json', result)
                else:
                    result = run_training(engine, context, args.output, fingerprint, args.arm, resume=args.resume)
        finally:
            if args.output.exists():
                unchanged = original == tree_hashes(CHECKPOINT)
                write(args.output / 'original-checkpoint-integrity.json', {'unchanged': unchanged, 'before': original})
                if not unchanged:
                    raise RuntimeError('immutable original checkpoint changed')
        print(json.dumps({k: result[k] for k in ('status', 'arm', 'completed_updates') if k in result}))


if __name__ == '__main__':
    main()
