"""Frozen-input gates, evaluation and reporting for the bounded two-arm pilot."""

import argparse
import gc
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.architecture_diagnostics import GPU_lock, sha, suite_metrics, write_json
from experiments.architecture_fit import tree_hashes
from experiments.contrast_scaling_training import apply_determinism, write
from experiments.intermediate_evaluation import (
    calibrated_responses,
    calibration_records,
    capture_case,
    fact_metrics,
    fit_calibration,
)
from experiments.intermediate_training import (
    CHECKPOINT,
    KINDS,
    ROOT,
    SETTINGS,
    AuxiliaryHead,
    canonical_hash,
    load_auxiliary,
    make_fingerprint,
    prepare_inputs,
)
from experiments.revised_metrics import prediction_view
from openjev.consistent_cached_inference import ConsistentCachedJudgmentEngine
from openjev.consistent_inference import canonical_score_prompts
from openjev.judgment_cli import load_judgment_engine
from openjev.judgment_model import MODEL_ID, REVISION, JudgmentEngine, frozen_digest
from openjev.judgments import assemble_response

MODELS = ('H0', 'A', 'B')
PROTOCOL = ROOT / 'reports/intermediate-supervision-v1/PROTOCOL.md'
LEGACY = ROOT / 'data/contrast-factorial-v1/legacy'
REGRESSION = ROOT / 'data/architecture-diagnostics-v1/development.json'


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def spec(path):
    return {'path': str(Path(path).absolute()), 'sha256': sha(path)}


def check_spec(value, label):
    if sha(value['path']) != value['sha256']:
        raise ValueError(f'{label} hash differs')


def verify_approval(approval, *, repo=ROOT, protocol=PROTOCOL):
    if approval.get('version') != 1 or approval.get('study') != 'intermediate-supervision-v1':
        raise ValueError('unknown approval identity')
    for key in ('data', 'original_checkpoint'):
        entry = approval[key]
        if tree_hashes(entry['path']) != entry['files']:
            raise ValueError(f'{key} tree changed')
    for kind in ('source_files', 'external_files'):
        for relative, expected in approval[kind].items():
            if sha(Path(repo) / relative) != expected:
                raise ValueError(f'{kind} changed: {relative}')
    if sha(protocol) != approval['protocol_sha256']:
        raise ValueError('protocol changed')


def verify_boundary(path):
    path = Path(path)
    if (path / 'COMPLETE').read_text().strip() != sha(path / 'checkpoint.json'):
        raise ValueError('checkpoint completion marker mismatch')
    manifest = read(path / 'checkpoint.json')
    actual = tree_hashes(path)
    actual.pop('checkpoint.json')
    actual.pop('COMPLETE')
    if manifest.get('format') != 'openjev-intermediate-v1' or actual != manifest['files']:
        raise ValueError('checkpoint file set/hash mismatch')
    if manifest['state']['update_phase'] != 'boundary':
        raise ValueError('not a complete optimizer boundary')
    return manifest


def verify_endpoint_evidence(output):
    endpoint = read(Path(output) / 'endpoint-lock.json')
    if set(endpoint['models']) != set(MODELS):
        raise ValueError('endpoint model population mismatch')
    for name, item in endpoint['models'].items():
        if tree_hashes(item['checkpoint']) != item['files']:
            raise ValueError(f'{name} locked checkpoint changed')
        if name != 'H0':
            check_spec(item['result'], 'locked training result')
            check_spec(item['boundary_manifest'], 'locked auxiliary boundary')
            verify_boundary(item['boundary'])
    check_spec(endpoint['approval'], 'approval')
    return endpoint


def validate_optimizer(optimizer, tensor_groups, recipe, step):
    groups = optimizer['param_groups']
    ids = [p for group in groups for p in group['params']]
    if (len(groups) != len(tensor_groups) or len(ids) != len(set(ids))
            or set(optimizer['state']) != set(ids) or not ids):
        raise ValueError('optimizer parameter population differs')
    for group, tensors, (lr, decay) in zip(groups, tensor_groups, recipe, strict=True):
        if len(group['params']) != len(tensors) or group['lr'] != lr or group['weight_decay'] != decay:
            raise ValueError('optimizer group population or recipe differs')
        for identifier, tensor in zip(group['params'], tensors, strict=True):
            state = optimizer['state'][identifier]
            if int(state['step']) != step:
                raise ValueError('optimizer update count differs')
            for key in ('exp_avg', 'exp_avg_sq'):
                moment = state[key]
                if moment.shape != tensor.shape or moment.dtype != tensor.dtype or not torch.isfinite(moment).all():
                    raise ValueError('optimizer moment shape/dtype/finite mismatch')


def verify_portable_core(path, tensors):
    from safetensors.torch import load_file
    path = Path(path)
    heads = load_file(str(path / 'readouts.safetensors'))
    adapters = load_file(str(path / 'adapter/adapter_model.safetensors'))
    actual = {**heads, **{'backbone.' + key.replace('.lora_A.weight', '.lora_A.default.weight')
                         .replace('.lora_B.weight', '.lora_B.default.weight'): value for key, value in adapters.items()}}
    if set(actual) != set(tensors) or any(actual[k].dtype != tensors[k].dtype or
            not torch.equal(actual[k], tensors[k]) for k in tensors):
        raise ValueError('portable core export differs from serialized training tensors')


def assessment_gate(output):
    output = Path(output)
    endpoint = read(output / 'endpoint-lock.json')
    lock = read(output / 'calibration-lock.json')
    if (set(endpoint['models']) != set(MODELS) or set(lock['fits']) != set(MODELS)
            or lock['endpoint_sha256'] != sha(output / 'endpoint-lock.json')):
        raise ValueError('assessment needs all fixed endpoints and separate fits')
    for value in lock['fits'].values():
        check_spec(value, 'calibration fit')
    return lock


def paired_effect(a, b, categories, *, draws=2000):
    if not categories or set(a) != set(b) or set(a) != set(categories):
        raise ValueError('paired family population differs')
    if any(not math.isfinite(v) for rows in (a, b) for v in rows.values()):
        raise ValueError('paired metric must be finite')
    strata = defaultdict(list)
    for key in sorted(categories):
        strata[categories[key]].append(key)
    groups = [strata[c] for c in sorted(strata)]

    def difference(groups):
        return statistics.mean(statistics.mean(b[k] - a[k] for k in group) for group in groups)

    rng = random.Random(42)
    samples = sorted(difference([[rng.choice(g) for _ in g] for g in groups]) for _ in range(draws))

    def percentile(q):
        position = q * (len(samples) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        return samples[lo] + (samples[hi] - samples[lo]) * (position - lo)

    return {'estimate': difference(groups), 'ci95': [percentile(.025), percentile(.975)],
            'families': len(categories), 'categories': len(groups), 'draws': draws, 'seed': 42,
            'definition': 'B minus A; paired family bootstrap within category; equal category means',
            'limitation': 'One-seed corpus uncertainty, not training-seed variability.'}


def suites(data):
    return {**{k: Path(data) / f'{k}-suite.json' for k in ('train', 'validation', 'calibration', 'confirmation')},
            'legacy_calibration': LEGACY / 'calibration.json', 'retention': LEGACY / 'retention.json',
            'regression': REGRESSION}


def preflight(data, output):
    from transformers import AutoTokenizer

    from experiments.intermediate_data import preflight as data_preflight

    output = Path(output) / 'preflight'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    oracle = data_preflight(data)
    if set(oracle['partitions']) != {'train', 'validation', 'calibration', 'confirmation'}:
        raise ValueError('all four data partitions required')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    engine = JudgmentEngine(None, tokenizer, max_input_tokens=1536, unit_batch_size=4)
    context = prepare_inputs(engine, data)
    counts, members = {}, {}
    for name, path in suites(data).items():
        suite, lengths = read(path), []
        members[name] = {c['id'] for c in suite['cases']}
        for case in suite['cases']:
            _, prompts, _ = engine.prepare(case['request'])
            lengths.extend(map(len, prompts))
        counts[name] = {'cases': len(suite['cases']), 'questions': sum(len(c['expected']) for c in suite['cases']),
                        'units': len(lengths), 'max_input_tokens': max(lengths), 'sha256': sha(path)}
    for i, name in enumerate(('train', 'validation', 'calibration', 'confirmation')):
        for other in ('train', 'validation', 'calibration', 'confirmation')[i + 1:]:
            if members[name] & members[other]:
                raise ValueError('fresh/training case overlap')
    if members['legacy_calibration'] & members['retention']:
        raise ValueError('legacy fit/assessment overlap')
    result = {'status': 'PASSED', 'suites': counts, 'schema_sha256': sha(Path(data) / 'feature-schema.json'),
              'schedule_sha256': canonical_hash(context['schedule']), 'families': 200, 'updates_per_arm': 200,
              'oracle_audit': oracle, 'created_utc': now()}
    write_json(output / 'schedule.json', context['schedule'])
    write_json(output / 'training-fingerprint-A.json', make_fingerprint(data, context, 'A'))
    result['schedule'] = spec(output / 'schedule.json')
    result['training_fingerprint_A'] = spec(output / 'training-fingerprint-A.json')
    with GPU_lock():
        apply_determinism(42)
        result['runtime'] = runtime_preflight(data)
    write_json(output / 'report.json', result)
    return result


def runtime_preflight(data):
    """Training-input-only numerical policy comparison, never assessment gold."""
    engine = load_engine_for({'checkpoint': str(CHECKPOINT)})
    before = frozen_digest(engine.model)
    schema = read(Path(data) / 'feature-schema.json')
    auxiliary = AuxiliaryHead(engine.model.binary.in_features, schema, device=engine.model.device).eval()
    chosen = {}
    for case in read(Path(data) / 'train-suite.json')['cases']:
        chosen.setdefault(case['category'], case)
    rows = []
    for case in chosen.values():
        plain = capture_case(engine, case)
        observed = capture_case(engine, case, auxiliary)
        if plain['logits'] != observed['logits'] or plain['response'] != observed['response']:
            raise ValueError('GPU auxiliary hook changed canonical cached final outputs')
        compiled, prompts, kinds = engine.prepare(case['request'])
        full, _ = canonical_score_prompts(engine.model, prompts, kinds, max_input_tokens=1536)
        response = assemble_response(compiled, full.cpu().tolist(), details=True)
        differences, flips = [], 0
        for qid, question in case['request']['questions'].items():
            full_answer, cached = response['answers'][qid], plain['response']['answers'][qid]
            if question['type'] == 'noul':
                differences.append(abs(full_answer['noul'] - cached['noul']))
                flips += (full_answer['noul'] > .5) != (cached['noul'] > .5)
            else:
                a, b = full_answer['probabilities'], cached['probabilities']
                differences.extend(abs(a[k] - b[k]) for k in a)
                flips += max(a, key=a.get) != max(b, key=b.get)
        rows.append({'case_id': case['id'], 'category': case['category'],
                     'hook_logit_max_difference': 0., 'full_cached_max_probability_difference': max(differences),
                     'full_cached_argmax_changes': flips, 'questions': len(case['request']['questions'])})
    if frozen_digest(engine.model) != before:
        raise ValueError('runtime preflight changed frozen body')
    return {'status': 'PASSED', 'rows': rows, 'frozen_body_sha256': before,
            'distinction': 'Hook/no-hook is strict identical. Full/cached differences are measured distinct policies, not a relaxed equivalence gate.'}


def lock_endpoints(data, output, training_root, approval_path):
    output, training_root = Path(output), Path(training_root)
    approval = read(approval_path)
    verify_approval(approval)
    preflight_report = read(output / 'preflight/report.json')
    check_spec(preflight_report['schedule'], 'preflight schedule')
    check_spec(preflight_report['training_fingerprint_A'], 'preflight training fingerprint')
    expected_fp = read(preflight_report['training_fingerprint_A']['path'])
    expected_common = {k: v for k, v in expected_fp.items() if k != 'arm'}
    schedule = read(preflight_report['schedule']['path'])
    expected_exposure = [{'family_id': row['family_id'], 'replay_ids': [r['id'] for r in row['replay']]}
                         for row in schedule]
    common, states = None, {}
    models = {'H0': {'checkpoint': str(CHECKPOINT), 'files': tree_hashes(CHECKPOINT), 'boundary': None}}
    start_core = None
    for arm in ('A', 'B'):
        root = training_root / arm
        result = read(root / 'result.json')
        if ((root / 'COMPLETE').read_text().strip() != sha(root / 'result.json') or result['status'] != 'COMPLETE'
                or result['completed_updates'] != 200 or not result['frozen_body_unchanged']
                or result['frozen_body_before'] != result['frozen_body_after']):
            raise ValueError('arm is not a complete unchanged-body 200-update run')
        if (result['exposure'] != expected_exposure or result['exposure_sha256'] != canonical_hash(expected_exposure)
                or [row['update'] for row in result['history']] != list(range(1, 201))
                or [row['family_id'] for row in result['history']] != [row['family_id'] for row in expected_exposure]
                or any(not row['finite_gradients'] for row in result['history'])):
            raise ValueError('endpoint history/exposure differs from approved prepared schedule')
        manifests = {step: verify_boundary(root / f'step-{step:04d}') for step in (0, 50, 100, 200)}
        fp = manifests[200]['fingerprint']
        if (fp['arm'] != arm or fp['settings'] != SETTINGS or fp['data_files'] != approval['data']['files']
                or fp['checkpoint_files'] != approval['original_checkpoint']['files']):
            raise ValueError('endpoint recipe/data/start differs')
        for relative, expected in fp['source_files'].items():
            if sha(ROOT / relative) != expected:
                raise ValueError('training source changed')
        identity = {k: v for k, v in fp.items() if k != 'arm'}
        if identity != expected_common:
            raise ValueError('training fingerprint differs from tokenizer preflight')
        if common is not None and identity != common:
            raise ValueError('arm fingerprints differ beyond auxiliary-gradient mode')
        common = identity
        for step, manifest in manifests.items():
            state = manifest['state']
            if (manifest['fingerprint'] != fp or state['arm'] != arm or state['completed_updates'] != step
                    or len(state['history']) != step or len(state['exposure']) != step
                    or state['history'] != result['history'][:step] or state['exposure'] != result['exposure'][:step]):
                raise ValueError('checkpoint/history/fingerprint prefix differs')
        payload = torch.load(root / 'step-0200/training.pt', map_location='cpu', weights_only=False)
        if payload['state'] != manifests[200]['state'] or payload['fingerprint'] != fp:
            raise ValueError('serialized state identity differs')
        core_groups = [[v for k, v in payload['core'].items() if '.lora_' in k],
                       [v for k, v in payload['core'].items() if k.startswith(('compatibility.', 'binary.'))]]
        if sum(map(len, core_groups)) != len(payload['core']):
            raise ValueError('unexpected trainable tensor population')
        validate_optimizer(payload['core_optimizer'], core_groups, [(5e-5, .01), (2.5e-5, .01)], 200)
        validate_optimizer(payload['aux_optimizer'], [list(payload['aux'].values())], [(2.5e-4, 0.)], 200)
        verify_portable_core(root / 'step-0200/core', payload['core'])
        if set(payload['rng']) != {'python', 'cpu', 'cuda'} or not payload['rng']['cuda']:
            raise ValueError('complete CPU/Python/CUDA RNG required')
        first = torch.load(root / 'step-0000/training.pt', map_location='cpu', weights_only=False)
        verify_portable_core(CHECKPOINT, first['core'])
        verify_portable_core(root / 'step-0000/core', first['core'])
        if any(torch.count_nonzero(v).item() for v in first['aux'].values()):
            raise ValueError('auxiliary readout was not initialized at zero')
        if start_core is not None and (set(start_core) != set(first['core']) or
                any(not torch.equal(start_core[k], first['core'][k]) for k in start_core)):
            raise ValueError('arms have different initial core tensors')
        start_core = first['core']
        states[arm] = result
        models[arm] = {'checkpoint': str((root / 'step-0200/core').absolute()),
                       'files': tree_hashes(root / 'step-0200/core'),
                       'boundary': str((root / 'step-0200').absolute()),
                       'boundary_manifest': spec(root / 'step-0200/checkpoint.json'),
                       'result': spec(root / 'result.json'), 'training_seconds': result['training_seconds']}
    if (states['A']['exposure'] != states['B']['exposure'] or
            states['A']['frozen_body_before'] != states['B']['frozen_body_before'] or
            len({x['family_id'] for x in states['A']['exposure']}) != 200):
        raise ValueError('arm exposure or frozen body differs')
    lock = {'models': models, 'approval': spec(approval_path), 'created_utc': now(),
            'common_fingerprint': common, 'data': str(Path(data).absolute()),
            'exposure_sha256': canonical_hash(states['A']['exposure']), 'selected_updates': 200,
            'frozen_body_sha256': states['A']['frozen_body_before'], 'selection': 'fixed before training'}
    write_json(output / 'endpoint-lock.json', lock)
    return lock


def load_engine_for(entry):
    raw = load_judgment_engine(checkpoint=Path(entry['checkpoint']), device='cuda', backend='fla',
                               unit_batch_size=4, max_input_tokens=1536, trainable=True)
    raw.model.eval()
    return ConsistentCachedJudgmentEngine(raw)


def capture_suite(engine, suite, output, *, auxiliary=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    captures = {}
    for index, case in enumerate(suite['cases']):
        capture = capture_case(engine, case, auxiliary)
        captures[case['id']] = capture
        write_json(output / f'{index:04d}.json', capture)
        write(output / 'progress.json', {'cases_complete': index + 1, 'cases_total': len(suite['cases']),
                                       'updated_utc': now()})
    write_json(output / 'captures.json', captures)
    return captures


def calibrate(data, output, name):
    output = Path(output)
    endpoint = verify_endpoint_evidence(output)
    target = output / 'calibration' / name
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    engine = load_engine_for(endpoint['models'][name])
    before = frozen_digest(engine.model)
    rows = []
    sources = {}
    for key, origin in (('calibration', 'new'), ('legacy_calibration', 'broad')):
        path = suites(data)[key]
        suite = read(path)
        captures = capture_suite(engine, suite, target / key)
        rows.extend(calibration_records(suite, captures, origin=origin))
        sources[key] = spec(path)
    fitted = fit_calibration(rows)
    if frozen_digest(engine.model) != before or before != endpoint['frozen_body_sha256']:
        raise ValueError('calibration frozen body changed')
    result = {'model': name, 'endpoint_sha256': sha(output / 'endpoint-lock.json'), 'sources': sources,
              'fits': fitted, 'records_sha256': canonical_hash(rows), 'records': len(rows),
              'execution_policy': engine.policy, 'created_utc': now()}
    write_json(target / 'fit.json', result)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def lock_calibration(output):
    output = Path(output)
    verify_endpoint_evidence(output)
    fits = {}
    for name in MODELS:
        path = output / 'calibration' / name / 'fit.json'
        fit = read(path)
        if fit['model'] != name or fit['endpoint_sha256'] != sha(output / 'endpoint-lock.json'):
            raise ValueError('calibration checkpoint misbinding')
        for value in fit['sources'].values():
            check_spec(value, 'calibration source')
        fits[name] = spec(path)
    lock = {'fits': fits, 'endpoint_sha256': sha(output / 'endpoint-lock.json'), 'created_utc': now()}
    write_json(output / 'calibration-lock.json', lock)
    return lock


def selective_metrics(suite, responses):
    rows = []
    for c in suite['cases']:
        for qid, q in c['request']['questions'].items():
            p = prediction_view(q, c['expected'][qid], responses[c['id']]['answers'][qid])
            rows.append((p['max_probability'], c['id'], qid, p['correct']))
    rows.sort(key=lambda x: (-x[0], x[1], x[2]))
    return {str(fraction): {'selected': max(1, math.ceil(fraction * len(rows))),
                           'error_rate': statistics.mean(not r[3] for r in rows[:max(1, math.ceil(fraction * len(rows)))])}
            for fraction in (.5, .75, .9, 1.)}


def unknown_predicate_metrics(suite, responses, raw):
    """Fresh Choice labels are contextual conjunctions, not a fixed Unknown class.

    Use only oracle-bound direct Noul queries of the unknown predicate. This is
    evaluation metadata, never an inference input or training-time target change.
    """
    rows = []
    for case in suite['cases']:
        rules = raw['cases'][case['id']]['program']['noul_rules']
        for index, rule in enumerate(rules):
            if rule != {'feature': 'unknown'}:
                continue
            qid = f'n{index + 1}'
            pred = prediction_view(case['request']['questions'][qid], case['expected'][qid],
                                   responses[case['id']]['answers'][qid])
            rows.append(pred)
    positive = sum(r['expected'] == 'true' for r in rows)
    tp = sum(r['expected'] == 'true' and r['correct'] for r in rows)
    predicted = sum(r['predicted'] == 'true' for r in rows)
    return {'questions': len(rows), 'gold_unknown': positive, 'predicted_unknown': predicted,
            'true_positive': tp, 'false_positive': predicted - tp,
            'false_negative': sum(r['expected'] == 'true' and r['predicted'] == 'false' for r in rows),
            'ambiguous': sum(r['predicted'] is None for r in rows),
            'recall': tp / positive if positive else None, 'precision': tp / predicted if predicted else None,
            'definition': 'Direct oracle-bound Noul query: is the operational state unknown? Not a three-way Choice classifier.'}


def assess(data, output, name):
    output = Path(output)
    cal_lock = assessment_gate(output)
    endpoint = verify_endpoint_evidence(output)
    entry = endpoint['models'][name]
    target = output / 'assessment' / name
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    engine = load_engine_for(entry)
    before = frozen_digest(engine.model)
    schema = read(Path(data) / 'feature-schema.json')
    aux = load_auxiliary(entry['boundary'], schema, device=engine.model.device) if name != 'H0' else None
    fits = read(cal_lock['fits'][name]['path'])['fits']
    temperatures = {'raw': {k: 1. for k in KINDS},
                    'global': {k: fits['global']['temperature'] for k in KINDS},
                    'per_primitive': {k: fits['per_primitive'][k]['temperature'] for k in KINDS}}
    # Actual GPU hook identity gate before bulk assessment, on training inputs only.
    train_suite = read(suites(data)['train'])
    gate_case = train_suite['cases'][0]
    gate_aux = aux if aux is not None else AuxiliaryHead(engine.model.binary.in_features, schema, device=engine.model.device).eval()
    original = capture_case(engine, gate_case)
    observed = capture_case(engine, gate_case, gate_aux)
    if original['logits'] != observed['logits'] or original['response'] != observed['response']:
        raise ValueError('auxiliary hook changed canonical serving output')
    write_json(target / 'capture-gate.json', {'passed': True, 'max_logit_difference': 0., 'case_id': gate_case['id']})
    report = {'model': name, 'endpoint_sha256': sha(output / 'endpoint-lock.json'),
              'calibration_sha256': sha(output / 'calibration-lock.json'), 'suites': {}}
    for key in ('train', 'validation', 'confirmation', 'retention', 'regression'):
        suite = train_suite if key == 'train' else read(suites(data)[key])
        captures = capture_suite(engine, suite, target / key, auxiliary=aux if key in ('train', 'validation', 'confirmation') else None)
        variants = {}
        for variant, temps in temperatures.items():
            responses = calibrated_responses(suite, captures, temps)
            variants[variant] = suite_metrics(suite, responses)
            variants[variant]['fixed_coverage_risk'] = selective_metrics(suite, responses)
            if key in ('validation', 'confirmation'):
                variants[variant]['unknown_predicate'] = unknown_predicate_metrics(
                    suite, responses, read(Path(data) / f'{key}-raw-records.json'))
        result = {'variants': variants}
        if aux is not None and key in ('train', 'validation', 'confirmation'):
            result['facts'] = fact_metrics(suite, captures, read(Path(data) / f'{key}-aux.json'), schema)
        report['suites'][key] = result
        write_json(target / key / 'metrics.json', result)
    # Small serving-only timed sample, category-balanced, no auxiliary hooks.
    chosen = {}
    for case in train_suite['cases']:
        chosen.setdefault(case['category'], case)
    timing = []
    for category, case in sorted(chosen.items()):
        engine.evaluate(case['request'], details=True)
        for _ in range(3):
            value = engine.evaluate(case['request'], details=True)
            timing.append({'category': category, 'case_id': case['id'], 'elapsed_ms': value['elapsed_ms']})
    report['serving'] = {'rows': timing, 'median_ms': statistics.median(r['elapsed_ms'] for r in timing),
                         'definition': '10 fixed training cases, one warmup and three repeats each; no auxiliary head calls'}
    if before != frozen_digest(engine.model) or before != endpoint['frozen_body_sha256']:
        raise ValueError('assessment frozen body changed')
    report['frozen_body_unchanged'] = True
    write_json(target / 'report.json', report)
    return report


def build_report(output):
    output = Path(output)
    assessment_gate(output)
    endpoint = verify_endpoint_evidence(output)
    reports = {name: read(output / 'assessment' / name / 'report.json') for name in MODELS}
    for name, report in reports.items():
        if (report['model'] != name or report['endpoint_sha256'] != sha(output / 'endpoint-lock.json')
                or report['calibration_sha256'] != sha(output / 'calibration-lock.json')
                or not report['frozen_body_unchanged']):
            raise ValueError('assessment identity/body mismatch')
    effects = {}
    for split in ('train', 'validation', 'confirmation', 'regression'):
        a, b = [reports[arm]['suites'][split]['variants']['raw']['primary'] for arm in ('A', 'B')]
        if a['family_categories'] != b['family_categories']:
            raise ValueError('paired category populations differ')
        effects[split] = paired_effect(a['family_values'], b['family_values'], a['family_categories'])
    limitations = [
        'One training seed; paired intervals describe this corpus, not run-to-run training variability.',
        'Fresh scenario/language frames and confirmation score-rule pairs are withheld; shared primitive ontology remains.',
        'A/B both include six repaired source-derived families; 194 selected input/target families unchanged.',
        'Some auxiliary facts are sparse or one-sided in fresh partitions; balanced accuracy excludes one-sided features.',
        'Fact decodability and joint correctness do not prove the final readout causally uses those facts.',
        'Full-prompt training and canonical cached serving are different floating-point conventions.',
        'Train facts are measured on endpoint representations, not as a claim of unseen generalization.',
        'The local GPU is shared with desktop applications; serving timing is descriptive, not an isolated performance benchmark.',
        'Legacy development/retention populations were inspected previously. No original quarantined TEST/Jev outcomes used.',
    ]
    result = {'status': 'COMPLETE', 'created_utc': now(), 'endpoints': endpoint,
              'models': reports, 'paired_primary_effects_B_minus_A': effects, 'limitations': limitations,
              'decision': 'Report and discuss; no automatic promotion or follow-on study.'}
    result['training_diagnostics'] = {}
    for arm in ('A', 'B'):
        state = read(endpoint['models'][arm]['result']['path'])
        history = state['history']
        result['training_diagnostics'][arm] = {
            'updates': len(history), 'training_seconds': state['training_seconds'],
            'mean_update_seconds': statistics.mean(r['seconds'] for r in history),
            'core_clip_fraction': statistics.mean(r['core_clipped'] for r in history),
            'aux_clip_fraction': statistics.mean(r['aux_clipped'] for r in history),
            'last_20_mean_final_loss': statistics.mean(r['final_loss'] for r in history[-20:]),
            'last_20_mean_auxiliary_loss': statistics.mean(r['auxiliary_loss'] for r in history[-20:]),
            'all_gradients_finite': all(r['finite_gradients'] for r in history),
            'exposure_sha256': state['exposure_sha256'],
            'milestones_retained': all((Path(endpoint['models'][arm]['boundary']).parent / f'step-{step:04d}/COMPLETE').is_file()
                                        for step in (0, 50, 100, 200)),
        }
    destination = output / 'final-report'
    destination.mkdir()
    write_json(destination / 'report.json', result)
    lines = ['# Intermediate-supervision A/B pilot', '',
             'Both arms completed 200 updates from the same H0. A trains detached fact probes; B also sends fact gradients into the shared representation.', '',
             '| Model | Fresh accuracy | Pair macro | NLL | Brier | ≥90% coverage | Confident errors |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in MODELS:
        m = reports[name]['suites']['confirmation']['variants']['raw']
        o = m['overall']
        confidence = o['confidence']['0.90']
        lines.append(f"| {name} | {o['accuracy']:.3f} | {m['primary']['category_macro']:.3f} | {o.get('nll', o.get('mean_nll'))} | {o.get('brier', o.get('mean_brier'))} | {confidence['coverage']:.3f} | {confidence['wrong']} |")
    effect = effects['confirmation']
    lines += ['', f"B−A fresh pair effect: {effect['estimate']:+.3f}; paired 95% interval {effect['ci95']}.", '',
              'Full report contains raw/global/per-primitive results, facts by feature, train/validation/retention, coverage-controlled risk and serving timings.', '',
              *[f'- {item}' for item in limitations], '', result['decision'], '']
    (destination / 'RESULTS.md').write_text('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'lock', 'calibrate', 'calibration-lock', 'assess', 'report'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    approval = read(args.approval)
    verify_approval(approval)
    if args.data.absolute() != Path(approval['data']['path']):
        raise ValueError('data path differs from approval')
    if args.command in ('calibrate', 'assess') and args.model is None:
        parser.error('model required')
    if args.command == 'preflight':
        result = preflight(args.data, args.output)
    elif args.command == 'lock':
        result = lock_endpoints(args.data, args.output, args.training_root, args.approval)
    elif args.command == 'calibration-lock':
        result = lock_calibration(args.output)
    elif args.command == 'report':
        result = build_report(args.output)
    else:
        with GPU_lock():
            apply_determinism(42)
            result = (calibrate if args.command == 'calibrate' else assess)(args.data, args.output, args.model)
    verify_approval(approval)
    print(json.dumps({'command': args.command, 'status': result.get('status', 'COMPLETE'), 'model': args.model}))


if __name__ == '__main__':
    main()
