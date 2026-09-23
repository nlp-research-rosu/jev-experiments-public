"""Strict, immutable teacher-target bindings for the calibrated screen."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def request_sha256(request):
    raw = json.dumps(request, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def labels_for(question):
    kind = question.get('type')
    if kind == 'noul':
        return ['false', 'true']
    criteria = question.get('criteria')
    if kind == 'choice' and isinstance(criteria, dict) and criteria and all(isinstance(x, str) for x in criteria):
        return list(criteria)
    if kind == 'score' and isinstance(criteria, list) and len(criteria) >= 2:
        return [str(i) for i in range(len(criteria))]
    raise ValueError('invalid question answer space')


def _expected(suite):
    if not isinstance(suite, dict) or not isinstance(suite.get('cases'), list) or not suite['cases']:
        raise ValueError('suite must contain cases')
    expected, case_ids = {}, set()
    for case in suite['cases']:
        cid, request = case.get('id'), case.get('request')
        if not isinstance(cid, str) or not cid or cid in case_ids:
            raise ValueError('duplicate or invalid suite case id')
        case_ids.add(cid)
        if not isinstance(request, dict) or not isinstance(request.get('questions'), dict) or not request['questions']:
            raise ValueError('invalid suite request')
        digest = request_sha256(request)
        for qid, question in request['questions'].items():
            if not isinstance(qid, str) or not qid:
                raise ValueError('invalid question id')
            expected[cid, qid] = (labels_for(question), digest)
    return expected


def _observation_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError('missing observation path')
    p = Path(value)
    p = p if p.is_absolute() else REPO_ROOT / p
    if not p.is_file():
        raise ValueError('observation file missing')
    return p


def _validate(value, suite_path, *, publishing):
    suite_path = Path(suite_path)
    if value.get('version') != 1 or value.get('source_suite_sha256') != file_sha256(suite_path):
        raise ValueError('target version/source suite mismatch')
    teacher = value.get('teacher')
    if not isinstance(teacher, dict) or not isinstance(teacher.get('model'), str) or not teacher['model']:
        raise ValueError('teacher metadata/model missing')
    expected = _expected(json.loads(suite_path.read_text()))
    rows = value.get('rows')
    if not isinstance(rows, list):
        raise ValueError('target rows must be a list')
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('target row must be an object')
        cid, qid = row.get('case_id'), row.get('question_id')
        if not isinstance(cid, str) or not isinstance(qid, str):
            raise ValueError('target case/question id invalid')
        key = cid, qid
        if key in seen or key not in expected:
            raise ValueError('duplicate or unknown target identity')
        seen.add(key)
        labels, request_hash = expected[key]
        if row.get('request_sha256') != request_hash:
            raise ValueError('target request hash mismatch')
        probabilities = row.get('probabilities')
        if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
            raise ValueError('target probability labels mismatch')
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()):
            raise ValueError('invalid target probability')
        if not math.isclose(math.fsum(probabilities.values()), 1, rel_tol=0, abs_tol=1e-6):
            raise ValueError('target probabilities not normalized')
        digest = file_sha256(_observation_path(row.get('observation_path')))
        if publishing and 'observation_sha256' not in row:
            row['observation_sha256'] = digest
        if row.get('observation_sha256') != digest:
            raise ValueError('observation hash mismatch')
    if seen != set(expected):
        raise ValueError('missing target identities')
    # Also reject non-serializable metadata and nonfinite supplemental values.
    json.dumps(value, allow_nan=False)
    return value


def write_targets(suite_path, rows, teacher, output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    value = {'version': 1, 'source_suite_sha256': file_sha256(suite_path),
             'teacher': copy.deepcopy(teacher), 'rows': copy.deepcopy(rows)}
    _validate(value, suite_path, publishing=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive temp and hard-link publication avoid partial or overwritten final artifacts.
    temp = output_path.with_name(output_path.name + f'.partial-{os.getpid()}')
    with temp.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.link(temp, output_path)
    temp.unlink()
    fd = os.open(output_path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return value


def load_targets(path, suite_path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('target artifact must be an object')
    return _validate(value, suite_path, publishing=False)
