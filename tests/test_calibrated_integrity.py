from pathlib import Path
import importlib.util
import json
import pytest
from tests.test_calibrated_jev import run


def test_consumption_rejects_transitive_archive_changes(tmp_path,monkeypatch):
    assert importlib.util.find_spec('experiments.calibrated_integrity'), 'transitive validator missing'
    from experiments.calibrated_integrity import load_verified_targets
    report,sent,suite,output=run(tmp_path,monkeypatch,[{}])
    assert report['status']=='complete'
    target=output/'targets.json'
    d=load_verified_targets(target,suite)
    capture=Path(d['rows'][0]['observation_path']);body=capture.with_name('response.body')
    body.write_bytes(body.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='archive'):load_verified_targets(target,suite)


def test_consumption_rejects_probabilities_not_derived_from_archive(tmp_path,monkeypatch):
    assert importlib.util.find_spec('experiments.calibrated_integrity'), 'transitive validator missing'
    from experiments.calibrated_integrity import load_verified_targets
    report,sent,suite,output=run(tmp_path,monkeypatch,[{}])
    target=output/'targets.json';d=json.loads(target.read_text())
    d['rows'][0]['probabilities']={'false':.8,'true':.2}
    target.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='probabilities'):load_verified_targets(target,suite)
