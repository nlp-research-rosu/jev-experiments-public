import copy
import hashlib
import json
from pathlib import Path
import pytest


def api():
    import importlib.util
    assert importlib.util.find_spec('experiments.calibrated_targets') is not None, 'target integrity module missing'
    from experiments import calibrated_targets
    return calibrated_targets


@pytest.fixture
def sample(tmp_path):
    questions={'n':{'type':'noul','instructions':'Occurred?'},'c':{'type':'choice','instructions':'Which?','criteria':{'B':'second','A':'first'}},'s':{'type':'score','instructions':'Level?','criteria':['none','all']}}
    suite={'cases':[{'id':'f/1','request':{'state':{'text':'Café'},'questions':questions},'expected':{'n':True,'c':'A','s':1}}]}
    path=tmp_path/'suite.json';path.write_text(json.dumps(suite))
    obs=tmp_path/'capture.json';obs.write_text('{"raw":true}')
    request=suite['cases'][0]['request']
    digest=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    rows=[{'case_id':'f/1','question_id':qid,'request_sha256':digest,'observation_path':str(obs),'probabilities':p} for qid,p in [('n',{'false':.2,'true':.8}),('c',{'A':.7,'B':.3}),('s',{'0':.1,'1':.9})]]
    return path,rows,obs


def test_roundtrip_exact_id_probability_and_observation_binding(sample,tmp_path):
    m=api();suite,rows,obs=sample;out=tmp_path/'targets.json'
    result=m.write_targets(suite,rows,{'model':'test-v1'},out)
    assert result['version']==1
    assert result['source_suite_sha256']==hashlib.sha256(suite.read_bytes()).hexdigest()
    loaded=m.load_targets(out,suite)
    assert loaded==result
    assert loaded['rows'][1]['probabilities']=={'A':.7,'B':.3}
    assert loaded['rows'][0]['observation_sha256']==hashlib.sha256(obs.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):m.write_targets(suite,rows,{'model':'test-v1'},out)


@pytest.mark.parametrize('change',['duplicate','missing','wrong_question','wrong_request','wrong_labels','nan','unnormalized','negative'])
def test_reject_corrupt_targets_before_publishing(change,sample,tmp_path):
    m=api();suite,rows,obs=sample;rows=copy.deepcopy(rows)
    if change=='duplicate':rows.append(rows[0])
    if change=='missing':rows.pop()
    if change=='wrong_question':rows[0]['question_id']='unknown'
    if change=='wrong_request':rows[0]['request_sha256']='0'*64
    if change=='wrong_labels':rows[0]['probabilities']={'0':.2,'1':.8}
    if change=='nan':rows[0]['probabilities']['true']=float('nan')
    if change=='unnormalized':rows[0]['probabilities']['true']=.7
    if change=='negative':rows[0]['probabilities']={'false':-.1,'true':1.1}
    out=tmp_path/'bad.json'
    with pytest.raises(ValueError):m.write_targets(suite,rows,{'model':'test-v1'},out)
    assert not out.exists()


def test_mutation_of_source_or_observation_invalidates_load(sample,tmp_path):
    m=api();suite,rows,obs=sample;out=tmp_path/'targets.json'
    m.write_targets(suite,rows,{'model':'test-v1'},out)
    obs.write_text('{"raw":false}')
    with pytest.raises(ValueError,match='observation'):m.load_targets(out,suite)
    obs.write_text('{"raw":true}')
    suite.write_text(suite.read_text()+'\n')
    with pytest.raises(ValueError,match='suite'):m.load_targets(out,suite)


def test_targets_do_not_copy_expected_labels_into_payload(sample,tmp_path):
    m=api();suite,rows,obs=sample;out=tmp_path/'targets.json'
    d=m.write_targets(suite,rows,{'model':'test-v1'},out)
    assert 'expected' not in json.dumps(d)
    assert m.labels_for({'type':'noul'})==['false','true']
    assert m.labels_for({'type':'choice','criteria':{'B':'two','A':'one'}})==['B','A']
    assert m.labels_for({'type':'score','criteria':['x','y','z']})==['0','1','2']
