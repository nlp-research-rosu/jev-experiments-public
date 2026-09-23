import json
from pathlib import Path
import importlib.util
import pytest
from experiments.calibrated_targets import request_sha256, write_targets, file_sha256, load_targets


def test_teacher_gate_checks_quality_and_preserves_disagreements(tmp_path):
    assert importlib.util.find_spec('experiments.calibrated_audit'), 'teacher audit implementation missing'
    from experiments.calibrated_audit import audit_teacher
    s={'cases':[], 'relations':[]}
    for i,(kind,expected,criteria) in enumerate([('noul',True,None),('choice','A',{'A':'yes','B':'no'}),('score',1,['zero','one'])]):
        q={'type':kind,'instructions':'Judge'}
        if criteria:q['criteria']=criteria
        s['cases'].append({'id':str(i),'family_id':str(i),'category':'test','request':{'state':'evidence','questions':{'q':q}},'expected':{'q':expected}})
    suite=tmp_path/'suite.json';suite.write_text(json.dumps(s))
    obs=tmp_path/'obs';obs.write_text('raw')
    rows=[{'case_id':c['id'],'question_id':'q','request_sha256':request_sha256(c['request']),'observation_path':str(obs),'probabilities':p} for c,p in zip(s['cases'],[{'false':.1,'true':.9},{'A':.1,'B':.9},{'0':.1,'1':.9}])]
    targets=tmp_path/'targets.json';write_targets(suite,rows,{'model':'test'},targets)
    gate={'audit_suite_sha256':file_sha256(suite),'minimum_teacher_accuracy':.8,'maximum_teacher_macro_primitive_nll':2,'nll_probability_floor':1e-12,'judgments':3}
    gp=tmp_path/'gate.json';gp.write_text(json.dumps(gate))
    d=audit_teacher(targets,suite,gp,target_loader=load_targets)
    assert d['eligible'] is False
    assert d['accuracy']==pytest.approx(2/3)
    assert len(d['disagreements'])==1
    assert d['disagreements'][0]['case_id']=='1'
    assert d['by_primitive']['choice']['accuracy']==0
    gate['minimum_teacher_accuracy']=.6;gp.write_text(json.dumps(gate))
    assert audit_teacher(targets,suite,gp,target_loader=load_targets)['eligible'] is True
    gate['audit_suite_sha256']='0'*64;gp.write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='suite'):audit_teacher(targets,suite,gp,target_loader=load_targets)
