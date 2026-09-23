import copy
import importlib.util
import json
import pytest
from openjev.judgments import compile_request,render_unit_messages


def module():
    assert importlib.util.find_spec('experiments.architecture_diagnostics'), 'diagnostic helpers missing'
    from experiments import architecture_diagnostics
    return architecture_diagnostics


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):return list(json.dumps(messages,sort_keys=True).encode())


class Engine:
    tokenizer=Tokenizer();max_input_tokens=100000
    def prepare(self,request):
        c=compile_request(request)
        return c,[self.tokenizer.apply_chat_template(render_unit_messages(u)) for u in c.units],[u.readout_kind for u in c.units]


def case(extra=False):
    c={'a':'Numeric value 2','b':'Numeric value 3'}
    if extra:c['c']='Numeric value 4'
    return {'id':'test-extra' if extra else 'test','family_id':'f','domain':'numeric','category':'numeric','variant':'v','request':{'state':{'task':'Use supplied options'},'questions':{'q':{'type':'choice','instructions':'Select the second-largest numeric value among the supplied candidates.','criteria':c},'n':{'type':'noul','instructions':'Does state contain a task?'}}},'expected':{'q':'b' if extra else 'a','n':True},'rationale':{'q':'Program oracle','n':'Explicit key'}}


def test_isolated_set_limitation_and_full_context_visibility_without_mutation():
    m=module();engine=Engine();a=case();b=case(True);before=copy.deepcopy(a)
    c,p,k=m.prepare_case(engine,a);_,p2,_=m.prepare_case(engine,b)
    assert p[:2]==p2[:2]
    _,full,_=m.prepare_case(engine,a,full_candidates=True);_,full2,_=m.prepare_case(engine,b,full_candidates=True)
    assert full[0]!=full2[0] and full[1]!=full2[1]
    assert full[-1]==p[-1] and full2[-1]==p2[-1]
    assert a==before


def test_metrics_bind_exact_population_and_handle_set_changes_separately():
    m=module();engine=Engine();a=case();b=case(True)
    suite={'cases':[a,b],'relations':[],'set_relations':[{'id':'set-flip','left':{'case_id':a['id'],'question_id':'q'},'right':{'case_id':b['id'],'question_id':'q'},'kind':'set_flip'}]}
    responses={}
    for c,logits in [(a,[3.,0.,3.]),(b,[0.,3.,0.,3.])]:
        compiled,_,_=m.prepare_case(engine,c)
        responses[c['id']]=m.response_from_logits(compiled,logits)
    metrics=m.suite_metrics(suite,responses)
    assert metrics['overall']['correct']==4 and metrics['overall']['questions']==4
    assert metrics['set_relations']['both_correct']==1
    with pytest.raises(ValueError):m.suite_metrics(suite,{a['id']:responses[a['id']]})


def test_input_limit_rejects_instead_of_truncating():
    m=module();engine=Engine();engine.max_input_tokens=1
    with pytest.raises(ValueError,match='token'):m.prepare_case(engine,case(),full_candidates=True)


def test_feature_extraction_reproduces_head_and_last_real_endpoint():
    import torch
    from types import SimpleNamespace
    from torch import nn
    m=module();assert hasattr(m,'extract_case_features'),'feature extractor missing'
    class Body(nn.Module):
        def forward(self,inputs,attention_mask,use_cache=False):
            positions=torch.arange(inputs.shape[1],dtype=torch.float32)[None,:,None]
            hidden=inputs.float()[:,:,None].repeat(1,1,4)+positions
            return SimpleNamespace(last_hidden_state=hidden)
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=Body();self.binary=nn.Linear(4,1);self.compatibility=nn.Linear(4,1,bias=False)
        def _batch(self,prompts):
            length=max(map(len,prompts));sizes=torch.tensor(list(map(len,prompts)));x=torch.tensor([p+[0]*(length-len(p)) for p in prompts]);mask=torch.arange(length)[None]<sizes[:,None];return x,mask,sizes
        def _read(self,h,kinds):return torch.where(kinds.bool(),self.binary(h).squeeze(-1),self.compatibility(h).squeeze(-1))
    e=Engine();e.model=Model();c,h,z=m.extract_case_features(e,case(),unit_batch_size=2)
    _,prompts,kinds=e.prepare(case()['request'])
    expected=torch.tensor([float(p[-1]+len(p)-1) for p in prompts])
    assert torch.equal(h[:,0],expected)
    assert not h.requires_grad and not z.requires_grad
    assert torch.allclose(e.model._read(h,torch.tensor(kinds)),z,atol=1e-6,rtol=0)


def test_training_relation_axis_alias_is_normalized_without_mutation():
    m=module();a=case();b=copy.deepcopy(a);b['id']='test-b';b['expected']['n']=False;b['request']['state']={'task':'different evidence'}
    rel={'id':'axis-test','kind':'flip','contrast_axis':'evidence','left':{'case_id':a['id'],'question_id':'n'},'right':{'case_id':b['id'],'question_id':'n'}}
    suite={'cases':[a,b],'relations':[rel]};before=copy.deepcopy(suite);responses={}
    for c in suite['cases']:
        compiled,_,_=m.prepare_case(Engine(),c);responses[c['id']]=m.response_from_logits(compiled,[0.0]*len(compiled.units))
    metrics=m.suite_metrics(suite,responses)
    assert metrics['primary']['pairs']==1 and 'evidence' in metrics['contrast_axes']
    assert suite==before
    suite['relations'][0]['axis']='rubric'
    with pytest.raises(ValueError,match='axis'):m.suite_metrics(suite,responses)
