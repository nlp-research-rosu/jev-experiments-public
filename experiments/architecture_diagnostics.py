"""Shared bounded architecture diagnostics; all original interfaces remain immutable."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from experiments.revised_metrics import validate_suite,prediction_view
from experiments.contrast_factorial_reporting import summarize_factorial
from openjev.judgments import compile_request,render_unit_messages,assemble_response

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/architecture-diagnostics-v1'
DATA=ROOT/'data/architecture-diagnostics-v1'
CHECKPOINT=ROOT/'checkpoints/calibrated-screen-v1/run-v1/H0/step-0400'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.pending')
    with tmp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(tmp,path);tmp.unlink()


def load_suite(path):
    suite=json.loads(Path(path).read_text());validate_suite(suite);return suite


@contextmanager
def GPU_lock():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'STUDY.lock').open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('architecture diagnostic GPU lock held') from e
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def prepare_case(engine,case,full_candidates=False):
    request=case['request'];compiled,prompts,kinds=engine.prepare(request)
    if full_candidates:
        prompts=list(prompts)
        for question in compiled.questions:
            if question.primitive=='noul':continue
            original=request['questions'][question.path[0]]
            for index in question.unit_indices:
                messages=render_unit_messages(compiled.units[index])
                messages[-1]['content']+='\nALL_CRITERIA:\n'+json.dumps(original['criteria'],sort_keys=True,separators=(',',':'),ensure_ascii=False)
                prompts[index]=engine.tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
    if any(len(p)>engine.max_input_tokens for p in prompts):raise ValueError('diagnostic prompt exceeds token limit; no truncation')
    return compiled,prompts,kinds


def response_from_logits(compiled,logits):
    return assemble_response(compiled,[float(x) for x in logits],details=True)


def suite_metrics(suite,responses):
    expected={c['id'] for c in suite['cases']}
    if set(responses)!=expected:raise ValueError('response case population mismatch')
    rows=[]
    for case in suite['cases']:
        answers=responses[case['id']]['answers']
        if set(answers)!=set(case['request']['questions']):raise ValueError('response question population mismatch')
        for qid,q in case['request']['questions'].items():
            rows.append({'case_id':case['id'],'question_id':qid,'family_id':case['family_id'],
                         'domain':case.get('domain',case.get('category','diagnostic')),'variant':case.get('variant','diagnostic'),
                         'layout':case.get('layout','diagnostic'),'prediction':prediction_view(q,case['expected'][qid],answers[qid])})
    normalized=copy.deepcopy(suite)
    for relation in normalized['relations']:
        if 'contrast_axis' in relation:
            if 'axis' in relation and relation['axis']!=relation['contrast_axis']:raise ValueError('conflicting relation axis metadata')
            relation['axis']=relation['contrast_axis']
    summary=summarize_factorial(normalized,rows)
    lookup={(r['case_id'],r['question_id']):r['prediction'] for r in rows}
    relations=[]
    for rel in suite.get('set_relations',[]):
        l=lookup[rel['left']['case_id'],rel['left']['question_id']];r=lookup[rel['right']['case_id'],rel['right']['question_id']]
        relations.append({**rel,'both_correct':l['correct'] and r['correct'],'predictions':[l['predicted'],r['predicted']]})
    summary['set_relations']={'pairs':len(relations),'both_correct':sum(x['both_correct'] for x in relations),'rows':relations}
    return summary


def extract_case_features(engine,case,*,unit_batch_size=4):
    import torch
    compiled,prompts,kinds=prepare_case(engine,case)
    model=engine.model;model.eval();features=[None]*len(prompts);scores=[None]*len(prompts)
    with torch.inference_mode():
        for q in compiled.questions:
            indices=list(q.unit_indices)
            for start in range(0,len(indices),unit_batch_size):
                selected=indices[start:start+unit_batch_size]
                inputs,mask,lengths=model._batch([prompts[i] for i in selected])
                out=model.backbone(inputs,attention_mask=mask,use_cache=False).last_hidden_state
                h=out[torch.arange(len(selected),device=out.device),lengths-1]
                z=model._read(h,torch.tensor([kinds[i] for i in selected],device=h.device))
                for row,index in enumerate(selected):features[index]=h[row].float().cpu();scores[index]=z[row].float().cpu()
    return compiled,torch.stack(features).clone(),torch.stack(scores).clone()


def _source_pins(data):
    paths=[Path(__file__),REPORT/'PROTOCOL.md',Path(data)/'manifest.json',ROOT/'src/openjev/judgments.py',ROOT/'src/openjev/judgment_model.py',ROOT/'src/openjev/runtime.py',ROOT/'experiments/architecture_heads.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def _verify_data(data):
    manifest=json.loads((Path(data)/'manifest.json').read_text())
    for spec in manifest['splits'].values():
        if sha(ROOT/spec['path'])!=spec['sha256']:raise ValueError('cohort split hash changed')
    for path,digest in manifest['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('frozen cohort source changed')
    return manifest


def _load_engine():
    from openjev.judgment_cli import load_judgment_engine
    from experiments.contrast_scaling_training import apply_determinism
    apply_determinism(42)
    engine=load_judgment_engine(checkpoint=CHECKPOINT,backend='fla',max_input_tokens=1536,unit_batch_size=4)
    engine.model.requires_grad_(False)
    return engine


def _heads(model):
    return {name:value.detach().float().cpu().clone() for name,value in {'compatibility.weight':model.compatibility.weight,'binary.weight':model.binary.weight,'binary.bias':model.binary.bias}.items()}


def _source_unchanged(pins):
    for name,digest in pins.items():
        if sha(ROOT/name)!=digest:raise ValueError('diagnostic source changed during execution')


def run_features(data,output,max_seconds=1200):
    import torch
    from torch.nn import functional as F
    from experiments.calibrated_targets import request_sha256,labels_for
    from experiments.contrast_factorial_training import tree_sha256
    from openjev.judgment_model import frozen_digest
    data,output=Path(data),Path(output);manifest=_verify_data(data);output.mkdir(parents=True,exist_ok=True)
    pins=_source_pins(data);before=tree_sha256(CHECKPOINT);start=time.monotonic()
    with GPU_lock():
        engine=_load_engine();heads=_heads(engine.model);model_before=frozen_digest(engine.model)
        for split in ['fit','calibration','development','retention']:
            cache_path=output/(split+'.pt')
            if cache_path.exists():raise FileExistsError(cache_path)
            suite_path=data/(split+'.json');suite=load_suite(suite_path);blocks=[];zs=[];groups=[];responses={};offset=0
            for case in suite['cases']:
                if time.monotonic()-start>max_seconds:raise TimeoutError('feature extraction time limit; partial observations retained')
                compiled,h,z=extract_case_features(engine,case);blocks.append(h);zs.append(z);responses[case['id']]=response_from_logits(compiled,z.tolist())
                for q in compiled.questions:
                    qid=q.path[0];question=case['request']['questions'][qid];labels=labels_for(question);gold=case['expected'][qid];gold=str(gold).lower() if q.primitive=='noul' else str(gold)
                    groups.append({'case_id':case['id'],'question_id':qid,'primitive':q.primitive,'labels':labels,'target_index':labels.index(gold),'indices':[offset+i for i in q.unit_indices],'request_sha256':request_sha256(case['request'])})
                offset+=len(h)
                write_json(output/(split+'-observations')/(request_sha256({'case_id':case['id']})+'.json'),{'case_id':case['id'],'response':responses[case['id']]})
            hidden=torch.cat(blocks);original=torch.cat(zs);kinds=torch.zeros(len(hidden),dtype=torch.bool)
            for g in groups:
                if g['primitive']=='noul':kinds[g['indices']]=True
            rebuilt=torch.where(kinds,F.linear(hidden,heads['binary.weight'],heads['binary.bias']).squeeze(-1),F.linear(hidden,heads['compatibility.weight']).squeeze(-1))
            difference=(rebuilt-original).abs().max().item()
            cache={'version':1,'checkpoint':str(CHECKPOINT),'checkpoint_id':engine.model_id,'source_suite_sha256':sha(suite_path),'hidden':hidden,'groups':groups,'heads':heads,'sourcepins':pins,'original_logits':original,'cpu_reference_logits':rebuilt,'head_reproduction_max_difference':difference}
            from experiments.architecture_heads import head_reproduction
            cache['head_reproduction']=head_reproduction(cache)
            write_json(output/(split+'-head-reproduction.json'),cache['head_reproduction'])
            if not cache['head_reproduction']['cpu_reconstruction']['passed'] or not cache['head_reproduction']['cpu_vs_gpu']['passed']:
                raise ValueError('feature readout probability/decision or numerical gate failed; no cache published')
            tmp=cache_path.with_suffix('.pending.pt');torch.save(cache,tmp);os.link(tmp,cache_path);tmp.unlink()
            write_json(output/(split+'-baseline.json'),{'responses':responses,'metrics':suite_metrics(suite,responses),'source_suite_sha256':sha(suite_path),'cache_sha256':sha(cache_path),'head_reproduction_max_difference':difference})
        after=frozen_digest(engine.model)
        if model_before!=after or tree_sha256(CHECKPOINT)!=before:raise ValueError('frozen model/checkpoint changed')
        _source_unchanged(pins)
        write_json(output/'report.json',{'status':'complete','checkpoint_tree_sha256':before,'frozen_model_before':model_before,'frozen_model_after':after,'sources':pins,'elapsed_seconds':time.monotonic()-start,'caches':{s:sha(output/(s+'.pt')) for s in ['fit','calibration','development','retention']}})


def run_context(data,output,max_seconds=1200):
    import torch
    from experiments.contrast_factorial_training import tree_sha256
    data,output=Path(data),Path(output);_verify_data(data);output.mkdir(parents=True,exist_ok=True);pins=_source_pins(data);before=tree_sha256(CHECKPOINT);started=time.monotonic();results={}
    with GPU_lock():
        engine=_load_engine()
        for split in ['development','sentinels']:
            suite=load_suite(data/(split+'.json'));parts={}
            for mode in ['isolated','full_candidates']:
                responses={};elapsed=0;tokens=0
                for case in suite['cases']:
                    if time.monotonic()-started>max_seconds:raise TimeoutError('context probe time cap; partial observations retained')
                    compiled,prompts,kinds=prepare_case(engine,case,full_candidates=mode=='full_candidates');scores=[None]*len(prompts)
                    engine.synchronize();start=time.perf_counter()
                    with torch.inference_mode():
                        for q in compiled.questions:
                            indices=list(q.unit_indices);values=engine.model.score_prompts([prompts[i] for i in indices],[kinds[i] for i in indices],unit_batch_size=4)
                            for i,z in zip(indices,values,strict=True):scores[i]=float(z)
                    engine.synchronize();elapsed+=time.perf_counter()-start;tokens+=sum(map(len,prompts))
                    response=response_from_logits(compiled,scores);responses[case['id']]=response
                    write_json(output/split/mode/(hashlib.sha256(case['id'].encode()).hexdigest()+'.json'),{'case_id':case['id'],'response':response,'prompt_lengths':[len(p) for p in prompts]})
                parts[mode]={'metrics':suite_metrics(suite,responses),'responses':responses,'model_seconds':elapsed,'input_tokens':tokens}
            results[split]=parts
        if tree_sha256(CHECKPOINT)!=before:raise ValueError('original checkpoint changed')
        _source_unchanged(pins)
        write_json(output/'report.json',{'status':'complete','checkpoint_tree_sha256':before,'checkpoint_id':engine.model_id,'sources':pins,'results':results,'elapsed_seconds':time.monotonic()-started,'interpretation':'Zero-shot input+length intervention; not a trained architecture comparison. Per-question scoring keeps binary inputs/batch shapes equal.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['context','features']);p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=float,default=1200);a=p.parse_args()
    (run_features if a.command=='features' else run_context)(a.data,a.output,a.max_seconds)


if __name__=='__main__':main()
