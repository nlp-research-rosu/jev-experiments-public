"""Read-only transitive raw-observation validation before consuming teacher targets."""
import json
import math
from pathlib import Path
from experiments.calibrated_targets import load_targets, request_sha256


def _equal(a,b):
    return set(a)==set(b) and all(math.isclose(a[k],b[k],rel_tol=0,abs_tol=1e-12) for k in a)


def load_verified_targets(path,suite_path):
    data=load_targets(path,suite_path)
    cases={c['id']:c for c in json.loads(Path(suite_path).read_text())['cases']}
    model=data['teacher']['model'];cache={}
    for row in data['rows']:
        case=cases[row['case_id']];qid=row['question_id'];question=case['request']['questions'][qid]
        observation=Path(row['observation_path']).resolve()
        if model=='jev-1.13.0':
            from experiments.jev_archive import ArchiveClient
            from experiments.jev_replay import make_payload,answer_view
            key=(str(observation),row['request_sha256'])
            if key not in cache:
                client=ArchiveClient(None,observation.parent.parent)
                response=client._cached(make_payload(case,model))
                if response is None or Path(client.last_record['archive_record']).resolve()!=observation:
                    raise ValueError('Jev raw archive integrity or request binding failed')
                cache[key]=response
            dummy=False if question['type']=='noul' else 0 if question['type']=='score' else next(iter(question['criteria']))
            actual=answer_view(question,dummy,cache[key]['answers'][qid])['probabilities']
        elif model=='Qwen/Qwen3.5-4B':
            from experiments.calibrated_qwen_teacher import average_orders
            stored=json.loads(observation.read_text())
            manifest=json.loads((observation.parent.parent/'manifest.json').read_text())
            labels=['false','true'] if question['type']=='noul' else sorted(question['criteria']) if question['type']=='choice' else [str(i) for i in range(len(question['criteria']))]
            expected={'manifest_sha256':request_sha256(manifest),'case_id':case['id'],'question_id':qid,
                      'request_sha256':request_sha256(case['request']),'question_sha256':request_sha256(question),'labels':labels}
            if (stored.get('binding')!=expected or stored.get('request')!=case['request']
                or stored.get('teacher')!=data['teacher'] or manifest.get('teacher')!=data['teacher']
                or manifest.get('source_suite_sha256')!=data['source_suite_sha256']):
                raise ValueError('Qwen raw archive identity binding failed')
            orders=stored.get('orders',[])
            if len(orders)!=2:raise ValueError('Qwen raw archive orders incomplete')
            for order,name,ordered in zip(orders,['canonical','reversed'],[labels,list(reversed(labels))]):
                if (order.get('binding')!=expected or order['prepared']['order']!=name
                    or order['prepared']['labels']!=ordered
                    or json.loads((observation.parent/(name+'.json')).read_text())!=order):
                    raise ValueError('Qwen raw archive order binding failed')
            actual=average_orders(labels,orders[0]['raw_code_logits'],orders[1]['raw_code_logits'])
            if not _equal(actual,stored['probabilities']):raise ValueError('Qwen observation probabilities differ from logits')
        else:
            raise ValueError('unsupported calibrated-screen teacher identity')
        if not _equal(actual,row['probabilities']):
            raise ValueError('target probabilities differ from raw observation')
    return data
