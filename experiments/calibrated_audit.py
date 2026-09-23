"""Teacher qualification against frozen, separate development judgments."""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from experiments.calibrated_targets import file_sha256
from experiments.calibrated_integrity import load_verified_targets


def audit_teacher(targets_path, suite_path, gate_path, *, target_loader=load_verified_targets):
    targets=target_loader(targets_path,suite_path)
    suite=json.loads(Path(suite_path).read_text())
    gate=json.loads(Path(gate_path).read_text())
    if gate['audit_suite_sha256'] != file_sha256(suite_path):
        raise ValueError('audit suite differs from frozen gate')
    lookup={(r['case_id'],r['question_id']):r for r in targets['rows']}
    rows=[]
    for case in suite['cases']:
        for qid,q in case['request']['questions'].items():
            label=case['expected'][qid]
            label=str(label).lower() if q['type']=='noul' else str(label)
            p=lookup[case['id'],qid]['probabilities']
            top=max(p.values()); winners=[k for k,v in p.items() if abs(v-top)<=1e-12]
            predicted=winners[0] if len(winners)==1 else None
            rows.append({'case_id':case['id'],'question_id':qid,'primitive':q['type'],
                         'category':case.get('category','unspecified'),'expected':label,'predicted':predicted,
                         'correct':predicted==label,'probabilities':p,
                         'nll':-math.log(max(p[label],gate['nll_probability_floor'])),
                         'brier':sum((v-float(k==label))**2 for k,v in p.items())})
    if len(rows)!=gate['judgments']:raise ValueError('audit judgment count mismatch')
    def summary(items):
        return {'count':len(items),'accuracy':statistics.mean(x['correct'] for x in items),
                'nll':statistics.mean(x['nll'] for x in items),'brier':statistics.mean(x['brier'] for x in items)}
    by_primitive={k:summary([x for x in rows if x['primitive']==k]) for k in ['noul','choice','score']}
    by_category={k:summary([x for x in rows if x['category']==k]) for k in sorted({x['category'] for x in rows})}
    accuracy=statistics.mean(x['correct'] for x in rows)
    macro=statistics.mean(x['nll'] for x in by_primitive.values())
    tests={'accuracy':accuracy>=gate['minimum_teacher_accuracy'], 'macro_primitive_nll':macro<=gate['maximum_teacher_macro_primitive_nll']}
    return {'status':'complete','eligible':all(tests.values()),'checks':tests,'teacher':targets['teacher'],
            'accuracy':accuracy,'macro_primitive_nll':macro,'by_primitive':by_primitive,'by_category':by_category,
            'targets_sha256':file_sha256(targets_path),'audit_suite_sha256':file_sha256(suite_path),
            'gate_sha256':file_sha256(gate_path),'gate':gate,'disagreements':[x for x in rows if not x['correct']],
            'limitations':['Teacher audit is inspected development, not independent confirmation.','Teacher estimates are not gold distributions.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--targets',required=True);p.add_argument('--suite',required=True)
    p.add_argument('--gate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit_teacher(a.targets,a.suite,a.gate)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ['eligible','accuracy','macro_primitive_nll','checks']}))


if __name__=='__main__':main()
