import json,pathlib,re,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=pathlib.Path(__file__).resolve().parents[2]/'data'/'kleverbench-judge-v1'; SUITE=HERE/'suite'
def prove(case):
    src=SUITE/case; lane=case.split('__')[0]
    d=HERE/'prove_ws'/case; d.mkdir(parents=True,exist_ok=True)
    for f in ('spec.k','verification.k'): shutil.copy(src/f,d/f)
    s=json.loads(subprocess.run(['kprover','session','start','--project','.','--semantics',lane],cwd=d,capture_output=True,text=True,timeout=120).stdout)
    sid,ws=s['sessionId'],pathlib.Path(s['workspaceDir'])
    for f in ('spec.k','verification.k'): shutil.copy(src/f,ws/f)
    mod=re.search(r'^module\s+(\S+)',(src/'spec.k').read_text(),re.M).group(1)
    subprocess.run(['kprover','prove','--session',sid,'--spec','spec.k','--spec-module',mod,'--verification','verification.k','--verification-module','VERIFICATION'],cwd=d,capture_output=True,text=True,timeout=1800)
    ts=sorted((d/'.kprover'/'sessions'/sid).glob('proof-*/task.json'),key=lambda p:p.stat().st_mtime)
    if not ts: ts=sorted((d/'.kprover'/'sessions'/sid).rglob('task.json'),key=lambda p:p.stat().st_mtime)
    t=json.loads(ts[-1].read_text()) if ts else {}
    r=t.get('result') or {}
    return case, r.get('outcome') or r.get('valid'), (t.get('timing') or {}).get('totalMilliseconds')
cases=['imp__abs-times__ref','imp__abs-times__vacuous','imp__abs-times__narrowed','imp-swap__abs-times__narrowed']
with ThreadPoolExecutor(max_workers=4) as ex:
    for f in as_completed([ex.submit(prove,c) for c in cases]):
        print(f.result(),flush=True)
