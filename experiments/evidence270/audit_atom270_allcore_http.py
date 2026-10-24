"""Recompute the fixed matrix and compare recorded baseline reference tokens."""
import hashlib,json
from pathlib import Path
from audit_http265 import audit_result
A=Path(__file__).resolve().parent
p=A/'atom270-qwen4b-allcore-http.json'
d=json.loads(p.read_text())
r={'audit':audit_result(d,'atom','native',expected_slots=4,expected_atom_devices=1),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'comparisons':{},'measurements':[]}
for label in ['b1','b4']:
 b=json.loads((A/f'atom270-qwen4b-tp1-{label}-native.json').read_text())
 assert b['prompts']==d['prompts']
 r['comparisons'][label]={k:{'tokens_equal':v['tokens']==b['references'][k]['tokens'],'first_difference':next((i for i,(x,y) in enumerate(zip(v['tokens'],b['references'][k]['tokens'])) if x!=y),None)} for k,v in d['references'].items()}
log=(A/'atom270-qwen4b-allcore-http.server0.log').read_text()
for marker in ['ATOM270_BINDING ','ATOM270_COMPILED ','ATOM270_ACTIVE_LANES ']:
 rows=[json.loads(line.split(marker,1)[1]) for line in log.splitlines() if marker in line]
 assert rows,marker
 r[marker.strip()]=rows
hist=r['ATOM270_ACTIVE_LANES'][-1]
assert len(hist)==5 and hist[4]>0,hist
outer=json.loads((A/'atom270-allcore-http-outer.json').read_text())
assert outer['state']=='complete' and outer['exit_code']==outer['cleanup_exit_code']==0
for b in d['batches']:
 r['measurements'].append({'input_tokens':b['input_tokens'],'output_tokens_per_request':b['output_tokens_per_request'],**b['summary']})
r['limits']='One timed batch per condition; historical references are from other cards. No stable performance gain or general semantic-quality claim.'
(A/'atom270-allcore-http-audit.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
