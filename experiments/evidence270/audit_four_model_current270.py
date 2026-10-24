"""Re-audit raw serving/quality evidence; keep missing and failed routes visible."""
import hashlib,json
from pathlib import Path
from audit_http265 import audit_result
from heldout_quality270 import score
A=Path(__file__).resolve().parent
result={'http':{},'route_reference_matches':{},'quality':{},'limits':'Historical task runtimes and isolated artifacts, not current-main acceptance. Same-route parity is not cross-engine parity; timings are not a matched speedup comparison. Missing routes remain missing.'}
for model,label,tp in [('qwen06b','qwen06b',1),('qwen4b','qwen4b',1),('koni','koni',2),('gemma','causal-gemma',1)]:
 for backend,batch in [('gpu',4),('atom',1),('atom',4)]:
  records={}
  for mode in ('native','whole','stream'):
   key=f'{model}/{backend}/b{batch}/{mode}'
   name=f'stream265-{model}-http-c16-{mode}.json' if backend=='gpu' else f'atom270-{label}-tp{tp}-b{batch}-{mode}.json'
   if (model,backend,batch,mode)==('gemma','atom',4,'stream'):name='atom270-timeline-gemma-b4-stream.json'
   p=A/name
   if not p.exists():result['http'][key]={'state':'missing','file':name};continue
   d=json.loads(p.read_text());entry={'file':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};records[mode]=d
   try:entry.update(state='passed',audit=audit_result(d,backend,mode,expected_slots=batch,expected_atom_devices=tp if backend=='atom' else 0))
   except (AssertionError,KeyError,ValueError) as e:entry.update(state='failed',reason=str(e),runtime_complete=d.get('passed'),parity_passed=d.get('parity_passed'))
   entry['failed_token_checks']=sum(not r['tokens_equal'] for b in d.get('correctness_batches',[]) for r in b['rows'])
   if name=='atom270-timeline-gemma-b4-stream.json':entry['limits']='Instrumented causal-export run; HTTP parity passes but max active lanes3, so full B4 occupancy remains unverified.'
   result['http'][key]=entry
  for a,b in [('native','whole'),('whole','stream')]:
   if a not in records or b not in records:continue
   x,y=records[a],records[b];assert x['prompts']==y['prompts']
   matches={k:v['tokens']==y['references'][k]['tokens'] for k,v in x['references'].items()}
   result['route_reference_matches'][f'{model}/{backend}/b{batch}/{a}-{b}']=matches
 fixture=json.loads((A/f'quality263-{model}-fixtures.json').read_text());cases={c['case']:c for c in fixture['cases']};assert len(cases)==36
 for route in ['native-gpu','rngd-gpu','native-atom','rngd-atom']:
  p=A/f'quality263-{model}-{route}.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()]
  assert len(rows)==len({r['case'] for r in rows})==36
  for r in rows:
   c=cases[r['case']];assert r['input_sha256']==c['input_sha256'] and 'error' not in r
   assert all(r[k]==v for k,v in score(r['text'],c['expected']).items())
  result['quality'][f'{model}/{route}']={'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cases':36,'answer_correct':sum(r['answer_correct'] for r in rows),'format_correct':sum(r['format_correct'] for r in rows)}
(A/'four-model-current270-audit.json').write_text(json.dumps(result,indent=2)+'\n')
from collections import Counter
print('HTTP',dict(Counter(r['state'] for r in result['http'].values())))
for k,v in result['http'].items():
 if v['state']!='passed':print(k,v)
for k,v in result['quality'].items():print(k,v['answer_correct'],v['format_correct'])
