"""Require completed real ATOM fault runs and independently compare retained tokens."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);models=('qwen06b','qwen4b','gemma','koni')
queue=json.loads((root/'stream265-fault-atom-queue.json').read_text())
assert len(queue)==4 and {r['model'] for r in queue}==set(models)
assert all(r['exit_code']==r['copy_exit_code']==0 for r in queue)
summary=[]
for model in models:
 d=json.loads((root/f'stream265-{model}-fault-atom.json').read_text())
 assert d['stage']=='complete' and d['width']==8192 and len(d['cases'])==3
 reuse,death,prefetch=d['cases']
 assert [x['case'] for x in d['cases']]==['cancel-reuse','producer-death','prefetch-source-death']
 assert all(x['idle'] and len(x['trigger'])==1 for x in d['cases'])
 for x in (reuse,death):
  assert x['trigger'][0]['spans']>0 and not x['trigger'][0]['admitted']
 for x,n in ((reuse,16),(prefetch,512)):
  assert x['tokens_exact'] and len(x['actual_tokens'])==n and x['actual_tokens']==x['expected_tokens']
 assert death['error'] and prefetch['error'] and not death['poisoned'] and not prefetch['poisoned']
 assert 0<prefetch['trigger'][0]['active_tokens']<512 and prefetch['trigger'][0]['victim_slot']==-1
 summary.append(dict(model=model,cases=3,survivor_tokens=512,tokens_at_source_death=prefetch['trigger'][0]['active_tokens']))
print(json.dumps(summary,indent=2))
