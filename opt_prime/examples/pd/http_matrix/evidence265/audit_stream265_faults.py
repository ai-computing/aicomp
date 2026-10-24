"""Audit real GPU fault-run assertions and require successful process exits."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);queue=json.loads((root/'stream265-fault-queue-v2.json').read_text())
models=('qwen06b','qwen4b','gemma','koni')
assert {r['label'] for r in queue}==set(models) and len(queue)==4
assert all(r['exit_code']==0 for r in queue)
rows=[]
for model in models:
 d=json.loads((root/f'stream265-{model}-fault-gpu-v2.json').read_text())
 assert d['stage']=='complete' and d['width']==8192 and len(d['runs'])==2
 cancel,death=d['runs']
 assert cancel['case']=='cancel-reuse' and cancel['tokens_exact'] and cancel['idle']
 assert death['case']=='producer-death' and death['idle'] and death['error']
 for case in (cancel,death):
  assert len(case['trigger'])==1 and case['trigger'][0]['spans']>0 and not case['trigger'][0]['admitted']
  assert not case['prefix_cache']
 assert 0<death['trigger'][0]['active_tokens']<512
 rows.append(dict(model=model,cancel_reuse='passed',partial_producer_death='passed',survivor_tokens=512,tokens_at_death=death['trigger'][0]['active_tokens']))
print(json.dumps(rows,indent=2))
