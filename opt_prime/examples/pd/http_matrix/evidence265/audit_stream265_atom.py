"""Recompute long ATOM timings; reject incomplete/parity/ownership failures."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);summary=[]
for model in ('qwen06b','qwen4b','gemma','koni'):
 d=json.loads((root/f'stream265-{model}-long-atom.json').read_text())
 assert d['stage']=='complete' and d['width']==8192 and d['output_tokens']==32
 assert len(d['cases'])==16
 reference=d['cases'][0]['tokens']
 for index,c in enumerate(d['cases']):
  assert c['index']==index and c['warmup']==(index%8<2)
  assert c['prefetch']==(index>=8)
  assert c['stream']==(index%8 in (1,3,4,7))
  assert c['tokens_exact'] and c['tokens']==reference and c['idle']
  assert all(len(t)==32 for t in c['tokens'])
  assert bool(c['progress_spans'])==c['stream']
  assert sum(i['spans'] for i in c['imports'])==c['progress_spans']
  assert all(i['active_decode']==0 for i in c['imports'])
  for times,ttft in zip(c['arrivals_s'],c['ttft_s']):
   assert len(times)==32 and times==sorted(times) and times[0]==ttft and 0<ttft<=times[-1]<=c['elapsed_s']
 for prefetch in (False,True):
  rows=[c for c in d['cases'] if not c['warmup'] and c['prefetch']==prefetch]
  whole=[c for c in rows if not c['stream']];stream=[c for c in rows if c['stream']]
  assert len(whole)==len(stream)==3
  a=statistics.median(c['elapsed_s'] for c in whole);b=statistics.median(c['elapsed_s'] for c in stream)
  summary.append(dict(model=model,prefetch=prefetch,whole_median_s=a,stream_median_s=b,reduction_percent=100*(1-b/a),ttft_reduction_percent=[100*(1-statistics.median(c['ttft_s'][i] for c in stream)/statistics.median(c['ttft_s'][i] for c in whole)) for i in range(2)],early_spans=sum(e['spans'] for c in stream for e in c['imports'] if not e['source_done_after'])))
print(json.dumps(summary,indent=2))
