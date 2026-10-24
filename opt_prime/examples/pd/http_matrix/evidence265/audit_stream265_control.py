"""Audit balanced three-mode timing control and recompute medians."""
import json,sys,statistics
from pathlib import Path
root=Path(sys.argv[1]);models=('qwen06b','qwen4b','gemma','koni');plan=['whole','early','deferred','early','deferred','whole','deferred','whole','early','whole','early','deferred']
queue=json.loads((root/'stream265-control-queue.json').read_text())
assert len(queue)==4 and {r['label'] for r in queue}==set(models) and all(r['exit_code']==0 for r in queue)
summary=[]
for model in models:
 d=json.loads((root/f'stream265-{model}-control-gpu.json').read_text())
 assert d['stage']=='complete' and len(d['runs'])==12 and d['width']==8192
 reference=d['runs'][0]['tokens']
 for i,r in enumerate(d['runs']):
  assert r['mode']==plan[i] and r['warmup']==(i<3) and r['stream_layers']==(r['mode']!='whole')
  assert r['idle'] and r['tokens_exact'] and r['tokens']==reference
  assert len(r['tokens'])==2 and all(len(t)==32 for t in r['tokens'])
  assert bool(r['progress_spans'])==r['stream_layers']
  assert sum(t['spans'] for t in r['imports'])==r['progress_spans']
  for arrivals,ttft in zip(r['token_times_s'],r['ttft_s']):
   assert len(arrivals)==32 and arrivals==sorted(arrivals) and arrivals[0]==ttft and 0<ttft<=arrivals[-1]<=r['elapsed_s']
 stats={}
 for mode in ('whole','early','deferred'):
  rows=[r for r in d['runs'] if r['mode']==mode and not r['warmup']];assert len(rows)==3
  stats[mode]=dict(median_s=statistics.median(r['elapsed_s'] for r in rows),min_s=min(r['elapsed_s'] for r in rows),max_s=max(r['elapsed_s'] for r in rows),ttft_s=[statistics.median(r['ttft_s'][i] for r in rows) for i in (0,1)])
 summary.append(dict(model=model,modes=stats,early_vs_deferred_reduction_percent=100*(1-stats['early']['median_s']/stats['deferred']['median_s']),deferred_vs_whole_reduction_percent=100*(1-stats['deferred']['median_s']/stats['whole']['median_s'])))
print(json.dumps(summary,indent=2))
