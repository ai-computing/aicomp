"""Recompute paired long-context results without treating pending models as passes."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
report={'scope':'One RNGD and GPU, fixed width8192, sixteen alternating2048/8192 inputs, four execution slots, c16, fixed128 output; three measured batches/mode after two warmups. Host completion trace, not device-kernel overlap.','models':{}}
for label in ['qwen06b','qwen4b','gemma','koni']:
 p=root/f'stream265-{label}-c16-gpu.json'
 if not p.exists():report['models'][label]={'stage':'not_collected'};continue
 r=json.loads(p.read_text())
 if r.get('stage')!='complete':report['models'][label]={'stage':r.get('stage','running'),'runs':len(r['runs']),'error':r.get('error')};continue
 assert r['input_lengths']==[2048,8192]*8 and r['output_tokens']==128 and r['concurrency']==16 and r['decoder_slots']==r['producer_slots']==4 and r['width']==8192
 assert len(r['runs'])==8
 reference=r['runs'][0]['tokens']
 for i,run in enumerate(r['runs']):
  assert run['warmup']==(i<2) and run['idle'] and run['tokens_exact']
  assert run['stream_layers']==[False,True,False,True,True,False,False,True][i]
  assert bool(run['progress_spans'])==run['stream_layers']
  assert run['tokens']==reference and len(run['tokens'])==16 and all(len(t)==128 for t in run['tokens'])
  assert len(run['token_times_s'])==len(run['ttft_s'])==16
  for first,times in zip(run['ttft_s'],run['token_times_s']):
   assert len(times)==128 and first==times[0] and times==sorted(times) and 0<first<=times[-1]<=run['elapsed_s']
  assert sum(e['spans'] for e in run['imports'])==run['progress_spans']
  assert all(e['start_s']<=e['end_s'] for e in run['imports'])
 modes={}
 for mode in [False,True]:
  runs=[v for v in r['runs'] if not v['warmup'] and v['stream_layers']==mode];assert len(runs)==3
  summary={'median_elapsed_s':statistics.median(v['elapsed_s'] for v in runs),'median_ttft_s':[statistics.median(v['ttft_s'][i] for v in runs) for i in range(16)],'imports_before_source_done':sum(not e['source_done_before'] for v in runs for e in v['imports'])}
  assert summary==r['summary'][str(mode)]
  summary.update(elapsed_samples_s=[v['elapsed_s'] for v in runs],spans_completed_before_source_done=sum(e['spans'] for v in runs for e in v['imports'] if not e['source_done_after']))
  modes[str(mode)]=summary
 a,b=modes['False'],modes['True']
 report['models'][label]={'stage':'complete','modes':modes,'elapsed_reduction_pct':100*(1-b['median_elapsed_s']/a['median_elapsed_s']),'ttft_reduction_pct':[100*(1-y/x) for x,y in zip(a['median_ttft_s'],b['median_ttft_s'])],'exact_pairs_including_warmup':8,'exact_sequences_including_warmup':128}
(root/'stream265-c16-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{n:v[n] for n in ['stage','elapsed_reduction_pct'] if n in v} for k,v in report['models'].items()},indent=2))

queue_path=root/'stream265-c16-queue.json'
queue=json.loads(queue_path.read_text()) if queue_path.exists() else []
complete=all(v['stage']=='complete' for v in report['models'].values())
if complete:
 assert len(queue)==4 and {v['label'] for v in queue}==set(report['models'])
 assert all(v['exit_code']==0 for v in queue)
raise SystemExit(0 if complete else 2)
