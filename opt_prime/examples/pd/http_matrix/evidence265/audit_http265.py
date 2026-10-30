"""Audit HTTPc16 evidence. Token checks and timed SSE text checks are distinct."""
import json,math,statistics,sys
from pathlib import Path
CONDITIONS={(n,o) for n in (128,512,2048,8192) for o in (32,128)}
MODELS=('qwen06b','qwen4b','gemma','koni')

def audit_result(d,backend,mode):
 assert d['passed'] and d['parity_passed'], 'runtime or parity failure'
 assert d['slots_per_decoder']==(1 if backend=='atom' else 4)
 assert d['physical_atom']==int(backend=='atom') and d['physical_gpus']==int(backend=='gpu')
 assert d['physical_rngd']==int(mode!='native') and d['context']==9216
 assert not d['decoder_cache'] and not d['producer_cache']
 assert set(d['prompts'])=={'128','512','2048','8192'}
 assert all(len(v)==int(k) for k,v in d['prompts'].items())
 assert set(d['references'])=={f'{n}:{o}' for n,o in CONDITIONS}
 assert len(d['warmups'])==len(d['correctness_batches'])==len(d['batches'])==8
 warm=set()
 for r in d['warmups']:
  _,endpoint,n,o=r['id'].split('-');n,o=int(n),int(o);warm.add((n,o));ref=d['references'][f'{n}:{o}']
  assert endpoint=='0' and r['tokens_equal'] and r['tokens']==ref['tokens'] and len(r['tokens'])==o
 assert warm==CONDITIONS
 assert {(b['input_tokens'],b['output_tokens']) for b in d['correctness_batches']}==CONDITIONS
 for b in d['correctness_batches']:
  n,o=b['input_tokens'],b['output_tokens'];ref=d['references'][f'{n}:{o}']
  assert b['concurrency']==16 and len(b['rows'])==16 and len({r['id'] for r in b['rows']})==16
  assert all(r['tokens_equal'] and r['tokens']==ref['tokens'] and len(r['tokens'])==o for r in b['rows'])
 assert {(b['input_tokens'],b['output_tokens_per_request']) for b in d['batches']}==CONDITIONS
 for b in d['batches']:
  n,o=b['input_tokens'],b['output_tokens_per_request'];ref=d['references'][f'{n}:{o}'];rows=b['rows'];s=b['summary']
  assert b['concurrency']==16 and b['repeat']==0 and len(rows)==16 and len({r['id'] for r in rows})==16
  for r in rows:
   assert r['input_tokens']==n and r['output_tokens']==o and r['text_equal'] and r['text']==ref['text']
   assert all(math.isfinite(r[k]) for k in ['first_sse_s','e2e_s','tpot_est_s'])
   assert 0<r['first_sse_s']<=r['e2e_s'] and r['tpot_est_s']>=0
  assert s['requests']==16 and s['output_tokens']==16*o and s['all_text_equal']
  assert math.isfinite(s['wall_s']) and s['wall_s']>0
  expected={'output_tok_s':16*o/s['wall_s'],'median_first_sse_s':statistics.median(r['first_sse_s'] for r in rows),'median_e2e_s':statistics.median(r['e2e_s'] for r in rows),'median_tpot_est_s':statistics.median(r['tpot_est_s'] for r in rows)}
  assert all(math.isclose(s[k],v,rel_tol=1e-10,abs_tol=1e-10) for k,v in expected.items())
 return dict(stage='passed',conditions=8,warmups=8,correctness_responses=128,timed_text_responses=128)

def main():
 root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
 report={'scope':'HTTPc16 smoke, one measured batch per condition. Nonstreamed correctness retains token arrays; timed SSE retains text/counts. No statistical speedup claim.','cases':{},'route_reference_comparisons':{}}
 failed=False;pending=False
 for backend in ('gpu','atom'):
  queuepath=root/('stream265-http-atom-c16-queue.json' if backend=='atom' else 'stream265-http-c16-queue.json')
  queue=json.loads(queuepath.read_text()) if queuepath.exists() else []
  for model in MODELS:
   refs={}
   for mode in ('native','whole','stream'):
    stem=f'stream265-{model}-http-'+('atom-' if backend=='atom' else '')+f'c16-{mode}';p=root/(stem+'.json');key=f'{backend}/{model}/{mode}'
    if not p.exists():report['cases'][key]={'stage':'not_collected'};pending=True;continue
    try:
     d=json.loads(p.read_text());result=audit_result(d,backend,mode);refs[mode]=d
     q=[r for r in queue if r['model']==model and r['mode']==mode]
     if not q:result['stage']='awaiting_child_exit';pending=True
     else:
      assert len(q)==1 and q[0]['exit_code']==0 and all(x==0 for x in q[0].get('copy_exit_codes',[]))
     report['cases'][key]=result
    except (AssertionError,ValueError,KeyError) as e:report['cases'][key]={'stage':'failed','error':str(e) or type(e).__name__};failed=True
   for a,b in [('native','whole'),('whole','stream')]:
    if a in refs and b in refs:
     assert refs[a]['prompts']==refs[b]['prompts']
     matches={k:refs[a]['references'][k]['tokens']==refs[b]['references'][k]['tokens'] for k in refs[a]['references']}
     report['route_reference_comparisons'][f'{backend}/{model}/{a}-vs-{b}']=matches
     if a=='whole' and not all(matches.values()):failed=True
 (root/'stream265-http-c16-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report['cases'].items() if v['stage']!='not_collected'},indent=2))
 raise SystemExit(1 if failed else 2 if pending else 0)

if __name__=='__main__':main()
