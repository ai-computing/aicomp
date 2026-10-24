"""Audit completed ABBA data without converting parity failures to passes."""
import hashlib,json,math,statistics
from pathlib import Path
A=Path(__file__).resolve().parent
CONDITIONS={(512,128),(8192,128)}

def audit_arm(d):
 assert d['passed'] and d['slots_per_decoder']==4 and d['physical_atom']==1 and d['physical_rngd']==d['physical_gpus']==0
 assert d['context']==9216 and not d['decoder_cache'] and not d['producer_cache']
 assert set(d['prompts'])=={'512','8192'} and all(len(v)==int(k) for k,v in d['prompts'].items())
 assert set(d['references'])=={'512:128','8192:128'}
 assert len(d['warmups'])==len(d['correctness_batches'])==len(d['batches'])==2
 assert {(b['input_tokens'],b['output_tokens']) for b in d['correctness_batches']}==CONDITIONS
 assert {(b['input_tokens'],b['output_tokens_per_request']) for b in d['batches']}==CONDITIONS
 warm=set();flags=[];token_failures=0;text_failures=0
 for r in d['warmups']:
  _,endpoint,n,o=r['id'].split('-');n,o=int(n),int(o);warm.add((n,o));assert endpoint=='0'
  ref=d['references'][f'{n}:{o}'];assert len(ref['tokens'])==len(r['tokens'])==o
  equal=r['tokens']==ref['tokens'];assert r['tokens_equal']==equal;flags.append(equal)
 assert warm==CONDITIONS
 for b in d['correctness_batches']:
  n,o=b['input_tokens'],b['output_tokens'];ref=d['references'][f'{n}:{o}']
  assert b['concurrency']==16 and len(b['rows'])==len({r['id'] for r in b['rows']})==16
  for r in b['rows']:
   assert len(r['tokens'])==o
   equal=r['tokens']==ref['tokens'];assert r['tokens_equal']==equal;flags.append(equal);token_failures+=not equal
 for b in d['batches']:
  n,o=b['input_tokens'],b['output_tokens_per_request'];ref=d['references'][f'{n}:{o}'];rows=b['rows'];s=b['summary']
  assert b['concurrency']==16 and b['repeat']==0 and len(rows)==len({r['id'] for r in rows})==16
  equalities=[]
  for r in rows:
   assert r['input_tokens']==n and r['output_tokens']==o
   equal=r['text']==ref['text'];assert r['text_equal']==equal;equalities.append(equal);flags.append(equal);text_failures+=not equal
   assert all(math.isfinite(r[k]) for k in ['first_sse_s','e2e_s','tpot_est_s'])
   assert 0<r['first_sse_s']<=r['e2e_s'] and r['tpot_est_s']>=0
  assert s['requests']==16 and s['output_tokens']==16*o and s['all_text_equal']==all(equalities)
  assert math.isfinite(s['wall_s']) and s['wall_s']>0
  expected={'output_tok_s':16*o/s['wall_s'],'median_first_sse_s':statistics.median(r['first_sse_s'] for r in rows),'median_e2e_s':statistics.median(r['e2e_s'] for r in rows),'median_tpot_est_s':statistics.median(r['tpot_est_s'] for r in rows)}
  assert all(math.isclose(s[k],v,rel_tol=1e-10,abs_tol=1e-10) for k,v in expected.items())
 assert d['parity_passed']==all(flags)
 return {'responses':66,'parity_passed':all(flags),'failed_token_checks':token_failures,'failed_timed_text_checks':text_failures}

def main():
 outer=json.loads((A/'atom270-allcore-abba-outer.json').read_text());assert outer['state']=='complete' and outer['cleanup_exit_code']==outer['exit_code']==0
 queue=json.loads((A/'atom270-allcore-abba-arms.json').read_text());assert [x['arm'] for x in queue]==['a1','b1','b2','a2']
 result={'arms':{},'comparisons':{},'limits':'Same-card ABBA, two timed batches per variant/condition. Original parity failures remain failures; no equivalent-output speedup or general-quality claim.'};data={};bindings=[]
 for q in queue:
  arm=q['arm'];p=A/f'atom270-qwen4b-allcore-abba-{arm}.json';d=json.loads(p.read_text());data[arm]=d
  r=audit_arm(d);assert q['parity_passed']==r['parity_passed'] and q['exit_code']==(0 if r['parity_passed'] else 1)
  log=(A/f'atom270-qwen4b-allcore-abba-{arm}.server0.log').read_text()
  def marker(name):
   rows=[json.loads(l.split(name,1)[1]) for l in log.splitlines() if name in l];assert len(rows)==1;return rows[0]
  binding=marker('ATOM270_BINDING ');bindings.append(binding)
  assert marker('ATOM270_COMPILED ')==dict(batch_size=4,kv_blocks=5,context=9216,tensor_parallel_size=1)
  hist=marker('ATOM270_ACTIVE_LANES ');assert len(hist)==5 and hist[4]>0
  r.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),binding=binding,active_lane_histogram=hist)
  result['arms'][arm]=r
 assert all(b==bindings[0] for b in bindings)
 assert all(d['prompts']==data['a1']['prompts'] for d in data.values())
 result['reference_parity']={f'{a}-{b}':{k:v['tokens']==data[b]['references'][k]['tokens'] for k,v in data[a]['references'].items()} for a,b in [('a1','a2'),('b1','b2'),('a1','b1')]}
 for n,o in sorted(CONDITIONS):
  values={arm:next(b['summary'] for b in d['batches'] if b['input_tokens']==n) for arm,d in data.items()}
  entry={'measurements':values}
  for key in ['output_tok_s','median_first_sse_s','median_e2e_s']:
   a=statistics.mean(values[x][key] for x in ['a1','a2']);b=statistics.mean(values[x][key] for x in ['b1','b2'])
   entry[key]={'original_mean_of_two':a,'allcore_mean_of_two':b,'allcore_relative_change_pct':(b/a-1)*100}
  result['comparisons'][f'{n}:{o}']=entry
 (A/'atom270-allcore-abba-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
