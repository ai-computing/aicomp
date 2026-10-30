import argparse,hashlib,json,math,pathlib,statistics
parser=argparse.ArgumentParser(description="Audit native GPU short/long HTTP measurements")
parser.add_argument("artifacts",type=pathlib.Path)
args=parser.parse_args()
b=args.artifacts.resolve();q=json.loads((b/'edge201-gpu-queue.json').read_text())
assert q['state'] in ['running','complete','failed']
completed=[r for r in q['runs'] if r.get('exit_code')==0]
if q['state']=='complete':assert len(completed)==4
planned={'qwen06b','qwen4b','gemma','koni'}
keys=[r['model'] for r in q['runs']]
assert len(keys)==len(set(keys)) and set(keys)<=planned
if q['state']=='complete':assert set(keys)==planned
expected={(n,o,c) for n in [128,8192] for o in [32,128] for c in [1,4]}
report={'runtime_commit':q['runtime_commit'],'queue_sha256':hashlib.sha256((b/'edge201-gpu-queue.json').read_bytes()).hexdigest(),'sources':q['sources'],'models':{},'limits':'Historical experimental runtime, cache off, one timing batch per condition; not current-main certification, cache-on coverage or a speedup comparison.'}
for run in completed:
 model=run['model'];p=b/f'edge201-{model}-gpu-native.json';d=json.loads(p.read_text());old=json.loads((b/f'stream265-{model}-http-c16-native.json').read_text())
 assert d['passed'] and d['parity_passed'] and d['mode']=='native'
 assert d['physical_gpus']==1 and d['physical_rngd']==d['physical_atom']==0 and d['slots_per_decoder']==4 and d['context']==9216
 assert d['producer_cache'] is False and d['decoder_cache'] is False
 assert len(d['warmups'])==4 and len(d['correctness_batches'])==len(d['batches'])==8
 assert {(x['input_tokens'],x['output_tokens'],x['concurrency']) for x in d['correctness_batches']}==expected
 assert {(x['input_tokens'],x['output_tokens_per_request'],x['concurrency']) for x in d['batches']}==expected
 for n,ids in d['prompts'].items():assert len(ids)==int(n) and old['prompts'][n]==ids
 for warm in d['warmups']:
  n,o=map(int,warm['id'].split('-')[-2:]);ref=d['references'][f'{n}:{o}']
  assert warm['tokens_equal'] and warm['tokens']==ref['tokens'] and len(warm['tokens'])==o and warm['text']==ref['text']
 for batch in d['correctness_batches']:
  n,o,c=batch['input_tokens'],batch['output_tokens'],batch['concurrency'];ref=d['references'][f'{n}:{o}'];assert len(batch['rows'])==c
  assert all(r['tokens_equal'] and r['tokens']==ref['tokens'] and len(r['tokens'])==o for r in batch['rows'])
 for batch in d['batches']:
  n,o,c=batch['input_tokens'],batch['output_tokens_per_request'],batch['concurrency'];ref=d['references'][f'{n}:{o}'];rows=batch['rows'];s=batch['summary'];assert len(rows)==c
  assert all(r['text_equal'] and r['text']==ref['text'] and r['output_tokens']==o and 0<r['first_sse_s']<=r['e2e_s'] for r in rows)
  assert math.isclose(s['output_tok_s'],c*o/s['wall_s'],rel_tol=1e-10)
  assert math.isclose(s['median_first_sse_s'],statistics.median(r['first_sse_s'] for r in rows),rel_tol=1e-10)
  assert math.isclose(s['median_e2e_s'],statistics.median(r['e2e_s'] for r in rows),rel_tol=1e-10)
 cfg=json.loads((b/f'default201-{model}-gpu-config.json').read_text());assert cfg['servers'][0][1].endswith('/serve_batched.py') and '-c' not in cfg['servers'][0]
 report['models'][model]={'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'conditions':8,'warmups':4,'token_checks':sum(len(x['rows']) for x in d['correctness_batches']),'sse_checks':sum(len(x['rows']) for x in d['batches']),'historical_c16_reference_tokens_equal':{k:v['tokens']==old['references'][k]['tokens'] for k,v in d['references'].items()},'measurements':[{'input_tokens':x['input_tokens'],'output_tokens_per_request':x['output_tokens_per_request'],'concurrency':x['concurrency'],**x['summary']} for x in d['batches']]}
report['state']=q['state'];report['completed_arms']=len(completed)
(b/'edge201-gpu-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:{a:v for a,v in r.items() if a!='measurements'} for k,r in report['models'].items()})
