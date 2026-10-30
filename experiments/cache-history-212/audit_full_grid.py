import hashlib,json,math,pathlib,statistics,sys
b=pathlib.Path(sys.argv[1]);q=json.loads((b/'cachefull212r1-gpu-queue.json').read_text())
planned={(m,c) for m in ['qwen06b','qwen4b','gemma','koni'] for c in ['off','on']}
keys=[(r['model'],r['cache']) for r in q['runs']]
assert len(keys)==len(set(keys)) and set(keys)<=planned
expected={(n,o,c) for n in [128,512,2048,8192] for o in [32,128] for c in [1,4,16]}
report={'state':q['state'],'runtime_commit':q['runtime_commit'],'arms':{},'limits':'Single timing batch, native GPU, page256/chunk256; no PD, ATOM, default page64 or answer-quality certification.'}
for run in q['runs']:
 if 'exit_code' not in run:continue
 m,c=run['model'],run['cache'];p=b/f'cachefull212r1-{m}-{c}.json';d=json.loads(p.read_text())
 assert d['passed'] and d['mode']=='native'
 assert d['physical_gpus']==1 and d['physical_rngd']==d['physical_atom']==0 and d['slots_per_decoder']==4 and d['context']==9216
 assert d['producer_cache'] is False and d['decoder_cache']==(c=='on')
 cfg=b/f'cachefull212r1-{m}-{c}-config.json';assert hashlib.sha256(cfg.read_bytes()).hexdigest()==run['config_sha256']
 argv=json.loads(cfg.read_text())['servers'][0];assert argv[argv.index('--page-size')+1]==argv[argv.index('--chunk')+1]=='256'
 assert ('--prefix-cache' in argv)==(c=='on')
 assert len(d['warmups'])==8 and len(d['correctness_batches'])==len(d['batches'])==24
 assert {tuple(map(int,r['id'].split('-')[-2:])) for r in d['warmups']}=={(n,o) for n in [128,512,2048,8192] for o in [32,128]}
 assert set(d['references'])=={f'{n}:{o}' for n in [128,512,2048,8192] for o in [32,128]}
 assert {(x['input_tokens'],x['output_tokens'],x['concurrency']) for x in d['correctness_batches']}==expected
 assert {(x['input_tokens'],x['output_tokens_per_request'],x['concurrency']) for x in d['batches']}==expected
 for n,ids in d['prompts'].items():assert len(ids)==int(n)
 if c=='on':
  refpath=b/f'cachefull212r1-{m}-off.json';off=json.loads(refpath.read_text())
  assert off['prompts']==d['prompts'] and off['references']==d['references']
  assert hashlib.sha256(refpath.read_bytes()).hexdigest()==run['reference_sha256']
 mismatches=[]
 for row in d['warmups']:
  n,o=map(int,row['id'].split('-')[-2:]);ref=d['references'][f'{n}:{o}'];assert len(row['tokens'])==o
  equal=row['tokens']==ref['tokens'];assert row['tokens_equal']==equal
  if not equal:mismatches.append({'phase':'warmup','id':row['id']})
 for batch in d['correctness_batches']:
  n,o,cc=batch['input_tokens'],batch['output_tokens'],batch['concurrency'];ref=d['references'][f'{n}:{o}'];assert len(batch['rows'])==cc
  for row in batch['rows']:
   assert len(row['tokens'])==o;equal=row['tokens']==ref['tokens'];assert row['tokens_equal']==equal
   if not equal:mismatches.append({'phase':'token','id':row['id']})
 for batch in d['batches']:
  n,o,cc=batch['input_tokens'],batch['output_tokens_per_request'],batch['concurrency'];ref=d['references'][f'{n}:{o}'];rows=batch['rows'];s=batch['summary'];assert len(rows)==cc
  eq=[]
  for row in rows:
   assert row['output_tokens']==o and 0<row['first_sse_s']<=row['e2e_s']
   equal=row['text']==ref['text'];assert row['text_equal']==equal;eq.append(equal)
   if not equal:mismatches.append({'phase':'sse','id':row['id']})
  assert s['all_text_equal']==all(eq) and s['output_tokens']==o*cc and s['requests']==cc
  assert math.isclose(s['output_tok_s'],cc*o/s['wall_s'],rel_tol=1e-10)
  for key,field in [('median_first_sse_s','first_sse_s'),('median_e2e_s','e2e_s'),('median_tpot_est_s','tpot_est_s')]:assert math.isclose(s[key],statistics.median(r[field] for r in rows),rel_tol=1e-10)
 assert d['parity_passed']==(not mismatches) and run['exit_code']==int(bool(mismatches))
 report['arms'][m+'/'+c]={'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'conditions':24,'warmups':8,'token_checks':168,'sse_checks':168,'mismatches':mismatches,'parity_passed':not mismatches,'measurements':[{'input_tokens':x['input_tokens'],'output_tokens_per_request':x['output_tokens_per_request'],'concurrency':x['concurrency'],**x['summary']} for x in d['batches']]}
if q['state']=='complete':assert set(keys)==planned and len(report['arms'])==8
report['completed_arms']=len(report['arms'])
(b/'cachefull212r1-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:{kk:vv for kk,vv in v.items() if kk!='measurements'} for k,v in report['arms'].items()})
