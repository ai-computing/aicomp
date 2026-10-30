import argparse,hashlib,json,math,pathlib,statistics
parser=argparse.ArgumentParser(description="Audit saved ATOM native/whole/stream intermediate-length measurements")
parser.add_argument("artifacts",type=pathlib.Path)
args=parser.parse_args()
b=args.artifacts.resolve();q=json.loads((b/'mid201-atom-final1030-queue.json').read_text())
completed=[x for x in q['runs'] if x.get('exit_code')==0 and x.get('passed') and x.get('parity_passed')]
planned={(model,mode) for model in ['qwen06b','qwen4b','gemma','koni'] for mode in ['native','whole','stream']}
keys=[(r['model'],r['mode']) for r in q['runs']]
assert len(keys)==len(set(keys)) and set(keys)<=planned
if q['state']=='complete':assert len(completed)==12 and set(keys)==planned
expected={(n,o,c) for n in [512,2048] for o in [32,128] for c in [1,4]}
report={'runtime_scope':q['runtime_scope'],'queue_sha256':hashlib.sha256((b/'mid201-atom-final1030-queue.json').read_bytes()).hexdigest(),'sources':q['source_hashes'],'models':{},'limits':'Historical stream265 runtime with final PR266 HTTP override, one ATOM decode lane; c4 queued clients, cache off, no current-main/cacheon/physicalB4 certification.'}
for run in completed:
 model=run['model'];mode=run['mode'];p=b/f'mid201-atom-final-{model}-{mode}.json';d=json.loads(p.read_text());oldpath=b/f'stream265-{model}-http-atom-c16-native.json';old=json.loads(oldpath.read_text()) if oldpath.exists() else None
 previous_path=b/f'stream265-{model}-http-atom-c16-{mode}.json'
 previous_pd=json.loads(previous_path.read_text()) if previous_path.exists() else None
 assert d['passed'] and d['parity_passed'] and d['mode']==('native' if mode=='native' else 'pd')
 assert d['physical_gpus']==0 and d['physical_rngd']==int(mode!='native') and d['physical_atom']==1 and d['slots_per_decoder']==1 and d['context']==9216
 assert d['producer_cache'] is False and d['decoder_cache'] is False
 assert len(d['warmups'])==4 and len(d['correctness_batches'])==len(d['batches'])==8
 assert {(x['input_tokens'],x['output_tokens'],x['concurrency']) for x in d['correctness_batches']}==expected
 assert {(x['input_tokens'],x['output_tokens_per_request'],x['concurrency']) for x in d['batches']}==expected
 for n,ids in d['prompts'].items():assert len(ids)==int(n) and (old is None or old['prompts'][n]==ids)
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
 report['models'][model+'/'+mode]={'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'historical_c16_same_route_reference_tokens_equal':({k:v['tokens']==previous_pd['references'][k]['tokens'] for k,v in d['references'].items()} if previous_pd and previous_pd.get('passed') and all(k in previous_pd.get('references',{}) for k in d['references']) else None),'conditions':8,'warmups':4,'token_checks':sum(len(x['rows']) for x in d['correctness_batches']),'sse_checks':sum(len(x['rows']) for x in d['batches']),'historical_c16_native_reference_tokens_equal':({k:v['tokens']==old['references'][k]['tokens'] for k,v in d['references'].items()} if old and old.get('passed') and all(k in old.get('references',{}) for k in d['references']) else None),'measurements':[{'input_tokens':x['input_tokens'],'output_tokens_per_request':x['output_tokens_per_request'],'concurrency':x['concurrency'],**x['summary']} for x in d['batches']]}
report['whole_stream_reference_tokens_equal']={}
for model in ['qwen06b','qwen4b','gemma','koni']:
 if not all(any(r['model']==model and r['mode']==mode for r in completed) for mode in ['native','whole','stream']):continue
 whole=json.loads((b/f'mid201-atom-final-{model}-whole.json').read_text());stream=json.loads((b/f'mid201-atom-final-{model}-stream.json').read_text());native=json.loads((b/f'mid201-atom-final-{model}-native.json').read_text())
 assert whole['prompts']==stream['prompts']==native['prompts']
 eq={k:v['tokens']==stream['references'][k]['tokens'] for k,v in whole['references'].items()};assert all(eq.values());report['whole_stream_reference_tokens_equal'][model]=eq
 report['models'][model+'/whole']['same_run_native_reference_tokens_equal']={k:v['tokens']==native['references'][k]['tokens'] for k,v in whole['references'].items()}
report['native_differences']={}
metadata=json.loads((b/'mid201-atom-final-artifact-metadata.json').read_text())
for model in ['qwen06b','qwen4b','gemma','koni']:
 if not all(any(r['model']==model and r['mode']==mode for r in completed) for mode in ['native','whole']):continue
 native=json.loads((b/f'mid201-atom-final-{model}-native.json').read_text());whole=json.loads((b/f'mid201-atom-final-{model}-whole.json').read_text())
 eos=metadata[model]['generation_config']['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])
 if model=='koni':eos.update(x['id'] for x in json.loads((b/'mid201-atom-final-koni-eos.json').read_text()))
 details={}
 for key,ref in native['references'].items():
  a=ref['tokens'];z=whole['references'][key]['tokens'];idx=next((i for i,(x,y) in enumerate(zip(a,z)) if x!=y),None)
  ea=next((i for i,x in enumerate(a) if x in eos),None);ez=next((i for i,x in enumerate(z) if x in eos),None)
  details[key]={'equal':a==z,'first_difference_index':idx,'native_eos_index':ea,'whole_eos_index':ez,'equal_through_same_eos':ea is not None and ea==ez and a[:ea+1]==z[:ez+1],'native_token':a[idx] if idx is not None else None,'whole_token':z[idx] if idx is not None else None}
 report['native_differences'][model]={'eos_ids':sorted(eos),'references':details}
report['state']=q['state'];report['completed_arms']=len(completed)
(b/'mid201-atom-final-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:{a:v for a,v in r.items() if a!='measurements'} for k,r in report['models'].items()})
