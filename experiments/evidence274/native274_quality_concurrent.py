"""Frozen quality cases submitted in twelve-request cohorts by input-length stratum."""
import argparse,concurrent.futures,hashlib,json,threading,time,urllib.request
from pathlib import Path
from heldout_quality270 import score

def run(fixtures,endpoint,output,model,eos_token_ids):
 assert eos_token_ids and all(type(x) is int and x>=0 for x in eos_token_ids)
 data=json.loads(Path(fixtures).read_text())
 assert len(data['cases'])==36 and len({c['case'] for c in data['cases']})==36
 for case in data['cases']:
  assert hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest()==case['input_sha256']
 cohorts=[[c for c in data['cases'] if c['target_tokens']==n] for n in (0,2048,8192)]
 assert all(len(c)==12 for c in cohorts)
 with Path(output).open('x') as out:
  for cohort in cohorts:
   barrier=threading.Barrier(12)
   def request(case):
    digest=hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest()
    assert digest==case['input_sha256']
    body=dict(model=model,prompt=case['ids'],temperature=0,max_tokens=data['max_tokens'],ignore_eos=False,stream=False,return_token_ids=True,stop_token_ids=sorted(set(eos_token_ids)))
    req=urllib.request.Request(endpoint.rstrip('/')+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','x-request-id':'native274-quality-c12-'+case['case']})
    barrier.wait(timeout=30)
    start=time.monotonic()
    row=dict(case=case['case'],input_sha256=digest,input_tokens=len(case['ids']),expected=case['expected'],cohort=case['target_tokens'],request_start_monotonic=start)
    try:
     with urllib.request.urlopen(req,timeout=600) as response:payload=json.load(response)
     choice=payload['choices'][0]
     assert choice['prompt_token_ids']==case['ids'] and len(choice['token_ids'])<=data['max_tokens'] and choice['finish_reason'] in ['stop','length']
     row.update(text=choice['text'],tokens=choice['token_ids'],finish_reason=choice['finish_reason'],response=payload,**score(choice['text'],case['expected']))
    except Exception as e:row.update(error=repr(e),answer_correct=False,format_correct=False)
    row['request_end_monotonic']=time.monotonic();row['elapsed_seconds']=row['request_end_monotonic']-start
    return row
   with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    futures=[pool.submit(request,c) for c in cohort]
    rows=[]
    for future in concurrent.futures.as_completed(futures):
     row=future.result();rows.append(row);out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
     print(row['case'],row['answer_correct'],row['format_correct'],flush=True)
   assert len(rows)==12 and not any('error' in r for r in rows), 'Request failure; all completed rows preserved'

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['fixtures','endpoint','output','model']:p.add_argument('--'+n,required=True)
 p.add_argument('--eos-token-ids',nargs='+',type=int,required=True)
 a=p.parse_args();run(a.fixtures,a.endpoint,a.output,a.model,a.eos_token_ids)
