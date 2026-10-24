"""Frozen quality cases submitted in twelve-request cohorts by input-length stratum."""
import argparse,concurrent.futures,hashlib,json,threading,time,urllib.request
from pathlib import Path
from heldout_quality270 import score

def run(fixtures,endpoint,output):
 data=json.loads(Path(fixtures).read_text())
 assert len(data['cases'])==36 and len({c['case'] for c in data['cases']})==36
 cohorts=[[c for c in data['cases'] if c['target_tokens']==n] for n in (0,2048,8192)]
 assert all(len(c)==12 for c in cohorts)
 with Path(output).open('x') as out:
  for cohort in cohorts:
   barrier=threading.Barrier(12)
   def request(case):
    digest=hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest()
    assert digest==case['input_sha256']
    body=dict(model=data['model'],prompt=case['ids'],temperature=0,max_tokens=data['max_tokens'],ignore_eos=False,stream=False)
    req=urllib.request.Request(endpoint.rstrip('/')+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','x-request-id':'quality270-c12-'+case['case']})
    barrier.wait(timeout=30)
    start=time.monotonic()
    row=dict(case=case['case'],input_sha256=digest,input_tokens=len(case['ids']),expected=case['expected'],cohort=case['target_tokens'],request_start_monotonic=start)
    try:
     with urllib.request.urlopen(req,timeout=600) as response:payload=json.load(response)
     choice=payload['choices'][0]
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
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',required=True);p.add_argument('--endpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.fixtures,a.endpoint,a.output)
