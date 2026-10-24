"""Public native context-edge requests followed by same-server recovery checks."""
import argparse,hashlib,json,time,urllib.error,urllib.request
from pathlib import Path

def run(endpoint,model,output,capacity):
 if capacity<32:raise ValueError('capacity must be >=32')
 def body(n,m):return dict(model=model,prompt=[1000]*n,max_tokens=m,temperature=0,ignore_eos=True,stream=False,return_token_ids=True)
 normal=body(min(128,capacity//4),min(32,capacity//4))
 cases=[('below_edge',capacity-2,1,200),('exact_edge_one',capacity-1,1,200),('exact_edge_four',capacity-4,4,200),('prompt_at_capacity',capacity,1,400),('prompt_over_capacity',capacity+1,1,400),('total_over_capacity',capacity-1,2,400),('negative_output',len(normal['prompt']),-1,400)]
 with Path(output).open('x') as f:
  def request(name,kind,payload,expected,reference=None):
   start=time.monotonic();row=dict(case=name,kind=kind,request=payload,expected_status=expected)
   try:
    req=urllib.request.Request(endpoint.rstrip('/')+'/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    try:response=urllib.request.urlopen(req,timeout=180)
    except urllib.error.HTTPError as error:response=error
    with response:
     row['status']=response.status;raw=response.read().decode();row['raw_body']=raw
    row['passed']=row['status']==expected
    if row['status']==200:
     choice=json.loads(raw)['choices'][0];row['tokens']=choice['token_ids']
     row['passed'] &= choice['prompt_token_ids']==payload['prompt'] and len(choice['token_ids'])==payload['max_tokens']
     if reference is not None:row['passed'] &= row['tokens']==reference
    with urllib.request.urlopen(endpoint.rstrip('/')+'/health',timeout=10) as health:row['health_status']=health.status
    row['passed'] &= row['health_status']==200
   except Exception as error:row.update(error=repr(error),passed=False)
   row['elapsed_seconds']=time.monotonic()-start
   f.write(json.dumps(row)+'\n');f.flush()
   print(name,row.get('status'),row['passed'],flush=True)
   return row
  baseline=request('baseline','baseline',normal,200)
  if not baseline['passed']:raise RuntimeError('baseline failed; raw row retained')
  for name,n,m,status in cases:
   request(name,'boundary',body(n,m),status)
   request(name+'-recovery','recovery',normal,200,baseline['tokens'])

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['endpoint','model','output']:p.add_argument('--'+name,required=True)
 p.add_argument('--capacity',type=int,default=9216)
 a=p.parse_args();run(a.endpoint,a.model,a.output,a.capacity)
