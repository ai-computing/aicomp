"""Paired native HTTP log-probability observation; no token forcing or output repair."""
import argparse,hashlib,json,threading,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

def run(endpoint,model,source,output):
 source=Path(source);raw=source.read_bytes();prompts=json.loads(raw)['prompts']
 for n in [128,8192]:
  assert len(prompts[str(n)])==n and all(type(x) is int and x>=0 for x in prompts[str(n)])
 with Path(output).open('x') as out:
  for arm,instrumented in [('a1',False),('b1',True),('b2',True),('a2',False)]:
   for n in [128,8192]:
    for concurrency in [1,4,16]:
     gate=threading.Barrier(concurrency)
     def request(index):
      body=dict(model=model,prompt=prompts[str(n)],temperature=0,ignore_eos=True,stream=False,return_token_ids=True,max_tokens=128)
      if instrumented:body.update(logprobs=5,return_tokens_as_token_ids=True)
      row=dict(arm=arm,instrumented=instrumented,input_tokens=n,concurrency=concurrency,index=index,source_sha256=hashlib.sha256(raw).hexdigest(),request=body)
      start=time.monotonic()
      try:
       gate.wait(timeout=30)
       req=urllib.request.Request(endpoint.rstrip('/')+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','x-request-id':f'logprobs-{arm}-{n}-{concurrency}-{index}'})
       with urllib.request.urlopen(req,timeout=600) as response:row['response']=json.load(response)
       c=row['response']['choices'][0]
       assert c['prompt_token_ids']==body['prompt'] and len(c['token_ids'])==128 and c['finish_reason']=='length'
       if instrumented:
        lp=c['logprobs'];assert len(lp['top_logprobs'])==128 and len(lp['token_logprobs'])==128
        assert lp['tokens']==[f'token_id:{t}' for t in c['token_ids']]
      except Exception as e:row['error']=repr(e)
      row['elapsed_s']=time.monotonic()-start
      return row
     errors=[]
     with ThreadPoolExecutor(max_workers=concurrency) as pool:
      for f in as_completed([pool.submit(request,i) for i in range(concurrency)]):
       row=f.result();out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
       if 'error' in row:errors.append(row['error'])
     if errors:raise RuntimeError(errors)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['endpoint','model','source','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args();run(a.endpoint,a.model,a.source,a.output)
