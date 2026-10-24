"""Owned-server cold-start/reuse pairs with raw cache metrics and exact token controls."""
import argparse,hashlib,json,threading,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

def run(endpoint,model,source,output):
 raw=Path(source).read_bytes();data=json.loads(raw);digest=hashlib.sha256(raw).hexdigest();endpoint=endpoint.rstrip('/')
 for n in [128,512,2048,8192]:
  assert len(data['prompts'][str(n)])==n and all(type(t) is int and t>=0 for t in data['prompts'][str(n)])
  for count in [32,128]:assert len(data['references'][f'{n}:{count}']['tokens'])==count
 with Path(output).open('x') as out:
  def save(row):out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
  for n in [128,512,2048,8192]:
   for count in [32,128]:
    for concurrency in [1,4,16]:
     base=dict(input_tokens=n,output_tokens=count,concurrency=concurrency,source_sha256=digest)
     reset=dict(base,type='reset',target='all')
     try:
      req=urllib.request.Request(endpoint+'/reset_prefix_cache',data=b'{"target":"all"}',headers={'Content-Type':'application/json'})
      with urllib.request.urlopen(req,timeout=30) as response:reset['status']=response.status
      assert reset['status']==200
     except Exception as e:reset['error']=repr(e);save(reset);raise
     save(reset)
     for phase in ['cold_start','reuse']:
      def metrics(stage):
       with urllib.request.urlopen(endpoint+'/metrics',timeout=30) as response:text=response.read().decode()
       save(dict(base,type='metrics',phase=phase,stage=stage,raw=text))
      metrics('before');gate=threading.Barrier(concurrency)
      def request(index):
       body=dict(model=model,prompt=data['prompts'][str(n)],temperature=0,ignore_eos=True,stream=False,return_token_ids=True,max_tokens=count)
       row=dict(base,type='request',phase=phase,index=index,request=body)
       try:
        gate.wait(timeout=30);start=time.monotonic()
        req=urllib.request.Request(endpoint+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','x-request-id':f'cache276-{n}-{count}-{concurrency}-{phase}-{index}'})
        with urllib.request.urlopen(req,timeout=600) as response:row['response']=json.load(response)
        row['elapsed_s']=time.monotonic()-start;c=row['response']['choices'][0]
        assert c['prompt_token_ids']==body['prompt'] and len(c['token_ids'])==count and c['finish_reason']=='length'
        row['cache_off_reference_equal']=c['token_ids']==data['references'][f'{n}:{count}']['tokens']
       except Exception as e:row['error']=repr(e)
       return row
      start=time.monotonic();errors=[]
      with ThreadPoolExecutor(max_workers=concurrency) as pool:
       for future in as_completed([pool.submit(request,i) for i in range(concurrency)]):
        row=future.result();save(row)
        if 'error' in row:errors.append(row['error'])
      save(dict(base,type='batch',phase=phase,wall_s=time.monotonic()-start,errors=errors));metrics('after')
      if errors:raise RuntimeError(errors)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['endpoint','model','source','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args();run(a.endpoint,a.model,a.source,a.output)
