"""Measure native HTTP disconnect, observable drain, survivors and recovery."""
import argparse,json,time,threading,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def metric(raw,name):
 values=[float(l.rsplit(' ',1)[1]) for l in raw.splitlines() if l.startswith(name+'{') or l.startswith(name+' ')]
 if not values:raise ValueError('missing metric '+name)
 return sum(values)

def run(endpoint,model,output,lengths=(128,8192),concurrencies=(1,4,16)):
 endpoint=endpoint.rstrip('/')
 with Path(output).open('x') as out:
  def record(row):out.write(json.dumps(row)+'\n');out.flush()
  def metrics():
   with urllib.request.urlopen(endpoint+'/metrics',timeout=10) as r:return r.read().decode()
  def drain():
   deadline=time.monotonic()+60;observations=[];stable=0;previous=None
   while True:
    raw=metrics();observations.append({'at':time.monotonic(),'raw':raw})
    idle=metric(raw,'furiosa_llm_num_requests_running')==0 and metric(raw,'furiosa_llm_num_requests_waiting')==0
    gen=metric(raw,'furiosa_llm_generation_tokens_total')
    stable=stable+1 if idle and gen==previous else 0;previous=gen
    if stable>=3:return True,observations
    if time.monotonic()>deadline:return False,observations
    time.sleep(.5)
  def request(n,count,stream=False,gate=None,reference=None):
   body=dict(model=model,prompt=[1000]*n,max_tokens=count,temperature=0,ignore_eos=True,stream=stream,return_token_ids=True)
   if stream:body['stream_options']={'include_usage':True}
   row={'input_tokens':n,'max_tokens':count,'stream':stream}
   try:
    if gate:gate.wait(timeout=30)
    row['start']=time.monotonic()
    req=urllib.request.Request(endpoint+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as response:
     row['status']=response.status
     if stream:
      row['events']=[];row['closed_after_content']=False
      for line in response:
       if not line.startswith(b'data: '):continue
       if line[6:].strip()==b'[DONE]':row['received_done']=True;break
       event=json.loads(line[6:]);row['events'].append(event)
       choices=event.get('choices',[])
       if any(c.get('text') or c.get('token_ids') for c in choices):
        row['closed_after_content']=all(c.get('finish_reason') is None for c in choices)
        break
     else:
      row['response']=json.load(response);choice=row['response']['choices'][0]
      row['tokens']=choice['token_ids'];row['valid']=choice['prompt_token_ids']==body['prompt'] and len(row['tokens'])==count and choice['finish_reason']=='length'
      if reference is not None:row['tokens_equal']=row['tokens']==reference
    row['closed_at']=time.monotonic()
   except Exception as error:row['error']=repr(error)
   return row
  for n in lengths:
   baseline=request(n,128);record({'type':'baseline','input_tokens':n,'result':baseline})
   if not baseline.get('valid'):raise RuntimeError('invalid baseline; raw retained')
   for concurrency in concurrencies:
    idle,before=drain()
    if not idle:record({'type':'pre_round_drain_failure','observations':before});raise RuntimeError('not idle')
    gate=threading.Barrier(concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
     victim=pool.submit(request,n,1024,True,gate)
     survivors=[pool.submit(request,n,128,False,gate,baseline['tokens']) for _ in range(concurrency-1)]
     v=victim.result();ss=[f.result() for f in survivors]
    drained,after=drain()
    delta=metric(after[-1]['raw'],'furiosa_llm_generation_tokens_total')-metric(before[-1]['raw'],'furiosa_llm_generation_tokens_total')
    full=1024+(concurrency-1)*128
    recovery=request(n,128,reference=baseline['tokens'])
    row={'type':'round','input_tokens':n,'concurrency':concurrency,'before':before,'victim':v,'survivors':ss,'after':after,'drained':drained,'generation_delta':delta,'full_generation_budget':full,'generation_below_full_budget':(concurrency-1)*128<=delta<full,'recovery':recovery}
    record(row);print(n,concurrency,'closed',v.get('closed_after_content'),'drain',drained,'generation',delta,'/',full,'survivors_equal',sum(s.get('tokens_equal',False) for s in ss),'recovery',recovery.get('tokens_equal'),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['endpoint','model','output']:p.add_argument('--'+name,required=True)
 a=p.parse_args();run(a.endpoint,a.model,a.output)
