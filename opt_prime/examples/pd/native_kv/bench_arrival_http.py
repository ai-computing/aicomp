"""Deterministic open-loop mixed HTTP arrivals with scheduled-time latency."""
import argparse,concurrent.futures,json,random,statistics,threading,time
from pathlib import Path
from bench_http import request
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--rates',default='4,6');p.add_argument('--requests',type=int,default=48)
p.add_argument('--sse-reader',choices=('byte','read1'),default='read1')
p.add_argument('--native',default='http://127.0.0.1:19101');p.add_argument('--gpu',default='http://127.0.0.1:19111');a=p.parse_args()
if not 1<=a.requests<=512:p.error('requests must be 1..512')
def percentile(values,p):
 return sorted(values)[min(len(values)-1,int((len(values)-1)*p))] if values else None
cases=[]
for i in range(a.requests):
 n=(128,512,1024)[i%3];count=(32,128)[i%2];rng=random.Random(149000+i)
 cases.append(dict(model='Qwen/Qwen3-8B-FP8',prompt=[rng.randrange(1000,30000) for _ in range(n)],max_tokens=count,min_tokens=count,ignore_eos=True,stream=True,temperature=0))
rows=[]
for phase,rate in enumerate(map(float,a.rates.split(','))):
 if rate<=0:raise ValueError('rate must be positive')
 modes=[('native',a.native),('gpu',a.gpu)]
 if phase%2:modes.reverse()
 for mode,url in modes:
  request(url,cases[0],threading.Barrier(1),a.sse_reader)
  epoch=time.perf_counter()+0.2;cpu_start=time.process_time()
  def run(i):
   target=epoch+i/rate;time.sleep(max(0,target-time.perf_counter()));started=time.perf_counter()
   base=dict(index=i,input_tokens=len(cases[i]['prompt']),output_tokens=cases[i]['max_tokens'],scheduled_s=i/rate,start_delay_ms=(started-target)*1000)
   try:
    result=request(url,cases[i],threading.Barrier(1),a.sse_reader)
    base.update(result);base['scheduled_ttft_ms']=result['ttft_ms']+base['start_delay_ms'];base['scheduled_total_ms']=(time.perf_counter()-target)*1000;base['ok']=True
   except Exception as e:base.update(ok=False,error=type(e).__name__,status=getattr(getattr(e,'response',None),'status_code',None))
   return base
  with concurrent.futures.ThreadPoolExecutor(max_workers=min(a.requests,128)) as pool:
   results=list(pool.map(run,range(a.requests)))
  duration=time.perf_counter()-epoch;ok=[x for x in results if x['ok']]
  gaps=[b-a for x in ok for a,b in zip(x['chunk_arrival_ms'],x['chunk_arrival_ms'][1:])]
  row=dict(phase=phase,mode=mode,rate=rate,sse_reader=a.sse_reader,client_cpu_s=time.process_time()-cpu_start,requests=a.requests,client_workers=min(a.requests,128),successes=len(ok),errors=len(results)-len(ok),duration_s=duration,completed_tok_s=sum(x['completion_tokens'] for x in ok)/duration,
   ttft_p50_ms=percentile([x['scheduled_ttft_ms'] for x in ok],.5),ttft_p95_ms=percentile([x['scheduled_ttft_ms'] for x in ok],.95),total_p95_ms=percentile([x['scheduled_total_ms'] for x in ok],.95),max_text_gap_ms=max(gaps) if gaps else None,results=results)
  rows.append(row);Path(a.out).write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='results'}),flush=True)
