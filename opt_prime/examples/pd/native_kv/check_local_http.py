"""Local endpoint smoke/latency evidence; no native comparison in this tool."""
import argparse,concurrent.futures,json,random,statistics,threading,time
from pathlib import Path
import requests
from bench_http import request
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--url',default='http://127.0.0.1:19111');a=p.parse_args()
rows=[]
for concurrency in (1,4):
 for repeat in range(-1,3):
  rng=random.Random(651);tokens=[rng.randrange(1000,30000) for _ in range(512)]
  body=dict(model='Qwen/Qwen3-8B-FP8',prompt=tokens,max_tokens=128,min_tokens=128,ignore_eos=True,temperature=0,stream=True)
  barrier=threading.Barrier(concurrency+1)
  with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
   futures=[pool.submit(request,a.url,body,barrier) for _ in range(concurrency)]
   start=time.perf_counter();barrier.wait();results=[f.result() for f in futures];elapsed=time.perf_counter()-start
  row=dict(concurrency=concurrency,repeat=repeat,aggregate_tok_s=128*concurrency/elapsed,mean_ttft_ms=statistics.mean(r['ttft_ms'] for r in results),results=results)
  rows.append(row);Path(a.out).write_text(json.dumps(rows,indent=2)+'\n');print({k:v for k,v in row.items() if k!='results'},flush=True)
for count in (1,2):
 body=dict(model='Qwen/Qwen3-8B-FP8',prompt=[1000]*32,max_tokens=count,ignore_eos=True)
 r=requests.post(a.url+'/v1/completions',json=body,timeout=60);r.raise_for_status();assert r.json()['usage']['completion_tokens']==count
 print('count',count,'PASS',flush=True)
