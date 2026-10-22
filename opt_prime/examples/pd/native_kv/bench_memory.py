"""Paired disk-free KV path timings versus standalone native first-token HTTP.

Boundaries differ: external KV-ready excludes decoder final-token/LM-head work.
It must not be presented as a full-serving speedup over native TTFT.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

import requests
from memory_receiver import NativeKVClient


def main(a):
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    layout=json.loads(Path(a.layout).read_text())
    client=NativeKVClient(a.http,a.prefill_port,a.receiver_port,layout)
    session=requests.Session()
    rows=[];hashes={};rng=random.Random(139)
    try:
        for n in map(int,a.lengths.split(',')):
            tokens=[rng.randrange(1000,30000) for _ in range(n)]
            for rep in range(-a.warmups,a.repeats):
                modes=a.modes.split(',')
                if rep%2:modes.reverse()
                for mode in modes:
                    if mode=='native':
                        body={'model':a.model,'prompt':tokens,'max_tokens':1,'min_tokens':1,'temperature':0,'ignore_eos':True,'stream':False}
                        t=time.perf_counter();r=session.post(a.http+'/v1/completions',json=body,timeout=90)
                        elapsed=(time.perf_counter()-t)*1000;r.raise_for_status()
                        row={'mode':mode,'input_tokens':n,'complete_ms':elapsed,'response':r.json()}
                    else:
                        data,row=client.prefill(a.model,tokens,mode)
                        # Outside timing: same shape/order in both paths.
                        sha=hashlib.sha256(data.tobytes()).hexdigest()
                        if n in hashes and hashes[n]!=sha:raise AssertionError('KV bytes changed across repeated/paired requests')
                        hashes[n]=sha;row['sha256']=sha
                    row.update(repeat=rep,loadavg=os.getloadavg(),prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
                    rows.append(row)
                    with (out/'runs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('completed length',n,flush=True)
    finally:
        client.close();session.close()
    summary=[]
    for n in map(int,a.lengths.split(',')):
        for mode in a.modes.split(','):
            selected=[r for r in rows if r['input_tokens']==n and r['mode']==mode and r['repeat']>=0]
            entry={'input_tokens':n,'mode':mode,'samples':len(selected)}
            for metric in ('complete_ms','kv_ready_ms','parse_pack_ms'):
                values=[r[metric] for r in selected if metric in r]
                if values:entry[metric]={'median':statistics.median(values),'min':min(values),'max':max(values)}
            summary.append(entry)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--http',default='http://127.0.0.1:19101');p.add_argument('--prefill-port',type=int,default=19201)
    p.add_argument('--receiver-port',type=int,default=19202);p.add_argument('--model',default='Qwen/Qwen3-8B-FP8')
    p.add_argument('--layout',default=str(Path(__file__).with_name('qwen3_8b_layout.json')))
    p.add_argument('--modes',default='native,legacy,fast,views');p.add_argument('--lengths',default='128,512,1024');p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--repeats',type=int,default=7);p.add_argument('--out',required=True)
    main(p.parse_args())
