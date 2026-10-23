"""Matched scheduled native requests without GPU decode; not an inference benchmark."""
import argparse
import json
import random
import time
from pathlib import Path
from unittest.mock import patch
import torch
import pinned_receiver
from multiplex_receiver import MultiplexKVClient


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rate',type=float,default=6)
    p.add_argument('--requests',type=int,default=96)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    if not 0<a.rate<=100 or not 1<=a.requests<=512:p.error('invalid bounded load')
    torch.set_num_threads(1)
    original=pinned_receiver.PinnedAssembler
    class Timed(original):
        def __init__(self,*args,**kwargs):
            start=time.perf_counter()
            super().__init__(*args,**kwargs)
            self.data.allocation_ms=(time.perf_counter()-start)*1000
            self.data.receive_events=[]
        def receive(self,header,socket):
            start=time.perf_counter();cpu=time.thread_time()
            done=super().receive(header,socket)
            self.data.receive_events.append([start,time.perf_counter(),time.thread_time()-cpu])
            return done
    prompts=[]
    for i in range(a.requests):
        rng=random.Random(149000+i)
        prompts.append([rng.randrange(1000,30000) for _ in range((128,512,1024)[i%3])])
    layout=json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text())
    client=MultiplexKVClient('http://127.0.0.1:19101',19201,19202,layout)
    rows=[];index=0
    try:
        with patch.object(pinned_receiver,'PinnedAssembler',Timed):
            client.prefill_many('Qwen/Qwen3-8B-FP8',[prompts[0]],direct=True)
            epoch=time.perf_counter()+0.2
            def acquire():
                nonlocal index
                if index>=a.requests or time.perf_counter()<epoch+index/a.rate:return None
                context=(index,(time.perf_counter()-epoch-index/a.rate)*1000)
                tokens=prompts[index];index+=1
                return tokens,context
            def complete(context,packet,stats):
                i,lateness=context
                stats={k:v for k,v in stats.items() if k!='response'}
                rows.append(dict(index=i,input_tokens=len(prompts[i]),start_delay_ms=lateness,
                    allocation_ms=packet.allocation_ms,events=packet.receive_events,**stats))
            client.prefill_stream('Qwen/Qwen3-8B-FP8',acquire,complete,lambda:index==a.requests,
                                  max_pending=8,direct=True)
    finally:
        client.close()
        Path(a.out).write_text(json.dumps(dict(rate=a.rate,requested=a.requests,
            completed=len(rows),rows=rows),indent=2)+'\n')
    print(json.dumps({'completed':len(rows),'out':a.out}),flush=True)

if __name__=='__main__':main()
