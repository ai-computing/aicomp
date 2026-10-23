"""Single-request overlap prototype; not the continuous HTTP serving path."""
import argparse,concurrent.futures,json,random,threading,time
from pathlib import Path
import requests
import torch
from transformers import AutoTokenizer
from gpu_decoder import load_model
from gpu_norms import install_fused_norms
from gpu_batch_decoder import BatchGraphDecoder
from gpu_local_prefill import LocalPrefiller
from gpu_split_prefix import load_split_prefix
from multiplex_receiver import MultiplexKVClient
from bench_http import request
from check_live_quality import CASES,grade


class Experiment:
    def __init__(self,model_dir):
        self.device='cuda:0';self.name='Qwen/Qwen3-8B-FP8'
        model=load_model(model_dir,self.device,'marlin',fuse_mlp=True,fuse_qkv=True)
        install_fused_norms(model,'fused')
        self.graph=BatchGraphDecoder(model,16)
        self.prefiller=LocalPrefiller(model_dir,self.device)
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        eos=json.loads((Path(model_dir)/'generation_config.json').read_text())['eos_token_id']
        self.eos={eos} if isinstance(eos,int) else set(eos)
        self.pool=concurrent.futures.ThreadPoolExecutor(1)
        layout=json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text())
        self.client=self.pool.submit(MultiplexKVClient,'http://127.0.0.1:19101',19201,19202,layout).result()

    def run(self,tokens,skip,count):
        if not 0<=skip<len(tokens)-1:raise ValueError('invalid skip')
        torch.cuda.synchronize();start=time.perf_counter()
        future=self.pool.submit(self.client.prefill_many,self.name,[tokens],direct=True,start_pos=skip)
        prefix=None;gpu_prefix_ms=0
        if skip:
            gpu_start=time.perf_counter()
            _,prefix,_,_=self.prefiller.prefill([tokens[:skip]])
            gpu_prefix_ms=(time.perf_counter()-gpu_start)*1000
        packet,stats=future.result()[0]
        ready=time.perf_counter();import_start=ready
        if skip:load_split_prefix(self.graph,0,prefix,packet.host,skip,tokens[-1],count)
        else:self.graph.load_packed_lane(0,packet.host,tokens[-1],count)
        import_ms=(time.perf_counter()-import_start)*1000
        del prefix,packet
        output=[];first=None
        try:
            while self.graph.remaining[0]:
                output.append(self.graph.step(ignore_eos=True)[0])
                if first is None:first=time.perf_counter()
        finally:self.graph.cancel_lane(0)
        finish=time.perf_counter()
        end=next((i for i,t in enumerate(output) if t in self.eos),len(output))
        return dict(input_tokens=len(tokens),skip=skip,output_tokens=count,
            ttft_ms=(first-start)*1000,total_ms=(finish-start)*1000,
            ready_ms=(ready-start)*1000,gpu_prefix_ms=gpu_prefix_ms,import_ms=import_ms,
            native={k:v for k,v in stats.items() if k!='response'},
            transfer_bytes=72*(len(tokens)-1-skip)*2048,
            ids=output,text=self.tokenizer.decode(output[:end],skip_special_tokens=True))

    def close(self):
        try:self.pool.submit(self.client.close).result()
        finally:self.pool.shutdown(wait=True)


def long_cases():
    cases=[]
    for count in (40,80,120):
        for target in (0,count//2,count-1):
            records=[f'Record item{i:03d}: code Z{(i*7919+count*31)%100000:05d}.' for i in range(count)]
            expected=f'Z{(target*7919+count*31)%100000:05d}'
            prompt='Read these records.\n'+'\n'.join(records)+f'\nWhat is the code for item{target:03d}? Reply with only the code.'
            cases.append((f'retrieval-{count}-{target}',prompt,expected))
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=('perf','quality'),default='perf')
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--fractions',default='0,0.5,0.75')
    a=p.parse_args();fractions=list(map(float,a.fractions.split(',')))
    if not fractions or any(not 0<=f<1 for f in fractions) or not 1<=a.repeats<=10:p.error('invalid bounded experiment')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    engine=Experiment(a.model_dir);rows=[]
    def save():a.out.write_text(json.dumps({'mode':a.mode,'allocated_gpu_bytes':torch.cuda.memory_allocated(),
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'rows':rows},ensure_ascii=False,indent=2)+'\n')
    try:
        if a.mode=='perf':
            for n in (128,512,1024):
                rng=random.Random(139+n);tokens=[rng.randrange(1000,30000) for _ in range(n)]
                body=dict(model=engine.name,prompt=tokens,max_tokens=32,min_tokens=32,ignore_eos=True,temperature=0,stream=True)
                for repeat in range(a.repeats+1):
                    native=request('http://127.0.0.1:19101',body,threading.Barrier(1))
                    rows.append({'backend':'native','input_tokens':n,'repeat':repeat,'warmup':repeat==0,
                                 **{k:v for k,v in native.items() if k not in ('text','chunk_arrival_ms')}})
                    for f in (fractions if repeat%2==0 else list(reversed(fractions))):
                        result=engine.run(tokens,min(int(n*f),n-2),32)
                        result.pop('text');result.pop('ids')
                        result.update(backend='heterogeneous',fraction=f,repeat=repeat,warmup=repeat==0)
                        rows.append(result);save()
                        print(json.dumps({k:result[k] for k in ('input_tokens','fraction','repeat','ttft_ms','gpu_prefix_ms','ready_ms','import_ms')}),flush=True)
        else:
            for name,prompt,expected in CASES[:12]+long_cases():
                tokens=engine.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
                r=requests.post('http://127.0.0.1:19101/v1/completions',json=dict(model=engine.name,prompt=tokens,max_tokens=64,temperature=0),timeout=120)
                r.raise_for_status();text=r.json()['choices'][0]['text']
                row={'case':name,'input_tokens':len(tokens),'expected':expected,'native':{'text':text,'passed':grade(text,expected)},'variants':[]}
                for f in fractions:
                    result=engine.run(tokens,min(int(len(tokens)*f),len(tokens)-2),64)
                    result.update(fraction=f,passed=grade(result['text'],expected),matches_native_text=result['text']==text)
                    row['variants'].append(result)
                rows.append(row);save()
                print(json.dumps({'case':name,'native':row['native']['passed'],'variants':[(x['fraction'],x['passed']) for x in row['variants']]}),flush=True)
    finally:
        save();engine.close()

if __name__=='__main__':main()
