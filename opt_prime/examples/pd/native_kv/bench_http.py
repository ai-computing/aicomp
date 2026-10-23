"""Compare identical streaming requests over HTTP, including token delivery and queueing."""
import argparse,concurrent.futures,json,random,statistics,threading,time
from pathlib import Path
import requests


def iter_sse_lines(response, reader='read1'):
    if reader=='byte':
        yield from response.iter_lines(chunk_size=1)
        return
    if reader!='read1':raise ValueError('unknown SSE reader')
    if not hasattr(response.raw,'read1'):raise RuntimeError('SSE benchmark requires urllib3 read1')
    pending=b''
    while True:
        chunk=response.raw.read1(65536,decode_content=True)
        if not chunk:break
        lines=(pending+chunk).split(b'\n')
        pending=lines.pop()
        for line in lines:yield line.rstrip(b'\r')
    if pending:yield pending.rstrip(b'\r')


def request(url,body,barrier,reader='read1'):
    barrier.wait();start=time.perf_counter();first=None;text='';arrivals=[];reason=None;usage=None;done=False
    with requests.post(url+'/v1/completions',json=body,stream=True,timeout=180) as response:
        response.raise_for_status()
        for line in iter_sse_lines(response,reader):
            if not line.startswith(b'data: '):continue
            if line==b'data: [DONE]':done=True;break
            data=json.loads(line[6:])
            if 'error' in data:raise RuntimeError(data['error'])
            if data.get('usage'):usage=data['usage']
            for choice in data.get('choices',[]):
                if choice.get('finish_reason'):reason=choice['finish_reason']
                delta=choice.get('text','')
                if delta:
                    now=time.perf_counter()
                    if first is None:first=now
                    text+=delta;arrivals.append((now-start)*1000)
    end=time.perf_counter()
    if first is None or not done:raise ValueError('incomplete SSE response')
    if usage is None and reason!='length':raise ValueError('cannot infer generated token count')
    count=usage.get('completion_tokens') if usage else body['max_tokens']
    if count!=body['max_tokens']:raise ValueError('unexpected output token count')
    return {'ttft_ms':(first-start)*1000,'total_ms':(end-start)*1000,
            'tpot_ms':(end-first)*1000/(count-1) if count>1 else None,
            'tok_s':count/(end-start),'completion_tokens':count,'usage':usage,'finish_reason':reason,
            'text_chunks':len(arrivals),'chunk_arrival_ms':arrivals,'text':text}


def main(a):
    rows=[]
    for n in map(int,a.lengths.split(',')):
        rng=random.Random(139+n)  # Stable across selected lengths and concurrency.
        tokens=[rng.randrange(1000,30000) for _ in range(n)]
        body={'model':a.model,'prompt':tokens,'temperature':0,'max_tokens':a.output_tokens,
              'min_tokens':a.output_tokens,'ignore_eos':True,'stream':True}
        for rep in range(-a.warmups,a.repeats):
            modes=[('native',a.native),('gpu',a.gpu)]
            if rep%2:modes.reverse()
            for mode,url in modes:
                barrier=threading.Barrier(a.concurrency+1)
                with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
                    futures=[pool.submit(request,url,body,barrier,a.sse_reader) for _ in range(a.concurrency)]
                    start=time.perf_counter();barrier.wait()
                    results=[future.result() for future in futures];end=time.perf_counter()
                row={'mode':mode,'input_tokens':n,'output_tokens':a.output_tokens,'concurrency':a.concurrency,'repeat':rep,
                     'wall_ms':(end-start)*1000,'aggregate_tok_s':a.concurrency*a.output_tokens/(end-start),
                     'mean_ttft_ms':statistics.mean(r['ttft_ms'] for r in results),
                     'mean_tpot_ms':statistics.mean(r['tpot_ms'] for r in results),'requests':results}
                rows.append(row);Path(a.out).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k!='requests'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',default='http://127.0.0.1:19101');p.add_argument('--gpu',default='http://127.0.0.1:19111')
    p.add_argument('--sse-reader',choices=('byte','read1'),default='read1')
    p.add_argument('--model',default='Qwen/Qwen3-8B-FP8');p.add_argument('--lengths',default='128,512,1024')
    p.add_argument('--output-tokens',type=int,default=128);p.add_argument('--concurrency',type=int,default=1)
    p.add_argument('--warmups',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',required=True);main(p.parse_args())
