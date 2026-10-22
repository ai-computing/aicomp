"""End-to-end native prefill -> owned in-memory KV -> GPU decode.

Loopback ports may be SSH forwarded; output records the transport declaration.
Native comparator uses the same HTTP route, model and synthetic input tokens.
"""
import argparse,json,random,time
from pathlib import Path
from gpu_decoder import load_model,GraphDecoder
from memory_receiver import NativeKVClient


def native_request(session,url,model,tokens,count):
    body={'model':model,'prompt':tokens,'max_tokens':count,'min_tokens':count,'temperature':0,'ignore_eos':True,'stream':True}
    start=time.perf_counter();first=None;text='';chunks=0
    with session.post(url+'/v1/completions',json=body,stream=True,timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b'data: ') or line==b'data: [DONE]':continue
            data=json.loads(line[6:]);choices=data.get('choices',[])
            if choices and choices[0].get('text'):
                if first is None:first=time.perf_counter()
                text+=choices[0]['text'];chunks+=1
    end=time.perf_counter()
    if first is None:raise ValueError('native response did not generate text')
    return {'ttft_ms':(first-start)*1000,'tpot_ms':(end-first)*1000/(count-1),
            'total_ms':(end-start)*1000,'tok_s':count/(end-start),'text':text,'text_chunks':chunks}


def main(a):
    import torch,requests
    from transformers import AutoTokenizer
    torch.set_num_threads(a.cpu_threads);torch.set_num_interop_threads(1)
    model=load_model(a.model_dir,a.device,'marlin')
    if a.fused_norms:
        from gpu_norms import install_fused_norms
        install_fused_norms(model,a.norm_mode)
    graph=GraphDecoder(model,2048,attention='flash')
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    client=NativeKVClient(a.http,a.prefill_port,a.receiver_port,json.loads(Path(a.layout).read_text()))
    session=requests.Session();rows=[];rng=random.Random(139)
    try:
        for n in map(int,a.lengths.split(',')):
            tokens=[rng.randrange(1000,30000) for _ in range(n)]
            for rep in range(-a.warmups,a.repeats):
                modes=['native','heterogeneous']
                if rep%2:modes.reverse()
                for mode in modes:
                    if mode=='native':
                        row=native_request(session,a.http,a.model,tokens,a.output_tokens)
                    else:
                        start=time.perf_counter()
                        views,recv=client.prefill(a.model,tokens,'views')
                        ready=time.perf_counter()
                        graph.load_prefix(views.torch_layers(),tokens[-1],packed_import=a.packed_import)
                        imported=time.perf_counter()
                        generated,decode=graph.generate(a.output_tokens)
                        end=time.perf_counter()
                        row={'ttft_ms':(imported-start)*1000+decode['first_decode_ms'],
                             'tpot_ms':decode['tpot_ms'],'total_ms':(end-start)*1000,
                             'tok_s':a.output_tokens/(end-start),'receive_ms':(ready-start)*1000,
                             'import_ms':(imported-ready)*1000,'decode_ms':decode['decode_ms'],
                             'generated':generated,'text':tokenizer.decode(generated,skip_special_tokens=True),
                             'receive_detail':{k:v for k,v in recv.items() if k!='response'}}
                    row.update(mode=mode,input_tokens=n,output_tokens=a.output_tokens,repeat=rep,transport=a.transport,packed_import=a.packed_import,fused_norms=a.fused_norms,norm_mode=a.norm_mode if a.fused_norms else None,cpu_threads=a.cpu_threads)
                    rows.append(row);Path(a.out).write_text(json.dumps(rows,indent=2))
                    print(json.dumps({k:v for k,v in row.items() if k not in ('generated','text','receive_detail')}),flush=True)
    finally:
        client.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--cpu-threads',type=int,choices=range(1,33),default=4)
    p.add_argument('--packed-import',action='store_true')
    p.add_argument('--fused-norms',action='store_true')
    p.add_argument('--norm-mode',choices=['fused','torch-stats'],default='fused')
    p.add_argument('--http',default='http://127.0.0.1:19101');p.add_argument('--prefill-port',type=int,default=19201)
    p.add_argument('--receiver-port',type=int,default=19202)
    p.add_argument('--layout',default=str(Path(__file__).with_name('qwen3_8b_layout.json')))
    p.add_argument('--model',default='Qwen/Qwen3-8B-FP8');p.add_argument('--lengths',default='128,512,1024')
    p.add_argument('--output-tokens',type=int,default=128);p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--transport',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
