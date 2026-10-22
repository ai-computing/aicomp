"""GPU kernel diagnostic; excludes external native prefill/transport."""
import argparse,json,random,time
from pathlib import Path
from gpu_decoder import load_model,GraphDecoder


def main(a):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model=load_model(a.model_dir,a.device,a.weight_mode)
    if a.fused_norms:
        from gpu_norms import install_fused_norms
        print('Fused norms',install_fused_norms(model,a.norm_mode),flush=True)
    graph=GraphDecoder(model,capacity=2048,attention=a.attention)
    print('GRAPH_READY',flush=True)
    baseline=json.loads(Path(a.baseline).read_text())
    rows=[];rng=random.Random(139)
    with torch.inference_mode():
        for n in (128,512,1024):
            tokens=[rng.randrange(1000,30000) for _ in range(n)]
            ids=torch.tensor([tokens],device=a.device)
            prefix=model(ids[:,:-1],use_cache=True,logits_to_keep=1).past_key_values
            pairs=[(layer.keys,layer.values) for layer in prefix.layers]
            expected=next(row['generated'] for row in baseline if row['input_tokens']==n and row['repeat']==0)
            for rep in range(-1,3):
                start=time.perf_counter();graph.load_prefix(pairs,tokens[-1]);import_ms=(time.perf_counter()-start)*1000
                generated,metrics=graph.generate(128)
                row=dict(input_tokens=n,repeat=rep,weight_mode=a.weight_mode,attention=a.attention,fused_norms=a.fused_norms,norm_mode=a.norm_mode if a.fused_norms else None,prefix_copy_ms=import_ms,
                         generated=generated,bf16_tokens_equal=generated==expected,
                         bf16_matching_positions=sum(x==y for x,y in zip(generated,expected)),**metrics)
                rows.append(row);Path(a.out).write_text(json.dumps(rows,indent=2))
                print(json.dumps({k:v for k,v in row.items() if k!='generated'}),flush=True)
                if a.profile and n==128 and rep==0:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                        for _ in range(4):graph.graph.replay()
                        torch.cuda.synchronize()
                    profile.export_chrome_trace(a.profile)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--weight-mode',choices=['bf16','marlin'],default='marlin')
    p.add_argument('--norm-mode',choices=['fused','torch-stats'],default='fused')
    p.add_argument('--fused-norms',action='store_true');p.add_argument('--attention',choices=['sdpa','flash'],default='sdpa');p.add_argument('--profile');p.add_argument('--baseline',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
