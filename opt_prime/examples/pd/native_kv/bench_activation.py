"""Paired full-decode graph benchmark; excludes native prefill and HTTP."""
import argparse
import json
import time
from pathlib import Path
import torch
from gpu_decoder import load_model
from gpu_batch_decoder import BatchGraphDecoder,pack_prefix
from gpu_norms import install_fused_norms
from gpu_activation import install_fused_swiglu,install_fused_post_attention

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-dir',required=True)
p.add_argument('--out',required=True)
p.add_argument('--residual-norm',action='store_true')
a=p.parse_args()
torch.set_num_threads(1)
model=load_model(a.model_dir,fuse_mlp=True,fuse_qkv=True)
install_fused_norms(model,'fused')
baseline=BatchGraphDecoder(model,8)
install_fused_swiglu(model)
if a.residual_norm:install_fused_post_attention(model)
candidate=BatchGraphDecoder(model,8)
with torch.inference_mode():
    gen=torch.Generator(device='cuda').manual_seed(147)
    ids=torch.randint(1000,20000,(1,512),generator=gen,device='cuda')
    cache=model(ids[:,:-1],use_cache=True).past_key_values
    packed=pack_prefix([(x.keys.cpu(),x.values.cpu()) for x in cache.layers])
    last=int(ids[0,-1]);rows=[];expected=None
    for repeat in range(-1,5):
        modes=[('baseline',baseline),('fused_swiglu',candidate)]
        if repeat%2:modes.reverse()
        for name,graph in modes:
            for lane in range(8):graph.load_packed_lane(lane,packed,last,128)
            output=[[] for _ in range(8)]
            start=time.perf_counter()
            for _ in range(128):
                for lane,token in graph.step(ignore_eos=True).items():output[lane].append(token)
            elapsed=time.perf_counter()-start
            if expected is None:expected=output
            row={'mode':name,'repeat':repeat,'ms_per_step':elapsed*1000/128,'aggregate_tok_s':1024/elapsed,
                 'all_tokens_equal':output==expected,'matching_positions':sum(x==y for actual,ref in zip(output,expected) for x,y in zip(actual,ref))}
            rows.append(row);Path(a.out).write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(row),flush=True)
            assert output==expected
