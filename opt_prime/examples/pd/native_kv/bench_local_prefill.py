"""Measure GPU-local prefill with fixed-shape CUDA graphs; no HTTP/native costs."""
import argparse,json,time,gc
from pathlib import Path
import torch
from transformers import StaticCache
from gpu_decoder import load_model
from gpu_norms import install_fused_norms
from gpu_activation import install_fused_swiglu,install_fused_post_attention

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True)
p.add_argument('--weight-mode',choices=['marlin','bf16'],default='marlin')
p.add_argument('--lengths',default='128,512,1024');p.add_argument('--batches',default='1,4,8')
a=p.parse_args()
torch.set_num_threads(1)
model=load_model(a.model_dir,weight_mode=a.weight_mode,fuse_mlp=a.weight_mode=='marlin',fuse_qkv=a.weight_mode=='marlin')
install_fused_norms(model,'fused')
if a.weight_mode=='marlin':install_fused_swiglu(model)
install_fused_post_attention(model)
rows=[]
with torch.inference_mode():
 for batch in map(int,a.batches.split(',')):
  for n in map(int,a.lengths.split(',')):
   ids=torch.randint(1000,20000,(batch,n),generator=torch.Generator().manual_seed(147+n)).cuda()
   position=torch.arange(n,device='cuda');position_ids=position[None].expand(batch,-1)
   mask=torch.ones(n,n,dtype=torch.bool,device='cuda').tril()[None,None]
   cache=StaticCache(config=model.config,max_cache_len=n)
   cache.early_initialization(batch,8,128,torch.bfloat16,torch.device('cuda:0'))
   def forward():
    return model(ids,past_key_values=cache,cache_position=position,position_ids=position_ids,
      attention_mask={'full_attention':mask},use_cache=True,logits_to_keep=1).logits
   eager=[]
   for repeat in range(4):
    torch.cuda.synchronize();start=time.perf_counter();logits=forward();torch.cuda.synchronize()
    if repeat:eager.append((time.perf_counter()-start)*1000)
   expected_logits=logits.clone()
   expected_cache=[(layer.keys.clone(),layer.values.clone()) for layer in cache.layers]
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):forward()
   torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):captured=forward()
   graph_ms=[]
   for repeat in range(6):
    torch.cuda.synchronize();start=time.perf_counter();graph.replay();torch.cuda.synchronize()
    if repeat:graph_ms.append((time.perf_counter()-start)*1000)
   logits_equal=torch.equal(captured,expected_logits)
   cache_equal=all(torch.equal(layer.keys,k) and torch.equal(layer.values,v) for layer,(k,v) in zip(cache.layers,expected_cache))
   row={'weight_mode':a.weight_mode,'batch':batch,'input_tokens':n,'eager_ms':eager,'graph_ms':graph_ms,
        'logits_equal':logits_equal,'cache_equal':cache_equal,
        'argmax_equal':torch.equal(captured.argmax(-1),expected_logits.argmax(-1))}
   rows.append(row);Path(a.out).write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
   del graph,cache,expected_cache,captured,expected_logits,logits,ids,mask
   gc.collect();torch.cuda.empty_cache()
