"""Fixed-context logits and independent request-boundary diagnostics."""
import argparse,json
from pathlib import Path
import torch
from gpu_local_prefill import LocalPrefiller
from gpu_batch_decoder import BatchGraphDecoder
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True)
a=p.parse_args();torch.set_num_threads(1)
f=LocalPrefiller(a.model_dir)
prompts=[torch.randint(1000,20000,(n,),generator=torch.Generator().manual_seed(n)).tolist() for n in (32,128,257,512)]
def compare(a,b):
 pairs=[(x,y) for pair,other in zip(a,b) for x,y in zip(pair,other)]
 return dict(exact=all(torch.equal(x,y) for x,y in pairs),max_abs=max(float((x.float()-y.float()).abs().max()) for x,y in pairs),min_cos=min(float(torch.nn.functional.cosine_similarity(x.float().flatten(),y.float().flatten(),dim=0)) for x,y in pairs))
def top(logits):
 v,i=logits.flatten().float().topk(5);return dict(ids=i.tolist(),values=v.tolist(),margin=float(v[0]-v[1]))
with torch.inference_mode():
 first,packed,offsets,logits=f.prefill(prompts)
 _,ref,_,ref_logits=f.prefill(prompts,attention_reference=True)
 target=[(k[offsets[-2]:],v[offsets[-2]:]) for k,v in packed]
 reference=[(k[offsets[-2]:],v[offsets[-2]:]) for k,v in ref]
 changed=[[(t+123)%20000 for t in p] for p in prompts[:-1]]+[prompts[-1]]
 _,other,_,_=f.prefill(changed)
 other=[(k[offsets[-2]:],v[offsets[-2]:]) for k,v in other]
 _,single,_,_=f.prefill([prompts[-1]])
 result=dict(neighbor_isolation=compare(target,other),sdpa_reference=compare(target,reference),single_cache=compare(target,single),prefill_logits_max_abs=float((logits-ref_logits).abs().max()))
 graph=BatchGraphDecoder(f.model,1)
 context=prompts[-1]+[first[-1]]
 full=f.prefill([context])[3]
 result['full_context']=top(full)
 for name,layers in [('packed',target),('single',single),('sdpa',reference)]:
  graph.load_device_lane(0,layers,first[-1],1);graph.step(ignore_eos=True)
  values=graph.logits_by_width[1].clone()
  result[name]=top(values)
  result[name]['max_logit_difference']=float((values.flatten()-full.flatten()).abs().max())
 Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
