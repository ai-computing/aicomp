"""Assert prefix immutability and isolation of interleaved prefill sessions."""
import argparse,json
from pathlib import Path
import torch
from gpu_local_prefill import LocalPrefiller
from gpu_chunked_prefill import ChunkedPrefiller
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True);p.add_argument('--equal-length',action='store_true');a=p.parse_args()
torch.set_num_threads(1)
f=ChunkedPrefiller(LocalPrefiller(a.model_dir))
prompts=[torch.randint(1000,20000,(n,),generator=torch.Generator().manual_seed(n)).tolist() for n in ((512,)*4 if a.equal_length else (32,128,257,512))]
changed=[[(x+71)%20000 for x in p] for p in prompts[:-1]]+[prompts[-1]]
sessions=[f.start(prompts,128),f.start(changed,128)]
previous=[None,None];results=[None,None];checks=0
while not all(s.done for s in sessions):
 for index,s in enumerate(sessions):
  if s.done:continue
  old=previous[index];result=s.step()
  if old is not None:
   counts,snapshot=old
   for pair,ref in zip(s.layers,snapshot):
    for tensor,copy in zip(pair,ref):
     for lane,n in enumerate(counts):
      begin=s.offsets[lane]
      assert torch.equal(tensor[begin:begin+n],copy[begin:begin+n])
      checks+=1
  previous[index]=(list(s.processed),[(k.clone(),v.clone()) for k,v in s.layers])
  if result is not None:results[index]=result
begin=sessions[0].offsets[-2]
assert all(torch.equal(x[begin:],y[begin:]) for pair,other in zip(results[0][1],results[1][1]) for x,y in zip(pair,other))
assert results[0][0][-1]==results[1][0][-1]
try:sessions[0].step()
except RuntimeError:pass
else:raise AssertionError('completed session accepted another step')
result={'prefix_slice_checks':checks,'neighbor_target_kv_exact':True,'interleaved_sessions':2,'completed_reentry_rejected':True}
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
