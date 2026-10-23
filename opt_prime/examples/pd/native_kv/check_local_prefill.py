"""Live ragged prefill and direct device cache ownership check."""
import argparse,json
from pathlib import Path
import torch
from gpu_local_prefill import LocalPrefiller
from gpu_batch_decoder import BatchGraphDecoder
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True)
a=p.parse_args();torch.set_num_threads(1)
f=LocalPrefiller(a.model_dir)
prompts=[torch.randint(1000,20000,(n,),generator=torch.Generator().manual_seed(n)).tolist() for n in (32,128,257,512)]
with torch.inference_mode():
 first,packed,offsets,logits=f.prefill(prompts);rows=[]
 for lane,prompt in enumerate(prompts):
  single,layers,_,ref=f.prefill([prompt])
  candidate=[(k[offsets[lane]:offsets[lane+1]],v[offsets[lane]:offsets[lane+1]]) for k,v in packed]
  pairs=[(x,y) for pair,other in zip(candidate,layers) for x,y in zip(pair,other)]
  rows.append(dict(length=len(prompt),first_equal=first[lane]==single[0],kv_max_abs=max(float((x.float()-y.float()).abs().max()) for x,y in pairs),kv_min_cosine=min(float(torch.nn.functional.cosine_similarity(x.float().flatten(),y.float().flatten(),dim=0)) for x,y in pairs)))
 graph=BatchGraphDecoder(f.model,batch_size=1)
 for lane,prompt in enumerate(prompts):
  candidate=[(k[offsets[lane]:offsets[lane+1]],v[offsets[lane]:offsets[lane+1]]) for k,v in packed]
  graph.load_device_lane(0,candidate,first[lane],8)
  assert all(torch.equal(graph.keys[i][0,:len(prompt)],pair[0]) and torch.equal(graph.values[i][0,:len(prompt)],pair[1]) for i,pair in enumerate(candidate))
  actual=[graph.step(ignore_eos=True)[0] for _ in range(8)]
  context=prompt+[first[lane]];expected=[]
  for _ in range(8):
   token=f.prefill([context])[0][0];expected.append(token);context.append(token)
  single_first,single_layers,_,_=f.prefill([prompt])
  graph.load_device_lane(0,single_layers,single_first[0],8)
  single_actual=[graph.step(ignore_eos=True)[0] for _ in range(8)]
  rows[lane]['single_cache_matches']=sum(x==y for x,y in zip(single_actual,expected))
  rows[lane]['single_cache_actual']=single_actual
  rows[lane].update(continuation_matches=sum(x==y for x,y in zip(actual,expected)),continuation_total=8,actual=actual,expected=expected)
  Path(a.out).write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows[lane]),flush=True)
 print('allocated_GiB',torch.cuda.memory_allocated()/2**30,flush=True)
