"""Compare resumable prefill to full prefill, including mixed prompt lengths."""
import argparse,json,time
from pathlib import Path
import torch
from gpu_local_prefill import LocalPrefiller
from gpu_chunked_prefill import ChunkedPrefiller
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True)
p.add_argument('--no-paged',action='store_true')
a=p.parse_args();torch.set_num_threads(1)
local=LocalPrefiller(a.model_dir);chunked=ChunkedPrefiller(local);rows=[]
for lengths in ((32,128,257,512),(1024,)*4):
 prompts=[torch.randint(1000,20000,(n,),generator=torch.Generator().manual_seed(n+i)).tolist() for i,n in enumerate(lengths)]
 reference=local.prefill(prompts)
 for size in (2048,256,128,64):
  for repeat in (-1,0,1,2):
   state=chunked.start(prompts,size,paged=not a.no_paged);durations=[]
   while True:
    torch.cuda.synchronize();start=time.perf_counter();result=state.step();torch.cuda.synchronize();durations.append((time.perf_counter()-start)*1000)
    if result is not None:break
   tokens,layers,offsets,logits=result
   if repeat<0:continue
   errors=[float((x.float()-y.float()).abs().max()) for pair,other in zip(layers,reference[1]) for x,y in zip(pair,other)]
   cos=[float(torch.nn.functional.cosine_similarity(x.float().flatten(),y.float().flatten(),dim=0)) for pair,other in zip(layers,reference[1]) for x,y in zip(pair,other)]
   row=dict(lengths=lengths,chunk_tokens=size,repeat=repeat,step_ms=durations,total_ms=sum(durations),first_token_matches=sum(x==y for x,y in zip(tokens,reference[0])),requests=len(tokens),kv_max_abs=max(errors),kv_min_cosine=min(cos),logits_max_abs=float((logits-reference[3]).abs().max()))
   rows.append(row);Path(a.out).write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
