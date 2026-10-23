import argparse,json,time
from pathlib import Path
import torch
from gpu_local_prefill import LocalPrefiller
p=argparse.ArgumentParser()
p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True)
a=p.parse_args();torch.set_num_threads(1)
prefill=LocalPrefiller(a.model_dir);rows=[]
for batch in (1,4,8):
 for n in (128,512,1024):
  prompts=[torch.randint(1000,20000,(n,),generator=torch.Generator().manual_seed(n+i)).tolist() for i in range(batch)]
  times=[]
  for repeat in range(6):
   torch.cuda.synchronize();start=time.perf_counter()
   tokens,layers,offsets,logits=prefill.prefill(prompts)
   if repeat:times.append((time.perf_counter()-start)*1000)
  row={'batch':batch,'input_tokens':n,'prefill_ms':times,'first_tokens':tokens}
  rows.append(row);Path(a.out).write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
