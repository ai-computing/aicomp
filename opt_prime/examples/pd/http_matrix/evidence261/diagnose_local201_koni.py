"""Separate GPU batching/chunk effects from a retained exact RNGD KV export."""
import json
from pathlib import Path
from concurrent.futures import Future
import torch
from transformers import AutoModelForCausalLM
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.pd_batching import PDBatcher,PrefillOut
r=Path(__file__).resolve().parent
output=r/'local201-koni-diagnostic.json'
if output.exists(): raise FileExistsError(output)
torch.set_num_threads(4)
e=torch.load(r/'rngd-koni-8b-8192-parts221-layout.pt',weights_only=True,mmap=True)
d=json.loads((r/'local201-koni-matrix-a.json').read_text())
assert e['ids']==d['prompts']['8192']
model=AutoModelForCausalLM.from_pretrained('/tmp/serving-matrix-models/KONI-8B',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
class Replay:
 def submit(self,request,send_from=0,slot=None):
  assert list(request.input_ids)==e['ids']
  f=Future();f.set_result(PrefillOut(int(e['logits'].argmax()),send_from,e['keys'][:,:,send_from:],e['values'][:,:,send_from:]));return f
rows=[]
for mode in ['native256','native8192','retained_rngd_kv']:
 kw=dict(slots=4,max_context=9216,kv='paged',page_size=64,prefix_cache=False)
 b=PDBatcher(model,Replay(),**kw) if mode=='retained_rngd_kv' else ContinuousBatcher(model,prefill_chunk=int(mode[6:]),**kw)
 f=b.submit(PDRequest(mode,tuple(e['ids']),128));b.run_until_idle();tokens=f.result()
 native=d['references']['8192:128']['tokens'];live=next(x['rows'][0]['tokens'] for x in d['correctness_batches'] if x['input_tokens']==8192 and x['output_tokens']==128)
 row=dict(mode=mode,tokens=tokens,equals_native=tokens==native,equals_live=tokens==live,first_diff_native=next((i for i,(a,z) in enumerate(zip(tokens,native)) if a!=z),None));rows.append(row);print(json.dumps(row),flush=True)
 del b;torch.cuda.empty_cache()
(r/'local201-koni-diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')
