"""Dump RNGD fixture KV/logits for the numerical diagnostic."""
import random,torch,json
from types import SimpleNamespace
from huggingface_hub import snapshot_download
from opt_prime.accel.pd.prefill_server import _RngdQwen
rng=random.Random(0)
prompts=[[rng.randrange(1000,30000) for _ in range(rng.randint(16,96))] for i in range(12)]
a=SimpleNamespace(model=snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',local_files_only=True),model_id='Qwen/Qwen2.5-0.5B-Instruct',width='256',chip=0)
e=_RngdQwen(a)
records={}
for i in [1,6,11]:
 k,v,logits=e.prefill(prompts[i]);records[i]=dict(prompt=prompts[i],keys=k,values=v,logits=logits)
 print(i,logits.topk(3),flush=True)
torch.save(records,'rngd_diagnostic.pt')
