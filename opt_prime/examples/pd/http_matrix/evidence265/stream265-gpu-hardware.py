import argparse,json,os,time,hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker,PDBatcher
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--family',required=True);p.add_argument('--label',required=True);p.add_argument('--width',type=int,required=True);a=p.parse_args()
out=Path('/home/gspark/workspace/npu_offload/analysis/serving-matrix-201')/f'stream265-{a.label}-gpu.json'
if out.exists():raise RuntimeError('refuse overwrite')
print('loading GPU',a.label,flush=True)
model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16).to('cuda:0').eval();c=model.config.get_text_config()
base=AutoTokenizer.from_pretrained(a.model).encode('Seoul is the capital of South Korea. Count one two three.',add_special_tokens=True)
prompts=[tuple(base),tuple((base*((a.width+len(base)-1)//len(base)))[:a.width])]
env=dict(os.environ,PYTHONPATH='/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime:/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/examples/pd:/tmp/rngd-gpu201-src:/home/gspark/workspace/aicomp-live-pd-234/opt_prime/examples/pd/serving_matrix')
result={'model':a.label,'width':a.width,'cases':[],'scope':'same-route real RNGD to GPU token parity, not cross-backend quality or speedup'}
result['source_sha256']={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('prefill_server.py','/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/opt_prime/accel/pd/prefill_server.py'),('streaming_factories.py','/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/examples/pd/streaming_factories.py')]}
try:
 for stream in [False,True]:
  print('start mode',stream,flush=True)
  worker=SubprocessPrefillWorker('/tmp/stream265-python.sh',['--engine','factory','--engine-factory',f'streaming_factories:{a.family}','--model',a.model,'--width',str(a.width),'--chip','0'],slots=2,layers=c.num_hidden_layers,kv_heads=c.num_key_value_heads,head_dim=getattr(c,'head_dim',None) or c.hidden_size//c.num_attention_heads,max_context=a.width+64,env=env,stream_layers=stream)
  try:
   batch=PDBatcher(model,worker,slots=2,max_context=a.width+64,page_size=64,prefix_cache=False)
   futures=[batch.submit(PDRequest(f'{stream}-{i}',ids,8)) for i,ids in enumerate(prompts)]
   batch.run_until_idle();tokens=[f.result(timeout=1) for f in futures]
   case={'stream_layers':stream,'tokens':tokens,'stats':dict(batch.stats),'idle':batch.idle}
   result['cases'].append(case);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(case),flush=True)
  finally:worker.close()
 result['tokens_exact']=result['cases'][0]['tokens']==result['cases'][1]['tokens']
 assert result['tokens_exact']
 assert result['cases'][1]['stats']['progress_spans']>0
 result['stage']='complete'
except Exception as error:
 result['stage']='failed';result['error']=repr(error);raise
finally:out.write_text(json.dumps(result,indent=2)+'\n')
