"""Alternating same-engine whole/stream test; all runs and warmups retained."""
import argparse,hashlib,json,os,time,statistics
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker,PDBatcher
class ObservedBatcher(PDBatcher):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw);self.import_trace=[]
 def _import_progress(self,r):
  count=self.stats['progress_spans'];done_before=r.pending.done();start=time.perf_counter()
  super()._import_progress(r)
  if self.stats['progress_spans']>count:
   self.import_trace.append({'request':r.request.request_id,'spans':self.stats['progress_spans']-count,'start_s':start,'end_s':time.perf_counter(),'source_done_before':done_before,'source_done_after':r.pending.done()})
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--family',required=True);p.add_argument('--label',required=True);p.add_argument('--width',type=int,default=8192);a=p.parse_args()
root=Path('/home/gspark/workspace/npu_offload/analysis/serving-matrix-201');out=root/f'stream265-{a.label}-c4-gpu.json'
if out.exists():raise RuntimeError('refuse overwrite')
print('GPU loading',a.label,flush=True)
model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16).to('cuda:0').eval();c=model.config.get_text_config()
base=AutoTokenizer.from_pretrained(a.model).encode('Seoul is the capital of South Korea. Count one two three.',add_special_tokens=True)
prompts=[tuple((base*((n+len(base)-1)//len(base)))[:n]) for n in [a.width//4,a.width,a.width//4,a.width]]
env=dict(os.environ,PYTHONPATH='/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime:/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/examples/pd:/tmp/rngd-gpu201-src:/home/gspark/workspace/aicomp-live-pd-234/opt_prime/examples/pd/serving_matrix')
result={'model':a.label,'width':a.width,'input_lengths':list(map(len,prompts)),'output_tokens':128,'concurrency':4,'runs':[],
 'scope':'fixed compiled width, four queued repeated synthetic prompts, fixed128 output, caches off; host import trace is not device-kernel overlap'}
result['source_sha256']={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('prefill_server.py','/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/opt_prime/accel/pd/prefill_server.py'),('streaming_factories.py','/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/examples/pd/streaming_factories.py')]}
worker=None
try:
 worker=SubprocessPrefillWorker('/tmp/stream265-python.sh',['--engine','factory','--engine-factory',f'streaming_factories:{a.family}','--model',a.model,'--width',str(a.width),'--chip','0'],slots=4,layers=c.num_hidden_layers,kv_heads=c.num_key_value_heads,head_dim=getattr(c,'head_dim',None) or c.hidden_size//c.num_attention_heads,max_context=a.width+128,env=env,stream_layers=True)
 reference=None
 for index,stream in enumerate([False,True,False,True,True,False,False,True]):
  # Handshake required capability at startup. Change request mode only with no outstanding futures.
  with worker._lock:
   assert not worker._futures, "request mode switch requires drained source"
  worker.stream_layers=stream
  batch=ObservedBatcher(model,worker,slots=4,max_context=a.width+128,page_size=64,prefix_cache=False)
  arrivals=[[],[],[],[]];start=time.perf_counter()
  futures=[batch.submit(PDRequest(f'{index}-{i}',ids,128),on_token=lambda token,i=i:arrivals[i].append(time.perf_counter()-start)) for i,ids in enumerate(prompts)]
  batch.run_until_idle();tokens=[f.result(timeout=1) for f in futures];elapsed=time.perf_counter()-start
  if reference is None:reference=tokens
  row={'stream_layers':stream,'warmup':index<2,'elapsed_s':elapsed,'ttft_s':[v[0] for v in arrivals],'token_times_s':arrivals,'tokens':tokens,'tokens_exact':tokens==reference,'progress_spans':batch.stats['progress_spans'],'imports':batch.import_trace,'idle':batch.idle}
  result['runs'].append(row);out.write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({k:v for k,v in row.items() if k not in ['tokens','imports','token_times_s']}),flush=True)
  assert batch.idle and row['tokens_exact']
  del batch
 measured=[r for r in result['runs'] if not r['warmup']]
 result['summary']={str(mode):{'median_elapsed_s':statistics.median(r['elapsed_s'] for r in measured if r['stream_layers']==mode),'median_ttft_s':[statistics.median(r['ttft_s'][i] for r in measured if r['stream_layers']==mode) for i in range(4)],'imports_before_source_done':sum(not e['source_done_before'] for r in measured if r['stream_layers']==mode for e in r['imports'])} for mode in [False,True]}
 result['stage']='complete'
except Exception as error:
 result['stage']='failed';result['error']=repr(error);raise
finally:
 if worker is not None:worker.close()
 out.write_text(json.dumps(result,indent=2)+'\n')
