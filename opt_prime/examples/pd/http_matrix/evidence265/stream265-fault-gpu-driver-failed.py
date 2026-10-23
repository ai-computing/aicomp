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
root=Path('/home/gspark/workspace/npu_offload/analysis/serving-matrix-201');out=root/f'stream265-{a.label}-fault-gpu.json'
if out.exists():raise RuntimeError('refuse overwrite')
print('GPU loading',a.label,flush=True)
model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16).to('cuda:0').eval();c=model.config.get_text_config()
base=AutoTokenizer.from_pretrained(a.model).encode('Seoul is the capital of South Korea. Count one two three.',add_special_tokens=True)
prompts=[tuple((base*((n+len(base)-1)//len(base)))[:n]) for n in [a.width//4,a.width]]
env=dict(os.environ,PYTHONPATH='/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime:/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/examples/pd:/tmp/rngd-gpu201-src:/home/gspark/workspace/aicomp-live-pd-234/opt_prime/examples/pd/serving_matrix')
result={'model':a.label,'width':a.width,'input_lengths':list(map(len,prompts)),'output_tokens':512,'concurrency':2,'runs':[],
 'scope':'actual streaming cancellation/reuse and source death with active512-token decode; same-route parity, not performance'}
result['source_sha256']={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('prefill_server.py','/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/opt_prime/accel/pd/prefill_server.py'),('streaming_factories.py','/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/examples/pd/streaming_factories.py')]}
worker=None
try:
 worker=SubprocessPrefillWorker('/tmp/stream265-python.sh',['--engine','factory','--engine-factory',f'streaming_factories:{a.family}','--model',a.model,'--width',str(a.width),'--chip','0'],slots=2,layers=c.num_hidden_layers,kv_heads=c.num_key_value_heads,head_dim=getattr(c,'head_dim',None) or c.hidden_size//c.num_attention_heads,max_context=a.width+1024,env=env,stream_layers=True)
 from opt_prime.accel.pd.runtime import RequestCancelled
 def drain(batch,timeout=600):
  deadline=time.monotonic()+timeout
  while not batch.idle:
   assert time.monotonic()<deadline,'owner drain timeout'
   batch.step()
 def batcher(cache=False):
  return ObservedBatcher(model,worker,slots=2,max_context=a.width+1024,page_size=64,prefix_cache=False)
 worker.stream_layers=False
 b=batcher();reference=b.submit(PDRequest('reference',prompts[0],16));drain(b);expected=reference.result()
 worker.stream_layers=True
 cache=False
 b=batcher(cache);original=b._import_progress;trigger=[]
 def observe_cancel(r,**kw):
  before=b.stats['progress_spans'];original(r,**kw)
  if b.stats['progress_spans']>before and not r.pending.done() and not trigger:
   trigger.append(dict(spans=b.stats['progress_spans'],admitted=bool(getattr(r,'admitted',False))))
 b._import_progress=observe_cancel
 old=b.submit(PDRequest('cancel-partial',prompts[1],16));deadline=time.monotonic()+600
 while not trigger and not old.done():
  assert time.monotonic()<deadline;b.step()
 assert trigger and not trigger[0]['admitted'] and not old.done(),'did not intercept incomplete import'
 assert b.cancel('cancel-partial');b.step()
 try:old.result();raise AssertionError('cancelled request succeeded')
 except RequestCancelled:pass
 new=b.submit(PDRequest('reuse',prompts[0],16));drain(b)
 assert new.result()==expected and not b.poisoned
 result['runs'].append(dict(case='cancel-reuse',trigger=trigger,tokens_exact=True,idle=b.idle,prefix_cache=cache))
 out.write_text(json.dumps(result,indent=2)+'\n')
 b=batcher();reference_long=b.submit(PDRequest('reference-long',prompts[0],512));drain(b);expected_long=reference_long.result()
 b=batcher(cache);original=b._import_progress;trigger=[]
 def observe_death(r,**kw):
  before=b.stats['progress_spans'];original(r,**kw)
  if r.request.request_id=='die-partial' and b.stats['progress_spans']>before and not r.pending.done() and not trigger:
   assert any(x.request.request_id=='survivor' for x in b.decoding),'active decode ended before fault'
   trigger.append(dict(spans=b.stats['progress_spans'],admitted=False,active_tokens=len(next(x for x in b.decoding if x.request.request_id=='survivor').tokens)))
   worker._proc.kill()
 b._import_progress=observe_death
 survivor=b.submit(PDRequest('survivor',prompts[0],512))
 failed=b.submit(PDRequest('die-partial',prompts[1],16));drain(b)
 assert survivor.result()==expected_long,'producer failure changed active decode'
 assert trigger and not trigger[0]['admitted'],'did not intercept incomplete import'
 assert failed.exception() is not None and not b.poisoned
 assert worker._proc.wait(timeout=10)==-9 and not worker.ready
 result['runs'].append(dict(case='producer-death',trigger=trigger,error=repr(failed.exception()),idle=b.idle,poisoned=b.poisoned,prefix_cache=cache))
 result['stage']='complete'
except Exception as error:
 result['stage']='failed';result['error']=repr(error);raise
finally:
 if worker is not None:worker.close()
 out.write_text(json.dumps(result,indent=2)+'\n')
