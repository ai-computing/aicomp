"""Owned hardware check: same RNGD source, ATOM whole/partial import, optional prefetch."""
import argparse,hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
p=argparse.ArgumentParser()
for name in ['model','family','label','cache-dir']:p.add_argument('--'+name,required=True)
p.add_argument('--kv-blocks',type=int,required=True);a=p.parse_args()
out=Path(f'stream265-{a.label}-fault-atom.json')
if out.exists():raise RuntimeError('refuse overwrite')
result={'model':a.model,'width':8192,'output_tokens':16,'batch_size':1,'cases':[],'scope':'actual partial-import cancellation, lane reuse and producer death; not performance'}
result['code_sha256']={name:hashlib.sha256((Path('stream265-code/opt_prime')/name).read_bytes()).hexdigest() for name in ['accel/platforms/atom/batching.py','accel/pd/pd_batching.py','accel/pd/prefill_server.py','accel/pd/layer_progress.py']}
nodes=list(Path('/dev').glob('rbln[0-9]*'));assert len(nodes)==1
engine=AtomRblnEngine(devices=[int(nodes[0].name[4:])],max_seq_len=9216,batch_size=1,kv_blocks=a.kv_blocks,cache_dir=a.cache_dir)
worker=None
try:
 engine.start(a.model);c=engine.model.config.get_text_config()
 base=AutoTokenizer.from_pretrained(a.model,local_files_only=True).encode('Seoul is the capital of South Korea. Count one two three.',add_special_tokens=True)
 prompts=[tuple((base*((n+len(base)-1)//len(base)))[:n]) for n in (2048,8192)]
 def make_worker():
  return SubprocessPrefillWorker('./furiosa_prefill_python.sh',['--engine','factory','--engine-factory',f'streaming_factories:{a.family}','--model',a.model,'--width','8192','--chip','0'],slots=1,layers=c.num_hidden_layers,kv_heads=c.num_key_value_heads,head_dim=getattr(c,'head_dim',None) or c.hidden_size//c.num_attention_heads,max_context=8192,env=dict(os.environ),stream_layers=True)
 worker=make_worker()

 from opt_prime.accel.pd.runtime import RequestCancelled
 def drain(batch,timeout=600):
  deadline=time.monotonic()+timeout
  while not batch.idle:
   assert time.monotonic()<deadline,'owner drain timeout'
   batch.step()
 def batcher(cache=False):
  return engine.batcher(prefix_cache=cache,prefill_worker=worker,local_prefill=False,remote_prefetch=False)
 worker.stream_layers=False
 b=batcher();reference=b.submit(PDRequest('reference',prompts[0],16));drain(b);expected=reference.result()
 worker.stream_layers=True
 cache=a.family!='gemma' and a.kv_blocks>1
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
 assert not b.blocks.index,'cancelled incomplete prefix published'
 try:old.result();raise AssertionError('cancelled request succeeded')
 except RequestCancelled:pass
 new=b.submit(PDRequest('reuse',prompts[0],16));drain(b)
 assert new.result()==expected and not b.poisoned
 result['cases'].append(dict(case='cancel-reuse',trigger=trigger,tokens_exact=True,expected_tokens=expected,actual_tokens=new.result(),idle=b.idle,prefix_cache=cache))
 out.write_text(json.dumps(result,indent=2)+'\n')
 b=batcher(cache);original=b._import_progress;trigger=[]
 def observe_death(r,**kw):
  before=b.stats['progress_spans'];original(r,**kw)
  if b.stats['progress_spans']>before and not r.pending.done() and not trigger:
   trigger.append(dict(spans=b.stats['progress_spans'],admitted=bool(getattr(r,'admitted',False))))
   worker._proc.kill()
 b._import_progress=observe_death
 failed=b.submit(PDRequest('die-partial',prompts[1],16));drain(b)
 assert trigger and not trigger[0]['admitted'],'did not intercept incomplete import'
 assert failed.exception() is not None and not b.poisoned and not b.blocks.index
 assert worker._proc.wait(timeout=10)==-9 and not worker.ready
 result['cases'].append(dict(case='producer-death',trigger=trigger,error=repr(failed.exception()),idle=b.idle,poisoned=b.poisoned,prefix_cache=cache))
 # A new source is required after the deliberate death above.
 worker.close();worker=make_worker()
 b=batcher();reference_long=b.submit(PDRequest('reference-long',prompts[0],512));drain(b);expected_long=reference_long.result()
 b=engine.batcher(prefix_cache=False,prefill_worker=worker,local_prefill=False,remote_prefetch=True)
 survivor=b.submit(PDRequest('survivor',prompts[0],512));victim=b.submit(PDRequest('prefetched-victim',prompts[1],16))
 deadline=time.monotonic()+600;trigger=[]
 while not b.idle:
  assert time.monotonic()<deadline,'prefetch isolation timeout'
  b.step();r=b._prefetched
  if not trigger and r is not None and r.pending is not None and not r.pending.done():
   progress=getattr(r.pending,'kv_progress',None)
   if progress is not None and progress.coverage.pending:
    active=next((x for x in b.decoding if x.request.request_id=='survivor'),None)
    assert active is not None,'survivor ended before source fault'
    trigger.append(dict(active_tokens=len(active.tokens),victim_slot=r.slot))
    assert r.slot<0,'prefetched KV unexpectedly owns active decode lane'
    worker._proc.kill()
 assert trigger and victim.exception() is not None and survivor.result()==expected_long and not b.poisoned
 assert worker._proc.wait(timeout=10)==-9 and not worker.ready
 result['cases'].append(dict(case='prefetch-source-death',trigger=trigger,expected_tokens=expected_long,actual_tokens=survivor.result(),tokens_exact=True,error=repr(victim.exception()),idle=b.idle,poisoned=b.poisoned))
 result['stage']='complete'
except Exception as error:
 result['stage']='failed';result['error']=repr(error);raise
finally:
 if worker is not None:worker.close()
 engine.close();out.write_text(json.dumps(result,indent=2)+'\n')
