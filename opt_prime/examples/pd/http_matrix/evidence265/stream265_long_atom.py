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
out=Path(f'stream265-{a.label}-long-atom.json')
if out.exists():raise RuntimeError('refuse overwrite')
result={'model':a.model,'width':8192,'output_tokens':32,'batch_size':1,'cases':[],'scope':'alternating same-route timing, synthetic2K/8K fixed32output, B1 and two queued requests'}
result['code_sha256']={name:hashlib.sha256((Path('stream265-code/opt_prime')/name).read_bytes()).hexdigest() for name in ['accel/platforms/atom/batching.py','accel/pd/pd_batching.py','accel/pd/prefill_server.py','accel/pd/layer_progress.py']}
nodes=list(Path('/dev').glob('rbln[0-9]*'));assert len(nodes)==1
engine=AtomRblnEngine(devices=[int(nodes[0].name[4:])],max_seq_len=9216,batch_size=1,kv_blocks=a.kv_blocks,cache_dir=a.cache_dir)
worker=None
try:
 engine.start(a.model);c=engine.model.config.get_text_config()
 base=AutoTokenizer.from_pretrained(a.model,local_files_only=True).encode('Seoul is the capital of South Korea. Count one two three.',add_special_tokens=True)
 prompts=[tuple((base*((n+len(base)-1)//len(base)))[:n]) for n in (2048,8192)]
 worker=SubprocessPrefillWorker('./furiosa_prefill_python.sh',['--engine','factory','--engine-factory',f'streaming_factories:{a.family}','--model',a.model,'--width','8192','--chip','0'],slots=1,layers=c.num_hidden_layers,kv_heads=c.num_key_value_heads,head_dim=getattr(c,'head_dim',None) or c.hidden_size//c.num_attention_heads,max_context=8192,env=dict(os.environ),stream_layers=True)
 reference=None
 for index,(stream,prefetch) in enumerate([(s,p) for p in (False,True) for s in (False,True,False,True,True,False,False,True)]):
  with worker._lock:assert not worker._futures
  worker.stream_layers=stream
  b=engine.batcher(prefix_cache=False,prefill_worker=worker,local_prefill=False,remote_prefetch=prefetch)
  traces=[];original=b._import_progress
  def observed(r,**kw):
   before=b.stats['progress_spans'];active=len(b.decoding);done=r.pending.done()
   original(r,**kw)
   if b.stats['progress_spans']>before:traces.append(dict(spans=b.stats['progress_spans']-before,active_decode=active,source_done_before=done,source_done_after=r.pending.done()))
  b._import_progress=observed
  arrivals=[[],[]];start=time.perf_counter()
  f=[b.submit(PDRequest(f'{index}-{i}',ids,32),on_token=lambda token,i=i:arrivals[i].append(time.perf_counter()-start)) for i,ids in enumerate(prompts)]
  b.run_until_idle();tokens=[v.result(timeout=1) for v in f]
  if reference is None:reference=tokens
  elapsed=time.perf_counter()-start
  assert all(len(x)==32 for x in arrivals)
  row=dict(index=index,warmup=index%8<2,elapsed_s=elapsed,arrivals_s=arrivals,ttft_s=[x[0] for x in arrivals],stream=stream,prefetch=prefetch,tokens=tokens,tokens_exact=tokens==reference,progress_spans=b.stats['progress_spans'],imports=traces,idle=b.idle)
  result['cases'].append(row);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
  assert row['tokens_exact'] and b.idle and all(e['active_decode']==0 for e in traces)
  assert (row['progress_spans']>0)==stream
 result['stage']='complete'
except Exception as error:
 result['stage']='failed';result['error']=repr(error);raise
finally:
 if worker is not None:worker.close()
 engine.close();out.write_text(json.dumps(result,indent=2)+'\n')
