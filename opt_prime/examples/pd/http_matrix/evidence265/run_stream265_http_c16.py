"""Prepare or run c16 HTTP native/whole/stream checks; launch only after devices are free."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
runtime = Path('/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime')
harness = Path('/home/gspark/workspace/aicomp-heteroperf-248/opt_prime/examples/pd/http_matrix/benchmark.py')
models = [
 ('qwen06b','qwen','/home/gspark/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'),
 ('qwen4b','qwen','/tmp/serving-matrix-models/Qwen3-4B'),
 ('gemma','gemma','/tmp/serving-matrix-models/Gemma3-1B'),
 ('koni','koni','/tmp/serving-matrix-models/KONI-8B'),
]
bootstrap = """import sys,importlib.util,runpy
s=importlib.util.spec_from_file_location('opt_prime.accel.serve.llmd','/tmp/llmd240.py')
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
runpy.run_path('/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/examples/pd/serve_batched.py',run_name='__main__')
"""
env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCH_DEVICE_BACKEND_AUTOLOAD='0',
 HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
 PYTHONPATH=f'{runtime}:{runtime}/examples/pd:/tmp/rngd-gpu201-src:/home/gspark/workspace/aicomp-live-pd-234/opt_prime/examples/pd/serving_matrix')

def plan(label,family,model,mode):
 stem=f'stream265-{label}-http-c16-{mode}'
 output=root/(stem+'.json')
 argv=[sys.executable,'-c',bootstrap,'--model',model,'--device','cuda:0','--dtype','bfloat16',
       '--slots','4','--max-context','9216','--max-output','128','--kv','paged','--page-size','64',
       '--chunk','256','--port','{port}']
 if mode!='native':
  argv += ['--pd-prefill-python','/tmp/stream265-python.sh','--pd-prefill-args',
           f'--engine factory --engine-factory streaming_factories:{family} --width 8192 --chip 0']
 if mode=='stream':argv+=['--pd-prefill-stream-layers']
 config=dict(servers=[argv],metadata=dict(physical_gpus=1,physical_rngd=int(mode!='native'),physical_atom=0,
    slots_per_decoder=4,producer_cache=False,decoder_cache=False,producer_chunk=8192 if mode!='native' else 256,
    local_chunk=256,context=9216))
 configpath=root/(stem+'-config.json')
 cmd=[sys.executable,str(harness),'--model',model,'--mode','native' if mode=='native' else 'pd',
      '--server-config',str(configpath),'--lengths','128','512','2048','8192','--outputs','32','128',
      '--concurrency','16','--repeats','1','--startup-timeout','1800','--output',str(output)]
 # Compare stream to the same producer's whole path. Native-vs-whole remains a separate numerical comparison.
 if mode=='stream':cmd+=['--reference',str(root/f'stream265-{label}-http-c16-whole.json')]
 return output,configpath,config,cmd

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
 plans=[(label,mode,*plan(label,family,model,mode)) for label,family,model in models for mode in ['native','whole','stream']]
 for label,mode,out,cpath,config,cmd in plans:
  assert config['servers'][0].count('{port}')==1
  assert ('--pd-prefill-stream-layers' in config['servers'][0])==(mode=='stream')
  assert ('--reference' in cmd)==(mode=='stream')
  assert not out.exists(),f'Refuse overwrite: {out}'
 if not a.run:
  print(json.dumps({'prepared_cases':len(plans),'models':[v[0] for v in models],
   'concurrency':16,'execution_slots':4,'lengths':[128,512,2048,8192],'outputs':[32,128],
   'repeats':1,'scope':'HTTP correctness smoke; one measured batch per condition, not statistical performance proof'}));return
 rows=[]
 for label,mode,out,cpath,config,cmd in plans:
  if mode=='stream':
   ref=json.loads((root/f'stream265-{label}-http-c16-whole.json').read_text())
   if not ref.get('passed'):raise RuntimeError('Whole path did not finish; diagnose before stream comparison')
  cpath.write_text(json.dumps(config,indent=2)+'\n')
  print('START',label,mode,flush=True)
  with out.with_suffix('.log').open('x') as log:
   r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  rows.append(dict(model=label,mode=mode,exit_code=r.returncode))
  (root/'stream265-http-c16-queue.json').write_text(json.dumps(rows,indent=2)+'\n')
  print('END',label,mode,r.returncode,flush=True)
  data=json.loads(out.read_text())
  if not data.get('passed'):raise RuntimeError('Runtime failure; preserve artifacts and diagnose')
 raise SystemExit(int(any(r['exit_code'] for r in rows)))

if __name__=='__main__':main()
