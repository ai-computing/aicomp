"""Matched GPU PD baseline/copy-overlap serving runs; abort on failed parity."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
runtime=Path('/home/gspark/workspace/aicomp-chunk-copy-47')
harness=Path('/home/gspark/workspace/aicomp-heteroperf-248/opt_prime/examples/pd/http_matrix/benchmark.py')
models=[('qwen06b','/home/gspark/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'),('qwen4b','/tmp/serving-matrix-models/Qwen3-4B'),('koni','/tmp/serving-matrix-models/KONI-8B'),('gemma','/tmp/serving-matrix-models/Gemma3-1B')]
bootstrap="""import sys,importlib.util,runpy
spec=importlib.util.spec_from_file_location('opt_prime.accel.serve.llmd','/tmp/llmd240.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
runpy.run_path('/home/gspark/workspace/aicomp-chunk-copy-47/opt_prime/examples/pd/serve_batched.py',run_name='__main__')
"""
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONPATH=str(runtime/'opt_prime'),HF_HUB_OFFLINE='1')
for i,(label,model) in enumerate(models):
 for flag in ([False,True] if i%2==0 else [True,False]):
  stem=f'chunk47-{label}-overlap{int(flag)}-a';dest=root/(stem+'.json')
  if dest.exists():raise RuntimeError('Refusing overwrite '+str(dest))
  argv=[sys.executable,'-c',bootstrap,'--model',model,'--device','cuda:1','--dtype','bfloat16','--slots','4','--max-context','9216','--max-output','128','--kv','paged','--page-size','64','--chunk','256','--port','{port}','--pd-prefill-device','cuda:0','--pd-prefill-no-prefix-cache','--pd-prefill-chunk','256']
  if flag:argv+=['--pd-prefill-copy-overlap']
  config={'servers':[argv],'metadata':dict(physical_gpus=2,physical_atom=0,physical_rngd=0,slots_per_decoder=4,producer_cache=False,decoder_cache=False,producer_chunk=256,local_chunk=256,context=9216)}
  config_path=root/(stem+'-config.json');config_path.write_text(json.dumps(config,indent=2)+'\n')
  cmd=[sys.executable,str(harness),'--model',model,'--mode','pd','--server-config',str(config_path),'--lengths','128','8192','--outputs','128','--concurrency','1','8','--repeats','3','--output',str(dest),'--reference',str(root/f'perf248-{label}-native-long128.json')]
  print('START',label,flag,json.dumps(cmd),flush=True)
  with dest.with_suffix('.log').open('w') as log:run=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  print('END',label,flag,run.returncode,flush=True)
  if run.returncode:sys.exit(run.returncode)
  data=json.loads(dest.read_text())
  if not data['passed'] or not data['parity_passed']:raise RuntimeError('Parity failed '+stem)
