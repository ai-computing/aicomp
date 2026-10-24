import os,subprocess,json
from pathlib import Path
root=Path('/home/gspark/workspace/npu_offload/analysis/serving-matrix-201')
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_DEVICE_BACKEND_AUTOLOAD='0',PYTHONPATH='/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime:/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/examples/pd:/tmp/rngd-gpu201-src:/home/gspark/workspace/aicomp-live-pd-234/opt_prime/examples/pd/serving_matrix')
rows=[]
for label,family,model,width in [('qwen06b','qwen','/home/gspark/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca',128),('qwen4b','qwen','/tmp/serving-matrix-models/Qwen3-4B',128),('gemma','gemma','/tmp/serving-matrix-models/Gemma3-1B',128),('koni','koni','/tmp/serving-matrix-models/KONI-8B',512)]:
 log=root/f'stream265-{label}-source.log'
 print('start',label,flush=True)
 with log.open('x') as f:
  p=subprocess.run(['/tmp/codex-rngd-gpu-201/bin/python','/tmp/stream265-factory-hardware.py','--label',label,'--family',family,'--model',model,'--width',str(width)],env=env,stdout=f,stderr=subprocess.STDOUT)
 rows.append({'label':label,'exit_code':p.returncode});(root/'stream265-source-queue.json').write_text(json.dumps(rows,indent=2)+'\n')
 print('finished',label,p.returncode,flush=True)
 if p.returncode:break
