"""Finite ABBA native ATOM matrix; run only after all build sessions have exited."""
from pathlib import Path
import json, subprocess
root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
repo=root/'aicomp-integration'
python=root/'experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv/bin/python'
models={
 '05':Path('/home/gspark/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'),
 '15':Path('/home/gspark/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')}
commands=[]
for size in models:
 for index,chunk in enumerate((64,128,128,64)):
  build=f'atom-qwen{size}-c{chunk}-build'
  if size=='05' and chunk==64:build='atom-qwen05-c64-r1'
  config=json.loads((out/build/'result.json').read_text())
  assert config['passed']
  name=f'atom-qwen{size}-abba-{index}-c{chunk}'
  args=[str(python),'opt_prime/examples/pd/validate_atom_checkpoint.py',
        '--model-dir',str(models[size]),'--out',str(out/name),
        '--artifact',str(out/build/'compiled'),'--chunk',str(chunk),
        '--preflight-module-dir',str(root/'hetero-npu-p2p/examples/koni')]
  commands.append(dict(cwd=str(repo),argv=args,log=str(out/(name+'.log'))))
(out/'matrix-commands.json').write_text(json.dumps(commands,indent=2))
for cmd in commands:
 print('START',cmd['log'],flush=True)
 with open(cmd['log'],'w') as log:
  result=subprocess.run(cmd['argv'],cwd=cmd['cwd'],stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise SystemExit(result.returncode)
 print('PASS',cmd['log'],flush=True)
