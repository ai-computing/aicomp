import hashlib,json,sys
from pathlib import Path
hub=Path.home()/'.cache/huggingface/hub'
models=[('qwen06b','Qwen--Qwen3-0.6B',None),('qwen4b','Qwen--Qwen3-4B','Qwen3-4B'),('koni','KISTI-KONI--KONI-Llama3.1-8B-Instruct-20241024','KONI-8B'),('gemma','unsloth--gemma-3-1b-it','Gemma3-1B')]
result={}
for label,repo,alias in models:
 cache=hub/('models--'+repo)
 revision=(cache/'refs/main').read_text().strip() if (cache/'refs/main').exists() else None
 if sys.argv[1]=='local' and alias: folder=Path('/tmp/serving-matrix-models')/alias
 else: folder=cache/'snapshots'/revision
 files=sorted(set(folder.glob('*.safetensors')) | {folder/'config.json',folder/'generation_config.json',folder/'tokenizer.json'})
 row={}
 for p in files:
  if not p.exists():continue
  h=hashlib.sha256()
  with p.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
  row[p.name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
 assert any(name.endswith('.safetensors') for name in row)
 result[label]={'revision_ref':revision,'files':row}
print(json.dumps(result,indent=2))
