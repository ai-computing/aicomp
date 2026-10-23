"""Read-only support gates and cached ATOM generation; no implicit compilation."""
import json,gc,torch,traceback
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from opt_prime.accel.pd.npu.model import DenseModel
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
models=['Qwen/Qwen2.5-0.5B-Instruct','Qwen/Qwen3-0.6B','Qwen/Qwen3-8B-FP8','KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024','unsloth/gemma-3-1b-it']
physical=int(next(Path('/dev').glob('rbln[0-9]*')).name[4:])
result={}
for mid in models:
 r={};result[mid]=r
 path=snapshot_download(mid,local_files_only=True)
 cfg=json.loads((Path(path)/'config.json').read_text());r['model_type']=cfg['model_type']
 try:
  spec=DenseModel.load(path,mid);r['rngd_research_gate']={'supported':True,'slots':spec.caches}
 except Exception as ex:r['rngd_research_gate']={'supported':False,'error':str(ex)}
 e=AtomRblnEngine(devices=[physical],max_seq_len=2048,batch_size=1)
 if not (Path(e._dir(mid))/'config.json').is_file():
  r['atom_standalone']={'status':'not_run','reason':'No compiled B1 artifact; no implicit compile'}
 else:
  try:
   e.start(mid);tok=AutoTokenizer.from_pretrained(mid);outs=[]
   for text in ['What is the capital of France?','Compute 2 + 3.','한국의 수도는 어디인가요?']:
    ids=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,return_dict=False)
    a,info=e.generate(ids,max_new=16);b,_=e.generate(ids,max_new=16)
    outs.append(dict(prompt=text,tokens=a,text=tok.decode(a),repeat_equal=a==b,info=info))
   r['atom_standalone']={'status':'passed','outputs':outs}
  except Exception as ex:r['atom_standalone']={'status':'failed','error':repr(ex),'traceback':traceback.format_exc()}
  finally:e.close();gc.collect()
 print(json.dumps({mid:r},ensure_ascii=False),flush=True)
 Path('model_families.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
