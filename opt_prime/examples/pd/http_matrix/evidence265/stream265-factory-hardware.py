import argparse,hashlib,json,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from furiosa.torch._register import _register
_register()
import rngd_pd.engines.fsdk as sdk
sdk.open_device=lambda *a,**k:(sdk.set_fusion(4),sdk.set_mesh_kind(sdk.MeshKind.Mesh(2)))
from opt_prime.accel.pd.prefill_server import factory_engine,stream_result
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--family',required=True);p.add_argument('--label',required=True);p.add_argument('--width',type=int,required=True);a=p.parse_args()
root=Path('/home/gspark/workspace/npu_offload/analysis/serving-matrix-201');out=root/f'stream265-{a.label}-source.json'
if out.exists():raise RuntimeError('refuse overwrite of prior hardware result')
config=json.loads((Path(a.model)/'config.json').read_text())
args=SimpleNamespace(model=a.model,engine_factory=f'streaming_factories:{a.family}',width=str(a.width),chip=0,slots=1,
 layers=config['num_hidden_layers'],kv_heads=config['num_key_value_heads'],head_dim=config.get('head_dim') or config['hidden_size']//config['num_attention_heads'],max_context=a.width)
result={'model':a.label,'width':a.width,'engine_factory':args.engine_factory,'cases':[],
 'scope':'same RNGD source whole vs streamed KV/logits; not decoder parity or speedup',
 'source_sha256':{}}
for path in [Path('/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/opt_prime/accel/pd/prefill_server.py'),Path('/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime/examples/pd/streaming_factories.py')]:
 result['source_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
try:
 print('initializing',a.label,flush=True);engine=factory_engine(args);print('initialized',a.label,flush=True)
 tokenizer=AutoTokenizer.from_pretrained(a.model)
 base=tokenizer.encode('Seoul is the capital of South Korea. Count one two three.',add_special_tokens=True)
 for n in [len(base),a.width]:
  ids=(base*((n+len(base)-1)//len(base)))[:n]
  with torch.inference_mode():
   k,v,logits=engine.prefill(ids)
   for start in [0,n//2,n]:
    slab=np.zeros((1,2,args.layers,args.kv_heads,args.max_context,args.head_dim),dtype=np.uint16)
    spans=[];t=time.perf_counter()
    def publish(msg):spans.append(dict(span=msg['kv_span'],time_s=time.perf_counter()-t))
    other=stream_result(engine,dict(id=0,tokens=ids,slot=0,send_from=start),slab,publish)
    elapsed=time.perf_counter()-t
    kv=torch.from_numpy(slab[0,:,:,:,:n-start].view(np.int16)).view(torch.bfloat16)
    case={'input_len':n,'send_from':start,'keys_exact':torch.equal(k[:,:,start:].to(torch.bfloat16),kv[0]),
     'values_exact':torch.equal(v[:,:,start:].to(torch.bfloat16),kv[1]),'logits_exact':torch.equal(logits,other),
     'max_logit_error':float((logits-other).abs().max()),'token_whole':int(logits.argmax()),'token_stream':int(other.argmax()),
     'notifications':spans,'elapsed_s':elapsed}
    result['cases'].append(case);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in case.items() if k!='notifications'}),flush=True)
    assert case['keys_exact'] and case['values_exact'] and case['logits_exact']
 result['stage']='complete'
except Exception as error:
 result['stage']='failed';result['error']=repr(error);raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n')
