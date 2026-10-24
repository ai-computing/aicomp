import json,time,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from furiosa.torch._register import _register
_register()
import rngd_pd.engines.fsdk as sdk
sdk.open_device=lambda *a,**k:(sdk.set_fusion(4),sdk.set_mesh_kind(sdk.MeshKind.Mesh(2)))
from opt_prime.accel.pd.prefill_server import _RngdFull
from transformers import AutoTokenizer
model='/tmp/serving-matrix-models/KONI-8B'
out=Path('/home/gspark/workspace/npu_offload/analysis/serving-matrix-201/stream265-source-hardware.json')
engine=_RngdFull(SimpleNamespace(model=model,width='128',max_context=128,chip=0))
tokens=AutoTokenizer.from_pretrained(model).encode('Seoul is the capital of',add_special_tokens=True)
with torch.inference_mode():
 k,v,logits=engine.prefill(tokens)
 spans=[]; ks=[];vs=[];start=time.perf_counter()
 def emit(layer,first,keys,values):
  spans.append({'layer':layer,'first':first,'layers':keys.shape[0],'t_s':time.perf_counter()-start})
  ks.append(keys.clone());vs.append(values.clone())
 other=engine.prefill_stream(tokens,emit)
 result={'model':'KONI-8B','width':128,'input_tokens':tokens,'spans':spans,'completed_s':time.perf_counter()-start,
         'keys_exact':torch.equal(k,torch.cat(ks)),'values_exact':torch.equal(v,torch.cat(vs)),
         'logits_exact':torch.equal(logits,other),'token_whole':int(logits.argmax()),'token_stream':int(other.argmax()),
         'scope':'RNGD source parity only; no end-to-end decoder or speedup claim'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 assert result['keys_exact'] and result['values_exact'] and result['logits_exact']
