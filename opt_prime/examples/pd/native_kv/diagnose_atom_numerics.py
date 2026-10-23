"""Fixed Qwen2.5-0.5B diagnostic; see ATOM_NUMERICAL_EXPERIMENT.md."""
import json,torch,numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
from opt_prime.accel.platforms.atom.batching import bf16_to_dlfloat
records=torch.load('rngd_diagnostic.pt',weights_only=True)
reference=json.load(open('atom_original.json'))['outputs']['aggregated']
e=AtomRblnEngine(devices=[int(next(Path('/dev').glob('rbln[0-9]*')).name[4:])],max_seq_len=2048,batch_size=8,kv_blocks=17)
e.start('Qwen/Qwen2.5-0.5B-Instruct')
b=e.batcher(prefix_cache=False)
cpu=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',dtype=torch.float32).eval()
report={}
def top(x):
 v,t=x.float().flatten().topk(5)
 return dict(argmax=int(x.argmax()),tokens=t.tolist(),values=v.tolist(),margin=float(v[0]-v[1]))
def trace(prompt,i):
 out=[]
 for j,t in enumerate(reference[i][:3]):
  ids=torch.full((8,1),b.pad,dtype=torch.int64);ids[0,0]=t
  pos=torch.zeros((8,1),dtype=torch.int32);pos[0,0]=len(prompt)+j
  logits=b.decoder(input_ids=ids,cache_position=pos,block_tables=b.blocks.table([0])).logits[0,-1].float()
  out.append(logits.clone())
 return out
def encoded(k,v):
 x=torch.stack([k,v],dim=1).to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)
 return bf16_to_dlfloat(x).reshape(48,2,k.shape[2],64)
with torch.inference_mode():
 for i,r in records.items():
  p=r['prompt'];n=len(p);b.blocks.admit(0,p);block=b.blocks.lane_block[0]
  native=b._prefill(0,p).logits[0,-1].float().clone()
  raw=[]
  for m,buf in zip(b.blocks.metas,b.blocks._staging()):
   b.blocks.io_rt.get_kv_cache_for_layer(m['name'],block,0,n,buf)
   raw.append(buf[block,:,:n].copy())
  raw=np.stack(raw)
  native_steps=trace(p,i)
  hf=cpu(torch.tensor([p]),use_cache=True)
  ck=torch.stack([l.keys[0] for l in hf.past_key_values.layers]);cv=torch.stack([l.values[0] for l in hf.past_key_values.layers])
  kv_metrics=[]
  for rk,cv_ in zip(torch.stack([r['keys'],r['values']],dim=1).flatten(0,1),torch.stack([ck,cv],dim=1).flatten(0,1)):
   delta=rk.float()-cv_.float();kv_metrics.append(dict(rms=float(delta.square().mean().sqrt()),relative_rms=float(delta.square().mean().sqrt()/cv_.float().square().mean().sqrt()),max_abs=float(delta.abs().max())))
  item={'rngd_vs_cpu_fp32_kv':kv_metrics,'native_prefill':top(native),'native_steps':[top(x) for x in native_steps],'sources':{}}
  for name,k,v,logits in [('rngd',r['keys'],r['values'],r['logits']),('cpu_fp32',ck,cv,hf.logits[0,-1])]:
   enc=encoded(k,v)
   b.blocks.import_span(0,0,k,v)
   exact=0
   for z,(m,buf) in enumerate(zip(b.blocks.metas,b.blocks._staging())):
    buf.fill(0);b.blocks.io_rt.get_kv_cache_for_layer(m['name'],block,0,n,buf)
    exact+=int(np.array_equal(buf[block,:,:n],enc[z]))
   steps=trace(p,i)
   item['sources'][name]=dict(prefill=top(logits),prefill_native_max_abs=float((logits-native).abs().max()),
     readback_exact_slots=exact,kv_encoded_equal_fraction_by_slot=[float(np.mean(x==y)) for x,y in zip(raw,enc)],
     steps=[dict(**top(x),native_max_abs=float((x-y).abs().max()),native_rms=float((x-y).square().mean().sqrt())) for x,y in zip(steps,native_steps)])
  # Same ATOM-produced cache roundtrip: isolates import/readback mechanics with no conversion.
  for m,buf,x in zip(b.blocks.metas,b.blocks._staging(),raw):
   buf[block,:,:n]=x;b.blocks.io_rt.set_kv_cache_for_layer(m['name'],block,0,n,buf)
  again=trace(p,i)
  item['native_raw_roundtrip_steps_exact']=[bool(torch.equal(x,y)) for x,y in zip(native_steps,again)]
  report[i]=item;print(json.dumps({i:item}),flush=True)
  b.blocks.release(0,None)
Path('numerical_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
e.close()
