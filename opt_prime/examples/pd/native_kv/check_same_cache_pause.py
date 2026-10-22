"""Hold native KV fixed while changing only the GPU pause schedule."""
import argparse,json
from pathlib import Path
import torch
from multiplex_receiver import MultiplexKVClient
from gpu_native_pack import pack_native_views
from gpu_decoder import load_model
from gpu_norms import install_fused_norms
from gpu_batch_decoder import BatchGraphDecoder
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1)
client=MultiplexKVClient('http://127.0.0.1:19101',19201,19202,json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text()))
packets=[]
try:
 for n,t in ((512,1000),(128,1001)):
  views,_=client.prefill('Qwen/Qwen3-8B-FP8',[t]*n,'views');packets.append((pack_native_views(views),t))
 rows_native=client.prefill_many('Qwen/Qwen3-8B-FP8',[[1000]*512 for _ in range(8)])
 parallel=[(pack_native_views(views),1000) for views,_ in rows_native]
 native_equal=[torch.equal(packet.view(torch.int16),parallel[0][0].view(torch.int16)) for packet,_ in parallel]
 differing_slots=[]
 for packet,_ in parallel:
  differing_slots.append(next((i for i in range(72) if not torch.equal(packet[i].view(torch.int16),parallel[0][0][i].view(torch.int16))),None))
finally:client.close()
model=load_model(a.model_dir,fuse_mlp=True,fuse_qkv=True);install_fused_norms(model,'fused');graph=BatchGraphDecoder(model,16)
rows=[]
for label,mixed in (('shared',False),('mixed',True),('native_parallel',False)):
 def reset():
  for lane in range(16):graph.cancel_lane(lane)
  for lane in range(16):
   packet,token=parallel[lane%8] if label=='native_parallel' else packets[lane%2 if mixed else 0];graph.load_packed_lane(lane,packet,token,128)
 def run(stall):
  reset();outputs=[[] for _ in range(16)];snap=None;checked=False;steps=0
  while any(graph.remaining):
   paused=(0,) if stall and len(outputs[0])==20 and any(graph.remaining[1:]) else ()
   if paused and snap is None:
    pos=int(graph.position[0]);snap=(pos,int(graph.input_id[0,0]),graph.remaining[0],[(k[0,:pos].clone(),v[0,:pos].clone()) for k,v in zip(graph.keys,graph.values)])
   result=graph.step(ignore_eos=True,paused=paused)
   for lane,token in result.items():outputs[lane].append(token)
   if paused:
    assert int(graph.position[0])==snap[0] and int(graph.input_id[0,0])==snap[1] and graph.remaining[0]==snap[2]
    if not any(graph.remaining[1:]):
     assert all(torch.equal(k[0,:snap[0]],pair[0]) and torch.equal(v[0,:snap[0]],pair[1]) for k,v,pair in zip(graph.keys,graph.values,snap[3]));checked=True
   steps+=1
   if steps>300:raise AssertionError('no progress')
  return outputs,checked,steps
 baseline,_,_=run(False);paused,checked,steps=run(True)
 equal=baseline==paused
 row=dict(source=label,mixed_lengths=mixed,outputs_per_lane=128,lanes=16,pause_valid_prefix_exact=checked,paused_matches_unpaused=equal,matching_positions=sum(x==y for a,b in zip(baseline,paused) for x,y in zip(a,b)),positions=2048,identical_input_lanes_equal=all(baseline[i]==baseline[i%8 if label=='native_parallel' else i%2 if mixed else 0] for i in range(16)),steps=steps)
 if label=='native_parallel':row.update(native_packets_equal=native_equal,first_differing_slots=differing_slots,distinct_output_sequences=len({tuple(x) for x in baseline[:8]}))
 rows.append(row);Path(a.out).write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
 assert checked and equal
