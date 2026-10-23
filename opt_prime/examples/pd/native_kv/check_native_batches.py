"""Inspect validated batch layout without persisting KV payloads."""
import argparse,json
from pathlib import Path
from memory_receiver import NativeKVClient
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
client=NativeKVClient('http://127.0.0.1:19101',19201,19202,json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text()))
rows=[]
def received(slots,values):
 indices=[int(x) for x in slots]
 rows.append(dict(slots=indices,contiguous_slots=indices==list(range(indices[0],indices[0]+len(indices))),positions=values.shape[2],wire_contiguous=values.transpose(0,2,1,3).flags.c_contiguous))
try:
 _,stats=client.prefill('Qwen/Qwen3-8B-FP8',[1000]*512,'views',on_batch=received)
 result=dict(batches=rows,count=len(rows),bytes=stats['bytes'])
 Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'count':len(rows),'all_contiguous':all(x['contiguous_slots'] and x['wire_contiguous'] for x in rows)}),flush=True)
finally:client.close()
