"""Verify the validated suffix API against full native exports on controlled prompts."""
import argparse,json
from pathlib import Path
import numpy as np
import memory_receiver as mr
p=argparse.ArgumentParser(description='Bounded diagnostic: verify native suffix export against full export')
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
layout=json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text())
client=mr.NativeKVClient('http://127.0.0.1:19101',19201,19202,layout,timeout=15)
out=[]
try:
 for length,skip in [(128,64),(512,256),(1024,512)]:
  tokens=[1000]*length
  full,full_stats=client.prefill('Qwen/Qwen3-8B-FP8',tokens,'views')
  tail,stats=client.prefill('Qwen/Qwen3-8B-FP8',tokens,'views',start_pos=skip)
  equal=all(np.array_equal(f[:,skip:,:],t) for f,t in zip(full.slots,tail.slots))
  out.append({'length':length,'skip':skip,'tail_bytes':tail.nbytes,'matches_full_suffix':equal,
              'full_stats':{k:v for k,v in full_stats.items() if k!='response'},
              'suffix_stats':{k:v for k,v in stats.items() if k!='response'}})
  print(length,skip,equal,stats['kv_ready_ms'],flush=True)
  del full,tail
finally:
 client.close()
 a.out.write_text(json.dumps(out,indent=2)+'\n')
