"""Diagnostic only; production full-transfer validation remains unchanged."""
import argparse,json,struct,time,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import memory_receiver as mr
from protocol import control as original_control
p=argparse.ArgumentParser(description='Bounded diagnostic: verify native suffix export against full export')
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
layout=json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text())
client=mr.NativeKVClient('http://127.0.0.1:19101',19201,19202,layout,timeout=15)
original_assembler=mr.ViewAssembler
out=[]
try:
 for length,skip in [(128,64),(512,256),(1024,512)]:
  tokens=[1000]*length
  full,full_stats=client.prefill('Qwen/Qwen3-8B-FP8',tokens,'views')
  class TailAssembler(original_assembler):
   def __init__(self,rid,n,on_batch=None):super().__init__(rid,n-skip,on_batch)
  def suffix_control(rid,kind,block_ids=()):
   if kind!=2:return original_control(rid,kind,block_ids)
   assert len(list(block_ids))==length-1
   data=bytearray(original_control(rid,kind,range(length-1-skip)))
   struct.pack_into('<I',data,len(data)-36+16,skip)
   return bytes(data)
  with patch.object(mr,'ViewAssembler',TailAssembler),patch.object(mr,'control',suffix_control):
   tail,stats=client.prefill('Qwen/Qwen3-8B-FP8',tokens,'views')
  equal=all(np.array_equal(f[:,skip:,:],t) for f,t in zip(full.slots,tail.slots))
  out.append({'length':length,'skip':skip,'tail_bytes':tail.nbytes,'matches_full_suffix':equal,
              'full_stats':{k:v for k,v in full_stats.items() if k!='response'},
              'suffix_stats':{k:v for k,v in stats.items() if k!='response'}})
  print(length,skip,equal,stats['kv_ready_ms'],flush=True)
  del full,tail
finally:
 client.close()
 a.out.write_text(json.dumps(out,indent=2)+'\n')
