"""CPU packing comparison using retained native-layout synthetic frames."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from memory_receiver import KVViews
from gpu_batch_decoder import pack_prefix
from gpu_native_pack import pack_native_views
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(1);rows=[]
for n in (127,511,1023,1711):
 views=KVViews(n)
 for first in range(0,72,4):
  raw=np.arange(4*n*8*128,dtype=np.uint16).reshape(4,n,8,128)
  for i in range(4):views.slots[first+i]=raw[i].transpose(1,0,2)
 reference=pack_prefix(views.torch_layers())
 for repeat in range(-2,7):
  modes=[('torch',lambda:pack_prefix(views.torch_layers())),('numpy',lambda:pack_native_views(views))]
  if repeat%2:modes.reverse()
  for name,fn in modes:
   start=time.perf_counter();packet=fn();elapsed=(time.perf_counter()-start)*1000
   exact=torch.equal(packet.view(torch.int16),reference.view(torch.int16))
   assert exact
   if repeat>=0:rows.append(dict(positions=n,mode=name,repeat=repeat,ms=elapsed,byte_exact=exact))
 Path(a.out).write_text(json.dumps(rows,indent=2)+'\n')
 print(n,{mode:float(np.median([x['ms'] for x in rows if x['positions']==n and x['mode']==mode])) for mode in ('torch','numpy')},flush=True)
