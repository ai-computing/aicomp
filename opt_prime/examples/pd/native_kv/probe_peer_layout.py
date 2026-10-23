"""Diagnostic only: admit precisely the two fixed bounded layouts, never arbitrary input."""
import argparse,json,copy,hashlib,time
from pathlib import Path
from unittest.mock import patch
from memory_receiver import NativeKVClient
import protocol
p=argparse.ArgumentParser(description='Diagnostic fixed peer-layout comparison, no GPU decode')
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
base=json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text())
protocol.validate_layout(base);clients={};rows=[]
try:
 for mode,port in [('default',19202),('head-major',19203)]:
  layout=copy.deepcopy(base)
  if mode=='head-major':
   axes=layout['dram_shape']['intra_chip_axes']['axes'];axes[0]['stride']=128;axes[1]['stride']=272321*128;axes[0],axes[1]=axes[1],axes[0]
  def validate_exact(candidate,expected=copy.deepcopy(layout)):
   if candidate!=expected:raise ValueError('unexpected bounded descriptor')
  with patch.object(protocol,'validate_layout',validate_exact):
   clients[mode]=NativeKVClient('http://127.0.0.1:19101',19201,port,layout,timeout=15)
 for n in [128,512,1024]:
  for repeat in range(4):
   for mode in (['default','head-major'] if repeat%2==0 else ['head-major','default']):
    packet,stats=clients[mode].prefill('Qwen/Qwen3-8B-FP8',[1000]*n,mode='views')
    rows.append(dict(mode=mode,input_tokens=n,repeat=repeat,sha256=hashlib.sha256(packet.tobytes()).hexdigest(),**{k:v for k,v in stats.items() if k not in ('response','mode','input_tokens')}))
    del packet
finally:
 for client in clients.values():client.close()
 a.out.write_text(json.dumps(rows,indent=2)+'\n')
print('completed',len(rows),flush=True)
