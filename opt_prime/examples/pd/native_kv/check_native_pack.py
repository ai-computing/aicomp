"""Compare both packing paths on exactly the same live native KV packet."""
import argparse,gc,json,random
from pathlib import Path
import torch
from memory_receiver import NativeKVClient
from gpu_batch_decoder import pack_prefix
from gpu_native_pack import pack_native_views
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1)
client=NativeKVClient('http://127.0.0.1:19101',19201,19202,json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text()))
rows=[]
try:
 for n in (128,512,1024):
  rng=random.Random(150+n);tokens=[rng.randrange(1000,30000) for _ in range(n)]
  views,stats=client.prefill('Qwen/Qwen3-8B-FP8',tokens,'views')
  reference=pack_prefix(views.torch_layers());packet=pack_native_views(views)
  exact=torch.equal(reference.view(torch.int16),packet.view(torch.int16))
  del views;gc.collect()
  owned=torch.equal(reference.view(torch.int16),packet.view(torch.int16))
  row=dict(input_tokens=n,positions=packet.shape[1],byte_exact=exact,owned_after_release=owned,cuda_pinned=packet.is_pinned('cuda'))
  rows.append(row);Path(a.out).write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
  assert exact and owned and row['cuda_pinned']
finally:client.close()
