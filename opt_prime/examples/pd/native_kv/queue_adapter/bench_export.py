import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from multiplex_receiver import MultiplexKVClient
torch.set_num_threads(1)
rows=[]
client=MultiplexKVClient('http://127.0.0.1:19101',19201,19202,json.loads(Path('qwen3_8b_layout.json').read_text()),timeout=120)
try:
 for c,n,repeats in [(1,128,12),(1,512,12),(1,1024,24),(4,1024,5)]:
  for repeat in range(repeats):
   results=client.prefill_many('Qwen/Qwen3-8B-FP8',[[1000]*n for _ in range(c)],direct=True)
   for packet,stats in results:
    digest=hashlib.sha256(packet.host.view(torch.uint8).numpy()).hexdigest()
    rows.append(dict(concurrency=c,input_tokens=n,repeat=repeat,warmup=repeat<2,sha256=digest,bytes=packet.nbytes,**{k:v for k,v in stats.items() if k!='response'}))
   del packet,results
  print('EXPORT',c,n,repeats,flush=True)
finally:
 client.close();Path(sys.argv[1]).write_text(json.dumps(rows,indent=2)+'\n')
