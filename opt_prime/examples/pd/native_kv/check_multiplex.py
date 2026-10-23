"""Live native KV concurrency: serial/multiplex byte parity on mixed lengths."""
import argparse
import json
import time
from pathlib import Path
import torch
from gpu_batch_decoder import pack_prefix
from multiplex_receiver import MultiplexKVClient

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',required=True)
a=p.parse_args()
torch.set_num_threads(1)
client=MultiplexKVClient('http://127.0.0.1:19101',19201,19202,
    json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text()))
prompts=[[1000+i]*n for i,n in enumerate([32,128,257,512])]
try:
    start=time.perf_counter()
    expected=[pack_prefix(client.prefill('Qwen/Qwen3-8B-FP8',tokens,'views')[0].torch_layers()) for tokens in prompts]
    serial_ms=(time.perf_counter()-start)*1000
    rows=[]
    for repeat in range(3):
        order=list(range(4)) if repeat%2==0 else [3,1,0,2]
        started=time.perf_counter();completed=[]
        actual=client.prefill_many('Qwen/Qwen3-8B-FP8',[prompts[i] for i in order],
            lambda i,kv,stats: completed.append(i))
        elapsed=(time.perf_counter()-started)*1000
        equality=[]; errors=[]
        for i,(kv,_) in enumerate(actual):
            observed=pack_prefix(kv.torch_layers())
            reference=expected[order[i]]
            equality.append(bool(torch.equal(observed,reference)))
            x,y=observed.float().flatten(1),reference.float().flatten(1)
            cos=torch.nn.functional.cosine_similarity(x,y,dim=1)
            errors.append({'equal_fraction':float((observed==reference).float().mean()),
                'max_abs':float((x-y).abs().max()),'min_slot_cosine':float(cos.min()),
                'first_k_cosine':float(cos[0]),'first_k_max_abs':float((x[0]-y[0]).abs().max())})
        row={'repeat':repeat,'order':order,'complete_order':completed,'total_ms':elapsed,
             'byte_equal':equality,'errors':errors,'stats':[stats for _,stats in actual]}
        rows.append(row)
        Path(a.out).write_text(json.dumps({'serial_ms':serial_ms,'rows':rows},indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
    serial_repeat=[]
    for tokens,reference in zip(prompts,expected):
        observed=pack_prefix(client.prefill('Qwen/Qwen3-8B-FP8',tokens,'views')[0].torch_layers())
        serial_repeat.append(bool(torch.equal(observed,reference)))
    Path(a.out).write_text(json.dumps({'serial_ms':serial_ms,'rows':rows,'serial_repeat_equal':serial_repeat},indent=2)+'\n')
    print('SERIAL_REPEAT',serial_repeat,flush=True)
finally:client.close()
