"""Profile one warmed local prefill; excludes model loading."""
import argparse,json
from pathlib import Path
import torch
from gpu_local_prefill import LocalPrefiller
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True)
a=p.parse_args();torch.set_num_threads(1)
f=LocalPrefiller(a.model_dir)
prompts=[torch.randint(1000,20000,(1024,),generator=torch.Generator().manual_seed(i)).tolist() for i in range(8)]
for _ in range(2):f.prefill(prompts)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
 f.prefill(prompts)
torch.cuda.synchronize()
rows=[dict(name=e.key,calls=e.count,self_device_us=e.self_device_time_total,cpu_us=e.self_cpu_time_total,shapes=e.input_shapes) for e in prof.key_averages(group_by_input_shape=True)]
rows.sort(key=lambda e:e['self_device_us'],reverse=True)
Path(a.out).write_text(json.dumps(rows,indent=2)+'\n')
print(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=25))
