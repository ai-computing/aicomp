"""Paired prefill measurement and separate CUDA traces; no serving-speed claim."""
import json, time, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.pd_batching import TorchPrefillEngine

root = Path(__file__).resolve().parent
torch.set_num_threads(4)
model_path = '/home/gspark/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True).to('cuda:0').eval()
engines = {flag: TorchPrefillEngine(model, max_context=9216, chunk=256, page_size=64,
                                  prefix_cache=False, out_device='cuda:1', copy_overlap=flag)
           for flag in [False, True]}
prompt = json.loads((root/'perf248-qwen06b-native-long128.json').read_text())['prompts']['8192']
request = PDRequest('chunk47', tuple(prompt), 1)
reference = engines[False].prefill(request)
engines[True].prefill(request)
rows = []
for repeat in range(3):
 for flag in ([False, True] if repeat % 2 == 0 else [True, False]):
  torch.cuda.synchronize(0); torch.cuda.synchronize(1)
  start=time.perf_counter(); out=engines[flag].prefill(request); elapsed=time.perf_counter()-start
  parity = out.token == reference.token and torch.equal(out.keys, reference.keys) and torch.equal(out.values, reference.values)
  row=dict(repeat=repeat,copy_overlap=flag,elapsed_s=elapsed,prefill_s=out.prefill_s,transfer_tail_s=out.transfer_s,parity=parity,kv_bytes=sum(t.numel()*t.element_size() for t in (out.keys,out.values)))
  rows.append(row);print(json.dumps(row),flush=True)
  assert parity
  del out
(root/'chunk47-qwen06-prefill.json').write_text(json.dumps(dict(model='Qwen3-0.6B',commit='0870dd3',torch=torch.__version__,input_tokens=8192,chunk=256,prefix_cache=False,rows=rows),indent=2)+'\n')
for flag in [False,True]:
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  out=engines[flag].prefill(request)
 prof.export_chrome_trace(str(root/f'chunk47-qwen06-trace-{int(flag)}.json'))
 print('TRACE',flag,flush=True)
 del out
