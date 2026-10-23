"""Real checkpoints through the paged continuous batcher (#127): token agreement with HF generate and throughput.

    python examples/pd/family_check.py --model google/gemma-4-26B-A4B-it --prompt 1500 --requests 8 --max-new 48

Prompts are random token IDs of about --prompt tokens (longer than Gemma's sliding window, so the windowed attention
is exercised); greedy decoding. The reference is HF `generate` per request (the model's own cache classes), one
request at a time; the batched run submits all requests to `ContinuousBatcher(kv="paged")`.
"""
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.torch_models import describe_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--requests", type=int, default=8)
ap.add_argument("--prompt", type=int, default=1500); ap.add_argument("--max-new", type=int, default=48)
ap.add_argument("--slots", type=int, default=8); ap.add_argument("--max-context", type=int, default=2048)
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--experts", default=None, help="transformers experts_implementation for MoE (e.g. grouped_mm)")
a = ap.parse_args()
t0 = time.time()
kw = {"experts_implementation": a.experts} if a.experts else {}
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), **kw).to(a.device).eval()
load = time.time() - t0
g = describe_model(model.config)
vocab = model.config.get_text_config().vocab_size
gen = torch.Generator().manual_seed(0)
reqs = [PDRequest(f"r{i}", tuple(torch.randint(1000, min(vocab, 30000) - 1, (a.prompt + 37 * i % 200,),
                                               generator=gen).tolist()), a.max_new) for i in range(a.requests)]
ref = []
torch.cuda.synchronize(); t0 = time.time()
with torch.inference_mode():
    for r in reqs:
        out = model.generate(torch.tensor([list(r.input_ids)], device=a.device), max_new_tokens=a.max_new,
                             do_sample=False, eos_token_id=None, pad_token_id=0)
        ref.append(out[0, len(r.input_ids):].tolist())
torch.cuda.synchronize(); seq_s = time.time() - t0
b = ContinuousBatcher(model, kv="paged", slots=a.slots, max_context=a.max_context, prefill_chunk=a.chunk)
f = b.submit(PDRequest("warm", tuple(range(1000, 1100)), 4)); b.run_until_idle(); f.result()
torch.cuda.synchronize(); t0 = time.time()
fs = [b.submit(r) for r in reqs]
b.run_until_idle(); torch.cuda.synchronize(); bat_s = time.time() - t0
out = [f.result() for f in fs]
div = [next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), None) for p, q in zip(ref, out)]
print(json.dumps(dict(
    model=a.model, dtype=a.dtype, experts=a.experts, chunk=a.chunk, load_s=round(load, 1), groups=[(x.kind, x.window, x.kv_heads, x.head_dim, len(x.layers))
                                                               for x in g.groups], recurrent_layers=len(g.recurrent),
    requests=a.requests, prompt=a.prompt, max_new=a.max_new,
    hf_sequential_tok_s=round(sum(map(len, ref)) / seq_s, 1), batched_tok_s=round(sum(map(len, out)) / bat_s, 1),
    speedup=round(seq_s / bat_s, 2), identical=sum(d is None for d in div), first_divergence=[d for d in div if d is not None],
    peak_gib=round(torch.cuda.max_memory_allocated(a.device) / 2 ** 30, 1))))
