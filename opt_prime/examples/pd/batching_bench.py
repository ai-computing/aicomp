"""Continuous batching with chunked prefill (#44): throughput vs one request at a time, and token agreement.

    python examples/pd/batching_bench.py --model unsloth/Llama-3.2-1B-Instruct --requests 32 --slots 8
"""
import argparse, json, statistics, time
import torch
from transformers import AutoModelForCausalLM
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--requests", type=int, default=32)
ap.add_argument("--slots", type=int, default=8)
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--max-new", type=int, default=64)
ap.add_argument("--dtype", default="bfloat16")
a = ap.parse_args()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).cuda().eval()
g = torch.Generator().manual_seed(0)
vocab = model.config.get_text_config().vocab_size
reqs = [PDRequest(f"r{i}", tuple(torch.randint(100, vocab - 100, (int(torch.randint(16, 512, (1,), generator=g)),),
                                                generator=g).tolist()), a.max_new) for i in range(a.requests)]


def run(slots, chunk):
    b = ContinuousBatcher(model, slots=slots, max_context=1024, prefill_chunk=chunk)
    b.submit(PDRequest("warm", (1, 2, 3, 4), 4)); b.run_until_idle()
    torch.cuda.synchronize(); t0 = time.time()
    fs = [b.submit(r) for r in reqs]
    b.run_until_idle(); torch.cuda.synchronize()
    return time.time() - t0, [f.result() for f in fs], dict(b.stats)


t1, seq, _ = run(1, 10 ** 6)                         # one at a time, whole-prompt prefill
tb, bat, stats = run(a.slots, a.chunk)
tokens = sum(len(x) for x in bat)
agree = [next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), None) for p, q in zip(seq, bat)]
res = dict(model=a.model, dtype=a.dtype, requests=a.requests, slots=a.slots, chunk=a.chunk, max_new=a.max_new,
           gpu=torch.cuda.get_device_name(0), sequential_s=round(t1, 2), batched_s=round(tb, 2),
           sequential_tok_s=round(tokens / t1, 1), batched_tok_s=round(tokens / tb, 1), speedup=round(t1 / tb, 2),
           mean_decode_batch=round(stats["decode_rows"] / max(stats["decode_steps"], 1), 2),
           identical_requests=sum(x is None for x in agree), first_divergence=[x for x in agree if x is not None])
print(json.dumps(res))
