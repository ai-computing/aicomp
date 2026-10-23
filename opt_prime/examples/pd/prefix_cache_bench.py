"""Prefix caching on the paged batcher (#46): requests sharing a long system prompt, with and without the cache.

    python examples/pd/prefix_cache_bench.py --model unsloth/Llama-3.2-1B-Instruct --prefix 1536 --requests 32
"""
import argparse, json, statistics, time
import torch
from transformers import AutoModelForCausalLM
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--requests", type=int, default=32)
ap.add_argument("--prefix", type=int, default=1536)
ap.add_argument("--slots", type=int, default=8)
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--max-new", type=int, default=32)
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--page-size", type=int, default=64)
a = ap.parse_args()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).cuda().eval()
g = torch.Generator().manual_seed(0)
vocab = model.config.get_text_config().vocab_size
rand = lambda n: tuple(torch.randint(100, vocab - 100, (n,), generator=g).tolist())
system = rand(a.prefix)
reqs = [PDRequest(f"r{i}", system + rand(int(torch.randint(16, 128, (1,), generator=g))), a.max_new)
        for i in range(a.requests)]


def run(cache):
    b = ContinuousBatcher(model, slots=a.slots, max_context=2048, prefill_chunk=a.chunk, kv="paged",
                          page_size=a.page_size, prefix_cache=cache)
    b.submit(PDRequest("warm", (1, 2, 3, 4), 4)); b.run_until_idle()
    torch.cuda.synchronize()
    first, fs, t0 = {}, [], time.time()
    for r in reqs:
        fs.append(b.submit(r, on_token=lambda t, rid=r.request_id: first.setdefault(rid, time.time())))
    b.run_until_idle(); torch.cuda.synchronize()
    total = time.time() - t0
    ttft = sorted(first[r.request_id] - t0 for r in reqs)
    return dict(total_s=round(total, 3), ttft_mean_s=round(statistics.mean(ttft), 3),
                ttft_p50_s=round(ttft[len(ttft) // 2], 3), prefill_chunks=b.stats["prefill_chunks"],
                hit_tokens=b.kv.hit_tokens, query_tokens=b.kv.query_tokens), [f.result() for f in fs]


off, ref = run(False)
on, out = run(True)
agree = [next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), None) for p, q in zip(ref, out)]
print(json.dumps(dict(model=a.model, dtype=a.dtype, requests=a.requests, prefix=a.prefix, slots=a.slots,
                      chunk=a.chunk, off=off, on=on, speedup_total=round(off["total_s"] / on["total_s"], 2),
                      speedup_ttft_mean=round(off["ttft_mean_s"] / on["ttft_mean_s"], 2),
                      identical=sum(x is None for x in agree), first_divergence=[x for x in agree if x is not None])))
