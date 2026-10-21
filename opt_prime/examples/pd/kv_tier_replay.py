"""Host-DRAM KV tier (#52): replay a Mooncake trace through the paged prefix cache, HBM only vs HBM + host DRAM.

    curl -LO https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl
    python examples/pd/kv_tier_replay.py --model unsloth/Llama-3.2-1B-Instruct --trace toolagent_trace.jsonl \
        --requests 1000 --pages 384 --host-gib 0 8

The trace gives each request's 512-token blocks as hash IDs. The replay scales a block to one page (64 tokens of
tokens derived from the ID), so prefix sharing is kept and prompts are 8x shorter. Requests run closed-loop at
--concurrency in trace order. Also reported: host->device reload time per block and the prefill rate, i.e. the
inputs of the reload-or-recompute rule.
"""
import argparse, json, statistics, time
import torch
from transformers import AutoModelForCausalLM
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--trace", required=True)
ap.add_argument("--requests", type=int, default=1000)
ap.add_argument("--max-blocks", type=int, default=28)
ap.add_argument("--max-new", type=int, default=16)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--pages", type=int, default=384)
ap.add_argument("--host-gib", type=float, nargs="+", default=[0, 8])
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--dtype", default="bfloat16")
a = ap.parse_args()
P, SCALE = 64, 512 // 64
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).cuda().eval()
vocab = model.config.get_text_config().vocab_size


def block(h):
    g = torch.Generator().manual_seed(1000003 * h + 7)
    return torch.randint(100, vocab - 100, (P,), generator=g).tolist()


reqs = []
for i, line in enumerate(open(a.trace)):
    if i == a.requests:
        break
    r = json.loads(line)
    ids = r["hash_ids"][:a.max_blocks]
    tail = P if len(r["hash_ids"]) > a.max_blocks else max(1, -(-(r["input_length"] - 512 * (len(ids) - 1)) // SCALE))
    prompt = [t for h in ids[:-1] for t in block(h)] + block(ids[-1])[:min(P, tail)]
    reqs.append(PDRequest(f"r{i}", tuple(prompt), a.max_new))


def replay(gib):
    b = ContinuousBatcher(model, slots=a.concurrency, max_context=(a.max_blocks + 1) * P, prefill_chunk=a.chunk,
                          kv="paged", pages=a.pages, page_size=P, prefix_cache=True, host_kv_gib=gib)
    b.submit(PDRequest("warm", (1, 2, 3, 4), 4)); b.run_until_idle()
    b.kv.hit_tokens = b.kv.query_tokens = 0
    torch.cuda.synchronize()
    t0, sent, ttft, futures, nxt = time.time(), {}, {}, {}, 0
    while nxt < len(reqs) or not b.idle:
        while nxt < len(reqs) and len(b.by_id) < a.concurrency:
            r = reqs[nxt]; nxt += 1
            sent[r.request_id] = time.time()
            futures[r.request_id] = b.submit(r, on_token=lambda t, rid=r.request_id: ttft.setdefault(
                rid, time.time() - sent[rid]))
        b.step()
    torch.cuda.synchronize()
    wall = time.time() - t0
    t = sorted(ttft.values())
    out = dict(host_gib=gib, wall_s=round(wall, 2), ttft_mean_ms=round(1e3 * statistics.mean(t), 1),
               ttft_p50_ms=round(1e3 * t[len(t) // 2], 1), ttft_p90_ms=round(1e3 * t[int(len(t) * .9)], 1),
               hit_rate=round(b.kv.hit_tokens / b.kv.query_tokens, 4), prefill_chunks=b.stats["prefill_chunks"])
    if b.kv.host is not None:
        out.update(host_blocks=b.kv.host.blocks, host_hit_rate=round(b.kv.host.stats["hit_tokens"] / b.kv.query_tokens, 4),
                   **{k: v for k, v in b.kv.host.stats.items() if k != "hit_tokens"})
    return out, {k: f.result() for k, f in futures.items()}, b


def reload_cost(b, n=256):
    """host -> device copy time per block, and the model's prefill rate, for the reload-or-recompute rule"""
    kv, host = b.kv, b.kv.host
    blocks = min(n, host.blocks)
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(blocks):
        kv.k[:, :, i * P:(i + 1) * P].copy_(host.k[i], non_blocking=True)
        kv.v[:, :, i * P:(i + 1) * P].copy_(host.v[i], non_blocking=True)
    torch.cuda.synchronize()
    per = (time.time() - t0) / blocks
    ids = torch.randint(100, vocab - 100, (1, a.chunk), device="cuda")
    with torch.inference_mode():
        model(ids); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            model(ids)
        torch.cuda.synchronize()
    rate = 10 * a.chunk / (time.time() - t0)
    return dict(block_bytes=host.block_bytes, reload_us_per_block=round(1e6 * per, 1),
                reload_gb_s=round(host.block_bytes / per / 1e9, 2), prefill_tok_s=round(rate),
                recompute_us_per_block=round(1e6 * P / rate, 1),
                break_even_gb_s=round(rate * host.block_bytes / P / 1e9, 3))


results, outputs, cost = [], {}, None
for gib in a.host_gib:
    res, outputs[gib], b = replay(gib)
    results.append(res)
    if b.kv.host is not None and cost is None:
        cost = reload_cost(b)
    print(json.dumps(res), flush=True)
base = a.host_gib[0]
same = {g: sum(outputs[g][k] == outputs[base][k] for k in outputs[base]) for g in a.host_gib}
print(json.dumps(dict(model=a.model, dtype=a.dtype, requests=len(reqs), pages=a.pages, concurrency=a.concurrency,
                      chunk=a.chunk, identical_to_first=same, reload=cost)))
