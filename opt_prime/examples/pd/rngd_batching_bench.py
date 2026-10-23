"""Continuous batching on RNGD through furiosa-llm (#122): batched vs one request at a time, same engine settings.

    python examples/pd/rngd_batching_bench.py --model KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024 \
        --fxb koni-s512.fxb --devices npu:3 --max-context 512 --concurrency 8 --requests 32

Builds `FuriosaLLMBatcher.from_artifact`, runs --requests random-token prompts (--min-prompt..--max-prompt) with
--max-new tokens each: first one at a time (sequential baseline), then all submitted at once (batched).
With --shared-prefix N every prompt starts with the same N tokens (prefix-cache hits).
"""
import argparse, json, random, time
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.platforms.rngd.batching import FuriosaLLMBatcher

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--fxb", required=True)
ap.add_argument("--devices", default="npu:0"); ap.add_argument("--max-context", type=int, default=512)
ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--requests", type=int, default=32)
ap.add_argument("--min-prompt", type=int, default=64); ap.add_argument("--max-prompt", type=int, default=320)
ap.add_argument("--max-new", type=int, default=64); ap.add_argument("--shared-prefix", type=int, default=0)
ap.add_argument("--host-kv-gib", type=float, default=0.0); ap.add_argument("--no-prefix-cache", action="store_true")
ap.add_argument("--skip-sequential", action="store_true")
ap.add_argument("--mode", default="rounds", choices=["rounds", "native"])
ap.add_argument("--quantum-busy", type=int, default=16); ap.add_argument("--quantum-idle", type=int, default=64)
a = ap.parse_args()
rng = random.Random(0)
shared = tuple(rng.randrange(1000, 30000) for _ in range(a.shared_prefix))
reqs = [PDRequest(f"r{i}", shared + tuple(rng.randrange(1000, 30000) for _ in range(rng.randint(a.min_prompt, a.max_prompt))),
                  a.max_new) for i in range(a.requests)]
t0 = time.time()
b = FuriosaLLMBatcher.from_artifact(a.model, a.fxb, devices=a.devices, max_context=a.max_context,
                                    max_concurrency=a.concurrency, prefix_cache=not a.no_prefix_cache,
                                    host_kv_gib=a.host_kv_gib, mode=a.mode, quantum_busy=a.quantum_busy,
                                    quantum_idle=a.quantum_idle)
load_s = time.time() - t0
f = b.submit(PDRequest("warm", tuple(range(1000, 1064)), 8)); b.run_until_idle(); f.result()
res = dict(mode=a.mode, model=a.model, devices=a.devices, concurrency=a.concurrency, requests=a.requests, max_new=a.max_new,
           shared_prefix=a.shared_prefix, host_kv_gib=a.host_kv_gib, load_s=round(load_s, 1))
seq = None
if not a.skip_sequential:
    t0 = time.time(); seq = []
    for r in reqs:
        f = b.submit(PDRequest(r.request_id + "-s", r.input_ids, r.max_new_tokens)); b.run_until_idle(); seq.append(f.result())
    dt = time.time() - t0
    res["sequential"] = dict(seconds=round(dt, 2), tok_s=round(sum(map(len, seq)) / dt, 1))
    print(json.dumps(res["sequential"]), flush=True)
h0, q0 = b.hit_tokens, b.query_tokens
first, last = {}, {}
t0 = time.time()
fs = []
for r in reqs:
    def tok(_t, rid=r.request_id):
        now = time.time(); first.setdefault(rid, now); last[rid] = now
    fs.append(b.submit(r, on_token=tok))
b.run_until_idle()
dt = time.time() - t0
out = [f.result() for f in fs]
tpot = [(last[r.request_id] - first[r.request_id]) / (len(o) - 1) for r, o in zip(reqs, out) if len(o) > 1]
res["batched"] = dict(seconds=round(dt, 2), tok_s=round(sum(map(len, out)) / dt, 1),
                      ttft_mean_s=round(sum(first[r.request_id] - t0 for r in reqs) / len(reqs), 3),
                      tpot_mean_ms=round(1e3 * sum(tpot) / len(tpot), 1),
                      prefix_hit_tokens=b.hit_tokens - h0, prompt_tokens=b.query_tokens - q0)
if seq is not None:
    res["batched"]["identical_to_sequential"] = sum(x == y for x, y in zip(seq, out))
    res["speedup"] = round(res["batched"]["tok_s"] / res["sequential"]["tok_s"], 2)
print(json.dumps(res), flush=True)
b.engine.shutdown() if hasattr(b.engine, "shutdown") else None
b.close()
