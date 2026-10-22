"""ATOM prefix caching and host tier (#123): own block tables over the optimum-rbln model.

    python examples/pd/atom_prefix_bench.py --model unsloth/Llama-3.2-1B-Instruct --devices 1 --batch 8 \
        --kv-blocks 17 --prefixes 4 --prefix-len 512 --requests 32

Requests use one of --prefixes shared prefixes (random tokens, --prefix-len) plus their own 16-127-token suffix and
--max-new new tokens; they are submitted with staggered arrivals. Runs: no prefix cache, device prefix cache, and
(with --host-gib) the device cache backed by a host tier. Reports throughput, TTFT, hit tokens and token agreement
with the no-cache run.
"""
import argparse, json, random, time
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--devices", type=int, nargs="+", default=[0])
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--kv-blocks", type=int, default=17)
ap.add_argument("--max-seq-len", type=int, default=2048); ap.add_argument("--prefixes", type=int, default=4)
ap.add_argument("--prefix-len", type=int, default=512); ap.add_argument("--requests", type=int, default=32)
ap.add_argument("--max-new", type=int, default=32); ap.add_argument("--host-gib", type=float, default=0.0)
ap.add_argument("--modes", nargs="+", default=["off", "device", "host"])
a = ap.parse_args()
rng = random.Random(0)
prefixes = [tuple(rng.randrange(1000, 30000) for _ in range(a.prefix_len)) for _ in range(a.prefixes)]
reqs = [PDRequest(f"r{i}", rng.choice(prefixes) + tuple(rng.randrange(1000, 30000) for _ in range(rng.randint(16, 127))),
                  a.max_new) for i in range(a.requests)]
e = AtomRblnEngine(devices=a.devices, max_seq_len=a.max_seq_len, batch_size=a.batch, kv_blocks=a.kv_blocks)
e.start(a.model)
result, outputs = dict(model=a.model, batch=a.batch, kv_blocks=a.kv_blocks, prefixes=a.prefixes,
                       prefix_len=a.prefix_len, requests=a.requests, compiled=e.compiled, runs={}), {}
for mode in a.modes:
    if mode == "host" and not a.host_gib:
        continue
    b = e.batcher(prefix_cache=mode != "off", host_kv_gib=a.host_gib if mode == "host" else 0.0)
    first, fs = {}, []
    t0 = time.time()
    for i, r in enumerate(reqs):
        fs.append(b.submit(r, on_token=lambda _t, rid=r.request_id: first.setdefault(rid, time.time())))
        for _ in range(i % 2):
            if not b.idle:
                b.step()
    b.run_until_idle()
    dt = time.time() - t0
    out = [f.result() for f in fs]
    run = dict(seconds=round(dt, 2), tok_s=round(sum(map(len, out)) / dt, 1),
               ttft_mean_s=round(sum(first[r.request_id] - t0 for r in reqs) / len(reqs), 3),
               hit_tokens=b.hit_tokens, prompt_tokens=b.query_tokens)
    if b.blocks is not None:
        run.update(b.blocks.stats, host_hit_tokens=b.blocks.host_hit_tokens)
    if outputs:
        run["identical_to_off"] = sum(x == y for x, y in zip(outputs["off"], out))
    outputs[mode] = out
    result["runs"][mode] = run
    print(json.dumps(dict(mode=mode, **run)), flush=True)
print(json.dumps(result))
e.close()
