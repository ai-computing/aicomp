"""Continuous batching on ATOM (#44 stage 3): the B-lane optimum-rbln decode graph vs one request at a time.

    python examples/pd/atom_batching_bench.py --model unsloth/Llama-3.2-1B-Instruct --batch 4 8 --devices 1 --ref-device 0

The reference runs every request alone through the batch-1 compiled model (`generate`, greedy) on --ref-device; the
batched runs submit all requests at once (staggered arrivals: a few iterations between submissions) to an
`AtomBatcher` over the batch-B model on --devices. Reports tok/s and per-request token agreement.
"""
import argparse, json, time
import torch
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--batch", type=int, nargs="+", default=[4, 8])
ap.add_argument("--devices", type=int, nargs="+", default=[0])
ap.add_argument("--ref-device", type=int, default=None)
ap.add_argument("--requests", type=int, default=32)
ap.add_argument("--max-new", type=int, default=64)
ap.add_argument("--max-seq-len", type=int, default=2048)
a = ap.parse_args()
g = torch.Generator().manual_seed(0)
from transformers import AutoConfig
vocab = AutoConfig.from_pretrained(a.model).vocab_size
reqs = [PDRequest(f"r{i}", tuple(torch.randint(100, min(vocab, 32000), (int(torch.randint(16, 512, (1,), generator=g)),),
                                               generator=g).tolist()), a.max_new) for i in range(a.requests)]
result = dict(model=a.model, requests=a.requests, max_new=a.max_new, runs={})

ref = None
if a.ref_device is not None:
    e = AtomRblnEngine(devices=[a.ref_device], max_seq_len=a.max_seq_len, batch_size=1)
    e.start(a.model)
    e.generate(reqs[0].input_ids, 4)
    t0 = time.time()
    ref = [e.generate(r.input_ids, a.max_new)[0] for r in reqs]
    dt = time.time() - t0
    result["sequential_b1"] = dict(seconds=round(dt, 2), tok_s=round(sum(map(len, ref)) / dt, 1), compiled=e.compiled)
    e.close()
    print(json.dumps(result["sequential_b1"]), flush=True)

for B in a.batch:
    e = AtomRblnEngine(devices=a.devices, max_seq_len=a.max_seq_len, batch_size=B)
    e.start(a.model)
    b = e.batcher()
    b.submit(PDRequest("warm", (1, 2, 3, 4), 4)); b.run_until_idle()
    t0, futures = time.time(), []
    for i, r in enumerate(reqs):
        futures.append(b.submit(r))
        for _ in range(i % 3):
            if not b.idle:
                b.step()
    b.run_until_idle()
    dt = time.time() - t0
    out = [f.result() for f in futures]
    run = dict(batch=B, seconds=round(dt, 2), tok_s=round(sum(map(len, out)) / dt, 1), compiled=e.compiled,
               decode_steps=b.stats["decode_steps"], mean_rows=round(b.stats["decode_rows"] / b.stats["decode_steps"], 2))
    if ref is not None:
        div = [next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), None) for p, q in zip(ref, out)]
        run.update(identical=sum(d is None and len(p) == len(q) for d, p, q in zip(div, ref, out)),
                   first_divergence=[d for d in div if d is not None],
                   speedup=round(run["tok_s"] / result["sequential_b1"]["tok_s"], 2))
    result["runs"][B] = run
    print(json.dumps(run), flush=True)
    e.close()
print(json.dumps(result))
