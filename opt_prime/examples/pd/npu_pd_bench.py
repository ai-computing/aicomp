"""Batched PD on NPUs (#120): prefill in another process / environment, decode lanes on ATOM.

    # CPU prefill -> ATOM decode (same environment)
    python examples/pd/npu_pd_bench.py --model Qwen/Qwen2.5-0.5B-Instruct --atom-devices 1 --batch 8 --kv-blocks 17 \
        --prefill-python $(which python) --prefill-args "--engine torch --device cpu --dtype float32"
    # RNGD prefill (Furiosa venv) -> ATOM decode
    python examples/pd/npu_pd_bench.py ... --prefill-python ~/venvs/furiosa-2026.4/bin/python \
        --prefill-args "--engine rngd-qwen --chip 0 --width 512" --prefill-pythonpath <aicomp>:<rngd_pd src>
    # every prompt prefilled on RNGD, 4 shared 256-token prefixes, ATOM prefix cache backed by a 4 GiB host tier
    python examples/pd/npu_pd_bench.py ... --remote-only --prefixes 4 --prefix-len 256 --prefix-cache --host-gib 4

Runs the same staggered requests twice: aggregated on ATOM (AtomBatcher prefills on ATOM) and PD (prefill worker,
KV imported into the lanes' blocks; by default each prompt goes to the side that should finish it first, with
--remote-only every prompt is prefilled remotely). Reports throughput, TTFT, TPOT, token agreement, prefix hits,
host-tier traffic and where prompts were prefilled.
"""
import argparse, json, os, random, shlex, sys, time
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
from transformers import AutoConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--atom-devices", type=int, nargs="+", default=[0])
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--kv-blocks", type=int, default=17)
ap.add_argument("--max-seq-len", type=int, default=2048); ap.add_argument("--requests", type=int, default=32)
ap.add_argument("--min-prompt", type=int, default=64); ap.add_argument("--max-prompt", type=int, default=480)
ap.add_argument("--max-new", type=int, default=64)
ap.add_argument("--prefill-python", required=True); ap.add_argument("--prefill-args", required=True)
ap.add_argument("--prefill-pythonpath", default=None); ap.add_argument("--prefix-cache", action="store_true")
ap.add_argument("--modes", nargs="+", default=["aggregated", "pd"])
ap.add_argument("--host-gib", type=float, default=0.0); ap.add_argument("--remote-only", action="store_true")
ap.add_argument("--prefixes", type=int, default=0, help="shared prompt prefixes (0: independent prompts)")
ap.add_argument("--prefix-len", type=int, default=256)
a = ap.parse_args()
rng = random.Random(0)
prefixes = [tuple(rng.randrange(1000, 30000) for _ in range(a.prefix_len)) for _ in range(a.prefixes)]
reqs = [PDRequest(f"r{i}", (rng.choice(prefixes) if prefixes else ()) +
                  tuple(rng.randrange(1000, 30000) for _ in range(rng.randint(a.min_prompt, a.max_prompt))),
                  a.max_new) for i in range(a.requests)]
cfg = AutoConfig.from_pretrained(a.model)
head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
e = AtomRblnEngine(devices=a.atom_devices, max_seq_len=a.max_seq_len, batch_size=a.batch, kv_blocks=a.kv_blocks)
e.start(a.model)
result, outputs = dict(model=a.model, batch=a.batch, requests=a.requests, prefill=a.prefill_args, runs={}), {}
for mode in a.modes:
    worker = None
    if mode == "pd":
        env = dict(os.environ)
        if a.prefill_pythonpath:
            env["PYTHONPATH"] = a.prefill_pythonpath
        t0 = time.time()
        worker = SubprocessPrefillWorker(a.prefill_python, shlex.split(a.prefill_args) + ["--model", a.model],
                                         slots=a.batch, layers=cfg.num_hidden_layers, kv_heads=cfg.num_key_value_heads,
                                         head_dim=head_dim, max_context=a.max_seq_len, env=env)
        result["prefill_start_s"] = round(time.time() - t0, 1)
    b = e.batcher(prefix_cache=a.prefix_cache, host_kv_gib=a.host_gib, prefill_worker=worker,
                  local_prefill=not a.remote_only)
    first, last, fs = {}, {}, []
    t0 = time.time()
    for i, r in enumerate(reqs):
        def tok(_t, rid=r.request_id):
            now = time.time(); first.setdefault(rid, now); last[rid] = now
        fs.append(b.submit(r, on_token=tok))
        for _ in range(i % 2):
            if not b.idle:
                b.step()
    b.run_until_idle()
    dt = time.time() - t0
    out = [f.result() for f in fs]
    tpot = [(last[r.request_id] - first[r.request_id]) / (len(o) - 1) for r, o in zip(reqs, out) if len(o) > 1]
    run = dict(seconds=round(dt, 2), tok_s=round(sum(map(len, out)) / dt, 1),
               ttft_mean_s=round(sum(first[r.request_id] - t0 for r in reqs) / len(reqs), 3),
               tpot_mean_ms=round(1e3 * sum(tpot) / len(tpot), 1), decode_steps=b.stats["decode_steps"],
               hit_tokens=b.hit_tokens, prompt_tokens=b.query_tokens)
    if worker is not None:
        run["prefilled"] = dict(b.stats_pd)
    if b.blocks is not None and a.host_gib:
        run.update(host_offloads=b.blocks.stats["host_offloads"], host_reloads=b.blocks.stats["host_reloads"],
                   host_hit_tokens=b.blocks.host_hit_tokens)
    if outputs:
        ref = outputs["aggregated"]
        run["identical_to_aggregated"] = sum(x == y for x, y in zip(ref, out))
        run["first_token_agree"] = sum(x[0] == y[0] for x, y in zip(ref, out))
    outputs[mode] = out
    result["runs"][mode] = run
    print(json.dumps(dict(mode=mode, **run)), flush=True)
    if worker is not None:
        worker.close()
print(json.dumps(result))
e.close()
