"""Batched PD (#120) vs aggregated continuous batching on two GPUs, same arrivals.

    python examples/pd/pd_batching_bench.py --model unsloth/Llama-3.2-1B-Instruct --requests 64 --rate 4 \
        --modes aggregated pd-thread pd-process

aggregated: one ContinuousBatcher (paged, chunked prefill) on --decode-device.
pd-thread:  PDBatcher on --decode-device, TorchPrefillEngine on --prefill-device in a thread of this process.
pd-process: the same with the prefill engine in its own process (CUDA IPC + peer copy).
Requests arrive as a Poisson process (--rate per second); prompts --min-prompt..--max-prompt random tokens,
--max-new tokens each. Reports throughput, TTFT, TPOT (mean, p90) and token agreement with the first mode.
"""
import argparse, functools, json, random, statistics, time
import torch
from transformers import AutoModelForCausalLM
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.pd_batching import PDBatcher, PrefillWorker, ProcessPrefillWorker, TorchPrefillEngine


def load(model_id, dtype, device):
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()


def engine_factory(model_id, dtype, device, max_context, chunk):
    return TorchPrefillEngine(load(model_id, dtype, device), max_context=max_context, chunk=chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--requests", type=int, default=64); ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--min-prompt", type=int, default=512); ap.add_argument("--max-prompt", type=int, default=1536)
    ap.add_argument("--max-new", type=int, default=128); ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=256); ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--prefill-device", default="cuda:0"); ap.add_argument("--decode-device", default="cuda:1")
    ap.add_argument("--modes", nargs="+", default=["aggregated", "pd-thread", "pd-process"])
    a = ap.parse_args()
    rng = random.Random(0)
    vocab = 32000
    reqs = [PDRequest(f"r{i}", tuple(rng.randrange(100, vocab) for _ in range(rng.randint(a.min_prompt, a.max_prompt))),
                      a.max_new) for i in range(a.requests)]
    t = 0.0
    arrivals = []
    for _ in reqs:
        t += rng.expovariate(a.rate)
        arrivals.append(t)
    model = load(a.model, a.dtype, a.decode_device)
    results, outputs = {}, {}
    for mode in a.modes:
        worker = None
        kw = dict(slots=a.slots, max_context=a.max_context, kv="paged")
        if mode == "aggregated":
            b = ContinuousBatcher(model, prefill_chunk=a.chunk, **kw)
        elif mode == "pd-thread":
            worker = PrefillWorker(TorchPrefillEngine(load(a.model, a.dtype, a.prefill_device), max_context=a.max_context,
                                                      chunk=a.prefill_chunk, out_device=a.decode_device))
            b = PDBatcher(model, worker, **kw)
        else:
            worker = ProcessPrefillWorker(functools.partial(engine_factory, a.model, a.dtype, a.prefill_device,
                                                            a.max_context, a.prefill_chunk), out_device=a.decode_device)
            b = PDBatcher(model, worker, **kw)
        for i in range(2):                                    # warm-up (compiles the flex decode kernel)
            b.submit(PDRequest(f"warm{i}", tuple(range(100, 100 + a.min_prompt)), 8))
        b.run_until_idle()
        torch.cuda.synchronize()
        first, last, futures, nxt = {}, {}, {}, 0
        t0 = time.time()
        while nxt < len(reqs) or not b.idle:
            now = time.time() - t0
            while nxt < len(reqs) and arrivals[nxt] <= now:
                r = reqs[nxt]; nxt += 1
                def tok(_t, rid=r.request_id):
                    ts = time.time() - t0
                    first.setdefault(rid, ts); last[rid] = ts
                futures[r.request_id] = b.submit(r, on_token=tok)
            if b.idle:
                time.sleep(max(0.0, min(0.005, arrivals[nxt] - (time.time() - t0)))) if nxt < len(reqs) else None
                continue
            b.step()
        torch.cuda.synchronize()
        wall = time.time() - t0
        out = {k: f.result() for k, f in futures.items()}
        ttft = sorted(first[r.request_id] - arrivals[i] for i, r in enumerate(reqs))
        tpot = sorted((last[k] - first[k]) / (len(v) - 1) for k, v in out.items() if len(v) > 1)
        pct = lambda xs, q: round(xs[min(len(xs) - 1, int(q * len(xs)))], 4)
        res = dict(mode=mode, wall_s=round(wall, 2), tok_s=round(sum(map(len, out.values())) / wall, 1),
                   ttft_mean_s=round(statistics.mean(ttft), 4), ttft_p50_s=pct(ttft, .5), ttft_p90_s=pct(ttft, .9),
                   tpot_mean_ms=round(1e3 * statistics.mean(tpot), 2), tpot_p90_ms=round(1e3 * pct(tpot, .9), 2),
                   decode_steps=b.stats["decode_steps"])
        if mode != "aggregated":
            s = b.stats
            res.update(prefill_s_mean=round(s["prefill_s"] / max(1, s["prefills"]), 4),
                       transfer_ms_mean=round(1e3 * s["transfer_s"] / max(1, s["prefills"]), 2))
        if outputs:
            ref = next(iter(outputs.values()))
            res["identical_to_first"] = sum(out[k] == ref[k] for k in ref)
        outputs[mode] = out
        results[mode] = res
        print(json.dumps(res), flush=True)
        if worker is not None:
            worker.close()
        del b
        torch.cuda.empty_cache()
    print(json.dumps(dict(model=a.model, dtype=a.dtype, requests=a.requests, rate=a.rate, prompt=[a.min_prompt, a.max_prompt],
                          max_new=a.max_new, slots=a.slots, chunk=a.chunk, results=results)))


if __name__ == "__main__":
    main()
