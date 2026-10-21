"""Replay the Azure LLM inference trace (conversation) against an OpenAI-style endpoint (#44 acceptance).

    curl -LO https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv
    python examples/pd/http_load.py --url http://127.0.0.1:8000 --model unsloth/Llama-3.2-1B-Instruct \
        --trace AzureLLMInferenceTrace_conv.csv --requests 200 --speed 1.0 --max-context 2048

Requests keep the trace's arrival times (divided by --speed) and its context/generated token counts, capped to fit
--max-context and --max-output. Prompts are random token IDs (so no prefix reuse), streamed `/v1/completions` with
`ignore_eos`. Reports per-request client TTFT and TPOT, HTTP status counts (429 = rejected for capacity), and the
server's own TPOT from /metrics (vllm:time_per_output_token_seconds), with client TPOT / server TPOT.
"""
import argparse, csv, json, random, statistics, threading, time
import urllib.request
from datetime import datetime


def load(path, n, max_context, max_output):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            t = datetime.strptime(r["TIMESTAMP"][:26], "%Y-%m-%d %H:%M:%S.%f").timestamp()
            out = max(1, min(int(r["GeneratedTokens"]), max_output))
            ctx = max(1, min(int(r["ContextTokens"]), max_context - out))
            rows.append((t, ctx, out))
            if len(rows) == n:
                break
    t0 = rows[0][0]
    return [(t - t0, c, o) for t, c, o in rows]


def server_tpot(url):
    text = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    vals = {}
    for line in text.splitlines():
        for k in ("vllm:time_per_output_token_seconds_sum", "vllm:time_per_output_token_seconds_count"):
            if line.startswith(k + "{"):
                vals[k] = float(line.rsplit(" ", 1)[1])
    return vals.get("vllm:time_per_output_token_seconds_sum", 0.0), vals.get("vllm:time_per_output_token_seconds_count", 0.0)


def one(a, i, ctx, out, rng, res):
    prompt = [rng.randrange(100, a.vocab) for _ in range(ctx)]
    body = json.dumps(dict(model=a.model, prompt=prompt, max_tokens=out, ignore_eos=True, stream=True)).encode()
    req = urllib.request.Request(a.url + "/v1/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "x-request-id": f"load-{i}"})
    t0, times, usage, status = time.time(), [], None, None
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as resp:
            status = resp.status
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                msg = json.loads(line[6:])
                if msg.get("usage"):
                    usage = msg["usage"]
                elif msg.get("choices") and msg["choices"][0].get("text"):
                    times.append(time.time())
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception as e:                                   # connection errors count as failures
        status = type(e).__name__
    n = usage["completion_tokens"] if usage else 0
    res[i] = dict(status=status, ctx=ctx, out=out, tokens=n, ttft=(times[0] - t0) if times else None,
                  tpot=((times[-1] - times[0]) / (n - 1)) if len(times) > 1 and n > 1 else None, e2e=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--trace", required=True); ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--speed", type=float, default=1.0, help="arrival-time compression (2 = twice the rate)")
    ap.add_argument("--max-context", type=int, default=2048); ap.add_argument("--max-output", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=32000); ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    plan = load(a.trace, a.requests, a.max_context, a.max_output)
    s0, c0 = server_tpot(a.url)
    res, threads, start = {}, [], time.time()
    for i, (t, ctx, out) in enumerate(plan):
        delay = start + t / a.speed - time.time()
        if delay > 0:
            time.sleep(delay)
        th = threading.Thread(target=one, args=(a, i, ctx, out, random.Random(i), res)); th.start(); threads.append(th)
    for th in threads:
        th.join()
    wall = time.time() - start
    s1, c1 = server_tpot(a.url)
    ok = [r for r in res.values() if r["status"] == 200 and r["tokens"] == r["out"]]
    statuses = {}
    for r in res.values():
        statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
    tp = sorted(r["tpot"] for r in ok if r["tpot"] is not None)
    tt = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    srv = (s1 - s0) / (c1 - c0) if c1 > c0 else None
    pct = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None
    summary = dict(requests=len(plan), offered_rps=round(len(plan) / (plan[-1][0] / a.speed or 1), 3), wall_s=round(wall, 1),
                   status=statuses, complete=len(ok), out_tokens=sum(r["tokens"] for r in ok),
                   tok_s=round(sum(r["tokens"] for r in ok) / wall, 1),
                   ttft_p50_s=pct(tt, .5), ttft_p90_s=pct(tt, .9),
                   client_tpot_mean_s=statistics.mean(tp) if tp else None, client_tpot_p90_s=pct(tp, .9),
                   server_tpot_mean_s=srv,
                   client_over_server_tpot=round(statistics.mean(tp) / srv, 3) if tp and srv else None)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
