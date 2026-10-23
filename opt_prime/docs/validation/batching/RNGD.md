# Continuous batching on RNGD (#122)

`platforms.rngd.batching.FuriosaLLMBatcher` serves RNGD through furiosa-llm 2026.4's native engine, behind the same
`BatchedService` HTTP layer as the GPU and ATOM batchers (`serve_batched.py --backend rngd --fxb ...`).

## Why our own round scheduler

With the artifacts we have (KONI-Llama3.1-8B, `koni-s512-tp8.fxb`, decode buckets B1/B4/B8), the native engine
serves concurrent fresh requests one after another:
- 16 concurrent requests: 61–64 tok/s, and the 16th request's first token arrives after 15 s;
- `LLM.generate` with 8 or 16 prompts: 58–61 tok/s;
- the scheduler settings we tried did not change this: profile-guided batching off, aggressive batching, the overlap
  loop, `max_num_batched_tokens`.

A fresh prefill waits until every running decode has finished: submitted while 7 requests decode, its first token
came 3.5 s later. Prompts already in the engine's prefix cache do batch (331–346 tok/s in decode).

The adapter therefore drives the engine in rounds:
1. **Prefill.** It prefills the waiting requests that fit (max_tokens=1: their first tokens).
2. **Decode quantum.** It decodes every active request for a quantum of Q tokens together. Each call carries the
   prompt plus the tokens so far, so the prefix cache turns the continuation into a short extend, and the engine
   batches these decodes.
3. **Quantum size.** Q is 16 while requests wait and 64 otherwise.

With the prefix cache disabled every quantum would recompute the context (47 tok/s), so the adapter falls back to the
native mode.

## Results

**Benchmark:** `examples/pd/rngd_batching_bench.py`, one RNGD (npu3, TP8), KONI-8B, context 512, 64 new tokens.

| run | tok/s | × one at a time | TTFT mean | TPOT mean | prefix hits |
|---|---:|---:|---:|---:|---:|
| one request at a time | 62.1 | 1.00 | — | — | — |
| rounds, 8 lanes, 32 requests | 239.4 | 3.86 | 3.65 s | 31.4 ms | (warm from the sequential run) |
| rounds, 16 lanes, 64 requests, cold | 259.0 | 4.17 | 7.02 s | 53.2 ms | 0 |
| rounds, 8 lanes, 64 requests, 256-token shared prefix | 247.2 | 3.98 | 7.64 s | 29.6 ms | 16128 / 24564 |
| the same with the host tier (`--host-kv-gib 8`) | **295.7** | **4.76** | 6.09 s | 24.7 ms | 16128 / 24564 |
| the same with the prefix cache off (rounds need it) | 47.1 | — | 39.4 s | 144 ms | 0 |

Correctness: 27/32 requests match the one-at-a-time run token for token. The rounds continue each request through
extend calls, whose bf16 numerics differ from one long decode.

**HTTP**, `serve_batched.py --backend rngd`, 8 lanes:
- smoke: " Seoul. Seoul is the largest city …";
- Azure replay of 60 requests at 0.15× speed: 60/60 with HTTP 200, no 429, TTFT p50 77 ms, client/server TPOT 0.99;
- /metrics shows `vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total`.

## KV events

furiosa-llm publishes no KV events. With `kv_events`, the adapter mirrors the prompt blocks it has sent (chained 64-bit
block hashes, LRU of `mirror_blocks`) and publishes BlockStored / BlockRemoved in vLLM's format. The mirror
approximates the engine's cache; prefix hits are the engine's own (`num_cached_tokens`).

## Limits

- Throughput depends on the artifact's buckets (here decode B≤8, context 512).
- Seeded sampling is deterministic per request, but the stream differs from a single engine call (one seed per
  quantum).
- The RNGD+ATOM host was shared with other jobs during these runs.
