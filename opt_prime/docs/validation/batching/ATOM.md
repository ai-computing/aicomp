# Continuous batching on Rebellions ATOM (#44 stage 3)

`platforms.atom.batching.AtomBatcher` schedules requests onto the lanes of the optimum-rbln decoder compiled with
`batch_size=B`. Each iteration it prefills at most one request (the SDK's prefill, one sequence written into its
lane's KV pages) and runs one step of the B-lane decode graph. A finished request frees its lane for the next one
(refill). Queueing, cancellation, sampling, stop tokens and the HTTP layer are those of the torch batcher:
`AtomRblnEngine(batch_size=B).batcher()` is served by `serve.batched.BatchedService` unchanged
(`examples/pd/serve_batched.py --backend atom`).

Setup: ATOM (RBLN-CA22) on the RBLN+ATOM host, one chip per engine. Software: rebel-compiler 0.11.0,
optimum-rbln 0.11.0.post1, transformers 5.8.1. Model: Llama-3.2-1B-Instruct compiled with max_seq_len 2048.

## Throughput and tokens

`examples/pd/atom_batching_bench.py --batch 4 8 --devices 1 --ref-device 0`. The load is 32 requests with random
16–511-token prompts and 64 new tokens each, submitted with staggered arrivals. The reference runs the batch-1
compiled model one request at a time (`generate`, greedy).

| engine | wall | tok/s | × B1 | mean active lanes per step | identical to B1 |
|---|---:|---:|---:|---:|---:|
| B1, one request at a time | 41.8 s | 49.0 | 1.00 | 1 | — |
| **B4** | 31.3 s | 65.5 | 1.34 | 3.96 | 22 / 32 |
| **B8** | 16.0 s | **128.1** | **2.61** | 7.71 | 22 / 32 |

Where the token differences come from:
- **The batcher is not the cause.** `AtomBatcher` over the B1 model gives the same tokens as `generate`.
- **The compiled graphs differ numerically.** B4 and B8 give identical outputs to each other, and both differ from
  B1 at the same 10 positions. A request alone in the B8 graph diverges from B1 exactly as it does inside a full
  batch.
- **Every divergence is a near-tie.** At each of the 10 points the fp32 CPU model's top-2 logit gap is 0.012–0.134,
  and B1 and B8 each pick the fp32 top-1 or top-2 token.
- **B8 is as close to fp32 as B1.** Against the CPU fp32 greedy output, 21 / 32 B1 requests and 20 / 32 B8 requests
  are fully identical.
- Raw data: `atom_b1_b8_fp32.jsonl` in the experiment record.

## HTTP under the Azure conversation trace (acceptance)

`examples/pd/http_load.py` replays the Azure LLM inference trace (conversation; first 150 requests), keeping its
arrival times and token counts:
- arrival times are scaled by `--speed`;
- outputs are capped at 128 tokens and contexts at 2048 − output;
- prompts are random token IDs, streamed `/v1/completions` with `ignore_eos`.

Server: `serve_batched.py --backend atom --slots 8` on one ATOM.

| trace speed | offered | HTTP 200 | 429 | tok/s | TTFT p50 / p90 | client TPOT | server TPOT | client / server |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2× | 0.59 req/s | 150 / 150 | 0 | 62.4 | 0.18 / 0.41 s | 38.0 ms | 38.3 ms | **0.99** |
| 0.5× (overload) | 1.48 req/s | 150 / 150 | 0 | 75.8 | 28.0 / 88.5 s | 80.2 ms | 80.3 ms | **1.00** |

Both acceptance criteria of #44 hold: no 429 below capacity, and client TPOT within 1.5× of server TPOT. Over
capacity, requests wait in the queue (TTFT grows) instead of being rejected; 429 is returned only when the
submission queue is full (1024).

The trace's prompts are long (mean ~930 tokens). The SDK prefill call is blocking and covers the whole prompt, so
each admission stalls the decode batch for one full prefill. That is why loaded throughput (75.8 tok/s) and TPOT
(80 ms) are worse than the decode-only benchmark above.

## Limits

- Prefill is one blocking SDK call per request; it is not chunked or interleaved with decode, because the SDK's
  internal 128-token chunking runs inside that call. Prefill on another stage (RNGD, PD) would remove the stall; it
  needs KV import into the SDK's pages, which optimum-rbln 0.11 does not expose.
- Idle lanes run the dummy token at position 0 of their own page, so a B8 step costs the same with 1 or 8 requests.
- No prefix caching on ATOM (#46 and #52 are torch-only): the SDK owns the pages.
