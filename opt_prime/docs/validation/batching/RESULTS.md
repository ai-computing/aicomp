# Continuous batching with chunked prefill (#44, stage 1: engine)

`opt_prime.accel.pd.batching.ContinuousBatcher` with a torch causal LM. Every iteration it admits waiting requests
into free slots, runs one prefill chunk, then one batched decode step over all decoding slots.

- **Setup:** A100 80GB, torch 2.13.0+cu130, transformers 5.15.0, `unsloth/Llama-3.2-1B-Instruct`.
- **Workload:** 32 requests with random prompts of 16–511 tokens, 64 new tokens each, all submitted at once.
- **Baseline:** the same batcher with one slot and whole-prompt prefill, i.e. one request at a time.
- **Method:** `examples/pd/batching_bench.py`, one run per row.

| dtype | slots | prefill chunk | one at a time | continuous batching | speed-up | mean decode batch | requests token-identical to one-at-a-time |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 8 | 256 | 130 tok/s | 562 tok/s | 4.33× | 7.45 | 20 / 32 |
| bf16 | 16 | 256 | 140 tok/s | 733 tok/s | 5.25× | 13.03 | 18 / 32 |
| bf16 | 8 | 64 | 141 tok/s | 447 tok/s | 3.18× | 6.37 | 21 / 32 |
| fp32 | 8 | 64 | 113 tok/s | 285 tok/s | 2.52× | 6.37 | **32 / 32** |

**Correctness.** In fp32 every request reproduces the one-at-a-time tokens exactly. The scheduling, per-slot masking
and positions, the slot KV writes and chunked prefill are therefore exact. In bf16 the batched kernels round
differently for different batch shapes, so near-tied greedy steps (frequent with random-token prompts) can diverge;
serving engines in general show the same effect.

**CPU tests** (`tests/test_pd_batching.py`, fp32 tiny Llama and Qwen3-MoE) pass on transformers 5.1, 5.8 and 5.15:
- staggered arrivals with prefill chunks of 1, 7 and 256 tokens: every request equals its single-request greedy run;
- chunked-prefill KV equals full-prefill KV;
- decode keeps running while a long prompt prefills;
- stop tokens, cancellation, slot reuse and the context limit.

**Not in this stage:**
- the llm-d stage service (slot table, admission, streaming per request);
- NPU batched graphs;
- hybrid or sliding-attention families, which are rejected.

## Paged KV (#108)

`ContinuousBatcher(kv="paged")` keeps the KV in a page pool, `[layers, kv_heads, pages × page_size, head_dim]`. A
request takes only the pages it needs (prompt + max new tokens). A decode step reads the pool **in place**: the
registered `opt_paged` attention runs a compiled `flex_attention` with a block mask built from the page tables, one
mask per step shared by all layers. The per-step prefix copy of the slot store (30–47% of a decode step) is gone.
Prefill chunks still gather the request's prefix once per chunk.

Same workload as above (32 requests, 64 new tokens, chunk 256), page size 64:

| dtype | KV | slots | continuous batching | vs one at a time | requests token-identical |
|---|---|---:|---:|---:|---:|
| fp32 | slot (above) | 8 | 285 tok/s | 2.52× | 32 / 32 |
| fp32 | **paged** | 8 | **426 tok/s** | 3.81× | **32 / 32** |
| bf16 | slot | 8 | 541 tok/s | 4.38× | 20 / 32 |
| bf16 | **paged** | 8 | **625 tok/s** | 4.62× | 18 / 32 |
| bf16 | slot | 16 | 722 tok/s | 5.48× | 18 / 32 |
| bf16 | **paged** | 16 | **899 tok/s** | 6.85× | 19 / 32 |
| bf16 | **paged** | 32 | **966 tok/s** | 7.73× | 21 / 32 |

As before, fp32 is token-exact and bf16 differences are batch-shape rounding. CPU tests run every batching case for
both KV layouts (flex attention eager on the CPU) on transformers 5.1, 5.8 and 5.15. GPU tests use the compiled
kernel.

Setup notes:
- flex decoding lowers here only with a 128-row query block; `Q_BLOCK = 128`, with the KV block equal to the page.
- The default pool equals `slots × max_context` (the same memory as the slot store). Pass `pages` to set a smaller
  KV budget: admission then waits for free pages, not for a free slot.

## Serving over HTTP (#44 stage 2)

`serve.batched.BatchedService` + `BatchedOwner` serve a `ContinuousBatcher` through `llmd.serve_stage` (the same
HTTP layer as the PD stage): /v1/completions, /v1/chat/completions, SSE, cancellation on disconnect, /metrics.
Requests are not limited to one lane: they wait for a free slot or free pages.

Run with `examples/pd/serve_batched.py --model unsloth/Llama-3.2-1B-Instruct --slots 32 --kv paged`. The load was
streaming chat requests with 64 new tokens each (`ignore_eos`), on an A100 in bf16:

| concurrent requests | wall | throughput | TTFT p50 | end-to-end p50 |
|---:|---:|---:|---:|---:|
| 1 | 0.76 s | 85 tok/s | 11 ms | 0.75 s |
| 32 | 1.97 s | **1037 tok/s** | 256 ms | 1.14 s |

## Prefix caching and KV events (#46)

With `ContinuousBatcher(kv="paged", prefix_cache=True)`, full prompt blocks (one block = one page) are
content-addressed with chained 64-bit hashes. A new request reuses the cached pages of its longest matching prefix and
prefills only the rest. The last prompt token is always computed, because it produces the first token's logits.
Shared pages are reference-counted and only read; writes go to the request's own pages. Unreferenced cached pages
stay in the pool until a new request needs them, then the least recently used go first.

`kv_events=serve.kv_events.ZmqKVEventPublisher(...)` publishes `BlockStored` / `BlockRemoved` once per step, in
vLLM's wire format (ZMQ PUB `[topic, seq, msgpack [ts, [event, ...]]]`, int block hashes, topic
`kv@<pod-ip>:<port>@<model>`), so the llm-d EPP's prefix-cache scorer can route to the pod that holds a prefix.
/metrics adds `vllm:cache_config_info` (`block_size`, read by the EPP), `vllm:prefix_cache_queries_total` and
`vllm:prefix_cache_hits_total`.

`examples/pd/prefix_cache_bench.py`, Llama-3.2-1B-Instruct on an A100 80GB PCIe (torch 2.13, transformers 5.15).
The load was 32 requests sharing a 1536-token system prompt, each with its own 16–127-token suffix and 32 new tokens;
8 slots, 256-token chunks, 64-token pages:

| dtype | cache | wall | TTFT mean | TTFT p50 | prefill chunks | cached prompt tokens | identical to no cache |
|---|---|---:|---:|---:|---:|---:|---:|
| fp32 | off | 12.51 s | 6.35 s | 6.55 s | 225 | 0 / 51581 | — |
| fp32 | on | **5.13 s** (2.44×) | **3.21 s** (1.98×) | 3.53 s | 81 | 36864 / 51581 | 32 / 32 |
| bf16 | off | 5.27 s | 2.52 s | 2.60 s | 225 | 0 / 51581 | — |
| bf16 | on | **2.89 s** (1.82×) | **1.57 s** (1.60×) | 1.71 s | 81 | 36864 / 51581 | 32 / 32 |

The first 8 requests are admitted together, before any prefix is registered, so they all miss. The remaining 24 hit
the 1536-token prefix (24 × 1536 = 36864). A prefix becomes reusable when its request finishes prefill; requests in
flight at the same time do not share prefill work.

CPU tests (`tests/test_pd_prefix_cache.py`) check:
- a shared prefix skips its prefill chunks and the tokens stay exact;
- a request with a fully cached prompt still computes its last prompt token;
- eviction under a small pool emits `BlockRemoved`;
- pages are reference-counted while concurrent requests share them;
- the event encoding and the ZMQ multipart message match the vLLM format;
- the metrics are present.

## Host-DRAM KV tier (#52)

`ContinuousBatcher(..., prefix_cache=True, host_kv_gib=N)` puts a `HostKVTier` behind the prefix cache:
- **Buffer.** One buffer of K/V blocks `[layers, kv_heads, page, head_dim]`, allocated once and pinned on CUDA. A block
  is contiguous and the buffer can be registered whole with a transfer engine.
- **Write-back.** When the device pool evicts a cached page, the page is written back before it is reused, ordered on
  the same stream.
- **Reload.** A later prefix hit reloads the block instead of recomputing it. The part of a match that is still on the
  device is reused; only the rest is reloaded.
- **Reload rule.** With `reload_bandwidth` and `prefill_rate` the tier reloads only when
  `block_bytes / bandwidth < page / prefill_rate`.
- **Events.** A host copy is published as `BlockStored` with `medium="CPU"` and no tokens, and host evictions as
  `BlockRemoved` with `medium="CPU"`. This is the tiered-prefix layout llm-d expects.
- **Metrics.** /metrics adds `airm_kv_host_blocks`, `airm_kv_host_capacity_blocks`, `airm_kv_host_hit_tokens_total`,
  `airm_kv_offload_blocks_total` and `airm_kv_reload_bytes_total`.

`examples/pd/kv_tier_replay.py` replays the Mooncake FAST'25 tool-agent trace (the first 2000 requests, ~150 s of
arrivals). Each 512-token trace block becomes one 64-token page of tokens derived from its hash ID, which keeps the
prefix structure with prompts 8× shorter (≤ 28 blocks). Requests run closed-loop at concurrency 8 in trace order, with
16 new tokens each. Llama-3.2-1B-Instruct, bf16, A100 80GB PCIe:

| device pages | host tier | hit rate | from host | TTFT mean | TTFT p50 | prefill chunks | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | — | 0.427 | — | 68.7 ms | 55.5 ms | 3398 | 91.9 s |
| 256 | 4 GiB (2048 blocks) | 0.461 | 0.033 | 66.1 ms | 48.7 ms | 3301 | 91.4 s |
| 256 | 16 GiB (8192 blocks) | **0.539** | 0.111 | **56.5 ms** (−18%) | **35.0 ms** (−37%) | 3070 | 86.5 s |
| 512 | — | 0.429 | — | 69.9 ms | 57.3 ms | 3393 | 94.4 s |
| 512 | 4 GiB | 0.467 | 0.038 | 66.9 ms | 46.0 ms | 3284 | 93.0 s |
| 512 | 16 GiB | **0.539** | 0.109 | **57.8 ms** (−17%) | **35.0 ms** (−39%) | 3070 | 88.6 s |

Reload versus recompute, measured on the same run:

| dtype | block | reload | prefill rate (512-token chunk) | recompute | break-even bandwidth |
|---|---:|---:|---:|---:|---:|
| bf16 | 2 MiB | 117 µs/block (17.9 GB/s) | 43.8k tok/s | 1460 µs/block | 1.44 GB/s |
| fp32 | 4 MiB | 199 µs/block (21.0 GB/s) | 5.7k tok/s | 11.2 ms/block | 0.38 GB/s |

For this 1B model on the A100, reloading a block is about 12× faster than recomputing it, so the reload rule always
picks reload.

Correctness: an fp32 replay (800 requests, 256 pages, host tier 0 vs 16 GiB) gives identical tokens for 800/800
requests. In bf16, 1804–1985 of 2000 match the HBM-only run. Hits change where prefill starts and how the chunks are
cut, which changes bf16 rounding; fp32 is exact.

Limits:
- Trace blocks are scaled to 64 tokens, so the absolute TTFT values apply only to the scaled prompts.
- Write-back happens on eviction, not write-through.
- Reload is synchronous within admission; it is not overlapped layer by layer (#47).
- The tier is local to one engine; there is no sharing across pods.
