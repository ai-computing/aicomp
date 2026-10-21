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
