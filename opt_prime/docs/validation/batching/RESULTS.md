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
