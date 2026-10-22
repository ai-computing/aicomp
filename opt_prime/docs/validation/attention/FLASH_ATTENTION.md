# SDPA vs FlashAttention-2 on A100 (#109)

**Decision: not adopted.** SDPA stays the only GPU attention path.
- **Prefill and training:** FA2 does not help. SDPA already runs its internal FlashAttention-2 kernel for causal
  inputs without padding.
- **Decode:** FA2's paged kernel would save about 12% of a batched decode step. It needs pages of a multiple of 256
  tokens, which would make prefix caching (#46) and the host tier (#52) four times coarser than our 64-token blocks.
  The cheaper step is to cut the block-mask build of the paged flex path (see below).

## Setup

- **Hardware and software:** A100 80GB PCIe, torch 2.13.0+cu130, transformers 5.15, bf16 (FA2 has no fp32 path).
- **FA2 build:** `flash-attn` has no wheel for this torch/CUDA and does not build here. The measured kernel is the
  prebuilt `kernels-community/flash-attn2` (version 2) from the HF `kernels` package.
- **Model:** Llama-3.2-1B-Instruct (32 query heads, 8 KV heads, head dim 64, 16 layers).
- **Command:** `PYTHONPATH=. python examples/bench/flash_attention_bench.py --model unsloth/Llama-3.2-1B-Instruct`.

## Results

**Kernel, causal prefill attention, B = 1, model head geometry**

| S | fwd SDPA | fwd FA2 | × | fwd+bwd SDPA | fwd+bwd FA2 | × |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.028 ms | 0.045 ms | 0.63 | 0.168 ms | 0.335 ms | 0.50 |
| 2048 | 0.146 ms | 0.143 ms | 1.02 | 0.626 ms | 0.623 ms | 1.00 |
| 4096 | 0.470 ms | 0.452 ms | 1.04 | 1.961 ms | 1.866 ms | 1.05 |
| 8192 | 1.751 ms | 1.720 ms | 1.02 | 6.895 ms | 6.426 ms | 1.07 |

Max abs difference between the two outputs: 0.001.

**Model prefill (forward, B = 1)**

| S | SDPA | FA2 | × | argmax agreement with fp32 (SDPA / FA2) |
|---:|---:|---:|---:|---|
| 512 | 10.77 ms | 10.72 ms | 1.00 | 0.969 / 0.953 |
| 2048 | 31.28 ms | 31.93 ms | 0.98 | 0.953 / 0.938 |
| 4096 | 63.43 ms | 62.80 ms | 1.01 | 0.938 / 0.953 |

**Training step (forward + backward, B = 2, S = 2048)**

| | SDPA | FA2 |
|---|---:|---:|
| step | 223.8 ms | 223.4 ms |
| loss | 12.45106 | 12.45165 |
| grad norm | 41.25 | 41.50 |
| peak memory | 15.04 GiB | 15.04 GiB |

**Decode, one layer, 32 sequences of mixed length (128–2047, mean 1092)**

| path | time |
|---|---:|
| SDPA, padded with a mask (the slot KV path) | 2955 µs |
| paged flex attention, page 64 (the batcher's #108 path) | 133 µs |
| block-mask build for the paged flex path (once per step, shared by the layers) | 491 µs |
| FA2 `flash_attn_with_kvcache`, block table, page 256 | **64 µs** |

Max abs difference against SDPA: 0.002 for both. FA2 rejects pages of 16, 64 or 128 tokens ("block size must be
divisible by 256").

A whole batcher decode step (32 rows, 16 layers) takes 13.1 ms. Paged-flex attention is 16 × 133 µs + 491 µs ≈ 2.6 ms
of it (20%); FA2 would take 16 × 64 µs ≈ 1.0 ms. That is about 1.6 ms, or 12% of the step.

## Why not now, and what next

- **Prefill and training.** OptimusPrime's `CachedScaledDotProductAttention`, the PD engines' HF `sdpa` and the
  batcher's chunked prefill run causal attention without a padding mask. SDPA already dispatches those to FA2, which
  is why the measured gain is 0.98–1.07×. Masked chunked prefill against a gathered prefix is not causal-only, but it
  runs once per chunk and its cost is in the linear layers.
- **Decode.** The gain is real but modest at 1B / 32 sequences, and it would grow with longer contexts. Taking it
  means:
  - 256-token pages, so prefix reuse and host offload would work in 256-token units (the hit rates in
    `../batching/RESULTS.md` were measured at 64);
  - a runtime fetch of a Hub kernel as a new dependency.
- **Follow-ups:**
  - Build the flex block mask once per batch change instead of once per step, on the device without host syncs.
    That is 491 µs (3.8% of the step) with no new dependency.
  - If decode attention grows (longer contexts, larger batches), measure FlashInfer's paged decode, which accepts
    small pages, before FA2.
