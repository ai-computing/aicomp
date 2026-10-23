# Shared KV store: LMCache-compatible chunks, Mooncake Store (#128)

`pd.kvstore.RemoteKVTier` puts the KV of every full prompt chunk into a shared store, and reads chunks back when a new
request misses the device cache and the host tier. Other engines can therefore reuse a prefix that one engine
computed. The store is used as follows:
- **When the prompt is done:** for each full chunk, the batcher copies it to the host and writes it from a background
  thread (keys that already exist are skipped).
- **On admission:** the batcher looks for consecutive chunks that start right after the device/host hit and writes
  them into the slot's pages with one span copy per chunk. The prefill then starts after them.

| piece | choice | why |
|---|---|---|
| keys | `LMCacheMPKeys`: blake3 chain over 256-token chunks (tokens as big-endian uint32), key `{model}@{kv_rank:08x}@{group}@{hash}` | the scheme of LMCache's multiprocess mode. It does not depend on Python `hash` or the environment, so other engines can compute it; ours equals LMCache 0.5.6's `TokenHasher` bit for bit (test vector in `tests/test_kvstore.py`) |
| keys (alternative) | `OptimusKeys`: our chained 64-bit block hashes | stores used only by Optimus Prime engines |
| value | `KV_2LTD` `[2, layers, chunk, kv_heads*head_dim]` in the KV dtype | LMCache's chunk layout |
| store | `MooncakeKVStore` (`MooncakeDistributedStore` put/get/is_exist; TCP or RDMA), `MemoryKVStore` (tests) | Mooncake Store is the shared tier that LMCache, vLLM and Dynamo deploy |

Usage: `ContinuousBatcher(..., prefix_cache=True, remote_kv=RemoteKVTier(store, keys, layers=, kv_heads=, head_dim=,
dtype=))`, or `serve_batched.py --prefix-cache --remote-kv mooncake.json --remote-kv-keys lmcache --remote-kv-model
<the vLLM model name>`.

## Verification
- **Keys.** `LMCacheMPKeys("Qwen/Qwen3-8B").chunk_hashes(tokens)` equals `lmcache.v1.multiprocess.token_hasher
  .TokenHasher(256, "blake3").compute_chunk_hashes(tokens)` (LMCache 0.5.6).
- **Two engines, one store** (tiny Llama, fp32, 16-token chunks, 8-token pages):
  - The first engine serves a 53-token prompt and writes 3 chunks.
  - A second engine, started with an empty device cache, serves a prompt with the same 50-token prefix. It reads
    48 tokens (3 chunks) from the store, prefills only the rest, and its tokens equal HF greedy.
  - This works with both key schemes, with `MemoryKVStore`, and with a real `mooncake_master` on localhost (TCP, HTTP
    metadata server; Mooncake 0.3.13, torch 2.14, transformers 5.20).

## Limits
- **Not verified against a live vLLM+LMCache instance.**
  - Matching keys is necessary for cache sharing but may not be sufficient: the value layout that LMCache's MP L2
    adapters write is unverified.
  - LMCache's in-process mode uses environment-dependent hashes and cannot be matched.
- **Uniform families only.** Hybrid families (sliding/global groups, recurrent states) keep the device and host
  tiers.
- **The ATOM/RNGD engines do not use the shared store yet.**
- **KV events for chunks served from the store are not published.**
