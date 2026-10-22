# Batched PD disaggregation (#120)

`pd.pd_batching.PDBatcher` is the decode side of batched PD:
- It schedules like the continuous batcher: paged KV, flex-attention decode over all lanes.
- A request's prompt is prefilled by a prefill worker that runs concurrently on another device or process.
- When the worker's result arrives, the KV span lands in the lane's pages with one indexed copy, the first token is
  emitted, and the request joins the next decode step.
- Decode steps never wait for a prefill.

Prefix reuse works on both sides:
- The prefill side (`TorchPrefillEngine`) keeps its own paged prefix cache, so a cached prompt prefix is not
  recomputed.
- Blocks already in the decode pool or its host tier are not transferred at all.

| worker | where the prefill runs | KV hand-off |
|---|---|---|
| `PrefillWorker(TorchPrefillEngine)` | another GPU (or CPU), thread of this process | copy to the decode device on a side stream |
| `ProcessPrefillWorker(factory)` | child process (spawn) | CUDA IPC handle, receiver thread copies, then acks |
| `SubprocessPrefillWorker(python, args)` | `pd.prefill_server` under another interpreter/venv (NPU SDKs) | shared-memory slab, one slot per decode lane |

The served form is
`examples/pd/serve_batched.py --pd-prefill-device cuda:0` or `--pd-prefill-python ... --pd-prefill-args ...`.

## GPU: 2×A100 80GB PCIe, Llama-3.2-1B-Instruct

**Benchmark:** `examples/pd/pd_batching_bench.py`, 128 requests with Poisson arrivals, random 512–1536-token
prompts, 128 new tokens, 32 decode slots, bf16, torch 2.13. "Aggregated" is one `ContinuousBatcher` on one GPU
(chunked prefill, 256 tokens). "PD" means prefill on GPU0 and decode on GPU1.

| rate | mode | tok/s | TTFT p50 / p90 | TPOT mean / p90 | KV hand-off |
|---:|---|---:|---:|---:|---:|
| 4/s | aggregated | 585 | 133 / 287 ms | 14.5 / 16.6 ms | — |
| 4/s | PD, thread | 590 | 39 / 52 ms | 11.2 / 11.8 ms | 2.5 ms |
| 4/s | PD, process | 590 | 38 / 48 ms | 10.8 / 12.5 ms | 2.3 ms |
| 8/s | aggregated | 1101 | 368 / 674 ms | 21.1 / 23.0 ms | — |
| 8/s | PD, process | 1124 | 39 / 54 ms | 11.1 / 11.7 ms | 2.3 ms |
| 16/s | aggregated (saturated) | 1122 | 3.47 / 6.02 s | 21.7 / 23.3 ms | — |
| 16/s | PD, thread | 1893 | 343 / 613 ms | 13.9 / 14.6 ms | 2.4 ms |
| 16/s | **PD, process** | **1969** | **178 / 354 ms** | **13.0 / 14.9 ms** | 2.2 ms |

Correctness:
- In fp32 (16 requests, 32 tokens), all three modes give identical tokens (16/16).
- In bf16, PD and aggregated agree on 97/128 requests. The rest is bf16 numerics: the prefill is one 2048-token chunk
  in PD and 256-token chunks in aggregated.

PD keeps TPOT flat under load, because decode never runs a prefill chunk. TTFT drops because prefill has a device to
itself. At 16 req/s the aggregated batcher saturates, while PD with two GPUs serves 1.75× its throughput.

**GPU peer copies on this host are unsafe.** Direct cuda:0 → cuda:1 copies deliver wrong KV and overwrite unrelated
memory on the destination GPU. This happens even though `torch.cuda.can_device_access_peer` reports True. On this host
the two GPUs are on different sockets (topology SYS) and the NVIDIA open kernel module is loaded. As a result, KV
between GPUs is staged through pinned host memory by default (`pd_batching.move_kv`); direct peer copies need
`OPT_PRIME_PEER_COPY=1` and should be enabled only on verified links (NVLink, one PCIe switch). The timings above use
the staged copy.

**HTTP**, Azure conversation trace (first 200 requests, 1.0× speed, ≤ 128 output tokens), `serve_batched.py`, 32
slots:

| mode | 429 | tok/s | TTFT p50 / p90 | client / server TPOT |
|---|---:|---:|---:|---:|
| aggregated | 0 | 205 | 6.2 / 29.3 s | 0.995 |
| PD (prefill GPU0, decode GPU1) | 0 | 231 | 2.2 / 21.5 s | 0.996 |

During these runs other users' jobs kept the host's 24 cores busy (load average about 85). The Python-launched decode
loop is CPU-bound, so the absolute numbers are lower than in the offline table: the same Azure plan replayed in
process, without HTTP, gave the same TPOT. Compare the rows with each other, not with the table above. The serving
change in this PR (handlers wake on results, and check for client disconnect every 0.5 s instead of polling every
50 ms) raised aggregated throughput from 161 to 205 tok/s under the same contention.

## Limits

- Torch PD needs uniform full-attention families, as the batcher does.
- `SubprocessPrefillWorker` engines:
  - `torch` (any device);
  - `rngd-qwen`: Qwen2-family dense models, context 512, prompts up to the compiled width.
- GPU↔RNGD direct DMA-BUF hand-off is #121.
- The prefill worker serves one request at a time; prefills of different requests do not batch.
