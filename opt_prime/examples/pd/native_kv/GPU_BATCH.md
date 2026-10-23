# Batched native-KV GPU decoding: first HTTP measurements

- **Date**: 2026-10-22
- **Task**: aicomp #146
- **Host / device**: GPU+RNGD host; A100 80GB PCIe and RNGD; NVIDIA 580.178.04, Furiosa 2026.4.0
- **Commit**: 0c8ebdf parent plus changes identified by gpu_batch_manifest.json

## Question

Can real batched GPU decode close the concurrent HTTP serving gap?

## Setup

Qwen/Qwen3-8B-FP8 revision 220b46e3b2180893580a4454f21f22d3ebb187d3.
Torch 2.10.0+cu128, vLLM 0.18.0, transformers 5.1.0 (GPU).
Native uses the existing TP8/2048 FXB, one RNGD, prefix caching disabled, poll_timeout_ms=1.
GPU uses native prefill on that RNGD plus an A100, Marlin W8A16, fused norms/MLP/QKV.
This is a larger hardware budget than native alone; arithmetic is not identical.

Batch KV has independent positions and disjoint paged storage per lane. One CUDA graph
runs batched projections and FA2 attention. HTTP admits waves of up to four requests,
including arrivals during the first native prefill. Prefill/import are still sequential;
new requests do not join an already decoding wave. Default server remains single-lane.

Input IDs now use a seed per length, independent of selected length order/concurrency.
Do not compare absolute before/after numbers as a controlled paired experiment with PR145:
its prompt selection differed and CPU contention is not controlled.

## Command

With MODEL_DIR pointing at the pinned local checkpoint, and the native server configured
as in GPU_HTTP.md:

```bash
python3 serve_gpu.py --model-dir "$MODEL_DIR" --norm-mode fused --fuse-mlp --fuse-qkv --batch-size 4
python3 bench_http.py --lengths 512 --concurrency 4 --repeats 5 --out http_c4_v2.json
python3 check_batch_gpu.py --model-dir "$MODEL_DIR" --output extended.json
python3 check_batch_http.py --model-dir "$MODEL_DIR" --out quality.json
python3 -m unittest discover -s . -p 'test_*.py'
```

## Results

HTTP SSE, input512/output128, concurrency4, one warmup excluded, five measured alternating runs:

| Configuration | Aggregate tok/s median | Mean TTFT median | Mean TPOT median |
|---|---:|---:|---:|
| Native RNGD | 336.16 | 155.29 ms | 10.70 ms |
| RNGD prefill + batched A100 decode | 276.23 | 542.30 ms | 10.35 ms |

- Earlier two-millisecond-only admission: GPU median265.12 versus native300.20 tok/s (three runs); GPU varied159.31–269.49 because some arrivals missed the wave. Admitting arrivals during prefill removed that observed split in the five-run follow-up.
- Isolated mixed-length batch:391.58 tok/s,10.22ms/step, excluding prefill/import/HTTP. Input lengths16/128/512/1024, output128 each.
- Single-lane versus batched generated token positions:128/128,128/128,93/128,128/128. The512 synthetic case diverges; no universal numerical parity claim.
- Earlier32-token check matched all128 positions at392.42 tok/s. The first implementation failed initialization because FA2 rejects simultaneous cu_seqlens_k and seqused_k. Corrected to paged KV and explicit block table.
- Hardware checks passed cancellation, EOS retirement, lane reuse, and replacement while a different lane keeps decoding.
- Fifteen question/retrieval cases: native, solo batched-server request, and concurrent mixed requests all produced matching text. This is a small smoke set, not a broad quality benchmark.
- Live HTTP cancellation: disconnect after the first streamed chunk; peer and subsequent request each completed128 tokens.
- Existing29 tests passed. New batch behavior is exercised by the hardware scripts above, not inferred from those29 tests.

## Conclusion

Concurrent throughput reaches82% of native in this experiment, but native parity is not
achieved. Serial native prefill/import increases TTFT. Next: overlap prefill with decode
and allow continuous admission, preserve fast single-request execution, investigate the
synthetic numerical divergence, and repeat full HTTP/quality/cancellation comparisons.
A consumer whose 32-event queue is full pauses only its own lane; if it stays full for 1 s (STALL_SECONDS) the request is cancelled so the wave can retire. Error delivery never blocks the GPU thread. This experimental worker is not production serving.

Research-workspace evidence: analysis/native_kv_batch/{first.log,second.json,extended.json,http_c4.json,http_c4_v2.json,quality.json,tests.log,source_manifest.json}.
