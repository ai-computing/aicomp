# Local prefill: paired native comparison and wave admission

- Date: 2026-10-22
- Task: aicomp #148, PR #149
- Commit: 6ac476d parent plus gpu_local_paired_manifest.json
- Hardware: one A10080GB PCIe versus one RNGD; GPU+RNGD host.
- Model and software: pinned Qwen3-8B-FP8 as GPU_LOCAL_PREFILL.md;
  native furiosa-llm2026.4.0, FXB SHA36d74b754f580bea893c68cc70147c615d1a9c4273a4580dbebaf87287400713.

## Question and change

Can local prefill approach native serving when measured in the same run?
Initial GPU width8/gather2ms has variable burst throughput. Added opt-in
`--local-single-width`: select a separate width1 cache/graph for a one-request
wave, with no mid-request graph switch. Gather8ms and wave tracing verify full
8-request admission in every measured tuned C8 wave. Defaults unchanged.
This is a GPU-local routing candidate, not improvement of RNGD-to-GPU transport.

## Commands

Native setup is the existing GPU_COLOCATED.md endpoint, prefix cache disabled.
From native_kv with MODEL_DIR set:

```bash
python3 serve_gpu.py --prefill-backend gpu --model-dir "$MODEL_DIR" --batch-size 8 --norm-mode fused --fuse-mlp --fuse-qkv
python3 bench_http.py --lengths 128,512,1024 --concurrency 1 --repeats 3 --out paired_c1.json
python3 bench_http.py --lengths 128,512,1024 --concurrency 8 --repeats 3 --out paired_c8.json
python3 check_batch_http.py --model-dir "$MODEL_DIR" --concurrency 8 --out paired_quality.json
# Stop the GPU server, then restart with:
python3 serve_gpu.py --prefill-backend gpu --model-dir "$MODEL_DIR" --batch-size 8 --local-single-width --native-batch-wait-ms 8 --norm-mode fused --fuse-mlp --fuse-qkv --profile-stages waves8.jsonl
# Repeat the same bench/quality commands, using tuned_ output names.
# After stopping both servers:
python3 check_pipeline_backpressure.py --prefill-backend gpu --local-single-width --model-dir "$MODEL_DIR" --batch-size 8 --out backpressure_v2.json
python3 -m unittest discover -s . -p 'test_*.py'
```

## Results

HTTP/SSE, output128; one warmup, three measured repeats; alternating endpoint order.
Medians of aggregate throughput and mean per-request TTFT. Native baseline varies
between tuning runs; do not attribute all differences to code. Hardware/arithmetic
are different. Tuned results:

| Clients | Input | Native tok/s | GPU tok/s | Native TTFT ms | GPU TTFT ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 103.17 | 116.46 | 21.22 | 32.04 |
| 1 | 512 | 91.92 | 112.16 | 51.98 | 56.54 |
| 1 | 1024 | 91.40 | 107.31 | 92.32 | 99.95 |
| 8 | 128 | 706.08 | 703.47 | 101.78 | 103.88 |
| 8 | 512 | 577.43 | 596.31 | 259.71 | 328.14 |
| 8 | 1024 | 447.67 | 495.66 | 498.85 | 632.18 |

Initial C1GPU throughput96.95/94.30/90.59tok/s for128/512/1024; tuned
116.46/112.16/107.31. Initial C8GPU medians372.71/594.62/296.58 with large
variation; tuned703.47/596.31/495.66 with full8-request waves. Source JSON preserves
ranges and baseline measurements. No steady-state Poisson or saturation measurement.
GPU process memory snapshots: fixed-width8 peak observed30852MiB; tuned snapshot
30760MiB. These are nvidia-smi process snapshots, not allocator peaks or cost parity.

Both initial/tuned15 question/retrieval cases match native, solo and concurrent
texts, with15 EOS stop responses. This does not erase known numerical mismatches.
Initial local backpressure test: stalled32-event consumer resumes with same text as
peer, but cancellation-followup text differs from prior batched peer. Failure log
preserved. The diagnostic was extended with an uncancelled single-request baseline
to distinguish cancellation corruption from width-dependent arithmetic; its original
cross-width equality result is still reported, not reclassified as parity.

## Conclusion

Throughput is near or above native in this measured matrix, but long-input C8 TTFT
is26-27% slower and C1 adds admission delay. The native-performance goal is not
complete. Retain experimental flags; remaining work includes first-token latency,
quality/numerical scope, steady load, and heterogeneous transport improvements.

Final validation: 37 regression tests pass. Extended local worker check passes
32-event stall/resume and peer128-token completion; cancellation followup128 tokens
matches the uncancelled single-request baseline. Single-versus-batched synthetic
text is NOT equal. This is a remaining numerical limitation, not a cancellation
parity claim across graph widths. All owned servers and checkers exited; GPU idle.
