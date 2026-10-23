# Colocated native RNGD prefill and A100 decode

- **Date**: 2026-10-22
- **Task**: Native-engine performance parity (aicomp #141 / PR #142)
- **Host / device**: GPU+RNGD host, Xeon Silver4410Y (24 logical CPUs), RNGD npu0, A10080GB PCIe cuda0, NVIDIA driver580.178.04; RNGD firmware2026.4.0 93d67f3
- **Commit**: a1c8a37e138fa2fc6a8643d9d3908229edf0b986

## Question

Does removing the measured1Gb/s inter-host transport eliminate the full-serving latency gap?

## Setup

Separate server venv using system-site-packages, furiosa-llm/native/common2026.4.0, transformers5.19.0. GPU client remains in its original torch2.10.0+cu128/transformers5.1.0 environment. Same pinned checkpoint220b46e3b2180893580a4454f21f22d3ebb187d3, same FXB SHA25636d74b754f580bea893c68cc70147c615d1a9c4273a4580dbebaf87287400713. Original global SDK2026.3 unchanged. Native compares1RNGD with heterogeneous1RNGD+1A100, explicitly unequal hardware budget. Local loopback TCP; no SSH tunnel.

Initial pip22 dependency resolution failed with internal AssertionError; isolated pip upgraded26.2.2 and install succeeded. First server startup failed importing inherited torchaudio2.7 against torch2.10. Installed torchaudio2.10 and pinned transformers5.19 only in the server venv. Next startup initialized native device/model successfully. Server startup performs8424 profiling samples before accepting HTTP.

## Commands

```bash
"$SERVER_VENV/bin/furiosa-llm" serve "$MODEL_DIR" --served-model-name Qwen/Qwen3-8B-FP8 --fxb "$FXB" --devices npu:0 --host 127.0.0.1 --port 19101 --no-enable-prefix-caching --experimental-kv-transfer-config '{"kv_connector":"TcpConnector","host":"127.0.0.1","port":19201,"poll_timeout_ms":1}'
```

## Results

See completed measurements below. Startup failure log retained at analysis/native_kv_colocated/startup_torchaudio_failure.log.

## Completed measurements

Native and heterogeneous alternating requests, seed139 synthetic IDs, lengths128/512/1024, output128, concurrency1, one warmup then three repeats. Both modes use the same local native server; heterogeneous adds an A100. Server completed startup; all72 requests across three benchmark runs completed. Load was not controlled; an observed local load average was31.82/26.57/26.42 on24 logical CPUs. Thread-count runs are sequential, not a paired causal comparison.

```bash
python3 "$CODE/bench_live_gpu.py" --model-dir "$MODEL_DIR" --packed-import --transport local-loopback --out analysis/native_kv_colocated/packed.json
python3 "$CODE/bench_live_gpu.py" --model-dir "$MODEL_DIR" --packed-import --fused-norms --norm-mode torch-stats --transport local-loopback --out analysis/native_kv_colocated/torch_stats.json
python3 "$CODE/bench_live_gpu.py" --model-dir "$MODEL_DIR" --packed-import --fused-norms --norm-mode torch-stats --cpu-threads 1 --transport local-loopback --out analysis/native_kv_colocated/torch_stats_t1.json
```

The last command uses a1c8a37 plus the added --cpu-threads benchmark argument (source hashes attached). Each run contains24 requests (3 lengths x4 repetitions x2 modes), hence72 completed requests total. No request concurrency is implied by that count.

| run | input | mode | TTFT ms | TPOT ms | tok/s | receive ms | import ms |
|---|---:|---|---:|---:|---:|---:|---:|
| packed | 128 | native | 26.91 | 10.84 | 91.22 | — | — |
| packed | 128 | heterogeneous | 87.32 | 11.47 | 82.90 | 62.80 | 7.31 |
| packed | 512 | native | 64.69 | 11.30 | 85.84 | — | — |
| packed | 512 | heterogeneous | 246.44 | 11.57 | 74.60 | 124.78 | 26.36 |
| packed | 1024 | native | 98.34 | 10.80 | 87.08 | — | — |
| packed | 1024 | heterogeneous | 351.40 | 11.60 | 70.13 | 308.46 | 88.65 |
| torch_stats | 128 | native | 22.30 | 10.32 | 96.04 | — | — |
| torch_stats | 128 | heterogeneous | 75.94 | 10.89 | 87.70 | 58.07 | 4.95 |
| torch_stats | 512 | native | 53.00 | 10.23 | 94.62 | — | — |
| torch_stats | 512 | heterogeneous | 160.72 | 10.99 | 82.18 | 136.57 | 12.42 |
| torch_stats | 1024 | native | 94.89 | 10.74 | 87.88 | — | — |
| torch_stats | 1024 | heterogeneous | 305.98 | 11.03 | 74.99 | 232.18 | 62.71 |
| torch_stats_t1 | 128 | native | 24.46 | 10.45 | 94.71 | — | — |
| torch_stats_t1 | 128 | heterogeneous | 72.80 | 10.89 | 87.88 | 55.28 | 6.61 |
| torch_stats_t1 | 512 | native | 53.83 | 10.69 | 90.58 | — | — |
| torch_stats_t1 | 512 | heterogeneous | 151.97 | 10.99 | 82.66 | 119.82 | 20.24 |
| torch_stats_t1 | 1024 | native | 98.34 | 10.62 | 88.46 | — | — |
| torch_stats_t1 | 1024 | heterogeneous | 307.63 | 11.03 | 74.92 | 260.76 | 35.30 |

Correctness: torch-stats norm and the1-thread copy setting preserve every tested live generated token relative to unfused packed import, across all lengths and repetitions. Native rendered-text equality remains limited (see summary.json), so this is preservation of our path, not native FP8 parity. TPOT is near native, but TTFT/total throughput still lag and resources differ. No native-speed completion claim.

Same-host receiving removes the1Gb/s link:1024-token receive is now hundreds of milliseconds rather than~1.5s. Export/receive plus host staging remain the bottleneck. Owned native server stopped via its sentinel; global environment unchanged. Raw records and summaries are under analysis/native_kv_colocated/.

Public summaries: gpu_colocated_summary.json and gpu_colocated_manifest.json.
