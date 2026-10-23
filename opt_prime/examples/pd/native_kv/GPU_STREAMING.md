# Streaming KV upload: correctness preserved, no stable latency gain

- **Date**: 2026-10-22
- **Task**: Native-engine performance parity (aicomp #141 / Draft PR #142)
- **Host / device**: GPU+RNGD host, same RNGD npu0 and A10080GB PCIe cuda0 as colocated record; GPU0 NUMA affinity2, CPUs6–9
- **Commit**: 4016813fb73cd0296802755971bb7decb3f44705 + streaming experiment; source hashes attached

## Question

Can uploading validated native KV batches while later batches arrive reduce first-token latency versus uploading all KV after reception?

## Setup

Same pinned Qwen3-8B-FP8 checkpoint, FXB, local2026.4 native server, original GPU environment and single-flight topology as experiments/2026-10-22-colocated-native-gpu.md. Torch-stats norm, CPU copy threads1. No native/weight/driver changes. External compilers consumed substantial CPU (three observed processes roughly731%,565%,234% CPU); none were modified or stopped. Local load observed25.22/28.84/26.97 on24 logical CPUs. These are contended experiments, not controlled production acceptance.

The optional callback runs only after each batch passes protocol validation. Its CPU copy retains bytes in owned pinned storage; async GPU upload and cache transpose run while subsequent batches arrive. Decoder admission requires all72 slots and a GPU completion fence. `--compare-imports` rotates native, packed and streaming ordering across repetitions. Native uses1RNGD; both GPU paths add1A100. Counts: first run36 requests, six-repeat run63, affinity run36, total135 completed requests.

## Commands

```bash
python3 -m unittest discover -s "$CODE" -p 'test_*.py' -v
python3 "$CODE/bench_live_gpu.py" --model-dir "$MODEL_DIR" --compare-imports --fused-norms --norm-mode torch-stats --cpu-threads 1 --transport local-loopback --out analysis/native_kv_streaming/paired.json
python3 "$CODE/bench_live_gpu.py" --model-dir "$MODEL_DIR" --compare-imports --fused-norms --norm-mode torch-stats --cpu-threads 1 --repeats 6 --transport local-loopback --out analysis/native_kv_streaming/repeat6.json
taskset -c 6-9 python3 "$CODE/bench_live_gpu.py" --model-dir "$MODEL_DIR" --compare-imports --fused-norms --norm-mode torch-stats --cpu-threads 1 --transport local-loopback --out analysis/native_kv_streaming/affinity.json
```

Seed139 synthetic IDs; input128/512/1024, output128, concurrency1. Warmup1 per mode/input excluded. First/affinity runs3 repeats, middle run6. Server command and setup from colocated record. `receive_ms` includes stream preparation/callback time; `import_ms` is only the work after reception. A smaller final import time alone is NOT an end-to-end gain.

## Results

Median values. Source JSON includes all paired differences, not just favorable cases.

| run | input | mode | TTFT ms | tok/s | final import ms |
|---|---:|---|---:|---:|---:|
| paired | 128 | native | 50.61 | 74.32 | — |
| paired | 128 | packed | 92.72 | 86.56 | 9.14 |
| paired | 128 | streaming | 117.53 | 85.26 | 0.19 |
| paired | 512 | native | 64.21 | 81.89 | — |
| paired | 512 | packed | 163.07 | 82.05 | 28.41 |
| paired | 512 | streaming | 158.30 | 82.33 | 0.18 |
| paired | 1024 | native | 113.31 | 80.62 | — |
| paired | 1024 | packed | 338.67 | 73.58 | 43.78 |
| paired | 1024 | streaming | 254.05 | 77.34 | 0.19 |
| repeat6 | 128 | native | 27.43 | 89.38 | — |
| repeat6 | 128 | packed | 64.07 | 88.39 | 5.81 |
| repeat6 | 128 | streaming | 81.54 | 87.36 | 0.22 |
| repeat6 | 512 | native | 55.52 | 85.55 | — |
| repeat6 | 512 | packed | 155.47 | 82.46 | 16.76 |
| repeat6 | 512 | streaming | 163.01 | 82.08 | 0.23 |
| repeat6 | 1024 | native | 100.33 | 87.91 | — |
| repeat6 | 1024 | packed | 275.38 | 76.36 | 31.77 |
| repeat6 | 1024 | streaming | 250.57 | 77.50 | 0.25 |
| affinity | 128 | native | 28.80 | 93.30 | — |
| affinity | 128 | packed | 70.45 | 88.01 | 7.96 |
| affinity | 128 | streaming | 107.41 | 85.84 | 0.19 |
| affinity | 512 | native | 57.22 | 88.37 | — |
| affinity | 512 | packed | 135.09 | 83.56 | 16.14 |
| affinity | 512 | streaming | 181.20 | 81.14 | 0.20 |
| affinity | 1024 | native | 93.05 | 83.65 | — |
| affinity | 1024 | packed | 293.93 | 75.53 | 31.02 |
| affinity | 1024 | streaming | 329.66 | 73.99 | 0.18 |

Correctness: all paired streaming outputs match packed outputs exactly for every prompt and repetition.21 unit tests pass, including validated-only callback dispatch, duplicate rejection, contiguous/reordered batches, dropped/modified input ownership, incomplete-prefix admission, and bitwise GPU cache comparison across different prefix lengths. This preserves our path; native FP8 quality remains unproven.

Six-repeat1024 TTFT medians275.38 packed vs250.57 streaming; paired changes include regressions. At128/512 streaming is slower. Affinity run1024 is293.93 packed vs329.66 streaming. Final import drops below1ms but callback staging adds CPU cost during receive. Restricting CPU affinity is not demonstrated to help under this contention.

## Conclusion

Do not enable streaming upload by default or claim a stable optimization. Keep the explicit experimental option and the negative results. More Python-copy tuning alone is unlikely to close the full-serving gap; next investigate decode kernel fusion/batching and native export cost, with real-prompt quality and controlled-load validation. No goal-completion claim. All owned hardware processes stopped after the runs.

Raw logs/results: analysis/native_kv_streaming/{paired,repeat6,affinity}.json/.log; summaries and source hashes alongside them. Unit test log retained locally.
