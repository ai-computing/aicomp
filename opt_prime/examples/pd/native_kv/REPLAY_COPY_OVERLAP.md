# Enqueue KV copies while decode graph is running

- **Date**: 2026-10-23
- **Task**: GPU prefix recomputation and native suffix transfer (aicomp #162)
- **Host / device**: GPU+RNGD host; A10080GB+RNGD, native reference one RNGD
- **Commit**: baseline9ab1a92; candidate source hashes in replay_overlap_manifest.json

## Question

Do separate copy streams actually overlap decode kernels, and does changing submission
order improve real HTTP serving? Prior improvement alone did not prove overlap.

## Setup

Same pinned Qwen3-8B-FP8 revision, FXB, SDK2026.4.0, torch2.10.0+cu128,
driver580.178.04 as GPU_OWNER_UPLOAD.md. Nsight Systems2023.4.4.54, CUDA graph node
tracing, CPU sampling/context-switch tracing disabled.16 decode lanes,8 pending native
requests, direct pinned receive, no GPU prefix recomputation, stage profiling enabled.

Baseline GPU owner drains uploads before graph launch. Candidate launches decode graph,
submits independent copy-stream staging through a callback, then reads output tokens.
No active decode-cache writes in that callback. Empty/all-paused cases still drain queued
uploads outside decode. Final cache import and ownership fences remain.

## Command

Native launch as previous records. From native_kv:

```bash
python3 -m unittest discover -s . -p 'test_*.py'
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --delay=15 --duration=180 --kill=sigterm --force-overwrite=true --output="$OUT/replay-owner-trace" python3 serve_gpu.py --model-dir "$MODEL_DIR" --pipeline --batch-size 16 --native-batch-size 8 --norm-mode fused --fuse-mlp --fuse-qkv --native-pack direct --rolling-native --stream-native-kv --profile-stages "$OUT/replay-owner-trace-stages.jsonl"
# From a second process after /health reports ready; same native remains alive:
python3 bench_arrival_http.py --rates 6 --requests 48 --out "$OUT/replay_owner_trace_http.json"
# Stop only the owned GPU server after benchmark completes; let nsys finish exporting.
nsys export --type=sqlite --force-overwrite=true --output="$OUT/replay-owner-trace.sqlite" "$OUT/replay-owner-trace.nsys-rep"
python3 summarize_cuda_overlap.py "$OUT/replay-owner-trace.sqlite" --out "$OUT/replay_owner_cuda_overlap.json"
# Restart candidate without nsys, same serving arguments:
python3 bench_arrival_http.py --rates 6 --requests 96 --out "$OUT/replay_owner6.json"
python3 bench_arrival_http.py --rates 12 --requests 384 --out "$OUT/replay_owner12.json"
python3 check_long_http.py --model-dir "$MODEL_DIR" --out "$OUT/replay_owner_quality.json"
```

Baseline trace uses same arguments,180 replaced by60seconds and baseline9ab1a92.
Successful candidate run uses a local harness: write target PID, wait for HTTP readiness,
run benchmark immediately, SIGTERM that owned PID, await profiler completion. Two failed
attempts retained: a60-second capture expired before GPU warmup request (connection refused),
and the first harness wrapper lacked the working directory in sys.path (import failed).
Neither failed attempt is used for performance/overlap claims.

## Results

74 tests pass8.483s. New graph tests first failed on the missing during_replay argument;
real CUDA graph tests verify callback occurs after launch and before token read, preserve
paused-lane state and leave inactive decode unchanged. SQLite interval tests first failed
on missing analyzer; overlapping kernels are merged, and other devices/processes/contexts
and small/input or D2D copies are excluded.

Each successful trace contains882 KV H2D copies (49 requests including warmup),
3,937,517,568bytes. Sizes are four slots of BF16[127/511/1023,8,128]:
1,040,384 /4,186,112 /8,380,416bytes. Compare copy GPU intervals against the union of kernel
GPU intervals in the same process/device/context. Fractions are weighted by individual
copy durations, not request latency, GPU utilization or end-to-end speedup.

|Trace|KV copy GPU time ms|Time overlapping kernels ms|Overlap fraction|Copies with overlap|
|---|---:|---:|---:|---:|
|before decode launch|160.803|0.887|0.55%|4/882|
|during decode replay|161.721|153.594|94.98%|762/882|

Candidate by input128/512/1024:61.51%/97.59%/98.97% overlap. Copy time itself remains
about the same, average3.3ms/request. This confirms real compute/copy concurrency, not
an improvement in PCIe bandwidth. Native NPU DMA is not present in this CUDA trace.
Profiler HTTP48/48 success both runs; those latencies are not unprofiled comparisons.

Unprofiled candidate:

|Load|Native TTFTp95 ms|Candidate TTFTp95 ms|Native/candidate tok/s|Success|
|---|---:|---:|---:|---|
|6/s96|162.25|251.29|441.84/439.08|96/96 both|
|12/s384|272.05|263.53|879.94/913.23|384/384 both|

Long retrieval9/9 native/solo/concurrent, all9texts equal.16 disconnected streams then16/16 recovery requests returned200 and8tokens, health200. All owned GPU/native/profiler processes stopped. Previous pre-launch streaming
6/s229-252ms and12/s255ms are contextual references, not contemporaneous unprofiled
A/B controls. No demonstrated end-to-end improvement from this scheduling change.

## Conclusion

Real copy/kernel overlap rises from0.55% to94.98%, but H2D only accounts for about3.3ms
per request on average. It cannot explain the much larger native-ready to KV-ready gap
(about94ms at1024 in the earlier matched stage measurements). Keep experimental/off by
default; native performance remains unmet. Next investigate native export block/DMA
processing and first prove any proposed larger transfer unit preserves complete KV.
Do not infer exact GIL causes, export internals, cost equivalence, or broad model quality.
