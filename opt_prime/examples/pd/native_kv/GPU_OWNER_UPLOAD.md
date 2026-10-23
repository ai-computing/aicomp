# Submit native KV copies on the GPU owner instead of the receiver

- **Date**: 2026-10-23
- **Task**: GPU prefix recomputation and native suffix transfer (aicomp #162)
- **Host / device**: GPU+RNGD host, A10080GB+RNGD; native reference one RNGD
- **Commit**: cb346be plus owner_stream_manifest.json hashes

## Question

Does removing CUDA submission from the receive loop eliminate the measured242ms
per-request receiver blocking while retaining early asynchronous KV uploads?

## Setup

Same model revision, FXB, SDK2026.4.0 and fixed16-lane decoder as DIRECT_STREAM_UPLOAD.md.
Native pending8, direct pinned receive, no GPU prefix recomputation, profiling enabled.
Receiver publishes validated ranges into a queue bounded by72 slots per lifetime credit.
Existing GPU owner allocates staging and enqueues copy-stream operations before decode
steps. Completion waits for receive and copy; cancellation/failure retains credit until
GPU owner drains queued chunks and fences any partial copy. Native producer joins before
shutdown drops queued source buffers. No changes to native engine or queue admission limits.

## Command

Start native as previous records. From native_kv:

```bash
python3 -m unittest discover -s . -p 'test_*.py'
python3 serve_gpu.py --model-dir "$MODEL_DIR" --pipeline --batch-size 16 --native-batch-size 8 --norm-mode fused --fuse-mlp --fuse-qkv --native-pack direct --rolling-native --stream-native-kv --profile-stages "$OUT/owner-stream-stages.jsonl"
python3 bench_arrival_http.py --rates 6 --requests 96 --out "$OUT/owner_stream6.json"
python3 bench_arrival_http.py --rates 12 --requests 384 --out "$OUT/owner_stream12.json"
python3 check_long_http.py --model-dir "$MODEL_DIR" --out "$OUT/owner_stream_quality.json"
# Restart without --stream-native-kv, keeping native alive; run same rates/counts
# to owner_baseline6.json and owner_baseline12.json. Restart with streaming again:
python3 bench_arrival_http.py --rates 6 --requests 96 --out "$OUT/owner_repeat6.json"
```

## Results

70 tests pass6.242s. New tests first failed on missing UploadQueue and stream_uploads
handoff argument. Tests include a real separate publication thread, pinned H2D byte/value
checks, partial cancellation/source reuse, bounded queue, and retained credits on native
failure/cancellation until the GPU owner consumes them. Existing full-transfer tests pass.

|Load|Native TTFTp95 ms|GPU-owner TTFTp95 ms|Native/GPU tok/s|Success|
|---|---:|---:|---:|---|
|6/s96|161.61|229.10|442.03/440.01|96/96 both|
|12/s384|241.97|255.08|881.19/913.39|384/384 both|

GPU-owner6/s median submission wall1.099ms/request (previous receiver-owner instrumented
run241.98ms), allocation0.374ms, native KV101.93ms, final import0.921ms.12/s median
submission1.27ms, admission0.353ms, native KV105.84ms, final import1.031ms. The earlier
6/s receiver-owner run failed16/96 with TTFTp956453ms; this regression is eliminated.

Long retrieval9/9 exact native/solo/concurrent, all9texts equal.16 streaming requests
closed after headers followed by16/16 recovery requests returning200 and8tokens; health200.
This does not cover every disconnect point or slow-reader behavior. Same-session full-transfer baseline6/s96:TTFTp95258.78ms,totalp951703.24ms,
439.49tok/s,96/96. Stream candidate229.10ms,totalp951665.07ms,440.01tok/s.
12/s384 full-transfer262.33ms,totalp951842.24ms,913.42tok/s,384/384;
stream255.08ms,totalp951815.29ms,913.39tok/s. Native references remain approximately
161ms at6/s but vary209-242ms at12/s. Throughput is essentially unchanged by streaming.
Stream/full/stream6/s sequence:229.10 ->258.78 ->251.76ms TTFTp95;
repeat native159.01ms, GPU438.97tok/s,96/96. Thus improvement spans2.7-11.5%,
not a stable11.5% gain. All owned GPU and native processes stopped.

## Conclusion

GPU-owner submission removes the severe receiver-thread regression. Candidate6/s TTFTp95
is11.5% below same-session full transfer, with no failures.12/s TTFTp95 is2.8% lower and
throughput unchanged. This does not establish native parity: low-load native is162ms.
There is no same-hardware/cost claim (native one RNGD vs RNGD+A100).
Even correct asynchronous submission does not prove DMA/compute overlap or a specific
hidden-copy fraction. The final device-to-device import and its fence remain.

## Stage boundaries and remaining uncertainty

At6/s1024 input, median native-ready87.24ms and KV-ready181.41ms in streaming;
full-transfer89.50/200.75ms. Median final GPU import1.20ms vs6.80ms. Prepared wait0.27ms
vs11.06ms also changes. The gain therefore cannot all be attributed to hidden H2D:
receiver/export timing and scheduling also changed. Next use CUDA events or a short
Nsight trace (nsys available) to verify when copy actually overlaps receive/decode, and
measure ready-to-KV export interval, which remains much larger than final GPU import.
A20ms prepared-queue idle wait can delay GPU consumption of newly published chunks;
measure queue residence before changing that scheduling policy. Per-request copy streams
and staging add GPU memory; peak extra allocation has not yet been measured.
