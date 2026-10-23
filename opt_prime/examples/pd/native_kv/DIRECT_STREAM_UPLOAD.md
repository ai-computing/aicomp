# Direct pinned KV receive with early CUDA copy submission

- **Date**: 2026-10-23
- **Task**: GPU prefix recomputation and native suffix transfer (aicomp #162)
- **Host / device**: GPU+RNGD host, A10080GB+RNGD; native reference one RNGD
- **Commit**: 281710d plus direct_stream_initial_manifest.json / direct_stream_manifest.json hashes

## Question

Can validated K/V slot groups be uploaded before the rest of native KV arrives,
without the CPU layout conversion in the earlier single-request streaming prototype?

## Setup

Same pinned Qwen3-8B-FP8 revision, FXB and SDK2026.4.0 as RECOMPUTE_SERVING.md.
Fixed16 decode lanes,8 pending native requests, direct receive. No GPU recomputation.
Receive callback submits each consecutive slot group to a per-request CUDA copy stream
and isolated device staging storage; an event gates final import into the graph cache.
Pinned source remains owned through completion/cancellation. Decode cache is never
written by the receiver thread. Last import retains a device-to-device copy and fence.

## Command

Native launch as previous paired records; run from native_kv:

```bash
python3 -m unittest discover -s . -p 'test_*.py'
python3 serve_gpu.py --model-dir "$MODEL_DIR" --pipeline --batch-size 16 --native-batch-size 8 --norm-mode fused --fuse-mlp --fuse-qkv --native-pack direct --rolling-native --stream-native-kv --profile-stages "$OUT/direct-stream-stages.jsonl"
python3 bench_arrival_http.py --rates 6 --requests 96 --out "$OUT/direct_stream6.json"
python3 check_long_http.py --model-dir "$MODEL_DIR" --out "$OUT/direct_stream_quality.json"
# Restart same candidate with allocation/submission wall-time counters, then:
python3 bench_arrival_http.py --rates 6 --requests 24 --out "$OUT/direct_stream_profile6.json"
```

## Results

67 tests pass6.170s. New tests initially failed on missing early callback, upload class
and multiplex callback API. The first payload equality assertion failed because fixture
BF16 bit patterns include NaNs; corrected comparison to uint8 byte equality. Tests
verify first chunk reaches GPU before remaining KV is supplied, complete byte equality,
partial-copy cancellation fencing, and callback-before-completion routing.

6/s96 native96/96,TTFTp95164.53ms,441.74tok/s; streaming80/96,16 errors,
TTFTp956453.26ms,290.46tok/s. This is a substantial regression, not an optimization.
Long retrieval9/9 native/solo/concurrent, all9texts equal.
Successful initial requests including warmup: median admission2836ms, native ready906ms,
native KV2090ms, final import3.37ms. GPU buffer allocation and copy-submission wall-time
instrumentation added to separate receiver blockage from asynchronous copy duration.

## Conclusion

Early H2D submission is implemented and tested, but this receiver-thread implementation
hurts end-to-end latency. Keep experimental/off by default. Do not treat smaller final
import or use of non_blocking as proof of performance benefit or measured overlap ratio.
Next isolate allocation, per-group submit and receiver polling costs before changing
thread ownership. Native performance goal remains unmet.

## Follow-up instrumentation

24-request6/s native24/24,TTFTp95148.68ms; streaming24/24,TTFTp953244.24ms.
Across24 completed requests (excluding warmup), median GPU buffer/stream allocation
10.15ms, sum of18 copy submissions241.98ms, slowest submission21.48ms. Median native
KV1829.42ms includes receiver callbacks and pending-request interference. One1024-token
request spends250.97ms in copy-submission calls. These are CPU wall intervals, not CUDA
DMA durations. Concurrent Python/GPU submission contention is a hypothesis, not proven
GIL attribution. The verified issue is lengthy GPU API work inside the receive loop.

Next controlled change: receive owner only publishes validated ranges into a bounded
queue; existing GPU owner submits copies on its separate copy stream. Retain per-request
storage/events and wait for receive plus copy completion before decode or credit return.
Repeat identical6/s96, then high-load if regression is removed. All owned servers stopped.
