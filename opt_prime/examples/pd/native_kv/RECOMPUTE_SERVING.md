# Chunked GPU prefix recomputation in bounded HTTP serving

- **Date**: 2026-10-23
- **Task**: GPU prefix recomputation and native suffix transfer (aicomp #162)
- **Host / device**: GPU+RNGD host; A10080GB + RNGD, native reference one RNGD
- **Commit**: 39d01d2 plus recompute_serving_manifest.json source hashes

## Question

Can overlapping GPU prefix work and native suffix transfer reduce low-load HTTP TTFT
without stalling active decode or losing correctness under concurrent admission?

## Setup

Same Qwen3-8B-FP8 revision, FXB and SDK2026.4.0 as prefix-overlap experiment.
Fixed16-lane Marlin decode,8 pending native requests,75% GPU prefix,256-token chunks.
Each request owns a lifetime credit until both producers finish and decode retires.
Cancellation during GPU work fences queued CUDA work before releasing GPU ownership;
native completion/failure must also arrive before the credit can return. Native socket
remains on its owner thread. Per-request source offsets are validated and snapshotted.
Default full-transfer serving is unchanged. Candidate stage profiling enabled.

## Command

Native launch as in prefix-overlap record; run from native_kv:

```bash
python3 -m unittest discover -s . -p 'test_*.py'
python3 serve_gpu.py --model-dir "$MODEL_DIR" --pipeline --batch-size 16 --native-batch-size 8 --norm-mode fused --fuse-mlp --fuse-qkv --native-pack direct --rolling-native --recompute-prefix 0.75 --recompute-chunk-tokens 256 --profile-stages "$OUT/recompute-serving-stages.jsonl"
python3 bench_arrival_http.py --rates 6 --requests 96 --out "$OUT/recompute_serving6.json"
python3 check_long_http.py --model-dir "$MODEL_DIR" --out "$OUT/recompute_serving_quality.json"
python3 bench_arrival_http.py --rates 12 --requests 384 --out "$OUT/recompute_serving12.json"
```

Corrected read1 SSE client; deterministic mixed128/512/1024 input and32/128 output.
Native then candidate, excluded warmup. No same-hardware/cost/capacity claim.

## Results

Initial61-test suite passed6.249s; then added cancellation/native-failure integration
coverage,4 rolling tests passed. First attempt to create the join test used an incorrect
relative path and failed import; corrected absolute path and ran tests successfully.

6/s96: native96/96,TTFTp95162.99ms,441.77tok/s; candidate96/96,180.00ms,439.96tok/s.
Long retrieval9/9 exact in native, solo and concurrent candidate; all9texts equal.
12/s384: native384/384,TTFTp95199.98ms,884.30tok/s; candidate301/384,
TTFTp952069.61ms,669.40tok/s,83 errors. Retain this regression; do not enable by default.
Cancellation smoke:16 streaming responses closed immediately after headers; all16
subsequent sequential recovery requests returned200 and8tokens, health200. This does
not prove every possible cancellation point or slow-reader/backpressure behavior.
Final full suite62 tests passed8.450s. Same-session full-transfer6/s:96/96,TTFTp95248.88ms,439.23tok/s;
its native reference157.55ms,442.09tok/s. Candidate TTFTp95 improves27.7%,
but total latency p95 worsens1684.80->2248.51ms (33.5%).
All83 high-load errors are HTTP503. All owned servers stopped.

## Conclusion

Actual HTTP candidate is closer to native at low load than previous rolling measurements,
but historical256ms is not a matched baseline. Native parity remains unproven. Chunked
GPU prefill consumes decode capacity; high-load regression is confirmed; next implement a measured load-aware recomputation policy.

## Root-cause investigation (Ponytail + Superpowers)

Applied ponytail09c434ab and systematic-debugging/verification-before-completion
from superpowers2ce8aa15. These prescribe smaller complete changes and evidence-first
root-cause analysis; they are not GPU profilers or automatic performance optimizers.

Stage median ms (successful completed requests; excluded warmup for12/s):

|Path|Admission wait|Native KV|Prepared wait|GPU import|First GPU token|
|---|---:|---:|---:|---:|---:|
|candidate6/s|0.17|68.84|11.32|1.60|96.69|
|candidate12/s|1693.32|71.89|11.78|1.68|1813.79|
|full6/s|0.23|103.11|11.04|3.50|134.25|

These show high-load delay accumulating before admission, not primarily inside native
KV delivery. Code runs one GPU prefix chunk before each decode iteration; GPU prefix
wall time includes interleaved decode and is NOT pure GPU kernel time. The leading
hypothesis is prefix work consuming decode service capacity, lengthening lifetime-credit
occupancy until the bounded input queue rejects requests. Confirm with decode-step/chunk
CUDA timing and a matched12/s full-transfer run before choosing a load threshold.
Do not simply enlarge queues: that hides rejection behind longer latency.

Next: preserve working full-transfer fallback, measure GPU duty and completed-request
occupancy, then test one recomputation admission policy against6/s and12/s. No native
parity, energy equivalence, or general quality claim. Slow-reader behavior remains untested.
