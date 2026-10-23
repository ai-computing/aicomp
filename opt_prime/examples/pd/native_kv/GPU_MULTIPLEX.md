# Concurrent native prefill on a request-ID-routed KV receiver

- **Date**: 2026-10-22
- **Task**: aicomp #146 / PR #147
- **Host / device**: GPU+RNGD host; A10080GB PCIe + RNGD
- **Commit**: aac8646 parent plus final hashes in gpu_multiplex_manifest.json

## Question

Can overlapping multiple native prefill requests reduce the remaining heterogeneous
HTTP serving gap, while keeping KV and cancellation isolated by request?

## Setup

Same pinned Qwen3-8B-FP8 checkpoint/FXB and SDKs as GPU_EIGHT.md. Native baseline:
one RNGD. Our path: RNGD prefill plus A100 Marlin W8A16/fused norms/MLP/QKV. Larger
hardware budget and different arithmetic. Eight GPU lanes; adaptive-single disabled.

MultiplexKVClient starts bounded concurrent HTTP calls, each owning its session.
One thread owns ZMQ, routes ready/payload frames by ID to independent validated
assemblers, ACKs only complete transfers and invokes callbacks after HTTP success.
Unexpected IDs, duplicate state and HTTP failures poison the client. Up to16 requests
are supported by the experiment; the measured configuration uses8. Native cancellation
is drained rather than abandoning its frames. Pipeline credits bound the total requests.

CPU packing remains on the prefill owner by default. Optional --async-pack moves it
to a separate worker; measured here but not beneficial. --profile-stages logs elapsed
native-ready/KV/HTTP-complete milestones and CPU-pack/GPU-import durations. Native
milestones are cumulative from batch submission, not separate pure kernel timings.

## Command

Native server as in GPU_HTTP.md. Run from native_kv with MODEL_DIR set:

```bash
python3 check_multiplex.py --out comparison.json
python3 serve_gpu.py --model-dir "$MODEL_DIR" --norm-mode fused --fuse-mlp --fuse-qkv --pipeline --batch-size 8 --native-batch-size 8 --native-batch-wait-ms 8 --profile-stages stages_wait8.jsonl
python3 bench_http.py --lengths 512 --concurrency 8 --repeats 5 --out wait8_c8.json
python3 check_batch_http.py --model-dir "$MODEL_DIR" --concurrency 8 --out wait8_quality.json
```

Initial HTTP runs used2ms admission and3 measured repeats. The separate packing
experiment adds --async-pack to the server and uses async_*.json output names.
Stop GPU HTTP before the worker check because it owns the same receiver socket:

```bash
python3 check_pipeline_backpressure.py --model-dir "$MODEL_DIR" --batch-size 8 --native-batch-size 8 --out backpressure.json
python3 check_pipeline_backpressure.py --model-dir "$MODEL_DIR" --batch-size 8 --native-batch-size 8 --async-pack --out async_backpressure.json
python3 -m unittest discover -s . -p 'test_*.py'
```

## Results

HTTP SSE input512/output128; one warmup excluded, alternating endpoint order:

| GPU configuration | Clients | Repeats | Native tok/s | GPU tok/s | Native TTFT | GPU TTFT |
|---|---:|---:|---:|---:|---:|---:|
| Concurrent prefill,2ms gather |4|3|320.69|306.62|159.23ms|273.82ms|
| Concurrent prefill,2ms gather |8|3|579.86|526.38|259.86ms|371.72ms|
| Concurrent prefill,8ms gather |8|5|567.46|541.07|267.48ms|375.86ms|
| Same + separate packing worker |8|5|576.90|535.35|262.35ms|383.21ms|

Best measured GPU throughput is95% of its same-run native baseline. Native variation,
separate tuning runs and uncontrolled host contention limit conclusions about the8ms
window. Separate packing shows no improvement and remains default off. C1, other
prompt lengths and longer-lived saturation are not covered by these performance runs.

Stage medians for512-token requests in the8ms run, including warmup: native ready
232.46ms, KV complete323.33ms, native HTTP complete334.86ms, CPU packing17.31ms,
GPU import3.50ms. Do not add cumulative milestones together. Receive/parse/other
callback work can contribute to the interval after readiness.

Fifteen question/retrieval cases match native, solo GPU and concurrent8-client text
for all three measured configurations. This is not broad quality or numerical parity.

The initial serial-versus-concurrent KV byte-equality assertion FAILED: only one of
four requests was byte-identical. Follow-up mixed lengths32/128/257/512 with reordered
requests shows serial repeat4/4 exact, first-layer K exact for every request, but later
layers differ. Minimum slot cosine~0.99924; maximum absolute difference reaches1.25.
This is consistent with native batched arithmetic differences, not proof of exact
transfer equivalence. Initial failure is retained in live.log/live.json; comparison.json
records the full diagnostics. Do not replace this failed strict criterion with an
unqualified parity claim.

Unit tests exercise interleaved payloads with different lengths and completion order,
unknown-ID poisoning and HTTP failure before receive. Existing packet/slot tests remain.

Final validation:33 tests pass. Both synchronous and asynchronous packing worker checks keep a32-event consumer stalled while its peer completes128 tokens; resumed text and cancellation followup match in the tested synthetic case. Owned servers stopped.

## Conclusion

Concurrent native prefill moves C8 throughput much closer to native, but first-token
latency and throughput still miss parity. Default native batch remains1; faster modes
are explicit experimental options. The next candidates are native export/receive costs
and GPU per-token fusion, with broader length/concurrency/quality validation. Adaptive
width switching and native batch numerical differences remain documented limitations.

Evidence: analysis/native_kv_multiplex/*.json, *.jsonl and *.log. Best synchronous
packing measurements precede the optional asynchronous-packing refactor; final source
hashes cover both paths and later test CLI changes. The async run measures the final
runtime implementation; final tests check both paths. No raw KV or weights published.
