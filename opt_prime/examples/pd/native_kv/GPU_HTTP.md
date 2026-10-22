# Native-KV GPU HTTP streaming: equivalent delivery and the batching gap

- **Date**: 2026-10-22
- **Task**: HTTP serving parity (aicomp #144), original native-engine-performance goal active
- **Host / device**: GPU+RNGD host; RNGD npu0, A10080GB PCIe cuda0; same hardware/driver/SDK as GPU_PROJECTION_FUSION.md
- **Commit**: base6ea611649620d44335d2872777208afb31e2b6be + feat/144-gpu-http; measured and final source hashes attached

## Question

Does the projection-fused GPU continuation retain its throughput advantage when both paths deliver tokens over HTTP SSE? What changes under concurrent requests?

## Implementation and setup

Loopback-only completions subset, owned single worker thread for native ZMQ/KV state and GPU graph. FIFO16 pending jobs,32 queued output events per request. Per-token graph replay then CPU readback and incremental text delivery; EOS stops generation, and cancellation is checked before the next replay. Unicode fragments ending in replacement characters are held until stable. Invalid requests are rejected, queued disconnects are canceled, poisoned native clients stop worker admission, and shutdown drains/cancels owned work. This is NOT continuous batching and is NOT registered as a production backend.

Same Qwen3-8B-FP8 revision220b46e3b2180893580a4454f21f22d3ebb187d3 and FXB SHA25636d74b754f580bea893c68cc70147c615d1a9c4273a4580dbebaf87287400713. Native SDK2026.4.0; GPU torch2.10.0+cu128, transformers5.1.0, vLLM0.18.0. Native1RNGD vs GPU path1RNGD+1A100; unequal hardware/precision remains explicit. Fully fused norm and projection options enabled for this experiment. Native prefix caching disabled. Single local native server serves the baseline and the GPU path's external prefill.

## Commands

Native launch is the same local command in GPU_COLOCATED.md. GPU server:

```bash
python3 "$CODE/serve_gpu.py" --model-dir "$MODEL_DIR" --norm-mode fused --fuse-mlp --fuse-qkv
python3 "$CODE/bench_http.py" --repeats 6 --out analysis/native_kv_http/c1.json
python3 "$CODE/bench_http.py" --lengths 512 --concurrency 4 --repeats 3 --out analysis/native_kv_http/c4.json
python3 "$CODE/check_http_quality.py" --model-dir "$MODEL_DIR" --out analysis/native_kv_http/quality.json
python3 -m unittest discover -s "$CODE" -p 'test_*.py' -v
```

Both endpoints receive the identical JSON/SSE client within a run. Requests specify greedy output128, min_tokens128 and ignore_eos=true, using seed139 synthetic IDs. One warmup per mode/length excluded;6 repetitions at c1,3 at c4. Mode ordering alternates; a client barrier releases all concurrent requests together. c1 tests lengths128/512/1024; c4 selects512 alone, so its RNG prompt differs from c1's512 prompt. Do not treat the across-run ratio as a perfectly matched scaling curve. Native SSE does not return usage here; token count128 is inferred from the explicit budget plus finish_reason=length. GPU returns actual usage128. No finish-before-budget case was counted. TTFT is the first nonempty text arrival; throughput includes connection setup and delivery through [DONE]. Native uses its standard HTTP stack, GPU uses Python's HTTP/1.0 ThreadingHTTPServer, and the common client makes a fresh connection for each request.

## Results

Medians of per-burst metrics. TTFT is the mean across requests within a burst, then median across repetitions. Latency distributions in raw JSON are small-sample diagnostics, not an SLO claim. Host contention was not controlled; use the same-run native comparison.

| concurrency | input | endpoint | aggregate tok/s | TTFT ms | TPOT ms |
|---:|---:|---|---:|---:|---:|
| 1 | 128 | native | 101.89 | 21.60 | 9.72 |
| 1 | 128 | gpu | 115.26 | 55.85 | 8.28 |
| 1 | 512 | native | 88.77 | 51.33 | 10.95 |
| 1 | 512 | gpu | 104.83 | 148.48 | 8.44 |
| 1 | 1024 | native | 94.43 | 93.16 | 9.92 |
| 1 | 1024 | gpu | 97.45 | 241.39 | 8.43 |
| 4 | 512 | native | 332.67 | 160.36 | 10.79 |
| 4 | 512 | gpu | 106.95 | 1935.64 | 8.37 |

At c1 the GPU server preserves the internal reference's complete rendered output in all18 measured cases. Both native and GPU finish with length for the fixed-budget performance workload. At c4 the GPU decoder remains fast, but FIFO serialization expands mean TTFT to~1.94s and limits aggregate throughput to~107 tok/s versus native~333. This is a scheduling/batching gap, not a solved serving-performance goal. Peak sampled request duration at c4 was4.81s GPU vs1.56s native.

## Quality and lifecycle

15 questions/retrievals up to1569 input tokens: native nonstream, GPU nonstream and GPU SSE rendered texts match in every case, and all finish with stop. Strict-format checker12/15 for every endpoint (shared formatting deviations); not broad model-quality proof.

Hardware smoke: Korean EOS output matches native, active-stream disconnect leaves the next request healthy. A request disconnected while queued was skipped before native prefill: exactly2 native requests for the active request plus follow-up, not3; follow-up HTTP200. Unit coverage includes partial Unicode, nonmonotonic decode rejection, backpressure cancellation, EOS/cancel avoiding extra graph replay, HTTP stream/nonstream, invalid fields and native-transfer failure stopping admission.29 tests pass.

Performance used the initial source_manifest.json snapshot. The subsequent final snapshot adds queued-disconnect detection, a shutdown race guard, bind-failure cleanup, native ValueError isolation and a stricter benchmark token-count guard. Hardware quality/queued-cancel checks used the queued-disconnect fix; final native-failure handling is covered by an injected-failure worker test. These lifecycle changes do not alter decoder kernels. Full final-source hashes are included separately.

## Conclusion and remaining work

Equivalent per-token HTTP delivery retains a c1 throughput advantage on this workload, with worse TTFT and extra hardware. It does not scale to native at c4. The next required optimization is batched GPU decode/admission (and parallel/native prefill scheduling), followed by consistent per-length prompts across concurrency, real-prompt quality and tail-latency validation. Native FP8/GPU W8A16 differences and earlier synthetic divergence remain. Goal stays active. Owned GPU server stopped cleanly with exit0, native server terminated via its owned process group; no other jobs stopped.

Artifacts: analysis/native_kv_http/{c1,c4,quality,smoke,queued_cancel,summary,source_manifest,source_manifest_final}.json and test logs. No binary/model/KV payloads published.
