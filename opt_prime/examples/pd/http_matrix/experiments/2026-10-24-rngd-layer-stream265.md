# RNGD layer-progress import toward within-request KV overlap

- **Date**: 2026-10-24
- **Task**: Layer-progress RNGD transfer (aicomp #265, original #47)
- **Host / device**: local unit/regression tests; no new streaming hardware experiment yet
- **Commit**: isolated perf/265-rngd-stream, base098169b plus dense-hybrid fix cherry-picked as8903b73; uncommitted implementation

## Question

Can completed RNGD layer groups be imported before final prefill/logits without publishing incomplete state or corrupting another request? GPU chunk-copy#254 and ATOM next-request prefetch#255 do not implement this source-to-decoder connection.

## Setup

The public research FullPrefillEngine.prefill_stream has per-group host output, but the current subprocess worker sends only whole-request completion. Work is isolated from the live #263 queue. First unit adds a thread-safe coverage/notification validator and partial dense model-layer import into reserved PagedKV pages. It does not yet alter worker messages, GPU/ATOM admission, or source adapters. No overlap or speedup claim.

Coverage validates integer geometry, source span bounds, overlap across every affected layer before any state mutation, complete layer/token coverage on finish, and no messages after closure. Draining notifications is separate from source completion; neither is a destination transfer fence. Cache import validates every affected hybrid group before writing and leaves lengths/cache publication unchanged. It rejects unreserved pages and overwrites of an admitted/shared prefix.

## Command

```bash
cd /home/gspark/workspace/aicomp-rngd-stream-265/opt_prime
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/gspark/venvs/pt213/bin/python -m pytest tests/test_pd_layer_progress.py tests/test_pd_dense_hybrid.py -q
# Full suite with the same environment:
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/gspark/venvs/pt213/bin/python -m pytest -q
```

## Results

- Coverage tests first11failed for missing implementation, then11passed.
- Partial receiver tests6failed before implementation (controlled removal in correct package working directory:6failed/11passed), then combined21passed/1existing flex-attention warning in5.18s.
- Additional admitted-prefix regression:1failed/17passed before guard; final focused22passed/1existing warning in4.93s.
- One initial focused invocation from the repository parent failed collection (`opt_prime.accel` unavailable with backend autoload disabled). Correct package-directory command above resolves it; failure is not an inference result.
- Initial full regression before the last admitted-prefix guard:555passed/15skipped/253subtests in160.70s. Final full run after guard:556passed/15skipped/253subtests in158.09s.13existing pinned-memory deprecation/flex-attention warnings retained.

Raw logs: /tmp/stream265-red.log, /tmp/stream265-import-red-controlled.log, /tmp/stream265-prefix-red.log, /tmp/stream265-focused-final.log, /tmp/stream265-full.log and /tmp/stream265-full-final.log.

## Conclusion

Only validation/import primitives are implemented. Still required: opt-in progress wire messages, source adapters, owner drain/commit/failure integration, ATOM cache-ownership handling, actual overlap traces, same-route exactness and long-context/large-KV performance. Existing #47 export-only0.1–4GB curve showed chunking slower but reduced incremental staging to~67MB; it does not prove RNGD streaming improvement. Preserve that negative control and measure the new path when connected.

## Opt-in receiver and GPU owner integration

Receiver now negotiates `kv_progress: 1`, validates layer/token coverage and exact final suffix length, exposes borrowed shared-memory slices, and wakes the scheduling owner on progress. The GPU owner imports partial spans without admitting decode or publishing prefix state until complete coverage; CUDA writes are fenced before cancellation can reuse pages. Default whole-result protocol remains unchanged. Real source adapters and ATOM owner integration remain pending: these are controlled-producer tests, not a hardware streaming speedup measurement.

Validation uses the same environment and package working directory as above. Additional commands:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/gspark/venvs/pt213/bin/python -m pytest tests/test_subprocess_layer_progress.py tests/test_pd_stream_admission.py -q
```

- Real subprocess/SHM protocol tests cover partial visibility, invalid/overlapping/holey coverage, producer exit, cancellation and immediate same-slot reuse, cached suffix offsets, and capability rejection.
- Real tiny Gemma decoder tests compare controlled streaming and whole-result producers on CPU and CUDA, including partial non-admission, failures, cancellation and reuse.
- CUDA page8 fixture failed in Triton (`K >= 16`); ordinary whole-result control reproduced the compiler error. That control's outer exception assertion also failed. CUDA tests now use page16; CPU remains page8. No production kernel/default changed.
- Capacity regression first failed: invalid streaming capacity raised synchronously. It now completes the request Future with an error while preserving producer readiness and successful subsequent requests.
- Boolean response-ID regression first failed: Python `True == 1` routed malformed progress to another in-flight request. Exact integer type validation now rejects it before lookup.
- Latest focused run:18 passed,15 warnings,9.44s. Pre-ID-guard full suite:573 passed,15 skipped,253 subtests,27 warnings,165.62s. Final post-guard full suite recorded below when complete.

Evidence logs retain the failing controls and successful reruns. No claim of source-compute/transfer overlap, hardware speedup, ATOM integration, or universal numerical parity.

Final post-ID-guard full regression: **574 passed, 15 skipped, 253 subtests passed**,27 warnings in164.46s. Exit0; `/tmp/stream265-final-regression.log`. Source adapters and ATOM streaming remain unimplemented.

## Source protocol and rngd-full adapter

The production prefill server now advertises capability only for engines with `prefill_stream`. `stream_result` validates the reservation, geometry and nonoverlapping coverage, clips source spans to a cached suffix, copies BF16 CPU data to SHM, then sends notifications. Completion requires exact coverage. The existing KONI/Llama rngd-full adapter exports each completed group before advancing the next group. Whole-prefill behavior is retained for the A/B control.

The serving entry point accepts `--pd-prefill-stream-layers` only with a subprocess producer and Torch decoder; ATOM/RNGD decoder owners are rejected until integrated. Producer capability negotiation rejects older workers. This does not yet connect the separate four-model long-context factory adapters.

- Initial source regressions:5 failures (missing implementation); source plus pipe tests15 passed in2.24s.
- Production server/receiver pipe integration initially failed because the executable fixture lacked the package on sys.path. Corrected fixture;6 source tests passed in2.73s. The engine alone is stubbed: production server request handling, SHM writes, progress notifications and receiver are exercised.
- CLI regressions:5 failures before wiring; combined source/server/CLI/receiver/CPU-CUDA owner36 passed,15 warnings in10.57s.
- Added fully cached suffix and invalid-reservation tests. Full suite and RNGD hardware source parity recorded separately below.

Reproduce focused verification from the package directory with the environment above:

```bash
python -m pytest tests/test_prefill_source_progress.py tests/test_serving_prefill_cache.py tests/test_subprocess_layer_progress.py tests/test_pd_stream_admission.py -q
```

The source adapter performs blocking CPU export per group; it permits the receiver to import while later groups run. Neither this sequencing nor CPU tests establish actual kernel/copy overlap or end-to-end performance improvement.

Runtime compatibility audit found that the installed rngd-pd snapshot exposes `_next_token`/`lmh` but not the `_logits` method expected by the existing server. Both whole and streaming adapters now use `_last_logits`: prefer `_logits` when available, otherwise invoke the same existing on-card head and retain all vocabulary logits rather than only argmax. A regression first failed for the missing helper, then all13 source tests passed in2.93s. Numerical parity on actual RNGD remains to be checked; the CPU fixture alone cannot establish it.

Hardware source parity setup: local RNGD chip0, KONI-8B, width128, group2, same full model weights as evidence263. Torch2.10.0+cpu, furiosa-torch2026.4.0, transformers5.8.1. Decoder is not run in this check. Loaded rngd-pd prefill_full.py SHA256 `0ac864695d8c2cedc5fb2ddadbc3d3d782bb21909a83f2b4d82944f32940e8ea`.

Exact launch (source paths are local runtime inputs):
```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
 PYTHONPATH=/home/gspark/workspace/aicomp-rngd-stream-265/opt_prime:/tmp/rngd-gpu201-src \
 /tmp/codex-rngd-gpu-201/bin/python /tmp/stream265-source-hardware.py \
 > /tmp/stream265-source-hardware.log 2>&1
```

Full regression collected before the final legacy-head helper:591 passed,15 skipped,253 subtests,27 warnings in165.79s. After the helper, all13 source tests passed in2.93s; the full run does not cover that final added test.

### Actual RNGD attempt: failed before parity

The process exited1. The group2 width128 graph compiled in74s, but the legacy FullPrefillEngine constructor failed while building its on-card LM head: rngd_pd.kernels.dsl_b1 imports missing `furiosa.native_llm_common`. Neither whole-prefill nor streaming inference ran; no KV/logit comparison or speedup result exists. The real runtime failure is retained in stream265-source-hardware.log. Next integration must use the already validated four-model long-context factory adapters, including KONI's partitioned graph with host FP32 norm/head, rather than treating this legacy path as hardware-validated. The `_last_logits` fallback is CPU-tested only and does not solve the constructor dependency.

## Validated four-model factory connection

Added an explicit trusted `--engine factory --engine-factory module:callable` loader using the same FP32 safetensors/config inputs as the prior research worker. Shared-memory geometry and width are checked before constructing the engine. `examples/pd/streaming_factories.py` wraps the previously measured Qwen, Gemma and KONI producers: the whole-result method is unchanged, and streaming preserves padding, layer order, per-layer attention inputs and each family's final head arithmetic. KONI uses its existing partitioned graph and host FP32 norm/head, avoiding the legacy native head dependency.

Dependencies: aicomp `04dab7cb6961103f9404c99e9fcbcaadac14b098` (`feat/234-live-model-pd`) serving_matrix and qwen3_rngd example directories, plus the measured rngd_pd runtime snapshot. Local qwen prefill/spec copies have identical SHA256 to that committed example. These examples are explicit experimental dependencies, not yet bundled into a production package.

Test-first results: factory loader6failed before implementation→6passed. Adapter tests6failed before implementation; first run2failed/10passed (Qwen fixture used a scalar instead of layer-parameter tuple; Gemma imported helpers before empty-input validation), corrected. Source/adapter/factory tests25passed. Production server/factory IPC added for both whole and streaming modes; combined27passed in6.47s. Hardware results below are separate.

Hardware source matrix: each model gets its tokenizer's short prompt plus an input exactly filling its compiled width. For each, compare whole output against streamed SHM with send_from=0, half the prompt and the entire prompt (empty suffix). Assert exact BF16 K/V and exact logits, record notification timestamps and first-token IDs. Width128 for Qwen0.6/Qwen4/Gemma, width512 for KONI (minimum validated partition width). These widths do not establish long-context performance.

Exact driver: `/tmp/stream265-source-queue.py`, invoking `/tmp/stream265-factory-hardware.py` in the isolated Furiosa2026.4 environment. Files refuse overwriting previous results. Same-route RNGD→GPU follow-up uses `/tmp/stream265-gpu-queue.py` and `/tmp/stream265-gpu-hardware.py`, eight output tokens, two concurrent requests, page64, caches off; whole versus streamed producer is the only mode change.

Hardware source results: **24/24 exact K/V and logits** across all four models (6 conditions each); max absolute logit error0 throughout. All nonempty suffixes emitted their first progress notification before final logits returned. All four child processes exited0. This proves source/SHM parity at tested widths128/512 only, not long-context decoder parity, pipeline overlap or speedup. Artifacts: stream265-{qwen06b,qwen4b,gemma,koni}-source.{json,log}, stream265-source-queue.json and dependency hashes. CPU times recorded in callbacks are not device-overlap measurements.

First real GPU integration failed before decoding: server passed a tuple to the Qwen factory's whole-prefill method, which concatenates a list for padding. Source-only comparisons passed because their driver passed lists. The production factory pipe regression reproduced it (whole mode failed, streaming mode passed); non-Torch server dispatch now passes a list consistently. Final combined tests27passed in4.82s. Original GPU failure JSON/log and queue result are preserved with `-failed-list` names; a fresh GPU queue follows. This demonstrates why direct producer parity does not substitute for actual server/decoder validation.

Actual RNGD→GPU integration completed: **8/8 output-token pairs exact** (two concurrently submitted requests per model, fixed8 output tokens). Whole versus streamed source modes both drained successfully. Streaming span imports: Qwen0.6=56, Qwen4=72, Gemma=52, KONI=64; whole mode0 for every model. Input lengths were short and128 (Qwen/Gemma), short and512 (KONI). This is same-route transport/scheduler parity, not nativeGPU equivalence or answer quality. Requests deliberately use fixed8 tokens without model EOS stop IDs; Gemma's short output contains turn markers, so this test must not be interpreted as normal-EOS quality evaluation. No speedup claim from these single runs. All owned subprocesses exited and GPU compute-process listing was empty before the full regression run started.

Final factory integration regression: **606 passed,15 skipped,253 subtests passed**,27 warnings in166.21s, exit0. Raw log stream265-factory-full.log. This includes the final tuple/list correction, real factory pipe tests and all existing source/owner tests.

Next performance matrix uses one capability-negotiated producer per model, changes request mode only after its future map and decoder are drained, and alternates modes after two warmup runs. Inputs2048/8192, compiled width8192, output32, concurrency2, page64, caches off. Three measured batches/mode. Host import trace records whether source completion was observed before/after destination import; it is not a device-kernel overlap trace. Drivers prepared as stream265-long-gpu.py and stream265-long-queue.py; results must be assessed separately from the completed short-width parity results.

## Long-context alternating same-engine results (runtime171e8c4)

Command: `python3 /tmp/stream265-long-queue.py > /tmp/stream265-long-queue.log 2>&1`. Exact queue and driver are retained in evidence265. Each model uses one capability-negotiated source process; changing `stream_layers` requires an empty source future map and drained decoder. Source width8192, decoder context8320/page64, mixed2048/8192-token synthetic repeated-text inputs submitted together, fixed32 output tokens without EOS stopping, no prefix cache. First two runs (whole/stream) are warmups; measured order whole/stream/stream/whole/whole/stream gives3 batches/mode. Engine startup and GPU model loading are excluded. Hardware: local RNGD chip0 and CUDA GPU0 (A10080GB); no concurrent benchmark/test jobs.

| Model | Whole median batch completion(s) | Stream median(s) | Reduction | TTFT reduction, first2048 / second8192 request |
|---|---:|---:|---:|---:|
| Qwen3-0.6B |3.637588|2.613433|28.15%|34.02% /33.58%|
| Qwen3-4B |7.305905|5.689632|22.12%|25.55% /25.98%|
| Gemma3-1B |3.936250|3.767887|4.28%|4.45% /4.22%|
| KONI-8B |7.598914|5.855674|22.94%|18.54% /25.32%|

All32 batches including warmups drained; all64 generated sequences (32tokens each) match their model's initial whole-result reference. All four processes exited0. TTFT includes queueing in this two-request workload, not isolated latency for a single input length. The auditor independently recomputes medians from raw records and checks warmup placement, token sequences, token arrival times and trace counts.

Across measured streaming batches, GPU imports completed before the source Future completed for all168/216/156/192 spans respectively (732 total). This verifies early destination import in the actual pipeline. It is a host-observed completion trace, not a device-kernel/DMA overlap trace. It does not isolate the benefit of earlier import from avoiding whole-KV host assembly; a deferred-notification control would separate those effects.

Limits: three samples/mode, one host, fixed compiled width8192 and concurrency2. Gemma improvement is small. This is fixed-output transport/scheduler parity, not semantic accuracy, normal-EOS behavior or native-vendor performance. Long-context byte-for-byte source KV comparison and streaming hardware fault tests remain to be extended; the direct source KV/logit exactness matrix above used widths128/512. ATOM partial owner integration remains pending. Whole transfer remains the default.

Artifacts: stream265-*-long-gpu.{json,log}, stream265-long-queue.json, stream265-long-summary.json, audit_stream265_long.py, stream265-long-gpu.py and stream265-long-queue.py. After completion, GPU compute-process listing and RNGD device-user checks were empty.
