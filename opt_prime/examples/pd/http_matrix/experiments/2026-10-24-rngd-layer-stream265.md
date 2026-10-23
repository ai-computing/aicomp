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

## Long-context KV exactness

`python3 /tmp/stream265-long-source-queue.py > /tmp/stream265-long-source-queue.log 2>&1` completed all four models with exit0. Driver stream265-long-source.py uses compiled width8192 and prompts2048/8192, each with send_from=0/half/full. All24 conditions have exact BF16 K/V and exact logits, max absolute logit error0. This closes the long byte-level source-parity gap for these deterministic synthetic inputs. Artifacts: stream265-*-long-source.{json,log} and stream265-long-source-queue.json. Source implementation matches b28408d; independent source hashes are in each JSON.

## ATOM partial ownership integration

Cherry-picked the existing B1 prefetch prerequisite3e16c33 (#255) as2e69570. Append-only conflicts in STATUS and serving tests were resolved preserving both sets of entries/tests. An initial relative-path mistake left conflict markers for a failed collection; corrected invocation43passed. No production inference result is associated with that invocation error.

Added per-layer block import using existing BF16→DLFloat conversion and SDK KV writes. Selected-layer geometry is checked before writes; hybrid local state maps to its lane and retains the prompt tail, while global state maps to its reserved block. The owner wakes on progress, imports only into reserved lanes, waits for complete coverage before admission, and leaves prefetch data untouched until its destination lane is available. Multi-lane hybrid caches defer imports until final admission because idle local cache slots can still be written by the decode graph. The actual B1 path supports early import.

Found and reproduced an existing prefix publication bug: remote reservation set `admitted` before prefill succeeded, allowing a failed/incomplete prompt block to enter the prefix index. Local and remote paths now mark it only after successful prefill/import. Cancellation during partial import cannot publish an incomplete prefix; cancellation after valid admission may still retain its complete prefix.

Test-first evidence:4failed/1passed before partial support; initial focused60passed. Extended identical-KV whole/stream decode, cancellation/reuse, SDK-write poisoning and hybrid lane/block mapping passed48tests. Multi-lane hybrid deferral regression failed before the guard; final49tests passed. ATOM CLI acceptance regression1failed/4passed before wiring, final26related tests passed. Full regression:626passed,15skipped,253subtests passed,27warnings in168.94s.

Actual ATOM check uses a separately allocated owned Pod, one RNGD and one ATOM, existing context9216/B1 compiled artifacts, sourcewidth512, short/512 inputs queued together and fixed8 output tokens. One live producer is reused across (whole, no-prefetch), (stream, no-prefetch), (stream,prefetch), (whole,prefetch), switching only when drained. Each import records active decode count; it must be zero for B1. New source package is isolated in the owned scratch directory; runtime base2e69570 plus candidate Atom/CLI changes, exact code SHA256 recorded in each result. Qwen0.6 completed all4modes token-exact; remaining models running at this checkpoint. No ATOM speedup or long-context claim yet.

Actual four-model ATOM result: all16 mode batches drained and all32 generated sequences exactly match the same-route whole/no-prefetch reference. Whole/stream × prefetch off/on, short/512 inputs, fixed8 outputs. Stream span totals across both streaming modes: Qwen0.6=112, Qwen4=144, Gemma=104, KONI=128; respectively44/36/99/96 spans completed host import before the source Future was done. All imports observed zero active decode requests, preserving B1 cache ownership. All four child/copy exits0. The owned Pod was deleted by the queue, and an independent get --ignore-not-found returned empty with exit0. These measurements establish integration parity and early host-observed import, not long-context ATOM speedup or device DMA overlap.

Reproduction drivers: stream265_atom.py, run-stream265-atom.sh, stream265-long-source.py. ATOM runtime is2e69570 plus candidate changes identified by SHA256 in each JSON. Full test log stream265-atom-full.log:626passed,15skipped,253subtests. Remaining: long ATOM timing, streaming hardware cancellation/producer failure, and device-overlap attribution.

## Long ATOM alternating timing extension

Runtime33d6cb0, isolated package hashes recorded per result. Owned Pod stream265long: one RNGD+one ATOM, B1 compiled context9216, sourcewidth8192, inputs2048/8192 queued together, fixed32 outputs, cache off. For each prefetch off/on stratum, modes whole/stream/whole/stream/stream/whole/whole/stream; first two warmups, three measured batches per mode. One live producer per model, switch only while drained; output tokens compared with initial same-route whole reference. Drivers stream265_long_atom.py and run-stream265-long-atom.sh, orchestration `python3 /tmp/stream265-long-atom-queue.py`. No performance result yet. Initial script upload used an SSH-config default port and failed before any benchmark; explicit accelerator-host port22 upload succeeded.

Follow-up checkpoint: existing long ATOM queue session64332 remained live; owned Pod phase Running and source process CPU time increased during initialization. No completed long ATOM batch yet; do not restart from observation timeouts. Prepared stream265_fault_atom.py for actual early-import cancellation/reuse and producer death, with explicit assertions that interception occurs before source completion/admission. Driver is syntax-checked only and has not run; it is not hardware evidence. Rechecked all24 long source exactness conditions and16 short ATOM mode batches from raw JSON to update ACCEPTANCE.md/stream265-acceptance.json, preserving earlier native-parity/content failures.

## Actual GPU streaming fault extension

Local GPU compute-process listing and RNGD device-user listing were empty before launch. Independent of the remote ATOM timing host. Runtime33d6cb0, sourcewidth8192, source chip0/GPU0, decoder two slots/context9216/page64/cacheoff. `python3 /tmp/stream265-fault-queue.py > /tmp/stream265-fault-queue.log 2>&1`, session27276, invokes stream265-fault-gpu.py for each model. Test intercepts an incomplete import, cancels it and compares a replacement request against same-route whole-source tokens. Then it compares an active512-token decode against its same-route reference while killing the source after partial import of a second8K prompt. Assertions require an actual partial-before-completion interception and an active survivor at the kill point. All failures retained; no result yet.

GPU fault driver first attempt failed in test instrumentation: after cancellation/reuse token comparison, it read ATOM-only `batch.poisoned` on PDBatcher, raising AttributeError. Qwen0.6 raw failure retained. Stopped only owned queue PID and its Qwen4 child; child exited through finally with KeyboardInterrupt, worker/device cleanup verified by empty GPU/RNGD users. Removed ATOM-only state assertions (GPU has no such quarantine field). Fresh driver/queue v2 write distinct `*-fault-gpu-v2.*` paths; no previous evidence overwritten. This is a harness fix, not a production engine fix.

Completed first long ATOM model Qwen3-0.6B:16batches drained and all32 sequences exact. Three measured batches/mode: prefetch-off whole5.125748s→stream3.297927s (35.66% reduction); prefetch-on4.877637s→2.963296s (39.25%). Other models still pending; no four-model ATOM performance claim yet.

GPU fault v2 checkpoint: Qwen0.6,Qwen4,Gemma each exited0, completed both cancellation/reuse and partial-producer-death tests, and drained. At producer death, the surviving512-token request had emitted5/6/4 tokens respectively; runtime assertions verified all512 tokens match the no-fault same-route reference. KONI remains live in source initialization; no completed claim for it. audit_stream265_faults.py requires allfour successful process exits and active_tokens strictly between0 and512; synthetic fixture accepted and a finished-survivor counterexample rejected. ATOM long Qwen4 source remains live; next ATOM fault scripts uploaded to owned scratch but no fault Pod allocated while timing continues.

Final GPU fault v2 result: allfour child exits0; auditor confirms allfour complete cancellation/reuse and partial-producer-death cases. Surviving512-token decode had5/6/4/10 tokens at source death (Qwen0.6/Qwen4/Gemma/KONI), and driver asserts its entire output equals the same-route no-fault reference. Failed request raises source-exit error; owners drain. Prefix caches off. This proves tested source/owner failure handling, not device-failure recovery, prefix restoration or native numerical equivalence. Raw JSON stores assertion outcomes/triggers, not full512-token sequences; the exact executed driver contains the equality assertions. After queue exit, local GPU compute listing and RNGD device-user listing were empty. Independent auditor rerun reproduces stream265-fault-summary.json byte-for-byte.

ATOM fault preparation extended (not run): after partial-import cancellation/reuse and producer death, start a fresh source and compare active512-token decode against its same-route reference while killing a prefetched8K request after progress appears. Assert victim has no destination lane yet and survivor is active. Store complete expected/actual token arrays for reuse and survivor cases. Updated isolated driver uploaded; running timing script/package is unchanged. Working notes refreshed to remove stale pre-implementation status and obsolete owned-Pod pointers.

Qwen4 long ATOM source startup diagnostic: read-only inspection inside the owned Pod reports source RSS38245120KiB, swap0,29threads; total Pod memory50786693120bytes. OOM/oom_kill counts0, memory pressure avg10/60/300 all0. CPU usage continues increasing, but this alone does not establish successful initialization; no Qwen4 timed batch yet. Existing startup timeout remains1800s, queue unchanged. Host-side /proc/io was denied; used only the owned container's /proc instead. Diagnostic snapshot stream265-qwen4b-init-diagnostic.json. No profiler attached and no live runtime/code changed.

Added two controlled-producer regressions for partial prefetch error/cancel while another request decodes: no partial destination import into the active lane, cancelled source reservation retained until its writer finishes, same BF16-source reference output, owner drains without poisoning. Focused command `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_DEVICE_BACKEND_AUTOLOAD=0 /home/gspark/venvs/pt213/bin/python -m pytest -q tests/test_atom_stream_progress.py tests/test_atom_batching.py` from worktree opt_prime:39passed in5.00s, log stream265-prefetch-fault-final.log. An initial append used the wrong relative path and failed before modifying tests; that initial invocation ran only pre-existing tests and is not new-test evidence. No production code changed. Qwen4 long ATOM now emitted actual timed batches after startup; measurements still in progress.

Qwen3-4B long ATOM completed with child/copy exits0, all16 batches/32sequences same-route exact and drained; every import observes zero active B1 decode. Three measured batches/mode: prefetch-off whole10.826179s→stream8.524409s (21.26% reduction), prefetch-on9.738162s→7.507991s (22.90%). Gemma starts next; KONI queued. Startup is excluded from these timings. These results do not establish native-vendor equivalence or full four-model ATOM performance yet.

Prepared audit_stream265_atom_faults.py: requires four successful child/copy exits, all three fault cases per model, actual partial-progress triggers, idle/unpoisoned owners, active survivor at source death and victim without a reserved lane, plus exact retained16/512-token arrays. Self-check accepts a complete synthetic fixture and rejects a single altered survivor token. This is auditor validation only; real ATOM fault runs remain pending. Existing long timing session64332 remains live, Gemma source initialization observed at2m27s with CPU time11m54s; no restart.

## GPU notification-timing control

To distinguish earlier destination import from avoiding whole-KV assembly, test-only stream265_control.py retains the production stream_result function and all per-layer compute/SHM writes; deferred mode buffers only progress descriptors until final source logits return, then publishes them in original order. Early mode delegates unmodified. Whole mode remains the existing control. No production package changes. A pure self-check verifies copy/logit/notification order and return-value preservation. A source launcher chooses mode from a fixed12-batch plan using sequential request IDs (two requests/batch); no wire API changes.

Four models, width8192, inputs2048/8192 queued together, fixed32 outputs, c2/page64/cacheoff; one warmup per mode followed by3 measured batches/mode in a balanced ordering. Same source process and GPU model per model; switch only while drained. Actual source/control hashes recorded. Command `python3 /tmp/stream265-control-queue.py > /tmp/stream265-control-queue.log 2>&1`. Local GPU/RNGD device users were empty before launch. Remote ATOM timing continues on a different host. This control can attribute host-pipeline timing, not prove device DMA overlap. No result yet.

First control result, Qwen0.6:12batches/24sequences exact, child exit0. Median completion whole3.464331s, early2.552257s, deferred2.676551s (three measured batches/mode). Early is4.64% faster than deferred; deferred is22.74% faster than whole. Thus the total improvement cannot be attributed solely to within-request early notification. Deferred retains layer-wise assembly/copy behavior and can still overlap one request's import with another request's prefill in this c2 workload. Early-vs-deferred estimates the incremental timing effect of within-request notification; deferred-vs-whole includes other source/host-path differences and is not a pure device-overlap measurement. Other models still running; no universal attribution claim.

Gemma long ATOM completed16batches/32sequences exact and drained, child/copy exits0, zero active B1 decode at every partial import. Prefetch-off whole4.116702s→stream3.713744s (9.79% reduction), prefetch-on3.807711s→3.380978s (11.21%);3measured batches/mode. KONI starts next. Qwen4 GPU notification control also completes12batches/24sequences exact: whole7.298562s, early5.691523s, deferred5.866724s; early2.99% faster than deferred. These first two control models both show most total whole-vs-stream gain persists with deferred within-request notification. Remaining control models pending.

Gemma GPU notification control completes12batches/24sequences exact, child exit0: whole median3.835957s, early3.711786s, deferred3.726879s. Early-vs-deferred difference is only0.40%; early samples3.71179/3.73507/3.67427s overlap deferred3.73673/3.72688/3.68459s, so these three repetitions do not establish a reliable additional early-notification gain for Gemma. Qwen0.6/Qwen4 early/deferred sample ranges are separated in this recorded run, but only three samples/model/mode and one c2 workload; no general statistical or device-overlap claim. KONI control and long ATOM source both remain live. Control auditor self-check accepted a synthetic balanced matrix and rejected an incorrect mode ordering.

## Completed four-model notification control

Allfour child exits0,48batches/96sequences including warmups exact and drained. Auditor verifies fixed mode ordering, warmups, three measured batches/mode, source span counts, token arrival times and exact output arrays; deterministic summary saved in stream265-control-summary.json. GPU compute-process and local RNGD user listings empty after queue exit.

|Model|Whole median(s)|Early(s)|Deferred(s)|Early reduction vs deferred|Deferred reduction vs whole|
|---|---:|---:|---:|---:|---:|
|Qwen0.6|3.464331|2.552257|2.676551|4.64%|22.74%|
|Qwen4|7.298562|5.691523|5.866724|2.99%|19.62%|
|Gemma|3.835957|3.711786|3.726879|0.40%|2.84%|
|KONI|7.453318|5.768218|5.869349|1.72%|21.25%|

Conclusion limited to this c2 synthetic2K/8K/fixed32 workload: most Qwen/KONI whole-vs-stream gain remains without within-request early notifications. The deferred control still changes source KV assembly and destination per-layer import relative to whole mode, and still permits cross-request overlap. It therefore does not isolate source assembly alone or prove device DMA overlap. Early-vs-deferred is the narrower incremental notification-timing comparison. Three samples/mode; Gemma ranges overlap. Preserve whole default; do not describe total speedup as an overlap percentage.
