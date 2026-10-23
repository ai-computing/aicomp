# HTTP model matrix

State: review; full goal remains active. GPU evidence remains as recorded. ATOM four-model cache-off lifecycle passes with48/48 concurrent equality; both Qwen device/host cache cases pass12/12 each. SIGPIPE fix tracked separately in #214. Remaining: RNGD routes, hybrid-state restoration, real transfer overlap and broader semantic testing.

## Production EOS correction (#229)

The common serve_batched factory now unions model generation EOS and tokenizer turn-end EOS. KONI config128001/tokenizer128009 previously continued into role delimiters; real GPU HTTP negative control ends length, fixed/stream/reuse end stop, ignore_eos remains length.22related CPU tests pass after3failed factory regressions. GPU and ATOM hardware both pass5/5 EOS/override/stream/reuse checks; NPU uses existing B1/context9216/k1 artifact. See serving-eos229 record/evidence229.

## Fresh hybrid PD (#231)

Gemma full-prefix KV imports now run through ATOM PD scheduling: 33 CPU tests and 18 real HTTP normal/reuse responses pass, with 20 remote / 0 local prefills including cancellations. Producer data is recorded RNGD KV replay. Live prefill, overlap, hybrid cache restoration and hardware batch >1 remain unverified. SIGPIPE log issue still depends on #214/#215. See hybrid-pd231 experiment and evidence231.

Geometry recovery follow-up: malformed remote heads/KV lengths fail only their request; real device import failures still poison the scheduler.37related tests pass. Hardware evidence remains c76e03a.

Long hardware follow-up on1e54786:6423–6428token inputs,10normal/reuse HTTP responses match nativeATOM;12remote/0local prefills including cancellation. Recorded RNGD KV replay, arithmetic wrong on both native/imported paths; no quality or live-transfer claim.

## Single-block static PD (#236)

Static B1 with one KV block now supports cache-off full-prefix PD. KONI live RNGD->ATOM HTTP14/14 output checks and16remote/0local prefills pass, including cancel/disconnect reuse.43CPU tests cover queued reuse, late cancellation and partial-prefix error recovery; cache/multi-lane guards remain. No new artifact, hardware batch or cache-restoration claim. See single-block-pd236 record and evidence236.

## Subprocess failure lifecycle (#238)

Structured SDK logs and cancelled/malformed replies previously killed the receiver and could strand later requests. Terminal errors now settle all queued Futures and reject subsequent submits. Failed startup and bounded close release the owned process and slab; fresh work succeeds on an explicitly created replacement worker. Nine CPU fault scenarios plus the existing tiny Torch subprocess path pass in the 64-test focused set. Full suite: 460 passed, 15 skipped, 135 subtests passed. Accelerator fault recovery and service-level automatic restart remain separate acceptance items. See prefill-worker-failure238 record.

## Explicit Torch producer policy (#241)

New opt-in controls disable producer prefix reuse and select producer chunk size; defaults remain cache-on/chunk2048. Public cache-off/chunk256 configuration passes all four real GPU PD HTTP paths:48/48 concurrent equality,12/12 native-PD serial equality at128/2048/8192tokens,84remote prefills and0producer hits. Lifecycle checks pass.20focused and464full tests pass (15skipped,135subtests). This is tested numerical reproducibility, not a speedup or arbitrary-input equality guarantee; original default-policy discrepancies remain in#240.

## Live GPU PD matrix (#240)

All four native/PD BF16 lifecycle configurations pass across128/2048/8192tokens. Default PD has35/48concurrent equality versus native48/48, and11/12native-PD serial equality. FP32 restores all divergent models; BF16 explicit cache-off/chunk256 controls#241 restore four-model48/48concurrent and12/12serial equality. Original divergences and validator module-identity failure are retained in evidence240. The self-contained HTTP-candidate loader was fixed and Qwen0.6B rerun passes. No matched speedup or compute/copy overlap claim.

## Producer-aware readiness (#246)

A real terminated subprocess producer previously left /ready200 and new requests failed500. The service now reports503 and rejects new admissions with retryable503 until explicit healthy replacement; single request errors preserve readiness. Pools require all configured members because routing still includes them.30focused tests and full468passed/15skipped/135subtests. Legacy workers without the optional readiness property retain their prior behavior. Historical #244 hardware evidence predates this new post-failure503 contract; no automatic restart claim.

## Batched producer timing (#249)

Candidate exports airm_pd_producer_seconds_total{stage="prefill"|"transfer"} from existing accumulated batcher durations. Repeated scrapes do not add observations; missing timing keys are omitted. Tiny real-model admission test covers fixed worker-reported durations and legacy omissions. These counters are worker-reported admitted-handoff sums, not request latency or proof of zero cost for workers that do not measure stages. Full CPU suite:467passed,17skipped,135subtests,1existing warning. See pd-stage-metrics249 experiment.

## Chunk-copy overlap (#47), 2026-10-24

Opt-in Torch GPU token-chunk copies implemented with two pending spans, pinned-host staging, source readiness events and completion before handoff. Focused CUDA correctness/failure checks pass; CPU suite 504 passed, 43 skipped, 253 subtests. Actual CUDA timeline, four-model serving speedup, large-KV curve and RNGD/ATOM layer-wise export remain open. Default remains whole-span transfer. See experiments/2026-10-24-chunk-copy47.md.

Qwen3-0.6B input8192 paired prefill: median997.98 ->955.64 ms (-4.24%), six runs KV/token exact. Separate CUDA traces show 27.91ms DtoH/source-kernel intersection (mostly27.47ms KV index gather), zero GEMM intersection. Both modes transfer939,524,096 bytes. Four-model HTTP A/B remains live; no serving speedup claim.

GPU HTTP blockA96batches all parity;long/c8 chunk-copy throughput changes: Qwen0.6-3.1%,Qwen4+2.5%,KONI+2.4%,Gemma-1.5%. Reverse-order blockB pending. Export-only0.1–4GB curve: chunked slower, but incremental source staging bounded at~67.1MB rather than4GB. Keep opt-in; no general serving speedup claim.

Reverse-order blockB completed: 192 total timed batches passed parity. Long/c8 A/B changes: Qwen0.6 -3.12%/+5.80%, Qwen4 +2.49%/+17.39%, KONI +2.38%/-1.53%, Gemma -1.54%/-0.28%. Qwen4 B baseline prefill slowed while transfer stayed constant, confounding the larger gain. Keep opt-in; no repeatable general throughput improvement. See evidence47/chunk47-four-model-serving-ab-summary.json.
#258 dense hybrid KV: actual Gemma failure reproduced; mapping/group validation corrected.31focused and505full CPU tests pass; live rerun pending.

#265 groundwork: coverage validation and reserved partial hybrid-cache import implemented;22focused and556full tests pass (15skip,253subtests). Admitted-prefix overwrite regression fixed. Not connected to worker/owner/source paths yet; no hardware overlap claim.

## Layer-progress receiver and GPU owner (#265)

Opt-in subprocess progress validation and incremental GPU imports implemented; partial state remains private until complete. CPU/CUDA focused18pass; full574pass,15skip,253subtests. Invalid boolean ID and capacity error regressions fixed. Controlled producers only: real RNGD source adapters, ATOM owner, and hardware overlap/performance remain pending. See layer-stream265 record/evidence265.

## Producer connection (#265)

Production SHM source protocol and rngd-full per-group export connected. Torch CLI opt-in added; unsupported owners rejected. Full591pass before final legacy-head helper; final source13pass. Actual KONI source parity running; no numerical or speedup claim yet. Four-model long-context factories and ATOM ownership integration remain pending.

Hardware attempt ended before inference: legacy FullPrefillEngine on-card head requires missing furiosa.native_llm_common. Group graph compiled74s; no KV/logit parity result. Continue with the already validated four-model factory adapters and KONI partitioned host-head path.
