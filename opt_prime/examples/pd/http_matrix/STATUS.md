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

## Matched performance block A (#248)

Four models times native/PD/DP2 completed144measured batches with parity. Inputs128/8192, outputs32, concurrency1/4, three repeats. DP2 leads all four at8192/c4. BlockA is ordered and has a short concurrent CPU-test window during KONI; quiet reversed-order replication and longer outputs remain. No within-request compute/copy overlap in this Torch baseline. See evidence248 and matched-pd-performance248 record.

BlockB reverse-model/mode repetition completed144additional batches with parity (288A/B total). Same-two-GPU DP2 leads all four at8192/c4. PD relative to native is variable: Gemma23.11->17.79tok/s,Qwen0.6B22.37->18.41,Qwen4B13.25->15.12,KONI14.47->14.02. No stable native-to-PD speedup claim. Runtime273952a adds stage counters; scheduling/compute unchanged. Longer-output and actual heterogeneous benchmarking ongoing.

External-backend benchmark candidate:trusted argv config and explicit startup deadline. Real ATOM nativeQwen0.6B12batches pass parity; firstPDattempt timed out at180seconds before readiness, raw failure retained. Retry with1800seconds is live. No heterogeneous speedup claim.3harness tests pass; no core serving changes.

Qwen0.6B actual RNGD-to-ATOM completed12PD batches with parity against nativeATOM;42remote/0localprefills. Readiness369.73s excluded from timing.8192input/c4 throughputnative22.25vsPD10.58tok/s. Current width8192producer is slower; shorter prompts also pad to8192. Other three model comparisons running. Failed180s startup record retained.
