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

## B1 next-request prefill (#253), 2026-10-24

Opt-in remote_prefetch / --pd-prefetch-next implemented for single-lane ATOM remote-only PD. Keep one source result reserved through owner import; cancelled writers drain before reuse. Prefix/host cache admission remains late, so early prefill transfers the full prompt.35focused tests and full511passed34skipped253subtests; real four-model hardware pipeline/performance validation pending. No speedup claim. See experiments/2026-10-24-atom-prefetch253.md.

KONI hardware prefetch validation complete:48timed batches exact,8fixed-output/4EOS groups pass, cancellation/reuse and producer-exit isolation/readiness503 pass. Same-warm-instance off/on3repeat c4 gains23.48–40.87%; c1 near-neutral. Host producer/decode overlap only in enabled mode. Remaining three models pending; no default promotion. See evidence253/atom253-koni-summary.json and experiment.

Qwen3-4B also passes48timed batches,8fixed/4EOS groups and live cancellation/producer-exit isolation. c4 median gains21.20–41.00%; c1 near-neutral. Qwen0.6/Gemma remain pending. This does not resolve the separate GPU decoder pre-EOS drift in#256.

Qwen3-0.6B: 48 prefetch off/on batches and fixed-output/EOS/lifecycle checks passed. At 8K input/output128/c4, 35.016 -> 48.231 tok/s (+37.74%); c1 roughly neutral. Gemma hardware validation remains pending.
