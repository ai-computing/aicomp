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

## Four-model streaming factories (#265)

Validated producers connected:24/24 actual RNGD source cases exact K/V/logits;8/8 real RNGD-GPU fixed8-token pairs exact, all four models. Width128/512 only. Factory tuple/list integration bug reproduced and corrected;27focused tests pass. Long-context repeated timing, hardware fault isolation under streaming and ATOM owner remain pending. Same-route parity does not waive previous cross-backend content counterexamples.

Final factory integration full regression:606passed,15skipped,253subtests in166.21s. Short-width source/GPU hardware checks complete; long-context matrix is the next measurement.

## Long-context streaming measurements (#265)

Runtime171e8c4: c2 mixed2K/8K input, fixed32output,3batches/mode after warmups. Median completion reduction28.2% Q06,22.1% Q4,4.3% Gemma,22.9% KONI. All64sequences incl warmups exact;732 GPU span imports complete before source Future. Published auditor reproduces raw-record summary. Host trace only; no universal speedup or semantic-equivalence claim. Long KV byte parity, hardware fault tests and ATOM partial ownership remain pending.
## B1 next-request prefill (#253), 2026-10-24

Opt-in remote_prefetch / --pd-prefetch-next implemented for single-lane ATOM remote-only PD. Keep one source result reserved through owner import; cancelled writers drain before reuse. Prefix/host cache admission remains late, so early prefill transfers the full prompt.35focused tests and full511passed34skipped253subtests; real four-model hardware pipeline/performance validation pending. No speedup claim. See experiments/2026-10-24-atom-prefetch253.md.

## ATOM streaming and long source parity (#265)

Long source24/24 conditions at2K/8K have exact KV/logits. ATOM partial import and B1 prefetch ownership implemented; failed/partial prompts no longer publish prefixes. Four models × four modes:16batches/32sequences same-route exact, all drain; owned Pod deleted and absence verified. Full626passed/15skipped/253subtests. Prerequisite #255 included. Long ATOM timing and streaming hardware faults remain pending; no ATOM speedup claim.

## Streaming hardware fault checks (#265)

Runtime33d6cb0: allfour GPU cancellation/reuse and partial-producer-death checks passed. Active512-token decode remains exact to same-route no-fault reference; kill at5/6/4/10 tokens. All child exits0 and local devices idle. Auditor reproduces summary. First harness-only AttributeError and interrupted follow-up retained. Long ATOM Q06 complete16batches/32sequences exact, median completion reduction35.66% without prefetch/39.25% with prefetch; remaining models running. ATOM hardware faults remain pending.

Qwen4 long ATOM completed16batches/32sequences exact: median completion reduction21.26% prefetchoff/22.90% on,3samples/mode. Gemma/KONI pending. Added two partial-prefetch failure/cancellation regressions; related39passed in5.00s. Production unchanged; previous full626-test run predates these two added tests.

## Notification-timing control (#265)

Allfour GPU control48batches/96sequences exact. Early-vs-deferred completion reduction4.64/2.99/0.40/1.72%; deferred-vs-whole22.74/19.62/2.84/21.25%. Most Qwen/KONI gain persists without within-request early notification; deferred still changes source assembly/destination import and permits cross-request overlap, so no pure source-assembly or DMA-overlap attribution. Three samples/mode; Gemma ranges overlap. Auditor reproduces summary. Gemma long ATOM complete16batches/32sequences exact,9.79/11.21% reduction without/with prefetch. KONI long ATOM and ATOM faults pending.

## Complete long ATOM timing (#265)

Allfour64batches/128sequences same-route exact; full auditor passes. Whole-vs-stream completion reductions35.66/21.26/9.79/19.19% without prefetch and39.25/22.90/11.21/16.84% with prefetch, Q06/Q4/Gemma/KONI. Three measured batches/mode, synthetic2K/8K fixed32output, B1/two queued requests. Timing Pod absence independently verified after termination; queue exited0. Separate owned ATOM hardware fault matrix now running; no fault result yet.

## Higher-load GPU streaming (#265)

C4/fixed128output, mixed2K/8K/2K/8K: allfour32batches/128sequences exact, exits0 and local devices idle. Whole-vs-stream median completion reductions20.72/16.44/2.26/19.34% (Q06/Q4/Gemma/KONI),3measured batches/mode. Auditor verifies outputs, ordering, arrivals/spans and reproduces summaries. STREAMING.md now maps source/transport/destination/CLI code and explicitly scopes same-host research-factory support. ATOM Q06 faults passed; other models still running.

## Complete actual ATOM streaming faults (#265)

Allfour models pass all3 actual cases (12 total), independently comparing retained16-token replacement and512-token survivor arrays. Producer death at survivor13/6/10/5tokens; owners drain without poisoning, incomplete prefixes remain private. All child/copy exits0; queue terminal0 and owned Pod absence verified. See experiments/2026-10-24-stream265-atom-faults.md and evidence265/stream265-atom-fault-summary.json. Native numerical differences and hybrid-prefix support remain separate limitations.

## Complete c16 queued streaming matrix (#265)

Four models complete32batches/512sequences exact, c16 queued behind4slots, mixed2K/8K fixed128output. Whole-to-stream median completion reductions25.20/20.02/2.33/20.91% (Q06/Q4/Gemma/KONI),3samples/mode. Full independent auditor and its malformed-evidence self-check pass; queue0/devices idle verified. See experiments/2026-10-24-stream265-c16.md. HTTPc16 runs are separate and still in progress; numerical boundaries remain.

## Qwen HTTP c16 streaming (#265)

Qwen0.6/Qwen4 native/whole/stream six modes independently pass eight conditions each. Whole/stream references exact. Qwen4 observed throughput22.1–27.2% above whole in single batches; native/PD numerical differences remain. Short-input PD remains slower, fixedwidth8192 cost is not yet isolated. Gemma/KONI and ATOM matrix pending. See experiments/2026-10-24-stream265-qwen-http.md and evidence265.

## Gemma HTTP c16 follow-up (#265)

Allthree modes independently pass8conditions each;whole/stream exact. Native strict4/8 differences occur afterEOS;throughEOS8/8 on these fixtures only. Observed+1.51–3.17% throughput is small and needs repeat measurements. Earlier Gemma content/numerical counterexamples remain. KONI GPU and remainingATOM cases active. See experiments/2026-10-24-stream265-gemma-http.md.

## GPU HTTP c16 matrix completed (#265)

Allfour models x3modes completed:96conditions,1536token checks,1536SSE text checks. Whole/stream exact8/8 permodel. Native/whole exact Qwen06=8/8,Qwen4=4/8,Gemma=4/8(EOStrim8/8),KONI=5/8;earlier numerical/quality failures retained. KONI observed throughput+20.94–33.93%,single batches only. RemainingATOMmatrix andwidth268 probes active. See experiments/2026-10-24-stream265-gpu-http-complete.md.

## ATOM HTTP progress (#265/#267)

Three models pass whole/stream controls, eight conditions each. Qwen4 and Gemma native results also audited; Gemma remains slower than native. KONI native180s timeout retained and600s diagnostic pending. Listener and width variants are not production changes. See experiments/2026-10-24-stream265-atom-http-progress.md.


## KONI native long-wait diagnostic

Separate600s native run passes264 responses across8 conditions, with original180s failure retained. At8K/128/c16 max firstSSE253.93s and max completion263.08s; no OOM/listen drops. Whole/stream remain running. See stream265-atom-http-progress record and evidence265/http267d-koni-native-complete-summary.json.

KONI whole HTTP diagnostic:264 responses pass,8K/128 throughput7.7847(native) ->9.3113(whole)tok/s; short inputs slower. Native/whole strict and EOS parity6/8. Stream is pending on a fresh physical-card allocation; no new runtime change. See stream265 ATOM HTTP progress record.

KONI stream HTTP complete: all3 routes264 responses each audited; whole/stream8/8 exact, native/whole6/8.8K/128 native7.7847,whole9.3113,stream10.8298tok/s; stream maxE2E189.105s. All4 model B1/c16 ATOM whole/stream controls complete; native equivalence and B4 remain open.

## Intermediate-length summary correction (2026-10-30)

Fixed an evidence-summary key collision: aggregate `output_tokens` previously overwrote request length. Published native/default/PD GPU summaries now include `output_tokens_per_request` separately. Corrected the native GPU2K/128c4 table, which had selected32tokens/request; raw results and correctness conclusions are unchanged. PD GPU and existing ATOM tables were checked and already selected the intended raw conditions. Portable ATOM raw auditor/selfcheck added under evidence265:10checks, including preserving request/aggregate lengths, duplicate-arm rejection and corrupt/missing raw results. Current ATOM final-HTTP matrix remains in progress; these scripts do not certify uncompleted configurations.

## Intermediate ATOM and short/long GPU coverage (2026-10-30)

Final-HTTP ATOM native/whole/stream comparison completes all four models at512/2048 input,32/128 output,c1/c4:96 conditions,528 responses. All within-route checks pass and whole/stream references match4/4 per model. Native/PD references match2/4 for each Qwen and4/4 for Gemma/KONI; Qwen pre-EOS differences and prior quality limits remain. One ATOM lane, caches off, research artifacts; c4 is queued clients. Owned Pod cleanup and absence verified. See experiments/2026-10-30-atom-mid201-final.md.

Native GPU short/long sweep completes four models at128/8192 input,32/128 output,c1/c4:32 conditions,176 responses, all16 reference arrays equal historical c16 controls. Four child exits and coordinator exit0; GPU compute-process query empty after completion. Portable raw auditor and8 selfchecks included. See experiments/2026-10-30-native-gpu-edge201.md. ATOM short/long c1/c4, heterogeneous GPU short/long, cache-on and supported export work remain separate; the full four-model goal is not complete.
