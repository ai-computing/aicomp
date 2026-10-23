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
