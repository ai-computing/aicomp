# Partial-prefix length controls for page256/chunk256

- Date: 2026-10-25; related #212/#201 and PR279.
- Runtime: a3497e57ac9d8286ad20032db4c842f26272e961; one A10080GB PCIe, driver580.178.04, torch2.13.0+cu130, Transformers5.15.0.
- Frozen models and other runtime settings match README.md/summary.json in this directory. No engine/default source changes.

## Question and setup

The original128-token seed shares no complete256-token page with the later prompt. Repeat the candidate configuration with280/536-token seed prompts, which share one/two full pages with the subsequent2K prompt. The actual common-prefix length is independently recomputed from saved token IDs for each tokenizer, not inferred from character counts. This proves the prompt geometry; aggregate hit counters prove reuse occurred but do not identify every allocation's hit length.

Each of four models runs seed280/cacheoff, seed280/cacheon, seed536/cacheoff, seed536/cacheon in separate processes. Page256, chunk256, BF16,4slots, context9216. Each process first collects serial seed/2048/8192 baselines, then mixed c4/c8 bursts with32output tokens and ignoreEOS, then cancellation/disconnect/recovery and invalid-input checks. Cache history remains within each process. Cold controls run immediately before their corresponding cache-enabled arm. One run per cell; no performance conclusion from run order or burst timings.

## Audited results

16/16 lifecycle runs pass; 192/192 concurrent response arrays equal their own serial baselines. Cache-enabled serial references equal the paired current cache-off references 24/24. All16child exits checked and GPU compute-process query empty after completion.

|Model|Seed length|Common prefix with2K|Full shared256 pages|Cacheoff concurrent equal|Cacheon concurrent equal|Cacheoff/on serial equal|Cacheon hit tokens|
|---|---:|---:|---:|---:|---:|---:|---:|
|qwen06b|280|268|1|12/12|12/12|3/3|35840|
|qwen06b|536|524|2|12/12|12/12|3/3|38912|
|qwen4b|280|268|1|12/12|12/12|3/3|35840|
|qwen4b|536|524|2|12/12|12/12|3/3|38912|
|gemma|280|267|1|12/12|12/12|3/3|35840|
|gemma|536|523|2|12/12|12/12|3/3|38912|
|koni|280|268|1|12/12|12/12|3/3|35840|
|koni|536|524|2|12/12|12/12|3/3|38912|

Per-run raw hashes, diagnostic-source hashes, failures (if any), lifecycle status, prefix hits and burst times are in seed-summary.json. The summary separately compares cold2K/8K arrays against the earlier128-seed cold experiment; historical/reference drift must not be conflated with cache-on differences. Full raw prompt/token arrays stay in the local experiment artifacts and were read by the auditor.

## Reproduction

Use the same frozen runtime and portable diagnostic-copy preparation in README.md with PAGE=256. Before writing the copy, additionally replace both `(128, 2048, 8192)` tuples with `(280, 2048, 8192)` or `(536, 2048, 8192)`; replace `complete(128)`, `body(128,` and `baseline[128]` with the chosen seed. This changes both the initial warmup and the short members of the mixed bursts and recovery checks. The measured copies also save `fixture_seed` and the constructed prompt token IDs before starting the HTTP server. Import path changes are for isolated execution only.

Run each copy once without --prefix-cache and once with it, using new result paths. Keep PYTHONPATH, environment, models, slots, precision and all other flags as documented in README.md. Exact local coordinator: run_cache212_seed.py; raw filenames: cache212-seed-{model}-{seed}-{0,1}.json. Every diagnostic-copy hash is recorded per arm.

## Limits and next step

These are additional prefix-length controls on the repeated-context fixture, not proof for arbitrary content, all model families or semantic quality. Cache eviction, host reload and per-allocation reuse traces remain separate. No GPU/RNGD/ATOM transport change or production cache-default change is justified solely by these measurements. #212 and the full four-model serving goal remain open.
