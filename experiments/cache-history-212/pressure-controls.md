# Device-cache pressure and host reload controls

- Date: 2026-10-25; related #212/#201 and PR279.
- Runtime: a3497e57ac9d8286ad20032db4c842f26272e961; A10080GB PCIe, driver580.178.04; torch2.13.0+cu130, Transformers5.15.0.
- Same four frozen models as summary.json; no production engine/default changes.

## Setup

One execution slot, BF16, context9216, page256/chunk256,36device pages. A request with8192input and32output tokens reserves33pages. Two generated prompts A/B have different first256-token blocks. Sequence A→B→A→A→B→A; greedy and ignoreEOS. Each model runs cacheoff, device-cache, then device+4GiB host-cache in separate processes. The host tier must have at least64blocks to hold both prompt block sets. This tests device pressure and host reload; it does not force host-tier eviction.

Record per-admission alloc return values without changing the allocator's inputs/results. After every request drains, record live page references, cache counters, removed events and host offload/reload counters. Assert that B evicts A, device-only A revisit reuses0tokens, immediate warm A reuses7936tokens, and host A revisit actually reloads blocks and reuses7936tokens. Each completed row must release all live page references. Hardware execution is one slot; no concurrency/batch scaling claim.

## Audited results

All12processes complete and pass pressure/admission/readiness checks. 72/72 saved response arrays equal their own cold A/B reference. Device/host arms match paired cacheoff arrays 48/48; their A/B reference arrays match 16/16. The independent auditor also rejects a negative control with the recorded reload increase removed. All12child exit statuses are0. Reviewed on2026-10-30; an unrelated GPU workload is currently active and was not modified. No contemporaneous global-idle snapshot is claimed.

|Model|Mode|Within-arm tokens equal|Device removed events|Host offloaded blocks|Host reloaded blocks|Host hit tokens|
|---|---|---:|---:|---:|---:|---:|
|qwen06b|off|6/6|0|0|0|0|
|qwen06b|device|6/6|125|0|0|0|
|qwen06b|host|6/6|119|61|87|22272|
|qwen4b|off|6/6|0|0|0|0|
|qwen4b|device|6/6|125|0|0|0|
|qwen4b|host|6/6|119|61|87|22272|
|gemma|off|6/6|0|0|0|0|
|gemma|device|6/6|125|0|0|0|
|gemma|host|6/6|119|61|87|22272|
|koni|off|6/6|0|0|0|0|
|koni|device|6/6|125|0|0|0|
|koni|host|6/6|119|61|87|22272|

The expected admission reuse sequences are off:[0,0,0,0,0,0], device:[0,0,0,7936,0,0], host:[0,0,7936,7936,7936,7936]. The final prompt page is recomputed to produce logits;8192cached tokens are neither expected nor claimed. Host-hit tokens count only restored host blocks, so they are lower than total reused tokens that can also include surviving device pages.

Per-run hashes, source hashes, actual capacities/counters, admission reuse and individual E2E timings are in pressure-summary.json. Timings are single sequential samples and are not a stable speedup estimate. Full arrays, before/after counters and prompts were independently audited from raw artifacts. Run the included `verify_pressure.py --input "$RESULT" --mode host` (or off/device) to recheck individual raw results; cross-mode comparisons are separately recorded in the summary.

## Reproduction

`pressure_http.py` is the measured diagnostic fixture with the runtime import path made portable (AICOMP_RUNTIME) and an existing-output guard added before model/device initialization. It uses existing build/HTTP/cache interfaces and wraps alloc solely to record returned counts. Set environment/model to the frozen versions above. The output filename must be new; the local coordinator enforced this before every arm.

```bash
export AICOMP_RUNTIME=/path/to/runtime-checkout
export PYTHONPATH="$AICOMP_RUNTIME/opt_prime:$AICOMP_RUNTIME/opt_prime/examples/pd"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON" pressure_http.py --model "$FROZEN_MODEL" --output "$NEW_OFF_RESULT"
# Device mode: add --prefix-cache with a new output path.
# Host mode: add --prefix-cache --host-kv-gib 4 with a new output path.
```

Exact site-local argv/exits/source hashes: cache212-pressure-queue.json. Raw names: cache212-pressure-{model}-{off,device,host}.json. Large models/data and site-specific paths/logs are not published.

## Limits

This establishes measured eviction/reload coverage for the two generated8K prompts in this one-slot GPU configuration. It does not test host-cache capacity eviction, concurrent transfers from multiple slots, faults during reload, semantic quality or NPU/heterogeneous paths. No new DMA overlap or default-setting safety is inferred. #212/#201 remain open for broader supported integration and correctness work.
