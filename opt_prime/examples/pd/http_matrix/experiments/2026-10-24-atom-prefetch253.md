# B1 ATOM next-request prefill pipeline hypothesis (#253)

- **Date**: 2026-10-24
- **Task**: B1 ATOM prefetch (aicomp #253)
- **Host / device**: Analysis of existing RNGD+ATOM host data; no new hardware run
- **Commit**: aicomp273952a; own worktree perf/253-atom-prefetch; implementation and regression tests added after RED

## Question

Can a bounded next-request prefill overlap current B1 decode without reusing a still-written shared source buffer?

## Setup

Source inspection: AtomBatcher._pd_step submits only while a decode lane is free. A B1 next-request reservation can reuse the existing single shared slot after the current request has been imported; an extra slab is not inherently required. Hold the reservation until import or acknowledged producer completion after cancellation. Admit decoder blocks only on the owner thread after the lane is free. Early prefill uses send_from=0 because the final cache state is not yet known; prefix hits can still be used during late admission but full transfer is required. Keep opt-in and remote-only for the first B1 implementation.

## Command

```bash
# RED first, then focused and full CPU validation after the GPU queue completed:
cd /home/gspark/workspace/aicomp-atom-prefetch-253
CUDA_VISIBLE_DEVICES='' PYTHONPATH=opt_prime /home/gspark/venvs/pt213/bin/python -m pytest opt_prime/tests/test_atom_batching.py -k b1_prefetch -xq
```

## Results

| config | metric | value | vs. baseline |
|---|---|---:|---:|
| scheduler regressions | RED | missing remote_prefetch option | reproduced before implementation |
| focused AtomBatcher + public factory tests | tests | 35 passed, 4.39s | CPU fake SDK with real tiny LM/KV |
| full opt_prime suite | tests | 511 passed,34 skipped,253 subtests,150.28s | one existing flex-attention warning |
| Qwen0.6B | first SSE / rest | 2.641s / 0.344s | heuristic pipeline ratio1.13 |
| Qwen4B | first SSE / rest | 4.540s / 1.310s | heuristic pipeline ratio1.29 |
| KONI8B | first SSE / rest | 4.553s / 2.223s | heuristic pipeline ratio1.49 |

Analysis: analysis/serving-matrix-201/atom253-pipeline-hypothesis.json, derived from existing #248 c1/input8192/output32 medians.

## Conclusion

The opt-in implementation and tests check second prefill starts during first decode, exact real-model outputs, reservation bound, cancellation without falsely releasing the writer, producer failure, and safe subsequent request reuse. Heuristic ratios assume perfect overlap and ignore import/resource contention; first SSE is not a pure producer stage, so these are not throughput predictions. Even ideal request pipelining is unlikely to close the Qwen0.6B measured gap by itself. Larger models offer more decode work to overlap. RED was observed before production changes. Real four-model RNGD-to-ATOM validation remains pending.

## Implemented behavior and verification

`AtomBatcher(remote_prefetch=True)` / `serve_batched.py --pd-prefetch-next` is opt-in, single-lane ATOM and remote-only. One queued request may start on the existing source slot while the current request decodes. Decoder cache admission/import remains owner-controlled. The prefetched result stays reserved until import; cancelling a client does not cancel its producer future or falsely release a still-written slot. The owner keeps draining an orphaned producer even when all clients have left. Unsupported routes are rejected before model loading. Default behavior is unchanged. No transport protocol changes or extra slab slot are required.

Commands after RED:
```bash
cd /home/gspark/workspace/aicomp-atom-prefetch-253
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=opt_prime /home/gspark/venvs/pt213/bin/python -m pytest opt_prime/tests/test_atom_batching.py opt_prime/tests/test_serving_prefill_cache.py -q
cd opt_prime
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. /home/gspark/venvs/pt213/bin/python -m pytest -q
```

The full run includes the added cancellation assertion that an orphaned producer keeps the batcher non-idle, followed by safe request-ID and source-slot reuse. Tests use a controlled future boundary to expose ordering, real TorchPrefillEngine KV, a fake Rebellions SDK import, and independent greedy outputs. Hardware throughput and a real producer/decode timeline remain unmeasured; no speedup claim. Logs: atom253-{red,cli-red,focused,full}.log in analysis/serving-matrix-201.

## Live warm-engine hardware validation started

New owned Pod `prefetch253` in namespace `codex-serving-matrix-201`, one RNGD and one assigned ATOM, CPU4requested/16limit,memory128Gi,4hourdeadline. Device bootstrap verified only its assigned ATOM opened. Prior matrix Pod was deleted. New physical assignment differs from the prior matrix, so off/on comparisons use the same warm instance rather than treating earlier absolute timings as a paired baseline.

Core3e16c33; candidate modules copied from its AtomBatcher, SubprocessPrefillWorker and BatchedService. HTTP candidate combines the previously tested9e442cd signal-safe writer with d0f6f12 readiness condition. Worker253 preserves worker234 protocol and records completed prefill+slab-write intervals using perf_counter on the same host clock as decoder calls. These are host-call intervals, not pure device kernel traces.

`atom_pipeline.py` starts the SDK/producer once, serves real nonstream HTTP, and switches remote_prefetch only between drained batches. Inputs128/8192,outputs32/128,c1/4,3repeats with off/on order alternating by repeat; no cache. Exact token references are warm off-mode outputs; native32-token references are used separately for actual normal-EOS checks. First-token timing is owner emission, not client SSE; cold load is excluded. Lifecycle tests cancel an in-flight prefetched request then check reuse, and terminate the owned producer while an already-imported request keeps decoding, checking503admission/readiness afterward.

The script contains assertions for geometry/allocation, every output, stop reasons, quiescent mode switches, cancellation, producer exit, draining, and no poisoned decoder. It has passed syntax compilation and actual SDK startup; remaining checks are pending the live run, not yet a pass claim.

Exact orchestrator command:
```bash
python3 analysis/serving-matrix-201/run_atom253.py > analysis/serving-matrix-201/atom253-queue.log 2>&1
```
Model order: KONI8B,Qwen3-4B,Qwen3-0.6B,Gemma3-1B. Expanded in-Pod arguments are in atom253-queue.log; remote wrapper run-pipeline253.sh uses the established bootstrap/SDK environment. Local raw outputs use atom253-<model>-a.json and .producer.jsonl.

Source snapshots for reproduction:
- `atom_pipeline253-initial.py` SHA256 `d8b7a0587afad6ef3713c35b215872d9dc685700bd317c72953b56bcb7be5ec0`
- `worker253.py` SHA256 `b84405509d6e3e3529d6fc2ce208463a624531a4efe62097ba8590836ff44b7f`
- `llmd253.py` SHA256 `849165a74a2046f7bc201d346201bf1fa98ef4d7ec0fe4bd71498614d5ec47fd`
- `worker_python253.sh` SHA256 `7fe175b5c13fb8a3d913d5f42af906e7916de2d5cb1cba64905e5e2cea8ca935`

## First warm KONI snapshot (incomplete)

Actual run remains live in the owned benchmark Pod. Cold load 434.679s excluded. Eight fixed-output checks and four normal-EOS checks completed. Six timed batches recorded; three-repeat matrix and fault lifecycle are still pending. For input128/output32, first paired repeat c1 off6.04447/on6.07041tok_s (+0.43%); c4 off6.07493/on8.58787 (+41.37%). Input128/output128/c1 off10.95522/on10.97215 (+0.15%). Do not generalize a single repeat to the final model result.

Correlated completed producer host-call intervals with decoder host-call intervals: measured off-mode 0/6 overlap, on-mode 3/6 overlap with union6.02735s. This includes prefill and KV export and is not device-kernel overlap proof. Snapshot tokens were rechecked against the warm off-mode reference. Analyzer union logic checked with overlapping intervals and exclusion of the same request. Files: atom253-koni-snapshot.json, atom253-koni-snapshot.producer.jsonl, atom253-koni-partial-summary.json; analyzer summarize_atom253.py under analysis/serving-matrix-201. Full results remain authoritative when the queue finishes.

Follow-up snapshot: ten timed batches completed, run still live. Input128/output128/c4 first pair off10.84835/on13.67179tok_s (+26.03%). Input8192/output32/c1 first pair off4.73979/on4.80128 (+1.30%); long-input concurrent pair still pending. Host-call overlap totals off0/11, on6/11 with15.42060s union across completed measured producer calls. No final speedup or lifecycle pass claim.

## Two-repeat KONI observation (34/48 batches)

All eight conditions have at least two completed off/on pairs. At c4, per-repeat throughput changes: input128/output32 +41.37%/+39.68%; input128/output128 +26.03%/+25.03%; input8192/output32 +31.57%/+32.92%; input8192/output128 +23.51%/+24.53%. Single-request pairs remain near neutral. Off-mode measured host overlap0/41; on-mode24/41 completed producer calls overlap another request decode, union64.2031s. Run still active; final repeat and lifecycle checks pending. Partial summary updated in analysis/serving-matrix-201/atom253-koni-partial-summary.json.

## Queue-latency analysis of the same 34-batch snapshot

Added paired latency summaries using the median of each completed batch's request medians (not a pooled percentile). At c4, off/on first-token owner-emission and E2E seconds:

| input/output | first-token off/on | E2E off/on |
|---|---:|---:|
|128/32|11.045/8.041|13.125/10.119|
|128/128|20.909/16.051|29.429/24.578|
|8192/32|14.437/11.223|16.658/13.446|
|8192/128|25.119/19.679|34.236/28.790|

These include queue waiting in the B1 decoder and two completed paired repetitions. First token is server owner emission, not client receipt/SSE. Rechecked all34timed batches' tokens against the warm off-mode references. Third-repeat/lifecycle completion remains pending.

## KONI complete: performance, correctness and lifecycle

The owned queue returned exit0 and advanced to Qwen3-4B. All48timed batches pass exact warm-off token parity;8fixed-output check groups and4normal-EOS groups pass. Three matched off/on repeats, alternating order, on the same loaded producer and decoder; cold434.679s excluded. Median tok/s:

| input | output | concurrency | off | on | change |
|---:|---:|---:|---:|---:|---:|
|128|32|1|6.074|6.070|-0.06%|
|128|32|4|6.075|8.558|+40.87%|
|128|128|1|10.939|10.936|-0.03%|
|128|128|4|10.873|13.674|+25.75%|
|8192|32|1|4.741|4.732|-0.19%|
|8192|32|4|4.783|6.293|+31.57%|
|8192|128|1|9.399|9.391|-0.08%|
|8192|128|4|9.409|11.618|+23.48%|

Lifecycle: cancelled prefetched request499, recovery request exact; producer termination fails the prefetched request500, new admission503 and readiness503. In both cases the already-imported current request retains its full128-token output; batcher drains, has no poisoned decoder, no remaining request IDs and no reserved prefetch.

Measured host-call timeline: off0/60 completed producer calls overlap another request decode; on36/60 overlap, union96.42009s. This is prefill+export host interval overlap, not device-kernel overlap proof. Raw2.23MB atom253-koni-a.json, producer trace and atom253-koni-summary.json. Analyzer rechecked all48token arrays against warm-off references. Other three models remain pending; do not generalize KONI's gain to them. B1/cache-off/width8192 research engine only; not native vendor engine.
