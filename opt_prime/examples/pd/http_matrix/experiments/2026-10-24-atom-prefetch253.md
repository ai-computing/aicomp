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
