# Native host shared-memory diagnostic — not P2P

User clarified that high-performance work targets the existing heterogeneous P2P
library. This experiment optimizes the separate native host-admission probe only;
it does not replace or improve the validated P2P route. No default serving change.

Added a sealed-size inherited memfd slab with a read-only consumer mapping,
request-number acknowledgement and sequential ownership. Removes NPZ I/O and
consumer stacking; device-to-host and host-to-device copies remain.

Final warmed A/B/B/A, Qwen0.5B and1.5B, B1/context512, four prompts/output32:

| Model | NPZ 8 requests s | Shared 8 requests s | Reduction |
|---|---:|---:|---:|
| 0.5B | 2.252818 | 1.916502 | 14.93% |
| 1.5B | 5.964626 | 5.018920 | 15.86% |

Request clock includes live RNGD calculation, host transfer, native admission and
32 decode steps; excludes startup and reference validation. Small controlled set,
not production throughput/SLO. Warmup included in parity:24requests768tokens per
model, all logits exact versus same-KV primer reference.38 CPU tests passed.
Initial0.5B r1/r2 lacked dedicated warmup and are preserved, not headline results.

## P2P distinction

`hetero-npu-p2p/src/hetero_p2p/atom_experimental.py` exports verified live eager
allocations at pinned thunk ABI. `qwen_atom_worker.py` supplies external KV tensor
addresses; `npu/engines.py` builds the plan and `pd/hetero_p2p.py` uses CopySession.
RNGD writes ATOM-owned exported buffers directly, then the owner marks KV updated.

Installed `rebel/kv_cache.py` native `set_kv_cache_for_layer` accepts **host** data.
The inspected native interface does not establish a device allocation/export
contract equivalent to our external-cache tensors. Native KV get/set support does
not by itself permit using those caches as P2P destinations. Do not guess static
cache addresses or bypass the exporter's live allocation checks.

Next optimization target is the existing external-cache P2P engine. Native engine
P2P needs a verified cache ownership/export adapter, separately from host staging.
