# Heterogeneous prefill/decode integration

This branch adds an optional synchronous PD execution layer to OptimusPrime.
It separates inference engines from KV transport. It does not replace the existing
PyTorch training or pipeline-parallel inference implementation.

## Architecture

```text
Caller / future llm-d service adapter
                 |
           PDRuntime (one lane)
             /           \
    PrefillEngine       DecodeEngine
             \           /
               Transport
                  |
       optional HeteroP2PTransport
                  |
     hetero_p2p.CopySession + checked backend
                  |
        supported driver / device pair
```

`from opt_prime import PDRuntime` and `from opt_prime.pd import ...` do not
initialize PyTorch, CUDA, or a vendor SDK. Existing `Optimus_Inference` imports
and behavior remain unchanged. Run from the repository with `PYTHONPATH=opt_prime`.
The optional P2P bridge imports `hetero_p2p` only when constructed. Install that
separate project in the same environment; no vendor implementation is vendored.

## Interface and lifetime contract

- `KVSpec`: immutable model revision, physical representation, versioned layout,
  and physical shape. Both engines and allocated caches must agree exactly.
  Include head order, sharding, padding and layout version in the layout contract.
  BF16, FP16 and DLFloat16 are distinct representations. Matching metadata is a
  prerequisite, not proof of numerical equivalence.
- `PDRequest`: request ID, tokenized prompt, total requested output token count.
- `PrefillEngine.prefill`: return a live source lease and the first sampled token.
  The lease payload carries engine-specific cache position/mask information.
- `DecodeEngine.reserve`: return an exclusively owned destination lease.
- `Transport.transfer`: return only after verified copy completion and visibility.
- `DecodeEngine.decode`: consume the first token, return only subsequent tokens,
  possibly fewer after EOS. For a one-token request, return an empty list. Fence
  all device accesses before returning. No asynchronous background access is allowed.
- `Transport.finish`: release its successful session lease after decode completes.

Each runtime is single-flight and rejects concurrent entry. Separate lanes require
separate engines/cache ownership and transport sessions; sharing a decoder's cache
between lanes is not safe. The endpoint adapter owns explicit cleanup and pool
recycling. Successful return drops runtime lease references; it does not free
vendor allocations. Endpoint methods that fail before returning a lease must
retain their own partially initialized resources.

Any execution exception poisons the lane and retains acquired leases. There is
no automatic retry/reset/free. Keep the runtime alive until a backend-specific
recovery process proves DMA and decoder activity have stopped. Process exit or
object garbage collection is not a recovery procedure.

## Connecting the current RNGD → ATOM implementation

```python
from opt_prime import PDRuntime
from opt_prime.pd.hetero_p2p import HeteroP2PTransport

# Real endpoint adapters must implement the contracts above.
transport = HeteroP2PTransport(
    checked_backend, build_plan_from_live_leases, mark_atom_cache_visible,
    source="rngd:3", destination="atom:2", spec=prefill_engine.spec,
)
runtime = PDRuntime(prefill_engine, decode_engine, transport)
tokens = runtime.generate(request)
```

These names are integration dependencies, not bundled hardware implementations.
`build_plan_from_live_leases` must produce the existing validated `TransferPlan`:
RNGD-owned source allocation, ATOM-owned exported DMA-BUF destinations and bounded
copy spans. The bridge checks the physical format and delegates completion receipt,
owner retention and decode sequencing to the real `CopySession`. Hardware backend
construction must retain its driver/SDK/helper ABI checks. The transfer adapter
never discovers raw addresses or attempts reverse import.

Current tested hardware support belongs to the separate library: same-host
RNGD → ATOM on its explicitly supported ABI. The new integration has only CPU
contract validation. Other NPU pairs, reverse transfers and network transports
require implementations and hardware validation. No automatic host fallback exists.

## Migration work still required

1. Extract KONI's manual RNGD prefill/source-packing operations into a
   `PrefillEngine`; preserve model revision and numerical-mode provenance.
2. Extract the ATOM external-KV worker into a `DecodeEngine`; carry valid token
   positions, padded context, first token, slot IDs and synchronization into its
   lease payload. Keep compiled artifact identity and capacity constraints.
3. Bind the existing live-allocation plan builder and post-copy cache visibility
   callback. Validate output parity and compare against the old worker before
   measuring throughput. Existing cross-native token divergence is unresolved;
   this interface change does not fix it.
4. Build an adapter for Optimus `KVCacheManager` only after defining PP/TP shard
   ownership, cache positions and rank collectives. `Optimus_Inference.forward`
   is not a ready-made portable cache import/decode-resume API. Its existing
   `generate` path remains independently usable.
5. Expose request admission/cancellation and streaming through a service adapter
   for llm-d. Current code provides no Kubernetes/EPP or multi-node integration.

A generic boundary allows each transport to keep its own constraints: the RNGD
64 MiB source limit, alignment and command grouping remain in its backend, not in
OptimusPrime's common interface. Conversion between representations should be an
explicit, independently validated adapter rather than a transport side effect.

## Validation

```bash
PYTHONPATH=opt_prime python3 -m unittest discover -s opt_prime/tests -v
# Also exercise the real optional CopySession with a fake CPU backend:
PYTHONPATH=opt_prime:/path/to/hetero-npu-p2p/src \
  python3 -m unittest discover -s opt_prime/tests -v
```

Tests cover request reuse, output-token accounting, incompatible revisions,
unsupported routes, transfer/decode failures, owner retention, lazy imports and
P2P completion-before-notify ordering. The CPU fixtures are not inference models
and do not measure hardware performance, accuracy, MFU or power.
