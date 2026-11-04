# Heterogeneous prefill/decode integration

This branch adds an optional synchronous PD execution layer to OptimusPrime.
It separates inference engines from KV transport. It includes executable Torch and KONI engine adapters. It does not replace the existing
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
   accel/pd/transport.py (Optimus glue)
                  |
   hetero-kv Session + engine (TorchCopyEngine,
   P2PBackendEngine over a checked hetero_p2p backend)
                  |
        supported driver / device pair
```

`from opt_prime import PDRuntime` and `from opt_prime.accel.pd import ...` do not
initialize PyTorch, CUDA, or a vendor SDK. Existing `Optimus_Inference` imports
and behavior remain unchanged. Run from the repository with `PYTHONPATH=opt_prime`.
KV transports (`opt_prime.accel.pd.transport`) run on the separate library
[hetero-kv](https://github.com/ai-computing/hetero-kv) (`pd` extra): it owns the
reserve -> transfer -> fence -> admit -> finish lifecycle, poisoning and explicit
routes; Optimus keeps its engine contracts and maps its caches onto hetero-kv
buffers. The RNGD -> ATOM path imports `hetero_p2p` only when a transfer runs.
Install that separate project in the same environment; no vendor implementation is vendored.

## Interface and lifetime contract

- `KVSpec`: immutable model revision, physical representation, versioned layout,
  and physical shape. Both engines and allocated caches must agree exactly.
  Include head order, sharding, padding and layout version in the layout contract.
  BF16, FP16 and DLFloat16 are distinct representations. Matching metadata is a
  prerequisite, not proof of numerical equivalence.
- `PDRequest`: request ID, tokenized prompt, total requested output token count.
- `PrefillEngine.prefill`: return a live source lease and explicit continuation.
  Full-prompt mode carries a sampled token, cache position equal to prompt length,
  and a one-token output prefix. Prefix mode carries the last prompt token, cache
  position equal to prompt length minus one, and an empty output prefix.
  The lease payload carries engine-specific cache/mask information.
- `DecodeEngine.reserve`: return an exclusively owned destination lease.
- `Transport.transfer`: return only after verified copy completion and visibility.
- `DecodeEngine.decode`: consume the continuation and return only tokens not
  already in its output prefix, possibly fewer after EOS. Fence
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
from opt_prime.accel.pd.transport import p2p_transport

# Real endpoint adapters must implement the contracts above.
transport = p2p_transport(
    checked_backend, build_plan_from_live_leases, mark_atom_cache_visible,
    source="/dev/rngd/npu3ch0", destination="atom:2", spec=prefill_engine.spec,
)
runtime = PDRuntime(prefill_engine, decode_engine, transport)
tokens = runtime.generate(request)
```

The snippet illustrates dependency injection; use the bundled KONI endpoint
adapters and explicit validation runner to bind existing hardware resources.
`build_plan_from_live_leases` must produce the existing validated `TransferPlan`:
RNGD-owned source allocation, ATOM-owned exported DMA-BUF destinations and bounded
copy spans. `p2p_transport` checks the physical format, maps the plan onto
hetero-kv buffers (one selection per DMA span) and runs it through a hetero-kv
`Session`: `P2PBackendEngine` executes each destination's spans as a validated
`TransferPlan` with the original command grouping and checks the completion receipt
as `CopySession` did; the session fences before `notify`, retains owners and
poisons on uncertainty (fenced = `CopySession` "ready", admitted = "decoding"). Other
device pairs use `SessionTransport` with any hetero-kv engine. Hardware backend
construction must retain its driver/SDK/helper ABI checks. The transfer adapter
never discovers raw addresses or attempts reverse import.

Current tested hardware support belongs to the separate library: same-host
RNGD → ATOM on its explicitly supported ABI. See the validation record for the new integration's measured scope. Other NPU pairs, reverse transfers and network transports
require implementations and hardware validation. No automatic host fallback exists.

## Implemented engine adapters

`TorchCausalLMEngine` wraps an already loaded dense GPT-2, Llama, Qwen2, Qwen3 or Mistral causal model.
It exports/imports actual KV through OptimusPrime's existing `KVCacheManager` and
uses explicit `TorchCacheTransport` tensor copies. This transport is selected by
the caller, never substituted for failed P2P. It supports single unpadded requests,
greedy or seeded sampled output (`SamplingParams`: temperature, top-k, top-p, seed
per position, so prefill and decode stages on different hosts continue one sequence),
early stop at stop token IDs, full-prompt or prefix-only prefill. Sliding attention,
quantized KV and sharded models are rejected or outside this adapter's contract. CPU validation uses independent no-cache full-sequence recomputation.
CUDA fences are implemented but CUDA hardware has not been validated.

`KoniRngdEngine` exports the existing bounded DLFloat prefix payload.
`KoniAtomWorkerEngine` reuses the existing ATOM allocation owner over a local Unix
socket. It carries the exact prompt boundary into the fenced iteration command,
requires matching generation/request/slot and mandatory full-logit reference
checks, and only returns after the worker's explicit device completion. The
transport builds its plan from live allocations. Device acquisition, export and
ABI validation remain with the existing library/worker, not a second driver layer.

The generic runtime is one lane, not a throughput scheduler. KONI validation uses
one active slot of a compiled B4 graph; it does not establish maximum utilization.
The hardware validation entry point is `examples/pd/validate_koni.py`; it requires
explicit artifact/config paths and the external hetero-npu-p2p repository. Its
quarantine file means uncertain allocations are deliberately retained in a live
process; do not kill it as a routine timeout cleanup.

## Installation and validation environment

```bash
python -m pip install -e ./opt_prime  # SDK-free contracts
# KV transports (hetero-kv; builds from source, needs a Rust toolchain):
python -m pip install -e './opt_prime[pd]'
# Optional isolated Torch validation environment (not the NPU SDK environment):
python -m pip install -e './opt_prime[torch-pd,pd]'
```

The Torch validation extra pins the versions actually tested. Keep vendor SDK
Python environments separate; the core package has no mandatory torch dependency.
KONI uses the existing RNGD Python environment and a separate ATOM owning process.
An SDK environment that streams layers (`pd.prefill_server`) or runs a transport
also needs hetero-kv installed.
The hardware runner accepts `--config` and `--hetero-repo` explicitly rather than
embedding local artifact paths in the engine package.

## Remaining integration boundaries

- Existing `Optimus_Inference` PP/TP shard ownership and rank collectives need a
  dedicated adapter before distributed CUDA cache handoff is supported. The
  current Torch adapter uses Optimus cache management, not its distributed scheduler.
- KONI output parity here compares P2P admission with host admission into the same
  decoder. Existing divergence from vendor-native token trajectories is unresolved.
- llm-d service lifecycle/streaming/cancellation and Kubernetes deployment belong
  on a separate feature branch layered on this engine. No llm-d dependency is
  introduced into the common runtime.
- Multi-lane batching and new models/transports need their own capability and
  parity validation; adding an endpoint name does not enable new hardware.

A generic boundary allows each transport to keep its own constraints: the RNGD
64 MiB source limit, alignment and command grouping remain in its backend, not in
OptimusPrime's common interface. Conversion between representations should be an
explicit, independently validated adapter rather than a transport side effect.

## Validation

```bash
PYTHONPATH=opt_prime python3 -m unittest discover -s opt_prime/tests -v
# Also exercise the hetero_p2p plan path with a fake CPU backend:
PYTHONPATH=opt_prime:/path/to/hetero-npu-p2p/src \
  python3 -m unittest discover -s opt_prime/tests -v
```

Tests include 24 actual GPT-2/Llama parity cases and cover request reuse, output-token accounting, incompatible revisions,
unsupported routes, transfer/decode failures, owner retention, lazy imports and
P2P completion-before-notify ordering on the hetero-kv session. The small randomly initialized CPU models test execution parity, not task quality.
Contract fixtures and CPU tests do not measure NPU performance, MFU or power.

Measured results and exact scope: [validation record](validation/RESULTS.md).

Contributor guide and module boundaries: [CONTRIBUTING_PD.md](CONTRIBUTING_PD.md).
