# Extending OptimusPrime heterogeneous PD

The integration lives in `opt_prime.pd`, on the `feat/optimus-heterogeneous-pd`
branch. It is an optional inference component, not a change to the original
training parallelism engine. The separate `feat/optimus-llmd` branch adds the
serving/control plane. Neither branch is the upstream main branch.

## Extension boundaries

| Layer | Files | Responsibility | New device/model work |
|---|---|---|---|
| Common lifecycle | `contracts.py`, `runtime.py` | Requests, continuation boundary, physical KV identity, fenced stages, retained owners on error | Usually none |
| Torch backend | `torch_models.py`, `torch_cache.py`, `torch_engine.py` | Dense HuggingFace model execution and explicit tensor copy | Add family policy and numerical tests; actual GPU validation still required |
| Dense NPU model contract | `npu/model.py` | Immutable checkpoint geometry, full-attention policy, QKV bias/tied-weight checks, per-head DMA alignment | Add explicit semantics; reject unsupported architectures |
| RNGD graph | `npu/rngd.py` | Model-sized biased attention/MLP prefill; device-resident BF16 KV | Implement supported compiler operations and precision policy |
| ATOM graph | `npu/atom.py` | Model-sized resident paged decode | Compile and validate exact input/cache contract |
| ATOM runtime | `npu/atom_runtime.py`, `npu/atom_device.py` | Pinned SDK runtime, physical-device binding, cache visibility | Replace version-specific implementation for a new SDK/device |
| Endpoint adapters | `npu/engines.py` | Adapt these graphs/worker to the common prefill/decode protocols | Preserve lease ownership and explicit completion fences |
| Transport | `hetero_p2p.py`, external `hetero-npu-p2p` library | Validate physical route/layout, export/import/copy completion | Implement a separately negotiated route; no implicit fallback |
| Serving | `llmd.py` on feature branch | Request validation, source preparation, serial owner, HTTP | Independent of model dimensions and vendor graph implementation |
| Experiments | `examples/pd/*qwen*.py` | Build artifacts, compare references, finite test runs and diagnostics | Extend measured cases; do not put test traffic into the serving engine |

A new model with the same explicitly supported semantics can share the graph
builders and change its immutable `DenseModel` specification. Qwen2.5-0.5B and
1.5B use the same implementation despite different hidden size, layers, GQA
ratio and head dimension. A model with sliding attention, MoE, Q/K normalization,
scaled RoPE or another cache type needs a dedicated implementation or an explicit
extension; changing its model name must not silently enable it.

A new NPU/GPU implements the existing `PrefillEngine` and/or `DecodeEngine`
protocol. A new copy mechanism implements `Transport`. It is not enough to
register a string such as `gpu:0`: the produced and consumed **physical** KVSpec
must match, including representation/layout, and returns must prove completion.
The generic runtime knows neither Qwen nor RNGD/ATOM. Importing its public API or
`npu.model` does not import a vendor SDK.

## What is still vendor-specific

This is an extensible design, not arbitrary-hardware support. RNGD currently uses
`rngd_pd.engines.fsdk` and its existing DSL primitive definitions. ATOM graph
construction reuses the established Llama projection/RMSNorm helpers; the
external runtime uses the existing tested step implementation. These optional
modules require the workspace backend dependencies; they are not yet independent
PyPI distributions.

ATOM external-KV operation is experimental and pinned to rebel0.11.0 and the
validated driver3.0.0 export ABI. The exporter uses live owned allocation records,
never guessed addresses. The main-thread RNGD DMA requirement and all lifetime
checks remain in the existing transport implementation. New hardware, drivers,
reverse directions and inter-node transfers require separate implementations and
validation; no universal P2P claim is made.

The NPU Qwen path currently uses B1, context512, compiled prefix widths32/64 and
greedy fixed-length decoding. It does not implement a production batch scheduler,
beam search, multimodal models, MoE, cancellation during in-flight DMA or general
streaming. GPU model support in the Torch adapter is an architectural path, not
an actual GPU benchmark from this work.

## Numerical and transfer contracts

Qwen attention projections include Q/K/V biases. Tied embeddings are checked and
explicitly used for lm_head. RoPE uses the checkpoint theta. The physical KV wire
representation is `dlfloat16-token-major-v1`, packed on RNGD and consumed by ATOM;
its FP16 tensor carrier is not FP16-valued KV.

Per-head DMA segments must be4KiB aligned. At head_dim64 the minimum padded
prefix is32 tokens; at head_dim128 it is16. The common planner still validates
all allocation bounds, non-overlap, descriptor grouping and completed commands.

The initial BF16 attention-score path exceeded the declared per-layer KV error
gate for1.5B (relative L2~0.14266). FP32 score accumulation/softmax in the compiled
RNGD graph reduced the maximum to~0.03582 and last-position logits relative L2
from~0.05088 to~0.01439 on the bring-up prompt. The higher-precision attention path
is now the default; the BF16 diagnostic option is retained explicitly.
This logical graph precision does not establish an NPU FP32 hardware peak.

Require three separate checks:

1. Model math against an independent FP32 reference, preserving disagreements.
2. Native graph execution on the actual target device, including physical binding.
3. P2P full-logit equality against host admission into the **same** decoder.

Check3 proves transfer correctness, not full numerical equivalence to FP32.
The finite runners retain exported owners if DMA/decode completion is uncertain.
They do not reset a board, recycle buffers after a timeout, or stop external jobs.

## Adding a backend

1. Keep the public protocol SDK-free. Put vendor imports and compiler details in
   an optional device module with a clear supported-version check.
2. Define immutable model identity and physical cache layout. Include padding,
   element representation, token axis, head/layer counts and ownership.
3. Implement and test prefill/decode continuation semantics independently of DMA.
   Verify biases, norms, positional encoding, tied weights and attention policy.
4. Implement a transport only where the SDK provides a verified allocation/export
   path. Validate all bounds and completion receipts; retain owners on failure.
5. Run finite actual-device numerical tests, then same-decoder transport parity.
   Record raw settings, compiler/runtime versions, commands, hashes and failures.
6. Connect the same endpoints to `PDRuntime`; reuse the llm-d service only after
   direct execution passes. Keep request queues separate from model kernels.

The compiler, standalone model-quality, P2P and optional llm-d runners are
separate so a failure can be attributed to its layer. `npu/provenance.py` records
small source/version snapshots; weights and compiled artifacts are excluded from
Git. The additional offline tests cover incompatible model semantics, SDK-free
imports and the head_dim64/128 alignment boundary.

## Installation and process isolation

Install the SDK-free package from a checkout with `python -m pip install
--no-deps -e /path/to/aicomp/opt_prime`. Importing `opt_prime.pd` does not require
Torch or a vendor SDK. The `torch-pd` extra pins the tested Torch/Transformers
pair; do not install that extra into the Furiosa environment, which uses a
different Torch version.

Keep RNGD and ATOM in separate vendor-prepared Python environments. In each,
install the local `opt_prime`, `hetero-npu-p2p` and `rngd_llmd` checkouts with
`python -m pip install --no-deps -e /absolute/path/to/checkout` (the Optimus
checkout path ends in `/opt_prime`). `--no-deps` deliberately preserves the SDK
stack; the environment must already contain the required NumPy, safetensors,
Transformers and device packages. This is a source integration, not an all-in-one
published wheel. Record all three Git revisions alongside SDK versions.

The ATOM worker additionally uses the transport repository's `examples/koni`
capture/export helpers, and the RNGD process needs its compiled `rngd-pdma`
helper. Supply `hetero_repo`, `helper`, `atom_python`, explicit device binding,
model revision and compiled artifact in the run config. Do not infer these from
a user's home directory. Driver/helper installation follows the transport
repository's instructions. Model weights and compiled artifacts stay external.

The checked-in validation configs are historical local paths. Copy them to a
new config, replace paths and select an idle physical card before running.
Compilation artifacts are matched against `model.json` before runtime creation.
Adding a backend should not add an unconditional vendor dependency to the public
package. Verify a clean SDK-free install separately from device execution.

## Explicit device selection in validation

`validate_qwen_rngd.py` accepts `--rngd-device-path` and
`--rngd-logical-index`; both must describe the same real device. It checks active
processes and opened device channels. Higher-precision attention now defaults
on, matching the PD engine; use `--no-accurate-attention` only to reproduce the
older BF16 diagnostic path. Historical runs retain their recorded setting.
ATOM selection accepts canonical physical names from the reported host group,
including indices above3, rather than assuming a four-card server. Idle/status,
active-context and actual opened-FD checks still apply. This naming extension
has offline tests; only physicalATOM1 was used in the new hardware validation.
Neither option changes driver groups or selects a busy card automatically.
