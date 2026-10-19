# Extending the PD engine

The public entry point is `opt_prime.pd`. Its import must remain independent of
Torch, CUDA and vendor SDKs. A new model family is not automatically a new NPU
backend: compiled graph, cache representation and ownership need separate proof.

## Module ownership

| Module | Responsibility | Must not do |
|---|---|---|
| `contracts.py` | Request, continuation, physical KV identity, engine/transport protocols | Import SDKs or infer device addresses |
| `runtime.py` | Single-flight execution and poisoned-lane owner retention | Model-specific layout conversion or service routing |
| `torch_models.py` | Explicit family support and cache geometry constraints | Load weights or silently accept hybrid/sliding caches |
| `torch_cache.py` | Optimus KVCacheManager storage and explicit tensor-copy transport | Choose a fallback after DMA failure |
| `torch_engine.py` | Model forward calls, continuation handling, greedy decoding | HTTP request handling or vendor driver calls |
| `koni.py` | Existing RNGD prefix and owned ATOM worker adaptation | Reimplement allocation/export ABI |
| `hetero_p2p.py` | Bridge to checked CopySession | Bypass completion validation |
| `llmd.py` (feature branch) | HTTP/control-plane ownership queue | Run DMA inside HTTP worker threads |

Earlier `opt_prime.pd.runtime.KVSpec` and
`opt_prime.pd.torch_engine.TorchCacheTransport` imports remain compatibility
exports. New code should import contracts from `opt_prime.pd`, and the Torch
transport from `opt_prime.pd.torch_cache`.

## Adding a Torch model

1. Inspect its physical cache type, attention policy, head geometry and exact
   `forward` contract. Similar class names are not compatibility evidence.
2. If it uses the existing dense DynamicCache layout, add its model type to
   `torch_models.py`; otherwise implement a separate cache adapter. Do not disable
   sliding attention in a user's checkpoint merely to pass the existing adapter.
3. Add seeded, offline model cases to `tests/test_pd_torch.py`. Compare every
   generated token and full logits against an independent no-cache oracle for
   full-prompt and prefix-only boundaries, short/long inputs and repeated requests.
4. Add a local pretrained-checkpoint run using `validate_checkpoint.py`. Record
   immutable revision and weight hashes; do not commit weights or credentials.
5. Test unsupported configurations explicitly. Never claim NPU support from a
   successful Torch run.

The current dense family list is GPT-2, Llama, Qwen2, Qwen3 and Mistral. Actual
sliding-window, hybrid, cross-attention and encoder-decoder caches are rejected.
A Qwen config's unused sliding-window field is allowed only when the actual
attention policy is full attention.

## Adding another device or transport

Implement `PrefillEngine` and `DecodeEngine` with the same physical `KVSpec` and
explicitly fenced returns. A prefill result must carry one of these boundaries:

- Full prompt cached: next token is already sampled; output prefix contains it.
- Prompt minus its last token cached: next token is that last input token; output
  prefix is empty and decoder must generate all requested tokens.

Implement `Transport.supports/transfer/finish` independently. `transfer` returns
only after all copies and destination visibility are established. Keep strong
allocation owners alive, including failures before a lease can be returned.
No raw pointer guessing or automatic fallback belongs in the common runtime.

Use one owner per mutable cache lane. After an uncertain device operation the
runtime is poisoned; it must not be reset by dropping its references. Device
recovery is a backend responsibility, not generic exception cleanup.

## Diagnostics and performance

`TorchCausalLMEngine(..., optimize_prefill=True, capture_logits=False)` is the
normal path. `logits_to_keep=1` avoids projecting every prompt position through
the vocabulary head. Argmax stays on the model device, with only the chosen token
materialized. This does not remove the per-token synchronization required by the
synchronous API.

Use `capture_logits=True` for parity checks. It copies full logits to CPU and is
intentionally excluded from the optimized serving path. To reproduce the old
reference path, also set `optimize_prefill=False`. These options do not alter the
KONI adapter or its mandatory experimental full-logit verification.

```bash
PYTHONPATH=opt_prime:/path/to/hetero-npu-p2p/src \
  python3 -m unittest discover -s opt_prime/tests -v
PYTHONPATH=opt_prime python3 opt_prime/examples/pd/benchmark_torch.py \
  --out /tmp/new-pd-ablation --repeats 9
```

Keep raw repetitions, compare identical weights/prompts/precision and balance run
order. Report prefill, transfer, decode and total timing separately. An improvement
on the small CPU models does not establish an improvement on the NPU testbed.

`PDRuntime.last_timings` reports seconds for fenced host calls: prefill,
reserve/validation, transfer, decode, finish, and total. The prefill adapter may
include KV packing; decode may include IPC and diagnostic logits checks. These
are not accelerator kernel times. A failed request leaves only completed stage
timings and no successful total; it still poisons and retains the lane.

## Validate a new checkpoint on actual NPUs

First establish native vendor execution independently from external-KV PD:

```bash
<atom-sdk-python> opt_prime/examples/pd/validate_atom_checkpoint.py \
  --model-dir /path/to/local/snapshot --out /path/to/new-run \
  --preflight-module-dir /path/to/hetero-npu-p2p/examples/koni \
  --chunk 64
```

This testbed runner uses physical ATOM2, vendor-owned KV, B1/context512, and
preserves source/artifact identity, physical ownership, warmups, repetitions and
failures. It loads only local weights. `--compile-only` does not open an NPU.
For measurements without compilation, pass `--artifact <build-run>/compiled`;
the runner rejects mismatched checkpoint, chunk, context or batch metadata.
Compare chunk64/128 in balanced order after compilation has finished.

Use `check_native_quality.py --model-dir ... --runs <run>/runs.jsonl ... --out ...`
for offline FP32 teacher-forced agreement on the saved NPU trajectories. Its CPU
reference time must not enter any NPU timing result. Repeatable tokens alone do
not establish numerical agreement or task accuracy.

A native checkpoint success does **not** register it with `KoniRngdEngine` or
`KoniAtomWorkerEngine`. Qwen requires Q/K/V projection biases, tied embedding
handling, checkpoint-specific layer/head dimensions and cache layouts in both
compiled graphs. Each new physical layout needs export/import/packing and fence
validation before it can use `HeteroP2PTransport`. Keep vendor-native baseline
support and end-to-end heterogeneous PD support as separate evidence columns.
