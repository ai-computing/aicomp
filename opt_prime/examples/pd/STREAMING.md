# RNGD layer streaming (opt-in design note)

`serve_batched.py --pd-prefill-stream-layers` lets a subprocess prefill producer
(`--pd-prefill-python`) export KV one layer or layer group at a time while a Torch
or ATOM decoder imports each finished span. It is off by default, and whole-KV
transfer stays the default path.

## Protocol

- The producer advertises `{"ready": true, "kv_progress": 1}`. A request with
  `stream_layers: true` gets `{id, kv_span: [layer_start, layer_end, token_start, token_end]}`
  messages, then the normal completion message.
- Each span is copied into the request's reserved shared-memory slot before it is
  announced. Published spans must not change until the request completes.
  Producers process requests serially, cancelled ones included, so an old request
  cannot overwrite a reused slot.
- `LayerProgress` rejects malformed or overlapping spans. At completion, every
  layer must cover the requested suffix exactly.

## Decode side

- `PagedKV.write_layer_span` writes into reserved pages only. Length, prefix-cache
  entries and recurrent state are published after completion and full coverage,
  and only once destination copies are fenced.
- `PDBatcher` (Torch) imports spans as they arrive.
- `AtomBatcher` imports only into a reserved lane. A B1 prefetched request
  (`--pd-prefetch-next`) may compute ahead, but its spans wait until the current
  decode releases the lane. Multi-lane hybrid caches defer imports until admission.
- An invalid source fails only its request. An SDK write failure poisons the owner.
  Incomplete prompts never enter the reusable prefix index.
- Recurrent (linear-attention) models are rejected at setup by the CLI,
  `SubprocessPrefillWorker` and `PDBatcher`, because their state is not part of
  the stream.

## Producers

- `prefill_server --engine rngd-full` streams through `_RngdFull.prefill_stream`.
- `--engine factory --engine-factory module:callable` loads a research engine. It
  is a trusted flag: the callable is imported and run as-is, and the checkpoint is
  first loaded into host memory as fp32 (4 bytes per parameter).
- The research factories used in #266 (`streaming_factories.py`) are not part of
  this package, because they depend on example modules that are not on this branch.

## What the measurements show

#266 measured end-to-end completion-time reductions against whole-KV transfer:
-20% to -35% for Qwen3-0.6B, Qwen3-4B and KONI-8B, and -2% to -10% for Gemma3-1B.
Most of that gain does **not** come from overlapping transfer with prefill.

A notification-delay control kept per-layer export but held every progress
message until the source finished:

| model | gain with delayed notification (deferred vs whole) | extra gain from early sending (early vs deferred) |
|---|---:|---:|
| Qwen3-0.6B | 22.7% | 4.6% |
| Qwen3-4B | 19.6% | 3.0% |
| Gemma3-1B | 2.8% | 0.4% |
| KONI-8B | 21.3% | 1.7% |

Most of the gain is still there when notification is delayed. It comes from the
research factories' slower whole-result export, which the per-layer export path
avoids, not from overlap. This change makes no claim that overlap improves
performance.
