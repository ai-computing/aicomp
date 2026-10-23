# Native ATOM decode without a prefill primer

2026-10-19. Implemented `NativeAtomDecodeLane` in `opt_prime/pd/npu/native_atom.py`.
The installed Optimum decoder accepts explicit `block_tables` and `attention_mask`.
For an exclusively owned B1 static cache with one 512-token block, block 0 and a
complete per-step mask remove the need to execute native prefill to initialize
wrapper metadata. No private allocator/free-list mutations are used.

## Validation

Physical ATOM rbln0, driver 3.0.0, Rebel 0.11.0, Optimum 0.11.0.post1.
Saved RNGD prefixes from pinned Qwen2.5 checkpoints; greedy 32-token continuations,
4 prompts × 2 repeats per model. The first decode occurs before any native prefill.
A throwing prefill guard covers each candidate call. Each candidate is compared
against native prefill followed by replacement with the identical RNGD KV.
All vocabulary logits are compared, not just selected token IDs. Subsequent
requests reuse dirty slots, including decreasing input lengths.

| Model | Requests / tokens | Exact full logits | Guard calls | Candidate total s | Primer reference total s | Reference prefill s |
|---|---:|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 8 / 256 | all | 0 | 1.477171 | 1.573087 | 0.055786 |
| Qwen2.5-1.5B | 8 / 256 | all | 0 | 3.904350 | 4.077893 | 0.128905 |

These are diagnostic sequential timings, not a controlled performance benchmark.
Candidate totals include host encoding, KV admission, synchronous decode and logits
materialization; exclude model loading, RNGD prefill, network, file loading and final
logit stacking. Different conversion allocation and fixed candidate/reference order
confound attribution. Do not claim an end-to-end PD speedup from this table.

## Ownership and limitations

The caller must exclusively own the vendor model for the entire lane lifetime;
no concurrent vendor scheduler/prefill invocation is allowed. A nonblocking lane
lock rejects overlapping requests. All KV is validated/encoded before admission.
Write/decode failures poison the lane and retain buffers; recreate the runtime
instead of reusing a potentially inconsistent slot. Every layer is overwritten
before the next request; the entire attention mask is reset at each step.

This is host admission through the native public KV API, not native P2P DMA.
It is a bounded library adapter plus a hardware validator, not yet wired into
live RNGD → llm-d serving. Only B1/context512/static one-block geometry and the
pinned runtime versions are accepted. Batch scheduling, cancellation, paged KV,
long contexts, KONI and other versions remain unvalidated. Existing numerical
quality differences between RNGD and native prefill remain unchanged: parity here
means parity with the same imported RNGD KV, not FP32 or application accuracy.

Initial q05-r1 failed with a validator NameError before any decoding; preserved.
Fixed validator q05-r2 and q15-r1 passed. 37 CPU regression tests passed, including
mask reset, invalid KV rejection and retained buffers on poisoned admission.
Raw results/provenance: `validation/native-no-primer/`; exact execution commands
and source snapshots are preserved there and in the research journal.

Next: integrate this adapter behind an explicit experimental engine selection,
then validate live RNGD admission and sidecar parity before A/B/B/A performance
measurement. Keep the existing custom P2P engine as the validated serving path.
