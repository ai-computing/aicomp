# Fixed-KV pause invariants and native concurrent cache differences

- Date: 2026-10-22
- Task: aicomp #150 / PR #151
- Commit: 3b2e12c parent plus same_cache_pause_manifest.json
- Hardware/software/model: NATIVE_PACK.md pinned Qwen3-8B-FP8, RNGD2026.4.0 native
  prefill and A10080GB PCIe Marlin GPU decode. No runtime optimization in this unit.

## Question

The strict live worker test compared outputs of separate native prefills and failed
on both Torch and NumPy packing. Does changing only the GPU pause schedule alter
output or valid KV, or do the native input caches already differ?

## Commands

Dedicated native endpoint as GPU_LOCAL_PAIRED.md; no GPU HTTP server:

```bash
python3 check_same_cache_pause.py --model-dir "$MODEL_DIR" --out same_cache_native_variation.json
python3 check_native_batches.py --out native_batch_layout.json
```

First diagnostic run covered shared/mixed caches only; its4096 matching positions
are retained locally in same_cache_pause.json. Expanded final script adds native
concurrent-prefill case and is the evidence summarized below.

## Method and results

One fixed16-lane graph, adaptive-single disabled,128 output tokens per lane.
Compare uninterrupted decoding against pausing lane0 after20 outputs while other
lanes finish, then resuming it. Check lane0 position, input ID and remaining count
throughout the pause and compare every element of its valid KV prefix at pause end.
The current uncommitted position is intentionally excluded from the prefix invariant:
inactive graph lanes may recompute it, but it is not part of the logical cache yet.

Cases:
1. One native512-token cache copied into all16 lanes.
2. Fixed native512/128-token caches alternating across16 lanes.
3. Eight native concurrently computed512-token caches, each reused in two lanes.

All three cases:2048/2048 token positions match paused/unpaused; total6144/6144.
Paused position/input/count and valid KV prefix remain unchanged. Lanes sharing
exactly the same packet produce the same output. Each paused run takes236 decode
steps rather than128, so it exercises a substantial stall rather than a no-op pause.

Native concurrent case uses eight identical prompts `[1000]*512`. Relative to packet0,
packets1..6 are not byte-identical; packet7 is identical. First differing slot is2
(second-layer K in the measured interleaved K/V layout). These eight packets yield
two different128-token output sequences. Both sequences reproduce with the same
fixed packet regardless of the pause schedule. This demonstrates that cross-peer
output equality is not a valid unconditional oracle when separate native computations
supply different caches. It does not identify the vendor-internal cause or prove
all possible live HTTP scheduling executions correct. Prior failed logs remain.

The measured native512-token transfer has18 validated batches; all batch slots and
wire-layout regions are contiguous (see native_batch_layout.json). No KV payload,
weights or vendor binaries are saved/published. This supports a future direct pinned
receive fast path but does not establish other builds/models use the same layout.

## Conclusion

The isolated GPU pause path passes stronger fixed-input invariants in these cases.
The previously failed cross-peer check conflated separately computed native inputs
with pause/copy correctness; new evidence separates those questions without claiming
native bitwise determinism. Retain original failures as numerical variability evidence.
Next optimize validated receiving into pinned storage and measure actual latency;
full native-performance goal remains active. All owned processes stopped.
