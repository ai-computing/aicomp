> Historical native-execution record. Subsequent actual Qwen RNGD→ATOM P2P
> results are recorded in [Qwen P2P validation](QWEN_PD_VALIDATION.md).

# Actual NPU model validation — 2026-10-19

The new checkpoint tests ran on physical ATOM2. KONI ran on physical RNGD3 →
ATOM2 with the existing checked P2P route. These results supersede no earlier
CPU result: they measure different implementations and hardware.

| Checkpoint | Native ATOM | Native RNGD FXB build | Heterogeneous P2P PD |
|---|---|---|---|
| KONI-Llama3.1-8B | Previously measured | Previously measured | Revalidated:24 requests,396 output tokens |
| Qwen2.5-0.5B-Instruct | New:72 measured runs | Dry-run rejected: no registry entry | Not implemented/validated |
| Qwen2.5-1.5B-Instruct | New:72 measured runs | Dry-run rejected: no registry entry | Not implemented/validated |

The FXB rejection applies to these unquantized checkpoints in installed SDK2026.4.
It does not prove a hardware limitation or that all Qwen variants are unsupported.
The runner uses the installed Optimum RBLN Qwen2 implementation, not the Torch PD
adapter executed on CPU. Physical NPU ownership snapshots are retained.

## ATOM compile-setting comparison

B1/context512; inputs32/128/256; outputs8/32; greedy fixed output counts. Compare
prefill chunks64/128 in64/128/128/64 order after all compilation has finished.
Each process warms each shape once, then measures three repeats: six samples per
model/input/output/chunk condition,144 measured runs and48 warmups in total.
The18 preliminary measurements overlapping compilation are retained separately.

For inputs128/256, chunk128 reduced median prefill time by approximately25–36%.
Whole-generation speedup was only1.011–1.105x in these conditions. For input32,
chunk128 was slower: total speedup0.940–0.994x. There is no universal winner;
retain a short-prompt graph if this native baseline becomes a serving backend.
The0.5B long-input decode samples also show timing spread, so a small total
difference should not be treated as a robust throughput improvement.

For example,1.5B with input256/output32 changed from61.95 to46.16ms prefill and
544.47 to527.50ms total. Its median single-request throughput was about58.8 to
60.7 output tok/s. This is not batched saturation throughput.

All six unique prompt/output trajectories per model produced identical tokens
across chunks and repetitions. Offline FP32 teacher-forced top1 agreed120/120
positions for each model. This is a numerical diagnostic on synthetic repeated
text, not a downstream task-accuracy evaluation or full-logit equality claim.

Environment: driver3.0.0, rebel-compiler0.11.0, optimum-rbln0.11.0.post1,
torch2.11.0+cpu and transformers5.8.1. The Torch package name does not mean that
the compiled vendor runtime runs on CPU. The artifact declares BF16; no native
BF16 hardware peak or cross-vendor arithmetic equivalence is inferred.

Full matrix and scatter plots: [results](validation/npu-models/analysis/RESULTS.md),
[graph](validation/npu-models/analysis/native-atom.svg).

## KONI stage profile

The B4 graph has one live lane, input lengths140/167/227/271, output1/32,
three repetitions. All24 requests passed same-decoder full-logit verification,
all1536 DMA commands completed successfully, and the worker exited normally.
No dedicated timed-loop warmup was excluded in this diagnostic profile.

| Output count | Prefill incl. packing | Transfer call | Decode incl. verification | Total |
|---:|---:|---:|---:|---:|
| 1 |59.62ms|17.25ms|74.02ms|150.67ms|
| 32 |57.21ms|16.45ms|2348.70ms|2425.44ms|

Each entry is the independent median of12 requests, so columns need not sum.
These are fenced host-wall call durations: transfer includes planning/import
and completion handling, while decode includes worker IPC and full-logit checks.
They are not isolated kernel times. This profile suggests prioritizing useful
decode batching and separating verification cost before optimizing the16ms
transfer stage. It does not establish that ATOM hardware alone consumes2.35s.
The generic single-flight runtime still does not replace the existing batched
scheduler, and this is not a saturated PD-vs-native comparison.

## Implementation boundaries and next extension

Added independent native-checkpoint, reference-quality and result-analysis
scripts under `examples/pd`, plus `PDRuntime.last_timings`. The common module
still imports no SDK; failures retain owners and never silently fall back to CPU.

New Qwen heterogeneous PD requires checkpoint-specific prefill/decode graphs
with Q/K/V biases, tied embeddings and geometry, then a validated physical KV
packer/exporter. For0.5B this means48 K/V tensors,2 KV heads,head_dim64; for1.5B,
56 K/V tensors,2 heads,head_dim128 (read from the pinned checkpoint). KONI's
64 tensors,8 heads,head_dim128 cannot be relabeled as either model. Native vendor
KV cache ownership alone is not permission or an API for external P2P admission.

Small raw evidence, hashes, commands, source snapshots, failures and ownership
records are in `validation/npu-models`. Weights, compiled binaries and large KV
arrays remain in the local experiment directory and are not committed. All NPU
processes started by these experiments exited; post-run device snapshots are
included. Power/MFU and concurrent arrival distributions were not measured here.
