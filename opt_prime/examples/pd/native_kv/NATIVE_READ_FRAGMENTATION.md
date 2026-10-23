# Native KV read fragmentation and an integration boundary

Issue: ai-computing/aicomp#192. Baseline engine source1aad009, SDK/native2026.4.0,
Qwen3-8B-FP8 revision220b46e3b2180893580a4454f21f22d3ebb187d3, TP8/max2048 FXB
SHA25636d74b754f580bea893c68cc70147c615d1a9c4273a4580dbebaf87287400713.
Measured on a GPU+RNGD host, using one dedicated RNGD. No native binary patch.

## Subsequent integration experiment

The earlier source/API audit below is historical. Submission and completion boundaries were subsequently located in the measured binary, and a build-pinned in-process adapter was tested without a debugger. The [reverse-only serving experiment](queue_adapter/SERVING_RESULTS.md) improved isolated handoff but not HTTP. Subsequent [partial physical-run coalescing](queue_adapter/PARTIAL_RESULTS.md) reduced mixed HTTP TTFT p95 by 13–14% in the measured ABBA comparison. The adapter remains experimental and opt-in; a supported vendor interface is still unavailable in the inspected installation.

## Evidence

At1024 input tokens, full1023-position KV export alternates between roughly40ms and
110ms after native readiness. Worker affinity,100ms idle delays, source offsets1/2/16
and nearby prompt lengths do not remove it. Interleaving a non-transfer native request
locks the measured requests into either fast or slow behavior depending on phase.
Changing destination block-ID order does not remove it. Unique-prefix requests with
prefix caching enabled remain slow; cache-off repeats alternate. These controls
implicate source allocation/order rather than a permanently slow DMA worker.

Read-only inspection of owned native submission rings resolves read fds to RNGD DMA
channels4–7. Slow requests add72 read entries with1023 segments each; fast requests add
72 single-segment KV reads, plus one other read. Internally consistent retained segment
arrays have lengths2048, source deltas-2048 and destination deltas+2048. This explains
how reverse-contiguous source blocks become many tiny transfers. Descriptor arrays may
already be freed after completion; inconsistent arrays were excluded. Retained snapshots
are not proof of every command's lifetime, and do not identify the allocator algorithm.
No raw KV or process memory snapshots are included here.

## Measured mechanism

[dma_probe](dma_probe/README.md) contains an independent owned-buffer hardware probe,
not a native-engine patch.126/126 outputs match the independent byte oracle;2unit tests
pass.20 warm samples per method/size, alternating AB/BA; total includes CPU restoration.

|2KB blocks|Reverse scatter ms|Contiguous read plus restore ms|
|---|---:|---:|
|127|0.23670|0.05611|
|511|0.87672|0.13593|
|1023|1.59404|0.27455|

The1023-block microbenchmark improves5.81x. This is not native TTFT/throughput evidence.
Numeric results are in coalesced_read_summary.json; source/runtime identities are in
coalesced_read_manifest.json. The tested local runtime is not assumed identical to the
published0.9.0 crate; reproducibility requires the identified local dependency.

## Required producer integration

The installed native extension embeds its engine: ELF dependencies contain system
libraries, not a separately replaceable Furiosa DMA runtime. The inspected Python
NativeLLMEngine interface has a serialized kv_transfer_config but no DMA callback or
external connector implementation hook. Local workspace inspection found no native
connector Rust source to patch. This is a limitation of the inspected installation,
not a claim that reverse engineering is prohibited or that no vendor interface exists.

A follow-up owned native run checked loaded mappings both before and after four real
KV exports (4/4 full hashes equal). The Furiosa extension mappings included llm_native
and native_torch; neither librenegade.so nor libmooncake_store.so, whose names occur in
the binary, was loaded in this TcpConnector path. ELF NEEDED inspection alone would
not have excluded dynamic loading; the live check narrows that gap. No replaceable
DMA library or coalescing configuration was established by this audit.

The proposed producer change belongs before DMA submission, in the code generating
read payloads (traces name DMAConnector::dma_read_payload and scatter-fragment planning):

1. Validate a reverse-contiguous run with equal block sizes and checked address bounds.
2. Read its physical ascending interval into owned staging, then restore logical token
   order. Preserve the existing path for unsupported/noncontiguous layouts.
3. Keep source allocation and staging alive through device completion, including request
   cancellation; do not publish/ACK before restoration. Drain owned transfers on shutdown.
4. Compare every exported KV slot against the existing producer, then run real HTTP
   arrival tests, cancellation/recovery, memory bounds and generation parity.

The probe awaits all transfers and has no serving cancellation path. It must not be
presented as a ready-made async adapter. Changing the Python receiver or this probe's
runtime cannot remove descriptors already submitted by the native producer. Native
performance remains unmet; no serving defaults are changed by this contribution.
