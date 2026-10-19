# Vendor-native engines and heterogeneous PD

Assessment:2026-10-19. Keep optimized vendor compute/scheduling where usable;
make OptimusPrime responsible for cross-vendor KV semantics, ownership and
orchestration. Do not promise native-engine throughput without a matched test.

## What is reused today

RNGD uses Furiosa CompileModule/compiler and device kernels. ATOM uses Rebel
compiled execution and vendor paged-attention with resident KV. We do not replace
the hardware compiler. However our Qwen path supplies its own graph, a B1 host
loop, prefix conversion and single-lane orchestration. It does not inherit the
full Furiosa-LLM or vLLM-RBLN scheduling/batching/sampling optimizations merely
because those lower-level APIs are used. The current worker also retains full
logits for validation, so it is not a production-speed server baseline.

## Verified integration surfaces

| Component | Evidence | Consequence |
|---|---|---|
| Furiosa-LLM2026.4 | Installed `api.py` accepts `kv_transfer_config`; `NativeLLMEngine` forwards it to the native engine | Integrate through its connector lifecycle where supported; a config argument alone is not an arbitrary external-buffer API |
| Furiosa native public stub | `generate`, `stream_generate`, `abort_request`, `reset_prefix_cache`; no general raw KV export/import method found in inspected stub | Need documented connector/block ownership contract or vendor extension, not invented Python methods |
| Rebellions0.11 runtime | `get/set_kv_cache` and per-layer variants with host tensor interfaces | A host-staged compatibility route exists in installed code; it does not prove zero-copy |
| Installed vLLM-RBLN0.11 | `RblnNixlConnector` source contains aligned host transfer buffers/HND layout | Do not treat newer documentation as installed device-to-device support |
| Host transport prerequisites | `nixl-rbln` absent; `/sys/class/infiniband` empty | Latest documented RoCE path cannot be exercised on this host as configured |
| Our tested route | Live ATOM export → RNGD PDMA write with DLFloat packing | Retain as version-pinned experimental transport; not a replacement implementation of NIXL or Mooncake |

Source paths and hashes are preserved in `validation/native-engine-optimization/native-audit.json`.
No SDK, driver, RSD group or RDMA settings were changed during this assessment.

Official references checked:

- [Furiosa2026.4 performance tuning](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/performance-optimization.html): native admission, batching and cache tuning are workload-dependent.
- [Furiosa Mooncake deployment](https://developer.furiosa.ai/latest/en/furiosa_llm/mooncake_deployment.html): provides a shared cache deployment among Furiosa-LLM pods; this does not establish RBLN KV compatibility.
- [RBLN PD connector](https://docs.rbln.ai/latest/software/model_serving/vllm_support/features/disaggregated-prefill-decode.html): latest docs describe NIXL device registration, RoCE and model/dtype/backend/layout compatibility checks. Shared connector naming does not make different physical representations compatible.
- [Furiosa Qwen2.5](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/models/qwen2.5.html): lists a vendor-precompiled0.5B model. Its public repository metadata was inspected, not executed as a native baseline here. The listing exposed artifact.json/binary_bundle.zip; loader/version and weight identity still need validation.

## Recommended architecture (proposal, not implemented native integration)

1. Keep vendor-native model execution and scheduling in separate processes.
2. Add per-vendor endpoint adapters for prepare/pin, export description, destination
   reserve/admit, resume decode, completion and release. Use vendor-supported
   hooks, not a sidecar that simply forwards two independent generate calls.
3. Negotiate model revision, weight/quantization identity, RoPE/position semantics,
   logical KV shape, physical block/shard layout, dtype/encoding and padding.
   Convert BF16/DLFloat and block layout explicitly where needed.
4. Select a validated transport per endpoint pair. Establish host staging first
   if native APIs expose only host tensors; independently validate direct-device
   registration before replacing that route. Never silently fall back mid-DMA.
5. Native schedulers retain control of block admission and batching. A pinned
   source lease lasts until transfer acknowledgement; destination becomes runnable
   only after copy completion/cache visibility. Abort must not recycle in-flight KV.
6. llm-d handles routing/request coordination. It is neither a representation
   converter nor a DMA implementation. Existing model-independent owner/service
   interfaces remain the control-plane boundary.

A shared wire contract is essential even if both engines adopt the same transfer
library. RDMA documentation for one vendor does not prove cross-vendor PCIe P2P.
Do not patch out compatibility hashes to make a handshake appear successful.

## How to measure the performance gap

Use the same pinned checkpoint, tokenizer/token IDs, actual compute precision,
context capacity, output limits and warmup. Include same-card-budget native
replicas as well as single-card diagnostic baselines. Sweep concurrency and
input/output lengths; report TTFT, TPOT p99, SLO goodput, total tokens/s and
board energy over the same interval. Separate model quality from transfer parity.

Decompose our time into source compute, packing/allocation, transfer, destination
admission, decode input preparation, vendor runtime and output materialization.
Use incremental experiments: native standalone → native host-admitted decode →
heterogeneous host transfer → validated direct transfer. This reveals whether
compute loss, scheduler underfill or interop overhead dominates. No native
throughput advantage has been measured in the current input-reuse experiment.

## Implemented improvement in this iteration

Optional `atom_reuse_inputs=true` reuses six ATOM input tensors, precomputes
RoPE/masks for context512, and keeps the B1 block index fixed. KV tensors,
compiled graph, precision, full-logit output and completion checks are unchanged.
The baseline still supplies all host-admitted reference logits even in optimized
runs. The option defaults off pending wider model/workload validation. This does
not resolve Qwen3B's pre-existing147/160 FP32 agreement.

The A/B/B/A results and raw per-step timings are recorded separately. These
intervals are host wall times, not NPU kernel profiler measurements. Async work
may be charged to output materialization; don't remove required fences based on
the apparent cost of one host interval.
