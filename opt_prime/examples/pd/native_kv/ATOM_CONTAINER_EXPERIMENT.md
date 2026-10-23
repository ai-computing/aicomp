# RNGD-to-ATOM backend validation in an allocated Pod

- **Date**: 2026-10-23
- **Task**: KV transfer backend separation (aicomp #194, PR #195)
- **Host / device**: RNGD+ATOM host; dedicated validation Pod requests one RNGD and one ATOM
- **Commit**: aicomp backend 1394760; existing optimus-rm benchmark snapshot on target

## Question

Does the portable ATOM block writer preserve hardware KV bits and RNGD-to-ATOM continuation output?

## Setup

Controller access through the accelerator host resolves the earlier Kubernetes authentication blocker. Created only namespace codex-kv194-validation / Pod kv194, with a 30-minute deadline, single ATOM and RNGD allocation, and read-only existing runtime/model mounts. Task output directory is writable. No other workloads or device-group settings changed.

Inside the Pod, existing host-group-only discovery is replaced in the validation process by a guarded mapping: exactly one exposed /dev/rbln device and exactly one runtime-visible device must exist. After loading, opened ATOM device descriptors must match the allocated physical device. This is a validation harness, not a general production discovery fix.

Qwen2.5-0.5B-Instruct; ATOM B8, context 2048, 17 KV blocks; RNGD research prefill width 256 on allocated chip 0; 12 prompts of 16–96 tokens, 16 generated tokens, remote-only PD. Native Qwen3-8B is outside this test.

## Command

Execute the retained run-pod.sh in the dedicated validation Pod. It calls container_bootstrap.py and check_atom_backend.py against the existing optimus-rm npu_pd_bench.py. Local analysis/local_prefill/native_dma/general/atom-pod/ retains exact launch artifacts; public reproduction uses role/configuration placeholders.

## Results

ATOM loaded on the allocated physical card, verified by open file descriptors. Aggregated 12-request/16-output-token baselines completed (348.0, 349.7, 359.1 tok/s on startup attempts; no PD comparison yet).

RNGD research prefill initialization aborted with device-index-out-of-range under the imported fusion=8/Mesh(2) setting. Disabling automatic vendor registration alone did not resolve it. A standalone one-element allocation reproduced the abort at fusion=8/Mesh(2), while fusion=4/Mesh(2) completed a CPU-to-RNGD-to-CPU round trip. The validation launcher now selects fusion=4 in its process-local compatibility layer. This changes the prefill execution configuration and precludes direct comparison to old fusion=8 results.

First successful portable-import run: 12 remote / 0 local prefills; 48/48 first-import cache slots match device readback exactly; 12 imports completed. ATOM aggregated 359.1 tok/s, mean TTFT 0.177s, TPOT 16.3ms. PD 56.4 tok/s, mean TTFT 2.096s, TPOT 8.3ms. Full output equality against aggregated is 9/12, first-token equality 11/12. This is not a correctness pass against native ATOM and not a speedup claim. A paired original-import run and a repeated portable run will distinguish transfer changes from prefill numerical differences.

Setup failures retained: direct controller SSH from the local machine was denied; the accelerator-host hop worked. Initial container Python imported the system Torch and failed torch_rbln ABI loading; setting HOME to the mounted runtime owner's home restored the validated Torch 2.11.0+cpu user installation. A probe used the wrong device-count method name; corrected to rebel.device_count(), which reports 1.

## Conclusion

Paired control completed on the same allocated devices/configuration:

| metric | original importer | portable importer |
|---|---:|---:|
| remote prefills / requests | 12/12 | 12/12 |
| exact first-import device readback slots | 48/48 | 48/48 |
| throughput tok/s | 59.9 | 60.3 |
| benchmark mean TTFT seconds | 1.669 | 1.637 |
| mean TPOT ms | 9.9 | 8.2 |
| mean import ms, excluding first request | 11.27 | 13.40 |
| outputs equal to ATOM aggregated | 9/12 | 9/12 |
| first tokens equal to ATOM aggregated | 11/12 | 11/12 |

Original versus portable PD outputs are token-exact for all 12 requests (16 generated tokens each). Aggregated outputs also match between runs. Thus the adapter preserves the existing RNGD research-prefill path on this sample; it does not establish equivalence of that prefill engine to ATOM-native prefill. Readback covers all 48 K/V slots of the first import, not every import.

This is a small, diagnostic paired run with first-import readback, startup/allocator variation and changed RNGD fusion configuration. No speedup claim: import overhead after the first request is ~2.13ms higher in this sample. Native Qwen3-8B->ATOM remains unvalidated. True source-side chunked export/ATOM DMA overlap is not implemented by this test.

Raw outputs and comparison summary are retained under analysis/local_prefill/native_dma/general/atom-pod/. Dedicated namespace codex-kv194-validation was deleted; no other workloads were changed.
