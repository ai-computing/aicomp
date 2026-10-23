# Backend separation validation

- Base: native DMA PR #193 (`a6f92cb`), combined with main `fd89265`.
- Related issue: #194. This is a draft; a limited RNGD research-prefill to ATOM paired validation has now passed adapter parity (see below).
- Local hardware: RNGD device 0 and A100 80GB PCIe GPU 0; native SDK 2026.4.0.
- Remote target: RNGD+ATOM host, ATOM runtime Rebel 0.11.0, optimum-rbln 0.11.0.post1.

## Automated checks

77 native-KV unittest cases pass, including CUDA byte equality with non-Qwen geometry, queue cancellation, ATOM chunk transposition/bounds, and existing source/protocol/lifecycle cases. CUDA pinned-memory calls emit existing PyTorch deprecation warnings; explicit CUDA selection is required with the registered RNGD backend.

The full opt_prime pytest run has 84 passes and 2 failures:
- `tests/test_kv_cache_batch.py::test_kv_cache_generation_matches_hf[1]`
- `tests/test_kv_cache_batch.py::test_kv_cache_generation_matches_hf[2]`

Both fail with an unregistered `transformers.cache_utils.DynamicCache` output pytree. Both reproduce on clean main `fd89265` in the same environment, without this change.

The portable C planner test uses 64-byte segments; adapter tests and ASan/UBSan pass, including reordered/partial/variable-length runs and completion cleanup.

## RNGD to GPU

The existing 15-case HTTP quality suite compares native, GPU non-streaming and GPU streaming outputs on Qwen3-8B-FP8. All 15 cases have equal text across the three paths. Strict expected-format grading is 12/15 on each path. The executable validation launcher and raw logs are retained in the local research record. No new latency or throughput improvement is claimed by this refactoring.

## RNGD to ATOM attempt

The existing Qwen2.5-0.5B RNGD research-prefill / ATOM B8 path was selected because matching compiled artifacts already exist. The new per-layer import adapter was installed only in the validation process; no import completed. Startup failed at physical-device mapping before model loading: all ATOM devices were in nonzero container groups, while this engine requires host group 0. An existing Kubernetes configuration was tried for the prior container validation route, but the API rejected authentication. Device assignments were not changed.

Therefore ATOM unit behavior is verified, but hardware byte readback, continuation correctness and serving measurements are pending authorized container access. Native Qwen3-8B to ATOM additionally needs compatible ATOM compiled weights and a validated native KV representation; the 0.5B research-prefill check would not establish that path.

## Follow-up: allocated container validation

The controller-hop access supplied by the user resolved the runtime access blocker. A dedicated one-RNGD/one-ATOM Pod was created and removed after testing; other workloads were untouched. See [full experiment](ATOM_CONTAINER_EXPERIMENT.md) and `atom_container_summary.json`.

Both importers passed 48/48 first-import device readback slots. All 12 PD outputs (16 tokens/request) match between the original and portable importer. Both agree with ATOM-native aggregate prefill on only 9/12 full outputs and 11/12 first tokens; this shared difference is not introduced by the new adapter. Throughput was 59.9 versus 60.3 tok/s in the small paired run; no speedup claim. Excluding the first import, import latency averaged 11.27 versus 13.40 ms.

The validation selected fusion=4/Mesh(2) because fusion=8/Mesh(2) aborts even on a one-element allocation in the single-card Pod. Automatic backend registration was isolated but did not alone fix that abort. This change is scoped to the validation launcher, not the default engine. Native Qwen3-8B-to-ATOM remains unvalidated.
