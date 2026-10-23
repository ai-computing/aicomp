# KV backend separation and RNGD-to-ATOM validation attempt

- **Date**: 2026-10-23
- **Task**: KV transfer backend separation (aicomp #194)
- **Host / device**: RNGD+GPU host (A100 80GB PCIe GPU 0, RNGD 0, Furiosa 2026.4.0); RNGD+ATOM host (Rebel 0.11.0, optimum-rbln 0.11.0.post1)
- **Commit**: main fd89265 plus PR193 a6f92cb; feat/194-kv-backends, dirty during verification

## Question

Can common KV transfer ownership and physical-run planning be separated from the RNGD ABI/CUDA backend, and can the same block-import interface serve RNGD-to-ATOM?

## Setup

Separate C planner with configurable alignment/byte budget; common bounded single-consumer transfer queue; dimension-derived CUDA upload; ATOM CPU BF16 layout conversion and per-layer write adapter. Native protocol and model-specific decoders retain validation. No transport protocol or device allocation changes.

## Command

From opt_prime/examples/pd/native_kv in the isolated worktree:

```bash
python3 -m unittest discover -p 'test_*.py'
gcc -O2 -Wall -Wextra queue_adapter/test_plan.c -o /tmp/test_portable_plan
/tmp/test_portable_plan
gcc -O2 -Wall -Wextra -pthread queue_adapter/test_adapter.c -o /tmp/test_general_adapter
/tmp/test_general_adapter
gcc -g -fsanitize=address,undefined -pthread queue_adapter/test_adapter.c -o /tmp/test_general_adapter_san
/tmp/test_general_adapter_san
```

Full suite: `python3 -m pytest` from opt_prime. Baseline comparison: same failing test file on clean main fd89265.

GPU hardware: `python3 analysis/local_prefill/native_dma/general/final/run_gpu.py`; launcher records exact native and GPU commands and stops its own processes. Same Qwen3-8B-FP8 weights/FXB as the prior partial-coalescing experiment. `check_http_quality.py` compares native, GPU and streaming GPU responses for 15 prompts.

ATOM target command, using the pre-existing optimus-rm checkout and runtime environment:

```bash
$ATOM_PYTHON check_atom_backend.py --benchmark "$OPTIMUS_RM/opt_prime/examples/pd/npu_pd_bench.py" \
 --report atom_backend.json --model Qwen/Qwen2.5-0.5B-Instruct --atom-devices 1 --batch 8 \
 --kv-blocks 17 --requests 12 --min-prompt 16 --max-prompt 96 --max-new 16 --remote-only \
 --prefill-python "$FURIOSA_PYTHON" --prefill-pythonpath "$PREFILL_PYTHONPATH" \
 --prefill-args '--engine rngd-qwen --chip 2 --width 256,512'
```

## Results

- New tests first failed on fixed Qwen geometry / missing backend injection; implementation passes. A first test setup used default pin_memory and failed due to registered RNGD allocator; corrected to explicit CUDA, consistent with existing code.
- 77 unittest cases pass; non-Qwen CUDA payload bytes match.
- Full pytest: 84 pass, 2 fail, both `test_kv_cache_generation_matches_hf` batch 1/2 with unregistered DynamicCache pytree. Both reproduce on clean main. An initial baseline invocation in an older local checkout lacked that test; retried on an isolated fd89265 checkout.
- Portable planner and adapter C tests pass; adapter ASan/UBSan pass.
- GPU hardware: 15/15 cases equal across native/nonstream/stream; strict expected-format score 12/15 each. Final planner parameterization rerun also completed.
- ATOM: zero imports, no generated-token/performance result. Runtime startup refused missing host-group mapping: all four physical cards are in groups 1–4. Device usage was zero, but this does not imply host runtime visibility. Existing Kubernetes config rejected authentication. No device groups or cluster resources were altered. SSH access works.

Artifacts: local analysis/local_prefill/native_dma/general/final/; ATOM attempt at codex-kv-backends-194/run.log and atom_backend.json on the RNGD+ATOM host. Public code and detailed boundaries in BACKENDS.md/BACKEND_VALIDATION.md.

## Conclusion

Reusable planning/ownership boundaries and CUDA correctness are established for this refactoring. ATOM writer is unit-tested and wired into a validation runner, not hardware-validated or enabled by default. Native Qwen3-8B -> ATOM is also not established by the planned 0.5B research-prefill validation. Need access to an existing allocated ATOM execution context, then readback and continuation parity. Scheduler latency optimization remains separate.
