# KV transfer backend boundaries (#194)

This change separates reusable transfer mechanics from the native RNGD protocol and device runtime. It does not make arbitrary engines KV-compatible.

## Implemented

- `queue_adapter/coalesced_plan.h`: host-side scatter/gather sorting and exact contiguous-run merging, with caller-provided alignment and byte budget. The planner retains the 25% segment-reduction criterion and 2048-segment safety bound. It has no driver ABI, native function offsets, or device discovery. `adapter.c` retains RNGD 2048-byte alignment, 4 MiB plans, pinned binary identity, device-0 selection and completion hooks.
- `kv_transfer_queue.py`: bounded thread-safe publication with a single device-owner consumer. A factory supplies `stage(first,last)`, `finish()`, `close()`. Close must fence outstanding copies before releasing source ownership. The caller validates publication ranges and retains lifetime credits through retirement. Backend exceptions propagate to the owner's existing shutdown path.
- `gpu_stream_upload.py`: CUDA stream/event implementation over CPU BF16 `[2*layers, positions, kv_heads, head_dim]`. Dimensions come from the packet. The compatibility `UploadQueue` defaults to 72 slots for the current native source; other sources supply their slot count and backend factory. CUDA-specific pinned allocation and fences remain here.
- `atom_stream_upload.py`: ATOM block writer with explicit target metadata, staging, block and position offset, and encoder. It transposes the host layout, converts BF16 bit patterns via the installed ATOM encoder, and calls the per-layer cache API. Geometry and duplicate ranges are checked; failed writes poison the transfer. `import_atom_span` adapts optimus-rm's block manager under its existing single-owner scheduler.

## Compatibility and limits

The native wire parser and Qwen3 GPU decoder still deliberately validate their original model contract. Model identity, revision, weights, RoPE, quantization and cache semantics must agree before connecting engines; tensor dimensions alone are not a compatibility handshake. No new wire protocol is introduced.

ATOM support requires exclusively owned blocks and a synchronous cache-write runtime (validation runner pins Rebel 0.11.0 and optimum-rbln 0.11.0.post1). It does not advertise asynchronous ATOM DMA. The current subprocess prefill source publishes whole completed KV; importing in chunks does not create source-side compute/copy overlap. On failure/cancellation the scheduler must retire the partially written block before reusing it. Source packets, runtime and staging remain alive across writes.

`check_atom_backend.py` injects the new adapter into an existing optimus-rm NPU PD benchmark in that benchmark process only. It does not patch an installed vendor library or change the default ATOM engine. Its first import checks device readback; this diagnostic run is not a clean throughput comparison.

TCP/ZMQ, NCCL, NIXL and hetero-npu-p2p are unchanged. A future device-memory transport can implement the same ownership contract, but requires its own capability and cancellation validation. ai-rm device allocation is independent of this change.

## Verification

```bash
python3 -m unittest discover -p 'test_*.py'
gcc -O2 -Wall -Wextra queue_adapter/test_plan.c -o /tmp/test_plan && /tmp/test_plan
gcc -O2 -Wall -Wextra -pthread queue_adapter/test_adapter.c -o /tmp/test_adapter && /tmp/test_adapter
gcc -g -fsanitize=address,undefined -pthread queue_adapter/test_adapter.c -o /tmp/test_adapter_san && /tmp/test_adapter_san
```

See `BACKEND_VALIDATION.md` for measured outcomes, including limited ATOM container parity.

## ATOM container reproduction

Use a dedicated Pod allocated exactly one ATOM and the required RNGD device. Retain the validated runtime/model environment; do not change another workload's device groups. `check_atom_container.py` maps the sole exposed physical ATOM to runtime index zero and verifies opened descriptors. It is a test harness, not a general device-discovery implementation.

```bash
export FURIOSA_PYTHON=/path/to/validated/furiosa/python
export PYTHONPATH=/path/to/optimus-rm/opt_prime:/path/to/rngd_llmd/src:/path/to/hetero-npu-p2p/src
$ATOM_PYTHON check_atom_container.py check_atom_backend.py \
  --benchmark /path/to/optimus-rm/opt_prime/examples/pd/npu_pd_bench.py \
  --report portable.json --model Qwen/Qwen2.5-0.5B-Instruct \
  --atom-devices "$ALLOCATED_PHYSICAL_ATOM" --batch 8 --kv-blocks 17 \
  --requests 12 --min-prompt 16 --max-prompt 96 --max-new 16 --remote-only \
  --prefill-python "$PWD/furiosa_prefill_python.sh" --prefill-pythonpath "$PYTHONPATH" \
  --prefill-args '--engine rngd-qwen --chip 0 --width 256'
# Repeat with --original-import and a distinct --report for the paired control.
```

The Furiosa shim registers only that vendor backend and selects fusion=4/Mesh(2) for the observed container limitation. The ATOM interpreter must resolve the same validated user-site/runtime libraries as the cached build. Set HOME accordingly when using a mounted environment; use writable task-local compiler/lease caches and retain the existing model artifacts.
