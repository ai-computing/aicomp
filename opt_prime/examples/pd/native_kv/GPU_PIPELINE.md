# Overlapped native prefill and continuous GPU lane admission

- **Date**: 2026-10-22
- **Task**: aicomp #146 / PR #147
- **Host / device**: GPU+RNGD host; A100 80GB PCIe and RNGD
- **Commit**: 4924097 parent plus source hashes in gpu_pipeline_manifest.json

## Question

Can overlapping native prefill with active GPU decoding reduce waiting time without
losing request isolation or single-request performance?

## Setup

Same pinned Qwen3-8B-FP8 checkpoint, FXB and SDK setup as GPU_BATCH.md. Furiosa2026.4.0;
Torch2.10.0+cu128, vLLM0.18.0, transformers5.1.0 on GPU; NVIDIA580.178.04.
Native uses one RNGD. Our path uses that RNGD for prefill plus an A100 for decoding.
The GPU path remains Marlin W8A16 with opt-in fused norms/MLP/QKV, not native FP8 arithmetic.

The prefill thread creates/uses/closes its own ZMQ client. A bounded credit budget
limits prefilling, prepared and decoding requests together to the configured lane count.
GPU admission happens between token steps, with a completed copy fence before reuse.
Output queues pause individual lanes instead of blocking the entire GPU worker.
Paused lanes retain their input token, position and remaining output budget.
Experimental --adaptive-single captures a second CUDA graph sharing lane0 KV. It is disabled by default after a stalled/resumed output mismatch. The cause remains unconfirmed.

## Command

Native server setup is in GPU_HTTP.md. Run from the native_kv example directory,
with MODEL_DIR pointing at the pinned local checkpoint:

```bash
python3 serve_gpu.py --model-dir "$MODEL_DIR" --norm-mode fused --fuse-mlp --fuse-qkv --pipeline --batch-size 4
python3 bench_http.py --lengths 512 --concurrency 1 --repeats 3 --out final_c1.json
python3 bench_http.py --lengths 512 --concurrency 4 --repeats 3 --out final_c4.json
python3 bench_http.py --lengths 512 --concurrency 8 --repeats 3 --out final_c8.json
python3 check_batch_http.py --model-dir "$MODEL_DIR" --out quality.json
```

For the experimental measurements, add `--adaptive-single` to the server command and use `adaptive_*.json` output names. It is not enabled by default.

After stopping the GPU HTTP server (the worker check owns the receiver socket), keep
native running for the backpressure test:

```bash
python3 check_pipeline_backpressure.py --model-dir "$MODEL_DIR" --out default_backpressure.json
python3 check_batch_gpu.py --model-dir "$MODEL_DIR" --count 32 --output adaptive_hardware.json
python3 -m unittest discover -s . -p 'test_*.py'
```

## Results

Actual HTTP SSE input512/output128; three measured alternating runs after one warmup.
Stable per-length prompt seed; concurrency8 still uses only four GPU lanes.
Default fixed-width pipeline results:

| Concurrency | Native tok/s | GPU tok/s | Native mean TTFT | GPU mean TTFT |
|---|---:|---:|---:|---:|
| 1 | 99.96 | 89.70 | 49.07 ms | 146.79 ms |
| 4 | 333.91 | 283.51 | 156.32 ms | 320.07 ms |
| 8 | 582.53 | 311.30 | 259.30 ms | 1073.99 ms |

Experimental --adaptive-single results (not the default):

| Concurrency | Native tok/s | GPU tok/s | Native mean TTFT | GPU mean TTFT |
|---|---:|---:|---:|---:|
| 1 | 95.91 | 103.22 | 56.87 ms | 153.88 ms |
| 4 | 329.93 | 282.85 | 169.00 ms | 319.75 ms |
| 8 | 566.73 | 304.67 | 276.79 ms | 1098.33 ms |

- Four-request GPU TPOT10.46ms versus native10.81ms; remaining gap is dominated by filling/queueing and serialized prefill/import. The old wave run had542.30ms TTFT; this is a separate run, not a controlled paired wave/pipeline measurement.
- Initial pipeline without width1 graph regressed c1 to89.70tok/s (native99.96); adaptive graph restores c1 to103.22tok/s. Native variation and only three repeats limit precision.
- Earlier pipeline five-run c4:286.68 versus native331.85tok/s, TTFT312.45 versus157.57ms, before per-lane backpressure/adaptive graph changes.
- Fifteen mixed questions/retrieval prompts match native, solo GPU and concurrent GPU text on the final graph configuration. No general quality equivalence claim.
- Hardware checks verify mixed lengths, cancellation, EOS, lane replacement while peers progress, and paused token/position preservation. All four32-token continuations match single-lane references. Prior128-token synthetic divergence remains unresolved; short checks do not supersede it.
- **Failed final adaptive check:** stalled/resumed text differed from its peer (adaptive_backpressure.log). Repeated with explicit --adaptive-single and reproduced (adaptive_mismatch.json retains both texts). Fixed-width final default_backpressure.json passes. The cause is not established; no state-isolation or numerical parity claim for adaptive switching. The option is disabled by default.
- Fixed-width real worker backpressure check fills one event queue to32 entries while a peer completes128 tokens; resuming the stalled consumer yields the same text. Cancellation then followup also yields128 tokens.
- Existing29 tests pass; batch/pipeline hardware behavior is separately exercised by the scripts above.

## Conclusion

This reduces the waiting imposed by wave prefill, but native throughput/TTFT parity is not achieved. The fixed-width pipeline regresses single-request throughput; the original serial server remains default. Experimental adaptive switching recovers speed but fails one synthetic equality check and is disabled by default. Four-lane capacity scales poorly
to eight clients. Native prefill calls remain serialized; CPU packing/GPU import can
interrupt the decode owner. Next: evaluate8-lane graphs with matching correctness checks,
move prefix packing off the decode owner, profile native export, and broaden quality tests.
Only lane0 has a width1 graph; other single remaining lanes still use the full graph.

Raw evidence in research workspace: analysis/native_kv_pipeline/. Earlier c4.json is an intermediate implementation; adaptive_c*.json, adaptive_quality.json,
adaptive_hardware.json, adaptive_backpressure.json and source_manifest.json describe the
adaptive implementation before it was made opt-in. final_c*.json and quality.json measure the fixed-width pipeline. default_backpressure.json rechecks the final default. source_manifest.json records the final code, including explicit adaptive opt-in and failure-result persistence added after the measurements.
