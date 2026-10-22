# Local prefill in request groups: first-token latency and streaming tradeoff

- Date: 2026-10-22
- Task: aicomp #148 / PR #149
- Commit: 24eef9c parent plus gpu_local_chunk_manifest.json
- Hardware/software/model: GPU_LOCAL_PAIRED.md; one A10080GB versus one RNGD,
  pinned Qwen3-8B-FP8 checkpoint, native2026.4.0. No hardware equivalence claim.

## Question and implementation

A warmed8x1024 prefill profiler shows596.32ms CUDA self time, of which aten::mm
490.68ms (82.28%). FA2 attention19.96ms (3.35%). Matrix sizes and full profile
are retained locally. Profiler timing is diagnostic, not the uninstrumented baseline.

Opt-in `--local-prefill-batch-size 4` prefills four requests at a time, immediately
queues their first output, and copies their GPU KV before computing the next group.
Then all eight decode lanes run together. The owner never blocks on full HTTP output
queues. Default group limit16 preserves full-wave prefill; native backend default
is unchanged. Grouping requests is NOT token-chunked prefill and does not overlap
prefill with decode: earlier groups wait after their first token. Numerical output
can depend on group size, as documented by earlier packed-prefill diagnostics.

## Commands

Native endpoint and environment from GPU_LOCAL_PAIRED.md. Run from native_kv:

```bash
python3 profile_local_prefill.py --model-dir "$MODEL_DIR" --out profile.json
python3 serve_gpu.py --prefill-backend gpu --model-dir "$MODEL_DIR" --batch-size 8 --local-single-width --local-prefill-batch-size 4 --native-batch-wait-ms 8 --norm-mode fused --fuse-mlp --fuse-qkv
python3 bench_http.py --lengths 128,512,1024 --concurrency 8 --repeats 3 --out chunk4_c8.json
python3 check_batch_http.py --model-dir "$MODEL_DIR" --concurrency 8 --out chunk4_quality.json
python3 bench_http.py --lengths 512 --concurrency 8 --repeats 5 --out chunk4_confirm512.json
# Stop HTTP servers before the direct worker check:
python3 check_pipeline_backpressure.py --prefill-backend gpu --local-single-width --local-prefill-batch-size 4 --peer-count 7 --model-dir "$MODEL_DIR" --batch-size 8 --out chunk4_backpressure.json
python3 -m unittest discover -s . -p 'test_*.py'
```

## Results

HTTP/SSE,128 output tokens,8 concurrent clients; one warmup then alternating endpoint
order. Table medians, using the5-repeat confirmation for512 because the initial
native run varied (initial median529.81tok/s,385.90ms TTFT). Other lengths3 repeats.

| Input | Native tok/s | GPU tok/s | Native TTFT ms | GPU TTFT ms | Native/GPU TPOT ms |
|---:|---:|---:|---:|---:|---:|
| 128 | 703.68 | 704.02 | 100.03 | 77.64 | 10.64/10.80 |
| 512 | 575.25 | 590.42 | 263.40 | 255.58 | 11.79/11.59 |
| 1024 | 450.38 | 492.25 | 483.41 | 482.60 | 13.73/12.53 |

Compared with the prior separate tuned full-wave run, GPU TTFT at1024 moves632.18
->482.60ms, throughput495.66->492.25tok/s. This is not a same-run full-wave A/B,
so native paired comparisons above are stronger than attribution of the entire delta.

Streaming caveat: at1024, first-to-second nonempty text-chunk gap median/max
is GPU176.50/339.20ms versus native86.87/92.21ms. At512 confirmation it is
GPU99.44/186.42ms versus native87.49/91.13ms. These are HTTP text delivery gaps,
not kernel token timings. Lower TTFT hides a longer early pause for the first group;
mean TPOT alone does not show it. Full per-request arrivals are retained locally.

15 mixed questions match native, solo and concurrent outputs. This is limited quality
coverage, not exact arithmetic parity or proof against the existing synthetic divergences.

## Conclusion

Request-group prefill approaches native average throughput/TTFT in this burst matrix,
but is not full latency parity: early streaming gaps remain and sustained mixed-load
behavior is unmeasured. Keep opt-in. Next investigate token-level chunked prefill and
decode scheduling to reduce pauses, plus steady-load tests. The heterogeneous native
KV path also still needs optimization; this GPU-local result does not complete that goal.

Final checks:37 regression tests pass. One stalled32-event consumer plus7 peers spanning both prefill groups complete128 tokens each; resumed text matches peers. Cancellation followup matches uncancelled same-width baseline; single-versus-batched synthetic text remains different. All owned servers/checkers exited; GPU idle.
