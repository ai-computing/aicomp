# Token-chunked local prefill with decode scheduling boundaries

- Date: 2026-10-22
- Task: aicomp #148 / PR #149
- Commit: cdb3fcb parent plus gpu_token_chunk_manifest.json
- Setup: GPU_LOCAL_PAIRED.md pinned Qwen3-8B-FP8, A10080GB versus one RNGD,
  same GPU/native SDKs and FXB. Hardware/arithmetic differ.

## Implementation

`--local-prefill-token-chunk 256` runs resumable prefill only when an earlier
request group still has active decode lanes. Between token chunks it flushes output,
handles cancellation/backpressure and advances active decode one step. Generated
text is flushed before returning to prefill. The first request group keeps ordinary
full prefill to avoid unnecessary overhead. Default token chunk0 leaves it disabled.

Each session owns KV, positions, prompt copies and last hidden states. Ragged
queries use per-request prefix lengths and absolute RoPE positions. Only initialized
KV enters FA2 attention, with a causal query suffix aligned to its prefix end.
Optional `--local-prefill-paged` reads fixed equal-length/multiple256 caches directly;
ragged lengths use gathered contiguous prefixes. Paged is default off and has only
microbenchmark/state validation here, not HTTP quality/performance validation.

## Commands

From native_kv with pinned MODEL_DIR; native server as GPU_LOCAL_PAIRED.md:

```bash
python3 check_chunked_prefill.py --model-dir "$MODEL_DIR" --no-paged --out token_chunks.json
python3 check_chunked_prefill.py --model-dir "$MODEL_DIR" --out token_chunks_paged.json
python3 check_chunked_state.py --model-dir "$MODEL_DIR" --out state.json
python3 check_chunked_state.py --equal-length --model-dir "$MODEL_DIR" --out state_paged.json
python3 serve_gpu.py --prefill-backend gpu --model-dir "$MODEL_DIR" --batch-size 8 --local-single-width --local-prefill-batch-size 4 --local-prefill-token-chunk 256 --native-batch-wait-ms 8 --norm-mode fused --fuse-mlp --fuse-qkv
python3 bench_http.py --lengths 512,1024 --concurrency 8 --repeats 3 --out token256_c8.json
python3 check_batch_http.py --model-dir "$MODEL_DIR" --concurrency 8 --out token256_quality.json
# Stop HTTP servers before the worker check:
python3 check_pipeline_backpressure.py --prefill-backend gpu --local-single-width --local-prefill-batch-size 4 --local-prefill-token-chunk 256 --peer-count 7 --prompt-tokens 512 --model-dir "$MODEL_DIR" --batch-size 8 --out token256_backpressure.json
python3 -m unittest discover -s . -p 'test_*.py'
```

Initial contiguous microbenchmark preceded paged implementation; --no-paged now
reproduces that branch. Paged microbenchmark preceded the final opt-in CLI gates.
A server was started then stopped before measurements to fix output flushing at
scheduling boundaries; HTTP measurements use the corrected implementation.

## Measurements

4x1024 prefill, three measured repetitions after warmup; synchronized per-step
wall times include scheduling-boundary overhead, exclude model load and allocation
of session buffers. Contiguous / paged median total ms:

| Token chunk | Contiguous total | Paged total | Contiguous max step | Paged max step |
|---:|---:|---:|---:|---:|
|2048|308.71|310.74|312.90|315.03|
|256|354.12|351.45|91.78|90.61|
|128|375.74|368.45|49.39|48.42|
|64|451.14|430.17|30.61|28.63|

HTTP/SSE8 clients,128 outputs,one warmup/three alternating measured runs;
contiguous token256 only. Median throughput/mean TTFT, maximum observed text gap:

| Input | Backend | tok/s | TTFT ms | Max text gap ms |
|---:|---|---:|---:|---:|
| 512 | native | 585.15 | 249.90 | 85.44 |
| 512 | gpu | 582.26 | 272.06 | 109.14 |
| 1024 | native | 459.09 | 460.36 | 87.65 |
| 1024 | gpu | 475.10 | 525.35 | 113.06 |

Earlier separate request-group-only run had first-to-second text gap up to339.20ms
at1024; new run maximum across ALL adjacent text chunks is113.06ms. This reduces
pauses but does not match native87.65ms. TTFT becomes worse:525.35ms versus460.36ms
native in this run. Overall throughput remains475.10 versus459.09tok/s at1024,
but512 is582.26 versus585.15. No claim that all metrics improved simultaneously.

## Correctness and limits

Single contiguous chunk matches full-prefill KV exactly in tested geometries.
Other chunk sizes/layouts are not bitexact. For4x1024, contiguous chunk128 first
tokens match3/4; chunk256 matches4/4, but KVmaxabs2.75. Paged full-size KVmaxabs4.0,
showing layout-dependent arithmetic too. These diagnostics return0 while reporting
mismatches; exit status is NOT a numerical parity pass.

Two interleaved sessions test initialized-prefix immutability with1728 slice checks,
neighbor target KV exact, and completed-session reentry rejection. Both ragged and
equal-length/paged variants pass. Worker test with512-token inputs spans both prefill
groups and multiple token chunks:32-event stalled consumer,7 peers128 outputs each,
resume text and cancellation followup all pass in this synthetic case.37 regression
tests pass.15 ordinary question/retrieval texts match native/solo/concurrent, but
those short prompts do not validate long-context chunking quality. Long-context
quality and varying chunk-size numerical sensitivity remain explicit follow-ups.

## Conclusion

Resumable token prefill provides scheduling boundaries and reduces long early pauses,
at an overhead/TTFT cost. Keep opt-in; optimize memory/launch costs and scheduling,
then evaluate sustained mixed traffic, long-context quality and native latency
criteria. GPU-local work does not resolve heterogeneous KV export bottlenecks.
All owned servers/checkers exited; GPU idle. Goal remains active.
