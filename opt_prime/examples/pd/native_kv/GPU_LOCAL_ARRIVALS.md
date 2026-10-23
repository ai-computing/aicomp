# Continuous local admission and deterministic open-loop arrivals

- Date: 2026-10-22
- Task: aicomp #148 / PR #149
- Commit: 2def957 parent plus gpu_local_arrival_manifest.json
- Setup: GPU_LOCAL_PAIRED.md; pinned Qwen3-8B-FP8, one A10080GB PCIe versus
  one RNGD, same GPU/native SDKs and FXB. Different hardware and arithmetic.

## Question and implementation

Burst throughput does not establish serving parity. With mixed input/output lengths,
the old wave worker waits for all current requests before admitting the next group.
This leaves free lanes unused and causes queueing under regular arrivals.

Opt-in `--local-continuous` admits queued requests into free lanes after each decode
step. Existing token-chunked prefill continues servicing earlier lanes during admission.
No cache moves or graph-width changes occur: this mode requires a fixed batch graph
and rejects --local-single-width. Output queues remain bounded and stalled lanes stay
reserved until drained/cancelled. Shutdown registry is rebuilt from active/admitting
jobs, bounded by lane capacity. Native/default worker selection is unchanged.

## Commands

From native_kv with MODEL_DIR pinned; native server as GPU_LOCAL_PAIRED.md:

```bash
# Wave baseline:
python3 serve_gpu.py --prefill-backend gpu --model-dir "$MODEL_DIR" --batch-size 8 --local-single-width --local-prefill-batch-size 4 --local-prefill-token-chunk 256 --native-batch-wait-ms 8 --norm-mode fused --fuse-mlp --fuse-qkv
python3 check_long_http.py --model-dir "$MODEL_DIR" --out long_quality.json
python3 bench_arrival_http.py --rates 4,6 --requests 48 --out arrival_wave.json
# Stop GPU server. Continuous8 uses the same command except replace
# --local-single-width with --local-continuous.
python3 bench_arrival_http.py --rates 4,6 --requests 48 --out arrival_continuous.json
python3 check_long_http.py --model-dir "$MODEL_DIR" --out long_quality_continuous.json
python3 bench_arrival_http.py --rates 6,6 --requests 96 --out arrival_continuous_confirm.json
# Stop/restart GPU with --batch-size16 (other continuous options unchanged).
python3 bench_arrival_http.py --rates 6,6 --requests 96 --out arrival_continuous16.json
python3 check_long_http.py --model-dir "$MODEL_DIR" --out long_quality_continuous16.json
# Stop HTTP servers before direct worker checks:
python3 check_pipeline_backpressure.py --prefill-backend gpu --local-continuous --local-prefill-batch-size 4 --local-prefill-token-chunk 256 --peer-count 7 --prompt-tokens 512 --model-dir "$MODEL_DIR" --batch-size 16 --out continuous16_backpressure.json
python3 -m unittest discover -s . -p 'test_*.py'
```

## Method and results

Arrivals are deterministic, equally spaced, not Poisson. Inputs cycle128/512/1024,
outputs alternate32/128 with EOS ignored. Each request has stable generated token IDs;
both endpoints receive the same sequence. One warmup request precedes each endpoint.
Endpoint order alternates across phases. Scheduled-time latency includes client start
lateness. p95 uses the sorted sample at floor((N-1)*0.95). Each finite run drains;
completed tok/s includes drain time and is NOT an asymptotic capacity estimate.
There is no hour-long soak, random arrival process or statistical equivalence claim.

48 requests, TTFT p95ms:
-4req/s: waveGPU1836.39 vs native160.77; continuous8GPU168.93 vs native154.92.
-6req/s: waveGPU3390.43 vs native159.71; continuous8GPU462.89 vs native165.07.
All48 requests succeed in every endpoint/phase. Native varies between tuning runs;
compare each candidate against its paired native run.

Extending6req/s to96 requests exposes queue growth at8 lanes: GPU p95TTFT857.17
and867.72ms in two repetitions, versus native166.31/165.09ms. Completed output
throughput GPU423.46/422.56 versus native441.64/441.64tok/s.

16-lane continuous,6req/s,96 requests, two repetitions:

| Repeat | Native/GPU completed tok/s | Native/GPU p95 TTFT ms | Native/GPU p95 total ms |
|---:|---:|---:|---:|
|0|441.62/439.78|163.40/183.09|2887.10/2495.40|
|1|441.80/439.88|164.40/183.15|2850.41/2498.29|

Every96-request phase completes without HTTP errors.16-lane GPU max observed text
gap83.62/72.19ms; native106.89/94.73ms. GPU process memory snapshot31596MiB,
not allocator peak or hardware-cost equivalence. p95TTFT remains~19ms higher than
native; total p95 is lower in this sampled workload. No conclusion about lower loads
or burst performance for16 lanes without measuring them. Full summary preserves
all tuning results and client start delays.

## Long-context quality

9 generated retrieval tasks contain40/80/120 keyed records (592/1152/1712 input
tokens), querying first/middle/last keys. Expected codes are graded exactly after
whitespace stripping. Wave8, continuous8 and continuous16 each match native/solo/
concurrent text9/9 and exact expected codes9/9 in every path. Concurrent clients8.
This exercises long prompts and slot reuse, but is limited retrieval coverage;
previous synthetic chunk/width numerical differences remain, not erased by this test.

## Conclusion

Continuous admission and sufficient lane capacity close much of the measured arrival
queueing gap.16 lanes approach native throughput/latency at6req/s in two finite runs;
this is not complete performance parity. More arrival patterns, load levels, numerical
quality and hardware cost remain. Native-KV heterogeneous transport still has separate
bottlenecks; this GPU-local optimization does not resolve them. Defaults remain opt-in.

Final validation:37 regression tests pass.16-lane continuous worker check with512-token
prompts passes32-event stalled consumer,7 peers128 outputs,resume/cancellation, and
a newcomer completing128 tokens BEFORE the stalled peer resumes. Active registry
remains bounded. Native/GPU client start-delay p95 in the16-lane comparisons is
14.95-16.45ms (included in scheduled TTFT), max21.26ms. All owned servers/checkers
exited; GPU idle. No merge performed; maintainer review remains required.
