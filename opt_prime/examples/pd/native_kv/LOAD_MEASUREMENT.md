# Correcting SSE load measurement and locating the remaining queueing gap

- Date:2026-10-22–23; issue:aicomp #156.
- Engine commit:604ad99 throughout. Benchmark changes identified by load_manifest.json.
- GPU+RNGD host:one RNGD native reference; one RNGD+A10080GB PCIe heterogeneous path.
- Same pinned Qwen3-8B-FP8 revision, FXB hash and SDK as DIRECT_RECEIVE.md.
  HTTP client:requests with urllib3 2.6.2. No CPU affinity restriction.
- Engine configuration:direct pinned receive,default native grouping,16 GPU lanes,
  native batch8,gather8ms,fused norms/QKV/MLP. No engine changes in this experiment.

## Problem found in the measurement client

The previous SSE reader used requests.iter_lines(chunk_size=1) to avoid buffering
first-token delivery. At high streaming rates that creates substantial per-byte Python
work. In the initial384-request12req/s GPU run, client scheduled-start lateness reached
p95523.49ms/max1114.47ms, so the workload and observed stream gaps were client-limited.
Do not interpret that run's2.95s TTFTp95 or2.14s maximum text gap as isolated engine
behavior. Raw results remain preserved, not relabeled as clean measurements.

Use urllib3 HTTPResponse.read1(65536,decode_content=True), splitting complete lines
while retaining partial bytes. Unlike a fixed-size blocking read, read1 returns available
data without waiting to fill64KiB. Preserve --sse-reader byte for reproducing the old
client. Record client CPU time, worker count, start lateness and rejected requests.
Extend requests to512 with at most128 client workers; visible start lateness remains
necessary to detect client saturation. Engine behavior and SSE contents are unchanged.

## Correctness

`python3 -m unittest test_bench_stream -v` passes (one test, two real HTTP subcases).
The server deliberately withholds its tail until the client receives the first event.
Chunked and close-delimited HTTP both deliver that event before EOF/full-buffer fill;
UTF-8 split across writes, CRLF, blank lines and DONE are retained correctly.
Live requests continue to validate output-token counts, terminal events and HTTP errors.
This change is a measurement correction, not an inference speedup or new quality claim.

## Commands and chronology

From native_kv with MODEL_DIR pinned, start native per GPU_LOCAL_PAIRED.md and GPU:

```bash
python3 serve_gpu.py --model-dir "$MODEL_DIR" --pipeline --batch-size 16 --native-batch-size 8 --native-batch-wait-ms 8 --norm-mode fused --fuse-mlp --fuse-qkv --native-pack direct --profile-stages load_stages.jsonl
# Initial legacy reader sweep; current code can reproduce it using --sse-reader byte:
python3 bench_arrival_http.py --sse-reader byte --rates 8,10,12,12,10,8 --requests 96 --out load_sweep.json
python3 bench_arrival_http.py --sse-reader byte --rates 8,12 --requests 384 --out load_long.json
# After the reader fix, restart GPU with identical engine flags; native remains alive:
python3 bench_arrival_http.py --sse-reader byte --rates 12 --requests 96 --out reader_byte96.json
python3 bench_arrival_http.py --sse-reader read1 --rates 12 --requests 96 --out reader_read1_96.json
python3 bench_arrival_http.py --sse-reader read1 --rates 12 --requests 384 --out reader_read1_384.json
python3 bench_arrival_http.py --sse-reader read1 --rates 6,6 --requests 96 --out reader_read1_6.json
python3 summarize_load.py --inputs load_sweep.json load_long.json reader_byte96.json reader_read1_96.json reader_read1_384.json reader_read1_6.json --out load_summary.json
```

Initial96 sweep used parent604ad99 harness. Initial384 run added the512-request limit
and128-worker bound before the SSE fix; those historical rows lack reader/CPU labels.
Corrected comparisons explicitly record reader type and CPU time. Inputs cycle128/512/
1024 tokens and outputs32/128 with deterministic seeds. Endpoint order alternates per
phase. Comparisons are sequential, not randomized; host variability remains.

## Evidence for client overhead

12req/s,96 requests per endpoint, all successful:

| Reader | Native/GPU client CPU seconds | GPU start lateness p95 ms | Native/GPU TTFTp95 ms |
|---|---:|---:|---:|
|byte|3.45/8.86|74.72|234.30/905.54|
|read1|0.91/0.82|7.37|214.54/822.96|

Client CPU drops substantially while the engine is unchanged. Do not attribute the
remaining TTFT difference or every cross-run millisecond to the reader alone.

## Corrected engine observations

| Offered load / requests | Native/GPU successes | Native/GPU tok/s | Native/GPU TTFTp95 ms | Native/GPU completion p95 ms |
|---|---:|---:|---:|---:|
|6/s,96 phase0|96/96|441.8/437.2|164.8/271.9|2944.5/1789.8|
|6/s,96 phase1|96/96|441.9/436.7|162.2/272.7|2891.2/1787.9|
|12/s,96|96/96|725.0/753.6|214.5/823.0|7608.8/2378.5|
|12/s,384|384/378|874.6/867.3|245.3/1562.2|11678.8/3083.9|

The384-request GPU run has six HTTP503 responses (1.56%); latency percentiles describe
successful requests only. Client-start lateness p95 is8.70ms native/9.24ms GPU, much
smaller than the old client. GPU TTFT medians by balanced request quartile rise565 ->
970 ->1323 ->1459ms, so an engine-side queueing/admission limit remains after fixing
the client. Native TTFT medians115 ->158 ->165 ->160ms; native completion is slower.
Counts/errors and quartile sampling are preserved in load_summary.json.

Initial byte-reader96 sweeps at8/10/12req/s and long384 runs are retained for audit,
but should not establish capacity. In particular, a short run showing higher GPU
throughput cannot establish sustainable capacity or mask503s in a longer corrected run.
Throughput includes finite-run drain; these are not steady-state saturation estimates.

## Conclusion and next action

The measurement client needed correction before tuning the server under load. Low-load
TTFT remains about272ms versus native162–165ms; higher-load TTFT and503s show the goal
is still unmet. The next implementation should measure admission wait, credit occupancy
and batch-barrier delay, then test wider decode/admission or rolling native submission
only against that evidence. Do not merely enlarge a queue to conceal rejection.

All owned servers stopped; GPU idle. No new model quality suite was needed for the
client-only change. Hardware budgets remain unequal; no performance-per-watt or full
native-parity claim. Compact summary omits generated texts; raw local logs are retained.
