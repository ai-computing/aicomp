# Native pipeline admission and batch-barrier timing

- Date:2026-10-23; issue:aicomp #158.
- Commit:fec193f parent (PR157) plus admission_manifest.json instrumentation hashes.
- Same pinned Qwen3-8B-FP8 revision, FXB, software and GPU+RNGD hardware as
  DIRECT_RECEIVE.md. One RNGD baseline versus RNGD+A100 heterogeneous path.
- Corrected read1 SSE client; native grouping unchanged,16 GPU lanes,native batch8,
  gather8ms,fused norms/QKV/MLP,direct pinned receive. No affinity restrictions.

## Question and instrumentation

Separate admission delay from native work and waiting for a GPU lane. Worker.submit
stamps enqueue time; producer stamps dequeue and native submission; owned packet enqueue
and GPU import stamp the prepared-cache wait. Record first GPU token and first text
readiness separately. Registry count is lock-protected; queue lengths are momentary
snapshots. Per-wave logs record requests, native call duration, credit-acquire wait,
pending/admitted counts and native completion callback offsets.

No admission policy, queue/credit limit, decoder arithmetic or native protocol changes.
Timing calls and optional wave-file logging can affect scheduling; these measurements
are not a performance optimization. A shared per-wave credit wait is not a per-request
component to add repeatedly. The accounting check sums each request's admission, gather/
validation, native completion, CPU packing, prepared wait, GPU import and first-step time.

## Commands

Start the dedicated native endpoint per GPU_LOCAL_PAIRED.md. Use a fresh STAGES path.
Run from native_kv with MODEL_DIR pinned:

```bash
python3 -m unittest discover -s . -p 'test_*.py'
python3 serve_gpu.py --model-dir "$MODEL_DIR" --pipeline --batch-size 16 --native-batch-size 8 --native-batch-wait-ms 8 --norm-mode fused --fuse-mlp --fuse-qkv --native-pack direct --profile-stages "$STAGES"
python3 bench_arrival_http.py --sse-reader read1 --rates 6 --requests 96 --out admission6.json
python3 bench_arrival_http.py --sse-reader read1 --rates 12 --requests 384 --out admission12.json
python3 summarize_admission.py --arrivals admission6.json admission12.json --stages "$STAGES" --out admission_summary.json
```

Stage records exclude each phase's one warmup in the summary. Wave records are written
to STAGES.waves.jsonl. The first test invocation from a repository root discovered zero
tests; it was not counted. Correct invocation from native_kv passes44 tests, including
the final wave instrumentation. An earlier44-test pass preceded that final addition.

## Results

All requests in these runs succeed. Deterministic mixed128/512/1024 input and32/128
output tokens; throughput includes finite-run drain, not sustainable capacity.

| Load | Native/GPU TTFTp95 ms | Native/GPU tok/s | Native/GPU completion p95 ms |
|---|---:|---:|---:|
|6/s,96 requests|161.49/245.10|442.28/439.34|2767.26/1685.75|
|12/s,384 requests|224.86/446.94|880.38/907.41|10973.35/2011.52|

This high-load run is substantially better than the earlier uninstrumented run
(1562ms GPU TTFTp95 and six503s). Do NOT attribute that difference to an optimization:
policy is unchanged, host placement/background state are uncontrolled, and instrumentation
may perturb scheduling. Keep the earlier failures. A candidate must be paired with a
fresh same-session baseline before claiming improvement.

Per-request p50/p95, excluding warmup (percentiles are not additive):

| Component |6/s ms|12/s ms|
|---|---:|---:|
|Admission wait|0.08/46.87|132.73/244.19|
|Gather/validation|8.36/8.63|0.95/8.99|
|Native completion|106.83/207.63|180.71/263.23|
|Prepared KV wait|4.99/10.22|6.41/10.89|
|GPU import|3.49/7.68|3.51/7.73|
|First GPU token since submit|136.22/236.05|325.24/436.54|

Per-request accounting residual is below0.10ms for every successful measured request;
first text readiness closely follows first GPU token here. HTTP/client TTFT includes
additional request parsing, scheduling and delivery outside these server timestamps.

At12/s:133 native waves, median3 requests (p954,max4); median wave245.58ms,p95287.42ms.
First-credit acquire wait median0.005ms,p950.008ms,max0.010ms: the producer is not
spending the observed admission delay blocked on its first semaphore acquisition.
95/133 waves have all16 credits occupied after gathering; additional credit limits may
still constrain wave size. Pending queue median0 before a wave but3 afterward.
At GPU import, active decode lanes median11,p9513,max14; prepared-packet wait stays small.

## Interpretation and next implementation

The current producer does not admit another request until prefill_many finishes its
whole wave. The large pre-submission wait, small first-credit wait, and pending arrivals
accumulating while native work runs support testing removal of that wave barrier.
They do not prove credits never matter or isolate every source of delay.

Next:an opt-in bounded rolling native submission loop should accept newly available
requests while previous KV transfers remain pending. Preserve the16-credit lifetime
budget, per-request IDs/validation/ACK, owned packet handoff and shutdown/error cleanup.
Test cancellation/interleaving and compare with a freshly rerun wave baseline; do not
simply enlarge a queue. Low-load native KV delivery cost remains an independent gap.

Owned servers stopped and GPU idle afterward. No new quality claim from telemetry.
Compact artifacts:admission_summary.json,admission_arrivals.json; raw local stage/wave
logs retained. Native-performance goal remains unmet.
