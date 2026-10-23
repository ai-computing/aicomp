# Bounded rolling native admission (#158)

Parent:62d300a (admission telemetry), stacked on PR157. Opt-in `--rolling-native`
requires native `--pipeline --native-pack direct`, with asynchronous packing off.
Default remains wave submission. Both use the same validated request-ID receiver.

The native socket owner replenishes pending requests without waiting for the last
request in an earlier wave. Native concurrency is capped by `--native-batch-size`;
the existing lifetime credit budget (`--batch-size`) still includes native,
prepared and active GPU work. Queue capacities and decoder arithmetic are unchanged.
Each request has an individual timeout; stopping drains existing transfers without
admitting new work. Completion hands off owned pinned storage; cancellation or errors
release unhanded credits. Completed tensors are not accumulated in a stream result list.

Validation includes interleaved IDs, admission of a third request while the second
still waits for KV (max pending two), HTTP failures, invalid IDs, ownership transfer,
cancellation and failure cleanup. Final full suite:49 tests pass, including the per-request deadline test. The first editing command
used an incorrect relative path and made no changes; its three passing tests were
pre-change checks, not candidate validation.

## Hardware comparison

Same GPU+RNGD host/model/FXB/SDK as ADMISSION.md and DIRECT_RECEIVE.md. RNGD native
reference versus RNGD+A100 pipeline; no equal hardware/power claim. Corrected read1
client, deterministic mixed128/512/1024 inputs and32/128 outputs. No stage profiling
for either candidate or wave baseline. Finite-run drain throughput is not capacity.

```bash
python3 -m unittest discover -s . -p 'test_*.py'
python3 serve_gpu.py --model-dir "$MODEL_DIR" --pipeline --batch-size 16 --native-batch-size 8 --native-batch-wait-ms 8 --norm-mode fused --fuse-mlp --fuse-qkv --native-pack direct --rolling-native
python3 bench_arrival_http.py --rates 6 --requests 96 --out rolling6.json
python3 bench_arrival_http.py --rates 12 --requests 384 --out rolling12.json
# Restart only the owned GPU server with the same flags except --rolling-native.
# Repeat both benchmark commands, writing wave6.json and wave12.json.
```

Measured core commit:7a89f6b (hashes in rolling_manifest.json).

|Mode/load|Native TTFTp95 ms|GPU TTFTp95 ms|Native/GPU tok/s|GPU success|
|---|---:|---:|---:|---:|
|rolling6/s|161.55|256.06|441.90/439.22|96/96|
|wave6/s|157.13|271.44|441.66/438.52|96/96|
|rolling12/s|300.32|269.14|877.98/913.17|384/384|
|wave12/s|295.64|1184.19|883.08/886.05|384/384|
|rolling12/s repeat|237.80|266.35|881.62/912.48|384/384|

The rolling/wave/rolling sequence supports removing the whole-wave barrier at high
load:GPU TTFTp95 falls77.3-77.5%, with throughput up3.0-3.1%. All phases succeed.
This is one wave run bracketed by two rolling runs, not a broad capacity study.
Native high-load TTFT varies238-300ms; rolling266-269ms is near that range, but low-load
rolling256ms remains slower than native162ms. Keep the option experimental and off by
default pending wider repetitions, burst/overload and cancellation/backpressure hardware
checks. No change to decoder arithmetic; long retrieval smoke tests are not full quality
evaluation. Goal remains unmet. Owned GPU/native servers stopped after measurement.
