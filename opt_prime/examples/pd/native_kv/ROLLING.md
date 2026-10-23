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
cancellation and failure cleanup. Full suite:48 tests pass; an additional per-request deadline test passes in the eight-test targeted suite. The first editing command
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

Results pending; do not recommend this flag until matched hardware checks complete.
