# Native KV delivery: matched arrivals with and without GPU decode

Issue #160; parent148e9ca (PR159). This unit changes diagnostic tools only. No
receiver validation, inference arithmetic, admission limits or production defaults change.
Same pinned Qwen3-8B-FP8, FXB, SDK and GPU+RNGD host as ROLLING.md/DIRECT_RECEIVE.md.
Native2026.4.0, torch2.10.0+cu128, A10080GB, one RNGD. Source hashes are in
interference_manifest.json. Dedicated owned native/GPU processes stopped afterward.

## Hypothesis and method

After rolling admission,1024-token native readiness is about93ms but KV completion
is about190ms. Earlier isolated measurements used different prompts and arrivals;
they cannot establish interference from GPU decode. Here use the exact same96
random token prompts (seed149000+i),128/512/1024 lengths and6req/s schedule.

1. Rolling GPU pipeline, direct pinned receive,16 lifetime credits and8 native pending
   requests, default5ms Python switch interval. Record stage and receive timings.
2. Same pipeline at0.1ms switch interval, testing interpreter scheduling effects.
3. Stop GPU server. Send the same scheduled prefill requests through the same native
   receiver,8 pending cap, without any GPU decode. No HTTP frontend or generated-output
   stage: this is a matched native receive experiment, NOT an inference throughput result.
4. Restart only native with kv_pipeline_group_layers=1, repeat the receive-only run.
   Default grouping emits18 messages; group1 emits72. Smaller groups satisfy the
   existing receiver bounds without relaxing validation. No generated-text check for
   group1, so it remains an unqualified candidate, not a recommended setting.
5. Restart default native with connector DEBUG/TRACE logging for one warmup128 and
   three timed requests128/512/1024 at1req/s. Use logs only to locate producer cadence;
   cold shapes, different rate and logging mean these are not comparable latency benchmarks.

The pipeline summaries discard the first warmup record. Native-only output already
excludes its warmup. All four96-request experiments completed successfully. Allocation
in native-only1024 is about1ms. Scheduled-start lateness medians are below1ms.

## Commands

From native_kv, start the dedicated native endpoint using GPU_LOCAL_PAIRED.md with
prefix caching disabled and TcpConnector poll_timeout_ms=1. MODEL_DIR points to the
pinned local checkpoint; OUT is a fresh result directory.

```bash
RECEIVE_PROFILE="$OUT/rolling_receive_profile.jsonl" python3 profile_pipeline_receive.py --model-dir "$MODEL_DIR" --pipeline --batch-size 16 --native-batch-size 8 --native-batch-wait-ms 8 --norm-mode fused --fuse-mlp --fuse-qkv --native-pack direct --rolling-native --profile-stages "$OUT/rolling_stage_profile.jsonl"
python3 bench_arrival_http.py --rates 6 --requests 96 --out "$OUT/rolling_profile6.json"
# Stop GPU; repeat above with SWITCH_INTERVAL=0.0001 and rolling_fast output names.
# Stop GPU; leave default native running:
python3 profile_native_arrivals.py --rate 6 --requests 96 --out "$OUT/native_only6.json"
# Restart native: add "kv_pipeline_group_layers":1 to its existing connector JSON.
python3 profile_native_arrivals.py --rate 6 --requests 96 --out "$OUT/native_group1_only6.json"
python3 summarize_rolling_profile.py --root "$OUT" --prefix rolling
python3 summarize_rolling_profile.py --root "$OUT" --prefix rolling_fast
# Restart default native with this environment prefix:
# RUST_LOG=info,furiosa_generator::connector::tcp=trace,furiosa_generator::next_gen::scheduler::transfer=debug
python3 profile_native_arrivals.py --rate 1 --requests 3 --out "$OUT/native_trace3.json"
```

## Results

HTTP6req/s:default-switch pipeline TTFTp95=255.32ms versus native160.30ms;
0.1ms-switch pipeline260.25ms versus native159.24ms. Throughput439.57/436.19tok/s
respectively. No improvement from the shorter switch interval; retain default.

Native KV-complete medians (ms since native HTTP submission):

| Input tokens | GPU pipeline5ms | GPU pipeline0.1ms | No GPU, default group | No GPU, group1 |
|---|---:|---:|---:|---:|
|128|42.94|45.38|40.86|45.61|
|512|105.71|107.42|102.38|98.33|
|1024|189.65|201.26|194.46|184.15|

At1024, first-to-last receiver span87/98/91/85ms respectively (last row group1).
Default pipeline measured receive-function wall work15.49ms versus12.60ms without GPU.
Pipeline prepared-cache wait11.07ms, GPU import6.80ms. These are medians; do not add
percentiles or infer independent causes from their sums.

The default native diagnostic trace at1024 contains18 groups of8,380,416 payload bytes.
First DMA-read log to last shipped log spans75.60ms. The next group's read log generally
precedes the preceding group's shipped log, consistent with pipelined export.
This is a producer-side elapsed interval including preparation/DMA/scheduling/send;
**it is not a measured75.60ms of pure DMA hardware time**. Receiver-thread or network
speed alone cannot explain it. The trace summary removes endpoints and request IDs.
The first parser attempt rejected nanosecond timestamps under Python3.10; truncating
fractional seconds to microseconds fixed metadata extraction, without rerunning hardware.

## Conclusion and next experiment

The matched default-group receive-only run does not reproduce a large GPU interference
penalty:1024 KV-complete194ms without GPU versus190ms with it. Do not pursue process
separation or interpreter tuning as an established fix. This is one controlled-input
sequence, not a universal proof that interference never matters.

Group1 slightly improves long inputs but worsens short inputs; no default recommendation,
no broad quality or end-to-end gain claim. Focus next on the native producer's readback
plan and DMA/packing breakdown, then test a bounded export-layout/transport mitigation
with equal-cache validation. The low-load native-performance goal remains unmet.
