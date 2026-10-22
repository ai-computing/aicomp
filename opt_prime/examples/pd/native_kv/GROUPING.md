# Bounded native KV grouping experiment

- Date:2026-10-22; issue:aicomp #154.
- Commit:604ad99 parent (PR151) plus grouping_manifest.json source hashes.
- Hardware:GPU+RNGD host; RNGD+A10080GB PCIe heterogeneous versus one RNGD native.
- Software/model/FXB:exact pinned revisions and hashes in DIRECT_RECEIVE.md,
  furiosa-llm/native2026.4.0, torch2.10.0+cu128, pyzmq27.1.0. No affinity restriction.

## Question and observed setting

Installed connector configuration metadata contains kv_pipeline_group_layers. Setting
it to4 produces18 messages;8 produces9. In this measured build those messages contain
4 and8 K/V slots respectively, covering all72 slots. Each attention layer has two slots.
Do not infer that the field counts attention layers from its name, or that untested
values/builds are compatible. The initial4 probe did not change the previously observed
18-message framing;8 established the framing effect.

## Receiver implementation

Add opt-in --native-kv-max-batch-slots8 to the direct native pipeline (default4).
The client sets the cap before connecting;36MiB ZMQ message bound covers the largest
8-slot payload plus metadata at2047 positions. Default retains32MiB and4-slot validation.
Only4/8 caps are accepted. Non-direct/single-flight legacy paths reject the larger cap.
The cap does not configure the sender: it must be set separately in the native CLI.

Every header keeps request-ID, metadata, position, uniqueness, slot and final-coverage
checks. Direct payload length and multipart checks are unchanged, as are poisoned-client
handling, owned pinned lifetime and the GPU import fence. Larger groups are not silently
accepted by default. No numerical/model changes or vendor binary patches.

## Commands

From native_kv. MODEL_DIR/FXB/NATIVE_CLI identify the pinned artifacts and2026.4 CLI;
OUT is a fresh result directory, with no servers occupying the test ports.

```bash
python3 -m unittest discover -s . -p 'test_*.py'
python3 bench_grouping.py --model-dir "$MODEL_DIR" --fxb "$FXB" --native-cli "$NATIVE_CLI" --out-dir "$OUT"
python3 summarize_grouping.py --input-dir "$OUT" --out grouping_summary.json
```

The controller runs8 ->4 ->8 native grouping, fresh owned native/GPU processes each.
Native uses npu:0, no prefix cache and TcpConnector with poll_timeout_ms1 and
kv_pipeline_group_layers set to the tested value. GPU always uses direct receive,
max-batch-slots8,16 lanes,native batch8,gather8ms,fused norms/QKV/MLP, Python switch5ms.
All exact child commands and cleanup are in bench_grouping.py. Each case runs
bench_arrival_http.py --rates6,6 --requests96. Both8 cases run check_long_http.py;
first8 case also stops GPU HTTP and runs check_same_cache_pause.py --direct
--max-batch-slots8 against the same native endpoint.

Separate initial isolated probes (GPU decode absent):

```bash
# Dedicated native endpoint first configured with group4, then restarted with group8:
python3 profile_direct_receive.py --max-batch-slots 8 --repeats 3 --out group4_profile.json
python3 profile_direct_receive.py --max-batch-slots 8 --repeats 3 --out group8_profile.json
```

## Correctness and bounds

46 unit tests pass. New real loopback TCP tests send8-slot frames at1/511/2047 KV
positions and compare exact uint16 bits against the existing4-slot canonical assembler
fed the SAME payload bytes split into4-slot frames. This includes maximum-size frames
and raw BF16 NaN patterns. Default4-slot and opt-in8-slot limits reject oversized slot
vectors before payload receive; invalid cap and non-direct extended paths reject.
Existing short/oversized/extra-frame and poisoned-reuse tests remain passing.

Both8-group serving runs:long retrieval592/1152/1712 tokens9/9 expected answers on
native, solo and concurrent GPU paths;9/9 text equality each. Same-cache pause tests:
6144/6144 output positions match uninterrupted runs, paused valid cache/input/position/
count preserved. These checks do not assert bit-exact separate native executions or
broad model quality equivalence.

## Performance results

Deterministic6req/s,96 mixed requests per phase, inputs128/512/1024,outputs32/128.
Two phases per launch alternate native/GPU endpoint order. All12 endpoint phases
complete96/96 successfully. Throughput includes finite-run drain, not saturation.

| Group/run | Messages/request | GPU TTFTp95 phases ms | Native TTFTp95 phases ms | GPU tok/s phases |
|---|---:|---:|---:|---:|
|8 first|9|263.45/283.83|164.29/167.23|439.09/438.59|
|4|18|275.02/257.61|163.17/164.92|437.65/438.92|
|8 repeat|9|282.29/248.43|160.59/154.26|439.71/439.51|

1024-token stage medians (64 samples each); ready/KV are cumulative from submission.

| Group/run | Native ready ms | KV ready ms | Receive wall/CPU ms | First-last receive span ms | GPU import ms |
|---|---:|---:|---:|---:|---:|
|8 first|96.77|205.25|14.18/13.85|89.45|6.81|
|4|90.56|191.61|15.76/15.36|92.47|6.81|
|8 repeat|88.47|195.49|16.20/15.94|86.99|6.81|

Initial isolated probes, after one excluded warmup and three repeats:
1024/one-request group4 KV-ready138.27ms versus group8 135.05ms;
1024/four concurrent requests per-request median359.31ms versus308.25ms.
These isolated probes use repeated-token inputs and different offered load, not the
mixed HTTP workload. The apparent concurrent microbenchmark gain does not establish
serving improvement. Full raw isolated timing samples are retained.

## Conclusion

The configuration's framing effect is verified, and8-slot interoperability works under
an explicit cap. But halving message count does NOT yield a consistent end-to-end TTFT
improvement; groups' timing ranges overlap and transfer span remains substantial.
Keep the default4-slot receiver and avoid recommending8 as a performance improvement.
No full native-performance claim. All owned benchmark servers stopped; GPU idle.

Next:measure higher offered loads to locate saturation and scheduler/transfer interference;
consider larger groups only with explicit bounds and measured benefit. Current evidence
does not justify assuming per-message overhead dominates total KV delivery time.
