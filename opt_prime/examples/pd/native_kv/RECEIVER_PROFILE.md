# Native KV receiver boundary profile

- Date:2026-10-22; task:aicomp #150; PR151.
- Commit:b6474a3 plus diagnostic hashes in receiver_profile_manifest.json.
- Same model revision, FXB, software and GPU+RNGD hardware as DIRECT_RECEIVE.md.
- No engine/default setting changes. No raw KV, output texts or vendor binaries saved
  in these profiling artifacts. Native prefix caching remains disabled.

## Question and method

Which receiver work accounts for native readiness-to-KV completion? First isolate
native prefill plus pinned receiver without GPU decode. Wrap existing validation and
receive methods with perf_counter/thread_time; preserve every original validation.
Each length/width has one excluded warmup and three repeats. Prompts repeat token1000
(or1000+lane); these are NOT the random mixed HTTP benchmark prompts.

Then run the full16-lane heterogeneous pipeline with receive timing enabled and vary
Python's global thread switch interval1ms ->5ms ->1ms. Each launch runs two phases
of96 mixed requests at6req/s, with native/GPU endpoint order alternated per phase.
Same native server persists through the sequence. No CPU affinity/NUMA or background
load control; this is exploratory attribution, not proof of a knob's causal benefit.

## Commands

From native_kv, start the dedicated native server per GPU_LOCAL_PAIRED.md first.
MODEL_DIR refers to the pinned model from DIRECT_RECEIVE.md.

```bash
python3 profile_direct_receive.py --out direct_receive_profile.json --repeats 3
# For interval/prefix pairs 0.001/switch1, 0.005/switch5, 0.001/switch1b:
SWITCH_INTERVAL="$INTERVAL" RECEIVE_PROFILE="${PREFIX}_receive.jsonl" python3 profile_pipeline_receive.py --model-dir "$MODEL_DIR" --pipeline --batch-size 16 --native-batch-size 8 --native-batch-wait-ms 8 --norm-mode fused --fuse-mlp --fuse-qkv --native-pack direct --profile-stages "${PREFIX}_stages.jsonl"
python3 bench_arrival_http.py --rates 6,6 --requests 96 --out "arrival_${PREFIX}.json"
# Stop each owned GPU process before launching the next interval.
```

The first1ms launch used an equivalent temporary wrapper with absolute source paths;
the5ms and repeated1ms launches used the checked-in relative-path launcher.

## Results

Isolated receive medians; each row excludes its cold run. Width4 values are per-request
medians over12 samples, not total batch latency. Ready/KV are cumulative milestones;
receive wall/CPU are summed calls, validation is included in receive.

| Input/width | Ready ms | KV-complete ms | Receive wall/CPU ms | Validation ms | First-to-last batch span ms |
|---|---:|---:|---:|---:|---:|
|128/1|21.90|33.72|4.17/4.18|1.00|8.30|
|512/1|49.04|68.52|9.27/9.28|1.17|17.03|
|1024/1|92.74|137.27|12.59/12.59|1.70|38.45|
|1024/4|225.33|361.96|15.71/15.72|1.50|86.31|

Single1024 has an outlier222.69ms total wall time; all raw samples, including cold
runs and the outlier, are retained. Thread CPU can exceed wall by tiny timer overhead.

Full-pipeline runs, every phase96/96 successful:

| Switch interval/run | GPU TTFTp95 phases ms | Native TTFTp95 phases ms | GPU tok/s phases |
|---|---:|---:|---:|
|1ms first|252.44/246.52|159.96/162.74|438.33/438.37|
|5ms|277.20/285.44|161.92/167.34|438.57/437.26|
|1ms repeat|262.91/271.98|164.77/164.18|438.76/438.95|

At1024 input tokens (64 samples per run), median summed receive wall/CPU times are
14.12/13.77ms,14.03/13.21ms,15.37/15.03ms respectively. First-to-last batch spans are
82.53ms,120.28ms,101.12ms. Median between-call gaps are4.07ms,5.87ms,5.31ms;
p95 gaps6.00ms,7.26ms,6.88ms. Most elapsed time therefore lies outside the timed
receive function, not in its validation/copy work.

These gaps also include routing/polling and other requests; they do NOT independently
identify network time, vendor readback, libzmq scheduling, GIL acquisition or NUMA
traffic. Small receive wall-minus-CPU excludes large waits INSIDE that function,
not GIL waits elsewhere. Isolated versus pipeline input/load differ, so their timing
difference cannot all be attributed to GPU contention.

## Conclusion

Header validation (~1.7ms at1024) is not a promising target for eliminating the roughly
90ms additional delivery interval. Reducing Python's switch interval is insufficient:
repeat1ms differs materially from first1ms, and all remain above native TTFT. Retain
the engine default. No new general-quality or full native-performance claim.

Next: fix CPU/NUMA placement for the receiver and libzmq I/O threads, measure transport
and decode interference with matched prompts/load, then choose process separation or
transport changes only if evidence supports them. All owned servers stopped afterward.
Compact results:receiver_profile_summary.json; raw isolated samples:direct_receive_profile.json.
