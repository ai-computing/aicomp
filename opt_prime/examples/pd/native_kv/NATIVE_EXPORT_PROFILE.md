# Native export spans and suffix-transfer feasibility

Issue #160, follow-up #162. Parent e6c4a92. Same pinned model, FXB, SDK and hardware
as INTERFERENCE.md; GPU decode is stopped for all these diagnostics. No production
engine, layout validator or transport default changes. Source hashes are recorded in
native_export_profile_manifest.json. Native servers are dedicated owned processes.

## Measurement

Unprivileged perf is denied (perf_event_paranoid=4); noninteractive sudo is unavailable.
No system permissions were changed. The installed native engine supports
FURIOSA_PROFILER_OUTPUT_PATH. Its JSON trace is flushed on shutdown and contains
Generator and DeviceRuntime spans. Raw traces remain local; only numerical summaries
are published.12 scheduled requests at2/s, lengths128/512/1024 repeated four times,
plus one excluded128 warmup. First occurrences of512/1024 can still be cold buckets.

Commands from native_kv; start the native endpoint as in GPU_LOCAL_PAIRED.md:

```bash
# Prefix the native launch environment; use a fresh trace path for each run.
# FURIOSA_PROFILER_OUTPUT_PATH="$OUT/native-export-profile.json"
python3 profile_native_arrivals.py --rate 2 --requests 12 --out "$OUT/native_export_profile12.json"
# Stop the owned native server to flush its trace.
python3 summarize_native_export.py --trace "$OUT/native-export-profile.json" --arrivals "$OUT/native_export_profile12.json" --out "$OUT/native_export_spans.json"
# Repeat with FURIOSA_PROFILER_ASYNC_SPAN_SEPARATE=1 and fresh output names.
```

Median observed spans, milliseconds (18 groups per request, four Stream::d2h calls
per group; groups contain four K/V slots):

| Input | Full read-span | Group duration | Scatter calls/group | D2H calls/group | After last submission/group |
|---|---:|---:|---:|---:|---:|
|128|7.36|0.362|0.021|0.064|0.233|
|512|33.81|2.202|0.040|0.151|1.751|
|1024|72.73|3.419|0.057|0.254|2.979|

The last column is unattributed elapsed time inside dma_read_payload after its last
recorded submission returns. It may include completion waiting, scheduling and
untraced postprocessing; it is NOT a direct hardware-DMA duration. Nested enqueue
spans are not added to their parent D2H spans. Per-group medians do not add to a
per-request median. The async-separate=1 run still exports X spans without an explicit
busy/wait split; it does not resolve the remaining attribution. Both runs complete12/12.

## Bounded peer-layout probe

Test exactly the original descriptor and a dense head-major permutation of the same
axes, shape, dtype, allocation extent and block sizes. Use distinct receiver ports so
an existing peer registration cannot mask the changed descriptor. Only the diagnostic
constructor admits these two exact descriptors; normal validate_layout stays strict.

```bash
python3 probe_peer_layout.py --out "$OUT/layout_compare.json"
```

Run128/512/1024, four repetitions with alternating order; exclude repeat0 from timing.
All12 paired KV hashes match, but no performance win across lengths:

| Input | Default KV-complete ms | Head-major KV-complete ms |
|---|---:|---:|
|128|38.83|27.63|
|512|109.48|105.39|
|1024|168.79|203.18|

Do not adopt this descriptor. Equality on these synthetic prompts is not general quality
validation. An earlier2-token probe used one endpoint and is not included in this table.
The first comparison script had a duplicate dict keyword and recorded zero rows; fixed
before the24-request run. No protocol error or engine failure occurred.

## Native suffix selection

A more promising path reduces transferred bytes while retaining native full prefill.
The control request's start-position field (u32 at root+16) selects the source suffix;
provide exactly length-1-skip destination IDs. The diagnostic assembler validates only
that shorter contiguous destination range, keeping all slot/header/payload checks.

```bash
python3 probe_suffix_transfer.py --out "$OUT/native_suffix_probe.json"
```

| Input / skipped prefix | Exported positions | Bytes | Equals corresponding full-native suffix |
|---|---:|---:|---|
|128 /64|63|9,289,728|yes, all72 slots|
|512 /256|255|37,601,280|yes, all72 slots|
|1024 /512|511|75,350,016|yes, all72 slots|

These are independent native executions on the same synthetic prompt, not two reads
of one captured cache. Every compared byte matches in these three cases. Full requests
run first and include cold-shape costs, so do NOT claim an end-to-end speedup from their
raw latencies. The1024 suffix request's native-ready/KV-ready milestones are94.35/135.69ms.
An initial assembler wrapper omitted the optional callback argument and failed before
the partial request was submitted; fixed before the three completed pairs.

## Next

Issue #162 tests GPU prefix recomputation concurrently with native full prefill, then
imports only native suffix KV. Causal prefix KV does not depend on later tokens, but GPU
and native arithmetic differ: suffix equality alone does not validate the combined cache.
Require meaningful generation/quality checks, paired latency and bounded memory/admission
before serving integration. Do not replace the full-transfer default based on this probe.
Native-performance goal remains active; no new end-to-end performance improvement claimed.
