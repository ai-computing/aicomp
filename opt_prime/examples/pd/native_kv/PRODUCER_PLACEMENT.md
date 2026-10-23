# Native producer placement follow-up

- Date:2026-10-22; issue:aicomp #152; PR153.
- Commit:cba6b9a plus producer_placement_manifest.json diagnostic hashes.
- Same pinned Qwen3-8B-FP8 revision, FXB hash, SDK and hardware as DIRECT_RECEIVE.md.
- GPU+RNGD host:one RNGD native reference versus RNGD+A100 heterogeneous path.
- RNGD npu0 PCI topology reports node1 (CPUs3–5); GPU0 reports node2 (CPUs6–9).

## Hypothesis and commands

Receiver-only locality did not reproduce a TTFT improvement. Hold the GPU process
on node2 and alternate native default/local/default launches to test producer locality.
MODEL_DIR, FXB and NATIVE_CLI identify the pinned artifacts and2026.4 environment;
OUT is a new result directory. Run from native_kv with no servers on the test ports:

```bash
python3 bench_producer_placement.py --model-dir "$MODEL_DIR" --fxb "$FXB" --native-cli "$NATIVE_CLI" --out-dir "$OUT" --producer-node 1 --gpu-node 2
python3 summarize_placement.py --input-dir "$OUT" --out producer_placement_summary.json
```

Each case starts fresh owned native and GPU servers. Native uses npu:0, no prefix
cache, TcpConnector poll_timeout_ms1. GPU uses direct receive,16 lanes,native batch8,
8ms gather window,fused norms/QKV/MLP, Python switch interval5ms. GPU always launches
under numactl --cpunodebind=2 --preferred=2. Candidate native prepends numactl
--cpunodebind=1 --preferred=1; references use default placement.

Each case runs deterministic6req/s,96 mixed requests twice with endpoint order
alternated. Inputs128/512/1024,outputs32/128. Candidate also runs nine long retrieval
cases. Process-group snapshots include native children if present; one native process
was observed in each snapshot. All owned groups are stopped before the next case.

## Placement verification and limitation

Candidate native threads all have affinity3–5; default native threads have0–23.
GPU threads remain6–9 in every case. However, preferred memory did NOT achieve native
node1 locality: only20.62MiB of2365.54MiB (0.9%) resides on node1;2019.38MiB (85.4%)
is on node2. References have only0.1% on node1. These snapshots distinguish the actual
placement from the requested policy. The experiment tests the combined launch policy,
not successfully localized native memory. It cannot reject a hard-local-memory
hypothesis, and we do not force memory allocation on the pressured node.

## Results

All twelve endpoint phases96/96 successful. Native baseline and heterogeneous hardware
budgets differ; tok/s includes finite-run drain, not saturation capacity.

| Native placement | GPU TTFTp95 phases ms | Native TTFTp95 phases ms | GPU tok/s phases |
|---|---:|---:|---:|
|default0|257.91/289.33|169.98/162.30|437.77/438.99|
|local1|325.15/335.64|183.67/183.53|436.53/436.77|
|default2|253.76/268.82|166.54/164.49|439.22/438.08|

1024-token medians,64 samples each. Native readiness/KV completion are cumulative;
receive span includes inter-message gaps.

| Placement | Ready ms | KV complete ms | Receive wall/CPU ms | Receive span ms | GPU import ms |
|---|---:|---:|---:|---:|---:|
|default0|92.88|201.06|14.80/13.57|95.76|6.80|
|local1|98.03|250.73|13.84/13.04|144.25|6.78|
|default2|92.52|194.39|13.65/12.89|90.57|6.78|

Candidate quality:9/9 expected long retrieval answers and9/9 native/solo/concurrent
text matches at592/1152/1712 input tokens. Reference cases did not repeat that suite.
The summary generator also reproduces the prior receiver-placement summary byte for
byte after adding producer snapshots. No engine/parser/arithmetic changes in this PR.

## Conclusion

Do not recommend this producer launch policy: it worsens native and heterogeneous
latency and does not realize the requested memory locality. Three available native
CPUs versus24 may impose a concurrency cost, but CPU and memory effects were not
separated. Defaults remain unchanged. Native-performance goal remains unmet.

Next hypothesis (issue154):installed connector configuration metadata contains
kv_pipeline_group_layers. The current18-message stream covers two layers per message.
Verify that option's actual framing effect before changing bounded receiver support;
then test whether grouping reduces inter-message overhead. This setting has NOT yet
been exercised, and its name alone is not evidence of supported behavior.

All owned processes exited and GPU was idle after the experiment. Compact metrics,
verified affinities and memory distribution are in producer_placement_summary.json.
