# GPU-pipeline CPU and NUMA placement experiment

- Date:2026-10-22; issue:aicomp #152.
- Commit:604ad99 prerequisite PR151 plus placement_manifest.json diagnostic hashes.
  Stacked experiment branch because the native direct receiver is not on main.
  Latest fetched main:62a783114afa3531beeb7955c66da2e6aaf4ba65.
- Hardware:GPU+RNGD host; native baseline one RNGD, heterogeneous one RNGD+A10080GB PCIe.
  GPU0 locality:NUMA node2, CPUs6–9.24 CPUs across8 NUMA nodes.
- Software/checkpoint/FXB:exact versions and hashes in DIRECT_RECEIVE.md.
- Native endpoint remains unbound, unchanged across all three GPU launches; no prefix cache.

## Hypothesis

GPU-local CPU and preferred host-memory placement may reduce KV message gaps and
TTFT variability. Limit changes to the owned GPU process and descendants; no global
CPU/memory settings or unrelated process affinities changed. This tests the receiver
side only, not native producer placement or hardware-isolated CPU cores.

## Setup and exact commands

Start the dedicated native server as GPU_LOCAL_PAIRED.md. Run from native_kv with
MODEL_DIR pointing to the pinned Qwen3-8B-FP8 checkpoint and OUT to a new result folder.

```bash
python3 bench_placement.py --model-dir "$MODEL_DIR" --out-dir "$OUT" --node 2
python3 summarize_placement.py --input-dir "$OUT" --out placement_summary.json
```

The controller alternates local0/default1/local2. Local launches prepend
`numactl --cpunodebind=2 --preferred=2`; default launches inherit unrestricted CPUs.
All use SWITCH_INTERVAL=0.005 and the timed direct receiver,16 GPU lanes,native batch8,
8ms gather wait,fused norms/QKV/MLP. The exact child commands are in bench_placement.py.
Each launch runs bench_arrival_http.py --rates6,6 --requests96: deterministic6req/s,
input128/512/1024, output32/128; native/GPU endpoint order alternates per phase.
A new GPU process is loaded each time; native stays alive. First local launch also
runs the nine long retrieval cases. No overnight soak or load-rate sweep is implied.

## Verified placement

Every observed thread in each local GPU process has affinity6–9. Default process
threads have0–23. numastat snapshots after the arrival benchmark show83.9% of process
resident memory on node2 for both local runs versus5.8% for default. Remaining pages
include shared/file-backed mappings; preferred placement does not promise hard binding.
Process memory totals are about2241–2249MiB at these snapshots, excluding GPU HBM.

## Results

All twelve endpoint phases complete96/96 requests successfully. Finite-run throughput
includes drain. Native and heterogeneous hardware budgets differ.

| Placement | GPU TTFTp95 phases ms | Native TTFTp95 phases ms | GPU tok/s phases |
|---|---:|---:|---:|
|local0|267.46/251.44|170.38/163.04|438.97/438.98|
|default1|266.58/262.02|161.25/162.70|438.98/439.16|
|local2|276.79/278.69|163.59/161.73|438.42/438.15|

1024-token stage medians (64 samples/run); ready/KV values are cumulative milestones,
not additive costs. Receive wall includes validation and copying. Span is first receive
entry through last receive exit and includes inter-message gaps.

| Placement | Native ready ms | KV ready ms | Receive wall/CPU ms | Message span ms | GPU import ms |
|---|---:|---:|---:|---:|---:|
|local0|92.99|195.41|13.05/12.32|89.30|6.78|
|default1|91.71|201.43|15.34/14.81|97.79|6.79|
|local2|91.01|221.09|13.22/12.69|118.33|6.78|

Local0 long retrieval592/1152/1712 tokens:9/9 exact expected answers on native, solo,
and concurrent GPU paths;9/9 texts equal. Other placements did not repeat this quality
suite. No decoder arithmetic or engine defaults changed.

## Conclusion and limitations

Receiver-local placement reduces timed receive work by about2ms but does not produce
a reproducible end-to-end benefit: the second local run is slower than default.
Do not promote it to an engine default or claim native parity. The large interval
between messages remains; GPU import is unchanged. Native producer CPU/memory and
background scheduling were not controlled, so sender-side locality/interference remains
an open hypothesis. The local restriction also reduces available CPUs24->4, which can
trade locality against contention.

Next: isolate native producer placement and its transfer cadence with matched prompts;
separate sender, libzmq and decode scheduling before implementing a new transport path.
All owned servers stopped after this experiment. Compact metrics, affinity and memory
snapshots are in placement_summary.json. Local raw files retain all request measurements,
server logs and receive intervals; generated text is excluded from the published summary.

Follow-up:PRODUCER_PLACEMENT.md tests producer default/local/default with GPU placement fixed. The candidate worsens latency; preferred native memory locality is not realized.
