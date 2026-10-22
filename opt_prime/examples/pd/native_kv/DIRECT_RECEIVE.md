# Native KV direct receive into pinned storage

- Date: 2026-10-22
- Task: aicomp #150; draft PR151
- Commit: f457736 plus source hashes in direct_receive_manifest.json.
- Hardware: GPU+RNGD host; one RNGD prefill and one A10080GB PCIe decode.
  Native baseline uses one RNGD only; hardware budgets are unequal.
- Software: furiosa-llm/native2026.4.0, torch2.10.0+cu128, transformers5.1.0,
  vLLM0.18.0, pyzmq27.1.0, NVIDIA580.178.04.
- Model: Qwen3-8B-FP8 revision220b46e3b2180893580a4454f21f22d3ebb187d3,
  TP8/2048 FXB SHA25636d74b754f580bea893c68cc70147c615d1a9c4273a4580dbebaf87287400713.

## Question and change

Can the separate host KV packing stage be removed without changing packet values or
GPU decode behavior? The opt-in `--native-pack direct` mode validates each header,
then receives the payload directly into an owned CUDA-pinned BF16 packet region.
It uses pyzmq recv_into (requires that API, available since26.4). libzmq still copies
payload bytes into this buffer: this is not zero-copy transport.

The 2026.4 observed layout sends consecutive slots; all18 batches in a live512-token
sample satisfy this. Unsupported nonconsecutive slots are rejected. Request IDs,
metadata bounds, slot uniqueness, positions, final completeness and frame lengths
remain validated. A truncated oversized message, short payload, or extra multipart
frame poisons the receiver; ACK is sent only after complete validated KV. Default
Torch mode is unchanged. Dedicated experimental loopback endpoints only.

## Commands

Run from native_kv with MODEL_DIR set to the pinned checkpoint. Native server setup
is in GPU_LOCAL_PAIRED.md, prefix caching disabled. Restart the dedicated native
endpoint after protocol errors. No other GPU benchmark runs concurrently.

```bash
python3 serve_gpu.py --model-dir "$MODEL_DIR" --pipeline --batch-size 16 --native-batch-size 8 --native-batch-wait-ms 8 --norm-mode fused --fuse-mlp --fuse-qkv --native-pack direct --profile-stages hetero_direct_stages.jsonl
python3 bench_arrival_http.py --rates 6,6 --requests 96 --out arrival_hetero_direct16.json
python3 check_long_http.py --model-dir "$MODEL_DIR" --out long_quality_hetero_direct16.json
# Stop GPU HTTP before running the standalone checker against native:
python3 check_same_cache_pause.py --model-dir "$MODEL_DIR" --direct --out direct_same_cache_pause.json
python3 -m unittest discover -s . -p 'test_*.py'
```

## Results

Deterministic6req/s,96 mixed requests per phase, input128/512/1024 and output32/128.
Two phases alternate endpoint order; every phase completes96/96 successfully.
Throughput includes finite-run drain and does not establish saturation capacity.

| Phase | Native/direct tok/s | Native/direct TTFT p95 ms | Native/direct completion p95 ms |
|---|---:|---:|---:|
|0|442.04/438.54|161.92/250.15|2765.89/1691.81|
|1|441.91/438.13|164.01/265.24|2983.69/1700.02|

Prior NumPy packing runs measured274/276ms TTFTp95 and Torch335/365ms. These were
separate tuning runs, not randomized interleaved direct/NumPy trials. Paired native
baselines and two repeats are retained; exact improvement attribution needs further
controlled runs. Direct mode still has1.5-1.6x native first-token latency.

Stage medians (ready and KV-complete are cumulative milestones from submission):

| Input tokens | Native ready ms | KV complete ms | Separate pack ms | GPU import ms |
|---:|---:|---:|---:|---:|
|128|21.88|41.22|0.003|1.14|
|512|50.04|101.61|0.003|3.50|
|1024|90.83|183.60|0.003|6.80|

The near-zero pack column measures packet lookup; CPU copies are now inside receive.
It must not be interpreted as zero CPU work or as GPU-direct transport.

43 unit tests pass, including real ZMQ exact payload and oversized-truncation tests.
Long retrieval592/1152/1712 tokens:9/9 exact expected answers and9/9 identical
native/solo/concurrent output. These are narrow quality checks, not general parity.
Same-cache pause checks:6144/6144 output positions match uninterrupted runs across
shared, mixed-length and parallel-native caches; paused valid KV/input/position/count
remain unchanged. Four separately computed native packets differ from packet0 in
this run, beginning at slot2, yielding two output sequences. Thus cross-peer equality
of separately computed native caches is not a pause correctness oracle. Earlier
failed strict peer-equality runs remain documented in NATIVE_PACK.md.

## Conclusion and next step

Separate host packing is removed with validated framing and owned pinned lifetime.
First-token latency remains above native; native readiness to KV completion is now
the largest visible additional interval (~93ms at1024). Separate producer serialization,
receiver validation/copy, and scheduling costs before selecting the next optimization.
No full native-performance or equal-cost claim. No weights, raw KV or vendor binaries
are published. Compact results are direct_receive_summary.json and
direct_same_cache_pause.json; source hashes identify the measured implementation.
