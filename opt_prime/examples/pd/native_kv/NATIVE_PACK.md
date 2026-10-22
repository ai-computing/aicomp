# Native KV packing: validated NumPy views to owned pinned storage

- Date: 2026-10-22
- Task: aicomp #150; related #146
- Commit: 2cee630 parent plus native_pack_manifest.json; baseline pipeline identical
  to ad2649e, benchmark scripts from PR149.
- Hardware: GPU+RNGD host, one RNGD native prefill plus A10080GB PCIe decode;
  native baseline uses one RNGD only. GPU process snapshot15970MiB.
- Software/model: same pinned Qwen3-8B-FP8 revision and FXB as GPU_LOCAL_PAIRED.md,
  furiosa-llm2026.4.0, torch2.10.0+cu128, transformers5.1.0,vLLM0.18.0.

## Question and implementation

The heterogeneous path approaches native throughput but pays additional KV delivery,
CPU packing and GPU import latency. At1024 input tokens, baseline median CPU pack
is33.40ms after KV completion207.97ms. Stage milestones are cumulative, not separate
kernel durations.

`--native-pack numpy` in the native pipeline copies validated uint16 NumPy slot views
directly into an owned CUDA-pinned BF16 packet using byte-preserving np.copyto. It
avoids72 temporary Torch views/copy dispatches. No protocol/parser changes, no alias
of transport frames, no model/arithmetic changes. Same packet layout and fenced GPU
import. Slot count, completion,shape,dtype and capacity are checked. Default remains
Torch packing; the option requires --pipeline.

## Commands

Native server as GPU_LOCAL_PAIRED.md; commands from native_kv with MODEL_DIR pinned:

```bash
python3 bench_native_pack.py --out native_pack_micro.json
python3 serve_gpu.py --model-dir "$MODEL_DIR" --pipeline --batch-size 16 --native-batch-size 8 --native-batch-wait-ms 8 --norm-mode fused --fuse-mlp --fuse-qkv --profile-stages hetero_stages.jsonl
python3 bench_arrival_http.py --rates 6,6 --requests 96 --out arrival_hetero16.json
python3 check_long_http.py --model-dir "$MODEL_DIR" --out long_quality_hetero16.json
# Stop GPU and native servers; restart native and the same GPU command with:
# --native-pack numpy --profile-stages hetero_numpy_stages.jsonl
python3 bench_arrival_http.py --rates 6,6 --requests 96 --out arrival_hetero_numpy16.json
python3 check_long_http.py --model-dir "$MODEL_DIR" --out long_quality_hetero_numpy16.json
# Stop GPU HTTP; keep the dedicated native endpoint for the worker check:
python3 check_pipeline_backpressure.py --model-dir "$MODEL_DIR" --batch-size 16 --native-batch-size 8 --native-pack numpy --peer-count 7 --prompt-tokens 512 --out native_pack_backpressure.json
python3 -m unittest discover -s . -p 'test_*.py'
```

## Results

Synthetic native-layout frames, alternating modes,2 warmups/7 measured iterations.
Checks compare raw16-bit packet contents, including NaN bit patterns. Timings include
allocation and view construction; no CPU affinity/NUMA control. These are isolated
copy measurements, not serving throughput:

| KV positions | Torch pack ms | NumPy pack ms |
|---:|---:|---:|
|127|3.46|2.40|
|511|16.40|13.68|
|1023|24.29|11.58|
|1711|40.03|21.00|

All measured packet bytes match. Tests mutate/free transport backing storage after
copy and verify owned packet contents, CUDA pinning, partial/malformed rejection.

Live native/GPU pipeline per-request CPU pack medians (input includes final prompt
position not sent in KV):128 tokens5.35->3.22ms;512 tokens18.20->12.85ms;
1024 tokens33.40->14.48ms. These stage samples include warmup/quality calls grouped
by input length;128/512/1024 are benchmark lengths. Full stage medians retained.

Open-loop deterministic6req/s,96 mixed requests per phase (inputs128/512/1024,
outputs32/128), two phases, alternating endpoint order. Completed tok/s includes
finite-run drain time. Each phase has its own native baseline:

| Packing/repeat | Native/GPU tok/s | Native/GPU p95 TTFT ms |
|---|---:|---:|
|Torch/0|441.63/437.43|165.84/364.87|
|Torch/1|441.79/437.66|162.88/334.67|
|NumPy/0|441.63/438.39|163.28/273.76|
|NumPy/1|441.82/437.83|168.02/276.37|

All96 requests succeed in each phase. Throughput is essentially unchanged;
first-token latency improves but remains far above native. Torch/NumPy tuning runs
are separate and lack controlled NUMA/host contention; paired native values and
repeated phases support, but do not prove exact attribution of every millisecond.
Hardware budgets differ: this path consumes RNGD+A100 versus native RNGD alone.

Baseline and candidate each match9/9 long retrieval answers exactly and native/solo/
concurrent text9/9 (592/1152/1712 tokens). This validates this transfer path on these
cases; previous GPU/native arithmetic differences remain documented elsewhere.

## Conclusion

Keep the faster byte-preserving packing path opt-in pending review. Native readiness
to KV completion remains a larger latency component than packing. Installed pyzmq27.1
exposes recv_into; next hypothesis is receiving validated payload frames directly
into pinned packet regions, with strict multipart/size handling and ownership tests.
That path is not implemented or measured here. Full native-performance goal remains
active; finite mixed-load/limited retrieval checks are not global parity.

## Validation follow-up and failed criterion

39 unit tests pass. A separate live packet check receives the SAME native KV once
and packs it both ways, avoiding separate native executions as a numerical oracle:

```bash
python3 check_native_pack.py --out native_pack_live.json
```

Inputs128/512/1024: all bytes match, CUDA pinning is true, and packet contents remain
unchanged after releasing transport views. No raw KV is saved or published.

The strict live backpressure test does NOT pass at16 lanes/7 peers: the candidate
with512 repeated-token input fails cross-peer text equality. Repeating the same
512-token test with --native-pack torch ALSO fails (first different character19).
Candidate128-token input also fails late cross-peer equality (character248).
Each compared peer reports128 generated tokens. This shows the failure is not
exclusive to NumPy packing, but does not prove its cause or backpressure correctness.
Do not reinterpret a baseline failure as a candidate pass. Failure logs and compact
diagnostics are retained; full16-lane synthetic output/paused-cache analysis remains
open. Nine long retrieval matches are narrower evidence and do not supersede this.

Reproduction uses the worker command above with --native-pack torch for the baseline,
and --native-pack numpy --prompt-tokens128 for the shorter candidate. A fresh dedicated
native server was used for the final same-packet check. All owned servers/checkers
exited; GPU idle. Draft review only; no main merge.
