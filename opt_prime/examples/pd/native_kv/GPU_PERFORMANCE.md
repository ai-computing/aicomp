# Native KV to A100: real hand-off and decode optimization

- **Date**: 2026-10-22
- **Task**: Native-engine performance parity (aicomp #141)
- **Host / device**: RNGD+ATOM host, RNGD npu3; separate GPU host, A100 80GB PCIe cuda:0
- **Commit**: f14d0e0294dc5166ffd9dadacfec0d89feb2685c + uncommitted perf/141-fp8-gpu-decode; exact Python hashes in analysis/native_kv_gpu_fp8/source_manifest.json

## Question

Can Marlin weight-only FP8, FlashAttention and contiguous KV upload close the full-serving performance gap?

## Setup

Pinned Qwen/Qwen3-8B-FP8 revision 220b46e3b2180893580a4454f21f22d3ebb187d3. Native FXB SHA256 36d74b754f580bea893c68cc70147c615d1a9c4273a4580dbebaf87287400713. Native SDK 2026.4.0, GPU torch 2.10.0+cu128, transformers 5.1.0, vLLM 0.18.0. Native is FP8 engine arithmetic; A100 Marlin is W8A16, not equivalent native FP8 arithmetic.

Seed 139, synthetic token IDs in [1000,30000), lengths 128/512/1024, 128 greedy output tokens, concurrency one, one warmup then three measured repetitions. Native and heterogeneous requests alternate order. Native uses one RNGD; heterogeneous uses one RNGD plus one A100 on another host. HTTP and ZMQ use loopback SSH forwarding (including a reverse receiver tunnel); this is NOT co-located PCIe performance. Prefix cache disabled, connector poll_timeout_ms=1. Native SSE first nonempty text defines native TTFT; token decoder timing defines heterogeneous TTFT. Native TPOT derived from stream duration, not engine events.

Host contention is not controlled: final remote load averages 77.35/51.78/52.11 and another user's profiling job on npu1 was visible. Native TPOT changed between runs, so cross-run end-to-end gains are not a controlled speedup claim. Component medians do not necessarily sum to total medians.

## Commands

Run from the research workspace; MODEL_DIR is the local mirror of the pinned checkpoint and CODE is the worktree's opt_prime/examples/pd/native_kv directory. Site paths are intentionally parameterized.

```bash
python3 "$CODE/bench_gpu_decode.py" --model-dir "$MODEL_DIR" --weight-mode marlin --attention flash --baseline analysis/native_kv_perf_2026_4/gpu_baseline.json --out analysis/native_kv_gpu_fp8/marlin_flash.json
# Same command plus --fused-norms, output marlin_flash_norm.json.
python3 "$CODE/bench_live_gpu.py" --model-dir "$MODEL_DIR" --transport ssh-loopback-forward --out analysis/native_kv_gpu_fp8/live_gpu_retry.json
python3 "$CODE/bench_live_gpu.py" --model-dir "$MODEL_DIR" --packed-import --transport ssh-loopback-forward --out analysis/native_kv_gpu_fp8/live_gpu_packed.json
python3 "$CODE/bench_live_gpu.py" --model-dir "$MODEL_DIR" --packed-import --fused-norms --transport ssh-loopback-forward --out analysis/native_kv_gpu_fp8/live_gpu_norm.json
python3 -m unittest discover -s "$CODE" -p test_protocol.py -v
python3 -m unittest discover -s "$CODE" -p test_memory_receiver.py -v
python3 -m unittest discover -s "$CODE" -p test_gpu_norms.py -v
```

Native command:
```bash
furiosa-llm serve Qwen/Qwen3-8B-FP8 --fxb "$FXB" --devices npu:3 --host 127.0.0.1 --port 19101 --no-enable-prefix-caching --experimental-kv-transfer-config '{"kv_connector":"TcpConnector","host":"127.0.0.1","port":19201,"poll_timeout_ms":1}'
```
Forward local ports 19101/19201 to the same loopback ports on the RNGD host, and remote 19202 to GPU-host loopback 19202, with SSH ExitOnForwardFailure enabled.

## Results

All table values are medians of three post-warmup repetitions.

| Run | Input | Mode | TTFT ms | TPOT ms | end-to-end tok/s | receive ms | import ms |
|---|---:|---|---:|---:|---:|---:|---:|
| live_gpu_retry | 128 | heterogeneous | 288.96 | 11.47 | 73.27 | 200.09 | 51.20 |
| live_gpu_retry | 128 | native | 44.57 | 9.87 | 99.70 | — | — |
| live_gpu_retry | 512 | heterogeneous | 861.43 | 11.57 | 54.91 | 751.95 | 73.64 |
| live_gpu_retry | 512 | native | 73.47 | 10.10 | 94.42 | — | — |
| live_gpu_retry | 1024 | heterogeneous | 1620.91 | 11.60 | 41.36 | 1492.18 | 116.81 |
| live_gpu_retry | 1024 | native | 148.17 | 11.27 | 79.98 | — | — |
| live_gpu_packed | 128 | heterogeneous | 250.91 | 11.47 | 74.91 | 233.78 | 4.61 |
| live_gpu_packed | 128 | native | 44.49 | 9.82 | 99.13 | — | — |
| live_gpu_packed | 512 | heterogeneous | 806.16 | 11.57 | 56.25 | 773.41 | 21.23 |
| live_gpu_packed | 512 | native | 73.00 | 10.00 | 95.27 | — | — |
| live_gpu_packed | 1024 | heterogeneous | 1531.92 | 11.60 | 42.59 | 1487.90 | 16.78 |
| live_gpu_packed | 1024 | native | 140.68 | 11.16 | 82.38 | — | — |
| live_gpu_norm | 128 | heterogeneous | 257.70 | 9.33 | 88.71 | 244.09 | 4.23 |
| live_gpu_norm | 128 | native | 43.87 | 13.58 | 72.32 | — | — |
| live_gpu_norm | 512 | heterogeneous | 935.05 | 9.43 | 60.02 | 816.88 | 15.21 |
| live_gpu_norm | 512 | native | 90.34 | 13.75 | 69.64 | — | — |
| live_gpu_norm | 1024 | heterogeneous | 1590.55 | 9.46 | 45.84 | 1562.79 | 16.97 |
| live_gpu_norm | 1024 | native | 181.81 | 12.89 | 70.72 | — | — |

GPU-only SDPA+Marlin ~17.7 ms/token; experimental atomic-add ~17.7, no useful gain. FA2 GQA removes repeated-KV copies and full-capacity SDPA work: ~11.47/11.57/11.60 ms/token. Fused RMSNorm ~9.33/9.43/9.46, but NOT accepted for exact behavior.

Correctness:
- Contiguous pinned upload preserved all 128 generated tokens at all three prompt lengths and all repetitions relative to original live import.
- GPU-only Flash vs BF16 baseline matches 114/128 positions at input128, 128/128 at 512 and 1024. Earlier conversational claim of complete equality was incorrect and has been corrected.
- GPU-only fused norm matches Flash tokens for all three cases, but live native-KV fused norm matches unfused only 37/128, 104/128, 128/128 positions respectively. Small RMSNorm rounding differences can propagate; norm unit test permits one BF16 ULP and is not a full-model correctness guarantee. Fused norm stays disabled by default.
- Packed live rendered text equals native for input1024 only (3/3 repetitions), not input128 or 512. Native arithmetic parity remains unproven.
- Protocol 10 tests, memory receiver 6 tests, GPU norm 2 tests passed. Added regression: HTTP503 propagates without waiting for KV timeout and poisons failed client.

Failures: first native server attempts on npu0 and npu1 exited EBUSY despite SMI showing no owner at inspection; cause unproven. The first client timed out after 90 seconds. No device reset or other-user termination was performed. npu3 initialized successfully. Owned native server and SSH tunnel stopped after runs.

## Conclusion

Real native-prefill -> in-memory KV -> GPU decode now works, without payload files in the request path. Packed import reduces 1024-token copy median 116.81 -> 16.78 ms; outputs retained. The inter-host SSH KV transfer dominates TTFT (~1.5 seconds at input1024), and full native-speed serving is NOT achieved. Latest faster fused-norm numbers fail exact output checks and cannot justify completion. Next: isolate precision/attention differences with teacher-forced logits, reduce transport or co-locate devices, then evaluate real prompts, concurrency and latency tails on controlled hardware.

Raw artifacts: analysis/native_kv_gpu_fp8/{summary,source_manifest,live_gpu_retry,live_gpu_packed,live_gpu_norm,marlin_flash,marlin_flash_norm}.json and matching logs. Large profiler trace is local-only and not uploaded.

---

# GPU arithmetic isolation and transport bound

- **Date**: 2026-10-22
- **Task**: Native-engine performance parity (aicomp #141)
- **Host / device**: GPU host, A100 80GB PCIe; RNGD+ATOM host for SSH transfer diagnostic
- **Commit**: f14d0e0294dc5166ffd9dadacfec0d89feb2685c + uncommitted perf/141-fp8-gpu-decode; source hashes attached

## Setup and commands

Same model revision and software as 2026-10-22-native-kv-gpu-live.md. No native server launched in this follow-up. GPU was idle before measurements. Six teacher-forced cases: synthetic lengths128/512/1024 seed139 plus English science, Python merge, and Korean heterogeneous-serving questions (exact prompts in check_gpu_logits.py). Each produces128 tokens using eager BF16 reference. Every variant starts with the same CPU-copied BF16 prefix and receives the same previous reference token at each step. Thus decode differences are isolated from prefill and autoregressive drift. This is not native FP8 accuracy validation.

```bash
python3 "$CODE/check_gpu_logits.py" --model-dir "$MODEL_DIR" --out analysis/native_kv_gpu_fp8/teacherforced.json
python3 "$CODE/bench_gpu_decode.py" --model-dir "$MODEL_DIR" --weight-mode marlin --attention flash --fused-norms --norm-mode torch-stats --baseline analysis/native_kv_perf_2026_4/gpu_baseline.json --out analysis/native_kv_gpu_fp8/marlin_flash_torch_stats.json
python3 -m unittest discover -s "$CODE" -p test_gpu_norms.py -v
```

SSH diagnostic: three trials each of aes128-gcm@openssh.com, chacha20-poly1305@openssh.com, aes128-ctr, port22 BatchMode=yes. Remote command `python3 -c 'import os; b=bytes(1024*1024); [(os.write(1,b)) for _ in range(144)]'`, stdout discarded locally. Time includes SSH startup; compression disabled, no connection multiplexing. Read the active routing NIC's sysfs speed on both hosts: **1000 Mb/s** each. This sends synthetic zeros, not model data. Exact local launcher command is retained in session; no site addresses published.

## Results

| Decode variant | teacher-forced top1 matches /768 |
|---|---:|
| BF16 static SDPA | 764 |
| Marlin static SDPA | 764 |
| Marlin FA2 | 766 |
| Marlin FA2 fused norm | 766 |

All mismatched reference top-two margins were <=0.25 (including ties). Some synthetic logit vectors differ substantially despite same argmax: worst cosine approximately0.677 for Marlin FA2 on synthetic1024. Do not infer full numerical equality or production quality from top1 agreement. The control shows cache shape/kernel arithmetic already changes results without weight quantization. Real-prompt sampling and native FP8 task quality remain open.

The new `torch-stats` RMSNorm keeps PyTorch variance and rsqrt, fusing only final multiplication/casts. Three GPU norm tests pass; new bitwise test covers widths128/4096, rows1/8/32/129 and scales0.001/1/100. GPU-only generation matches the unfused Flash path at every output token across three input lengths and three repetitions. Median TPOT10.895/10.994/11.027 ms (previous11.470/11.570/11.603). This is a sequential comparison, not paired controlled timing. Live native-KV parity for torch-stats remains untested; norm optimization stays opt-in.

SSH bulk: 150,994,944 bytes in 1.515–1.562 seconds (96.7–99.7 MB/s); cipher changes do not remove the bottleneck. Active1Gb/s route has an ideal payload floor of1.208 seconds for this byte count before overhead. This explains the observed ~1.5-second KV receive. No network configuration was changed.

Failure: first teacher-forced run completed synthetic references then failed on a real prompt because transformers returned BatchEncoding instead of a token list. Fixed with return_dict=False and explicit long input dtype; rerun completed all24 comparisons.

## Conclusion

Performance target remains unmet. Packet transport on the current1Gb/s inter-host route is a physical limit, while accuracy differences require more than token-sequence matching. We have a faster norm path preserving tested arithmetic, and a reproducible teacher-forced diagnostic. Next: evaluate native-KV torch-stats logits, improve device locality or select a genuinely faster available transport, and test real-prompt end-to-end behavior under matched resource budgets.

Artifacts: analysis/native_kv_gpu_fp8/teacherforced.json, ssh_bulk.json, marlin_flash_torch_stats.json and source_manifest_followup.json. Failed log teacherforced.log and completed teacherforced_retry.log remain local.

Public summary files: gpu_summary.json, gpu_teacherforced.json, gpu_ssh_bulk.json, gpu_source_manifest.json. Raw site logs and model/KV payloads are not distributed.
