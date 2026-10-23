# Projection fusion and real-prompt quality smoke checks

- **Date**: 2026-10-22
- **Task**: Native-engine performance parity (aicomp #141 / Draft PR #142)
- **Host / device**: GPU+RNGD host, RNGD npu0 and A10080GB PCIe cuda0; same firmware/driver/software as colocated record
- **Commit**: 766e27e973f7a6cb3303549f0e2dd89ff7166552 + projection/quality changes; source hashes attached

## Question

Can projection fusion reduce GPU decode time while retaining task behavior, and does the resulting native-KV continuation approach native throughput?

## Setup

Same pinned Qwen3-8B-FP8 revision220b46e3b2180893580a4454f21f22d3ebb187d3 and FXB SHA25636d74b754f580bea893c68cc70147c615d1a9c4273a4580dbebaf87287400713. Local native SDK2026.4.0; separate GPU torch2.10.0+cu128, transformers5.1.0, vLLM0.18.0. Native FP8 and GPU W8A16 arithmetic differ. Native uses1RNGD, heterogeneous uses1RNGD+1A100. Workload is single-flight; no batch/concurrency claim. Host load was not controlled; later observed10.85/11.27/16.66, lower than earlier contended runs. Use within-run native comparisons, not unrelated historical native numbers.

Gate/up weights and128x128 scale blocks are concatenated without requantizing. Decode Q/K/V projections are likewise combined; original QKV weights remain for ordinary HF prefill, costing approximately0.9GB extra GPU memory. Projection options are disabled by default. Fully fused RMSNorm remains opt-in because it changes some synthetic continuations.

## Commands

```bash
python3 "$CODE/check_live_quality.py" --model-dir "$MODEL_DIR" --out analysis/native_kv_quality/real_prompts.json
python3 "$CODE/check_live_quality.py" --model-dir "$MODEL_DIR" --fuse-mlp --out analysis/native_kv_quality/real_prompts_mlp.json
python3 "$CODE/check_live_quality.py" --model-dir "$MODEL_DIR" --fuse-mlp --long-only --out analysis/native_kv_quality/long_prompts_mlp.json
python3 "$CODE/check_live_quality.py" --model-dir "$MODEL_DIR" --fuse-mlp --fuse-qkv --out analysis/native_kv_quality/real_prompts_qkv.json
python3 "$CODE/bench_gpu_decode.py" --model-dir "$MODEL_DIR" --weight-mode marlin --attention flash --fused-norms --norm-mode torch-stats --fuse-mlp --baseline analysis/native_kv_perf_2026_4/gpu_baseline.json --out analysis/native_kv_quality/fused_mlp.json
python3 "$CODE/bench_gpu_decode.py" --model-dir "$MODEL_DIR" --weight-mode marlin --attention flash --fused-norms --norm-mode fused --fuse-mlp --fuse-qkv --baseline analysis/native_kv_perf_2026_4/gpu_baseline.json --out analysis/native_kv_quality/fused_qkv.json
python3 "$CODE/bench_live_gpu.py" --model-dir "$MODEL_DIR" --packed-import --fused-norms --norm-mode fused --fuse-mlp --cpu-threads 1 --repeats 6 --transport local-loopback --out analysis/native_kv_quality/fast_live.json
python3 "$CODE/bench_live_gpu.py" --model-dir "$MODEL_DIR" --packed-import --fused-norms --norm-mode fused --fuse-mlp --fuse-qkv --cpu-threads 1 --repeats 6 --transport local-loopback --out analysis/native_kv_quality/qkv_live.json
python3 -m unittest discover -s "$CODE" -p 'test_*.py' -v
```

The first two quality runs used the initial12-case version; add --short-only when reproducing them with the final script. Three long-context cases were then added (321/801/1569 input tokens); the final QKV run covers all15. Exact prompts and expected answers are in check_live_quality.py. Thinking disabled through the chat template, greedy decoding. Native stops on EOS; GPU computes128 tokens then truncates at the checkpoint's first EOS for scoring. Thus these quality timings are NOT a native/GPU speed comparison.

Performance runs retain prior seed139 synthetic prompts, input128/512/1024, output128, ignore EOS, warmup1, three GPU-only repetitions or six alternating native/live repetitions. Native is an HTTP SSE response; GPU continuation is an in-process fixed-budget CUDA graph call. It computes the first token with a fence then enqueues remaining replays, without providing per-token HTTP streaming. The faster GPU timing does not prove equivalent streaming server behavior. HTTP/streaming integration, EOS control and multi-request scheduling remain open.

## Quality results

All three norm variants produced exactly the native rendered text in the12 original cases, before and after MLP fusion. Three longer retrieval cases also match. With both MLP+QKV fusion, all three norm variants match native in all15 cases. Native and every GPU variant score12/15 on the strict-format checker. The three format failures are shared: `No.` rather than `no`, `서울특별시` rather than `서울`, and an arithmetic explanation rather than a bare integer. This is not a semantic-accuracy score or broad quality benchmark. The checker intentionally rejects booleans/floats in an integer JSON array; its tests pass.

GPU-only MLP+torch-stats generation preserves the previous Flash token sequence at every position for the three synthetic cases. Combined fully fused-norm/QKV configuration matches BF16 reference114/128 positions at input128 and128/128 at512/1024. Prior synthetic arithmetic divergence is not erased by the15-question smoke check.

23 unit tests pass. Real checkpoint execution validates every layer of the fused projections, but numerical equivalence across arbitrary inputs remains unproven.

## Performance results

GPU-only MLP+torch-stats TPOT10.66/10.77/10.80ms; prior torch-stats10.90/10.99/11.03 (sequential comparison). Combined QKV+MLP+fully fused norm TPOT8.21/8.31/8.35ms.

Median live path metrics, six repetitions each:

| configuration | input | mode | TTFT ms | TPOT ms | tok/s |
|---|---:|---|---:|---:|---:|
| fast_live | 128 | heterogeneous | 57.98 | 9.09 | 105.58 |
| fast_live | 128 | native | 21.37 | 9.68 | 102.39 |
| fast_live | 512 | heterogeneous | 131.63 | 9.19 | 98.50 |
| fast_live | 512 | native | 49.52 | 9.76 | 99.30 |
| fast_live | 1024 | heterogeneous | 239.68 | 9.23 | 90.68 |
| fast_live | 1024 | native | 89.41 | 9.68 | 97.04 |
| qkv_live | 128 | heterogeneous | 54.21 | 8.21 | 116.67 |
| qkv_live | 128 | native | 21.22 | 9.66 | 102.51 |
| qkv_live | 512 | heterogeneous | 119.97 | 8.31 | 108.89 |
| qkv_live | 512 | native | 48.96 | 9.66 | 100.24 |
| qkv_live | 1024 | heterogeneous | 217.95 | 8.35 | 100.11 |
| qkv_live | 1024 | native | 89.15 | 9.67 | 97.21 |

## Conclusion and limits

The combined configuration reaches1.138x/1.086x/1.030x native median throughput in this single-request execution-path benchmark. It still has much worse TTFT (approximately54/120/218ms vs21/49/89ms) and a larger hardware budget. GPU output delivery differs from native SSE; these results are not proof of production serving parity. Preserve opt-in flags and the disclosed precision differences. Next: compare both through equivalent HTTP streaming, test EOS/cancellation and real prompts under concurrency, and address TTFT/placement. Goal remains active. All owned native/GPU processes stopped after runs.

Artifacts: analysis/native_kv_quality/{summary,source_manifest,real_prompts,real_prompts_mlp,long_prompts_mlp,real_prompts_qkv,fused_mlp,fused_qkv,fast_live,qkv_live}.json and logs. No weights, FXB or KV payloads are published.
