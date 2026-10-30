# Qwen HTTP c16 streaming comparison

- **Date**: 2026-10-24
- **Task**: RNGD layer streaming (aicomp #265), partial four-model HTTP matrix
- **Host / device**: GPU benchmark host; A10080GB PCIe, driver580.178.04; PD additionally one RNGD with Furiosa2026.4 environment.
- **Commit**: production33d6cb08ec806f693dbecfd7d630331a3bd1bb2f; launch head5978c35d9d478a62c6e30819f03e7962925c1ae1. Later commits evidence/tests only; captured hashes in evidence265/stream265-http-c16-manifest.json.

## Question

Does layer streaming preserve same-producer HTTP outputs at c16, and how does latency vary with input length?

## Setup and command

BF16, four GPU execution slots and four PD source slots. Fixed source width8192, context9216, caches off. Each mode:8 warmups,128 retained-token correctness responses and128 timed SSE text responses. One measured batch per condition. Native uses one GPU; PD adds one RNGD.

```bash
/home/gspark/venvs/pt213/bin/python analysis/serving-matrix-201/run_stream265_http_c16.py --run
```

Executed from the research workspace. Retained driver uses benchmark.py from the issue248 worktree and llmd240.py HTTP module; hashes in manifest. Driver runs four models sequentially; only completed Qwen cases appear here. No runtime edits during runs.

## Results

Six modes exit0 and independently pass audit_result. Both models have identical whole/stream reference arrays. Qwen0.6 native/whole references match all8 conditions; Qwen4 native/whole differ before EOS in4/8 conditions (retained difference JSON). Same-route success does not resolve cross-backend numerical equivalence.

| Model | Input | Output | TTFT native / whole / stream (s) | Throughput native / whole / stream (tok/s) |
|---|---:|---:|---|---|
| qwen06b | 128 | 32 | 1.18 / 9.00 / 6.29 | 171.9 / 29.3 / 40.8 |
| qwen06b | 128 | 128 | 4.17 / 8.91 / 7.28 | 185.6 / 105.7 / 123.7 |
| qwen06b | 512 | 32 | 1.42 / 8.92 / 6.51 | 150.2 / 29.3 / 39.5 |
| qwen06b | 512 | 128 | 4.61 / 8.77 / 7.45 | 172.6 / 105.9 / 121.0 |
| qwen06b | 2048 | 32 | 2.91 / 9.62 / 7.09 | 81.9 / 27.5 / 36.5 |
| qwen06b | 2048 | 128 | 5.84 / 9.83 / 7.88 | 144.4 / 98.2 / 115.2 |
| qwen06b | 8192 | 32 | 12.89 / 11.72 / 9.44 | 20.1 / 24.5 / 28.2 |
| qwen06b | 8192 | 128 | 13.98 / 11.87 / 10.67 | 68.7 / 87.1 / 92.8 |
| qwen4b | 128 | 32 | 1.48 / 23.60 / 18.43 | 138.1 / 11.4 / 14.5 |
| qwen4b | 128 | 128 | 5.22 / 23.29 / 18.40 | 147.3 / 43.5 / 54.0 |
| qwen4b | 512 | 32 | 1.92 / 23.41 / 18.44 | 112.0 / 11.4 / 14.4 |
| qwen4b | 512 | 128 | 5.69 / 23.66 / 18.62 | 139.6 / 42.6 / 53.2 |
| qwen4b | 2048 | 32 | 4.13 / 24.50 / 19.60 | 58.7 / 10.9 / 13.5 |
| qwen4b | 2048 | 128 | 7.92 / 24.62 / 19.46 | 108.1 / 41.3 / 51.7 |
| qwen4b | 8192 | 32 | 20.26 / 28.52 / 23.09 | 12.9 / 10.1 / 12.3 |
| qwen4b | 8192 | 128 | 21.94 / 26.95 / 21.51 | 45.5 / 38.3 / 46.8 |

## Conclusion

Qwen4 stream improves observed whole-path throughput22.1–27.2% in this run. At8192/128 its46.77tok/s is near native45.46tok/s, with one additional accelerator. Short inputs remain substantially slower than native GPU. Fixed-width padding is a candidate explanation supported by code inspection, not isolated causal measurement. These single batches do not prove statistical speedup, equal-resource advantage, model quality or device DMA overlap. Gemma/KONI HTTP cases and ATOM candidate matrix remain in progress. Original ATOM reset is tracked separately in#267, not waived by GPU results.
