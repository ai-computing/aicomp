# Gemma HTTP c16 layer streaming

- **Date**: 2026-10-24
- **Task**: RNGD layer streaming (aicomp #265), partial four-model HTTP matrix
- **Host / device**: GPU benchmark host; A10080GB PCIe, driver580.178.04; PD additionally one RNGD, Furiosa2026.4 environment
- **Commit**: production33d6cb08ec806f693dbecfd7d630331a3bd1bb2f; launch head5978c35d9d478a62c6e30819f03e7962925c1ae1. Later commits evidence/tests only. Manifest retained in evidence265/stream265-http-c16-manifest.json.

## Question

Does Gemma layer streaming preserve whole-source output and improve HTTP latency under sixteen queued requests?

## Setup and command

Gemma3-1B BF16,4execution slots,4source slots,sourcewidth8192,context9216,cachesoff. Inputs128/512/2048/8192,outputs32/128,c16. Each mode8warmups,128retained-token checks,128timed SSE text checks. One measured batch per condition. Native uses one GPU; PD adds one RNGD. Exact driver and its environment/dependencies retained in evidence265/run_stream265_http_c16.py; it invokes the issue248 benchmark and llmd240 HTTP module, whose hashes are in the manifest.

```bash
/home/gspark/venvs/pt213/bin/python analysis/serving-matrix-201/run_stream265_http_c16.py --run
```

## Results

Allthree modes exit0 and independently pass audit_result. Whole/stream reference arrays match exactly in8/8conditions. Native/whole strict references match4/8: first differences at index8 for512input and24 for2048input, but firstEOS is index2 on both paths. Generation EOS IDs[1,106]; trimming through and including firstEOS yields8/8 equality. At8192 there is noEOS in32/128tokens and outputs remain exact. This post-hoc distinction preserves strict mismatches and does not waive earlier Gemma content/numerical failures.

| Input | Output | Native TTFT s | Whole TTFT s | Stream TTFT s | Stream throughput change vswhole % |
|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 1.44 | 12.78 | 12.50 | 2.27 |
| 128 | 128 | 5.01 | 12.70 | 12.63 | 1.51 |
| 512 | 32 | 1.66 | 12.83 | 12.42 | 3.17 |
| 512 | 128 | 5.19 | 12.83 | 12.49 | 1.56 |
| 2048 | 32 | 3.08 | 12.92 | 12.51 | 2.52 |
| 2048 | 128 | 6.68 | 12.84 | 12.53 | 1.97 |
| 8192 | 32 | 12.65 | 13.65 | 12.65 | 2.10 |
| 8192 | 128 | 13.94 | 13.48 | 13.11 | 2.19 |

## Conclusion

Observed throughput+1.51–3.17% is small and needs repeated measurements before claiming reliable improvement. Do not generalize Qwen's larger gains to Gemma. Native remains substantially faster for short prompts; fixedwidth cost remains unisolated. No device DMA-overlap, equal-resource advantage or model-quality claim. KONI GPU HTTP and remaining ATOM candidate cases are still active. Raw files stream265-gemma-http-c16-{native,whole,stream}.json, derived observations and EOS comparison are in evidence265.
