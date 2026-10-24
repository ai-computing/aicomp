# Four-model GPU HTTP streaming matrix completed

- **Date**: 2026-10-24
- **Task**: RNGD layer streaming (aicomp #265)
- **Host / device**: GPU benchmark host; A10080GB PCIe,driver580.178.04; PD additionally one RNGD,Furiosa2026.4 environment
- **Commit**: production33d6cb08ec806f693dbecfd7d630331a3bd1bb2f;launch head5978c35d9d478a62c6e30819f03e7962925c1ae1. Later changes evidence/tests only; captured hashes in evidence265/stream265-http-c16-manifest.json.

## Question and setup

Compare native GPU, whole RNGD KV handoff and layer streaming for Qwen3-0.6B,Qwen3-4B,Gemma3-1B,KONI-8B. BF16,sourcewidth8192,context9216,cachesoff,c16 behind4execution slots;inputs128/512/2048/8192,outputs32/128. Each of12modes has8warmups,128retained-token checks and128timed SSE text checks. One measured batch per condition, not statistical speedup evidence. Native uses one GPU; PD adds one RNGD. Exact driver, configuration and dependencies documented in the preceding Qwen/Gemma HTTP records.

## Command

```bash
/home/gspark/venvs/pt213/bin/python analysis/serving-matrix-201/run_stream265_http_c16.py --run
```

## Results

Queue session41030 exited0. All12child modes exit0 and independently pass audit_result. Totals:96conditions,96warmups,1536retained-token checks,1536timed SSE text checks. Whole/stream reference arrays identical for allfour models across all8conditions. Independent summary http265-four-model-gpu-summary.json in evidence265.

| Model | Native/whole exact references | Whole/stream exact references | Remaining numerical distinction |
|---|---:|---:|---|
| Qwen3-0.6B |8/8|8/8|Measured fixtures only|
| Qwen3-4B |4/8|8/8|Four differences beforeEOS;#256 remains|
| Gemma3-1B |4/8|8/8|All strict differences afterEOS;throughEOS8/8 on these fixtures;earlier#233/#263 failures remain|
| KONI-8B |5/8|8/8|Three differences beforeEOS;#261 remains|

KONI first divergent indices64/56/98 for128/2048/8192input with128output. EOS union128001/128009 verified; noEOS in these references. Short fixture and both fulloutputs exactly reproduce the earlier#261diagnosis. Head-onlyFP32 remains rejected as a general fix. Exact fixtures and hashes retained.

| KONI input | Output | Native TTFT s | Whole TTFT s | Stream TTFT s | Stream throughput change vswhole % |
|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 1.33 | 25.79 | 20.55 | 23.78 |
| 128 | 128 | 4.20 | 25.70 | 20.69 | 20.94 |
| 512 | 32 | 1.76 | 25.29 | 20.23 | 24.72 |
| 512 | 128 | 4.80 | 25.33 | 20.68 | 22.29 |
| 2048 | 32 | 4.39 | 26.05 | 20.60 | 24.35 |
| 2048 | 128 | 7.39 | 25.69 | 19.95 | 28.73 |
| 8192 | 32 | 21.29 | 29.50 | 22.12 | 33.93 |
| 8192 | 128 | 22.67 | 28.24 | 20.85 | 33.29 |

## Conclusion

KONI's observed stream throughput gain20.94–33.93% joins the larger Qwen gains; Gemma's1.51–3.17% needs repeats before a reliable improvement claim. No equal-resource advantage, device DMA overlap, native numerical equivalence or answer quality is established by this matrix. ATOM remaining models and width-cost experiments continue; overall four-model objective remains active. Once GPU queue exited and the RNGD device-user check was empty, separate issue268source probes started with Qwen0.6width512. Those measurements are not part of these unchanged HTTP runs.
