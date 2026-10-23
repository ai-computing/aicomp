# Six-thousand-token semantic handoff (all four models)

Final comparison count:31/32; unresolved Gemma arithmetic drift is tracked in #233. This completes this measurement matrix, not the full serving goal.

- **Date**: 2026-10-23
- **Task**: EOS-aware four-model validation (aicomp #227)
- **Host / device**: RNGD+ATOM host, one owned RNGD allocation and one ATOM allocation; GPU A10080GB. Furiosa 2026.4 environment; pinned artifacts from #227.
- **Commit**: c645fff (producer implementation unchanged from 5e19f0e).

## Question

Do the same four EOS-aware semantic questions preserve continuations across RNGD prefill and GPU/ATOM decode at approximately 6K input tokens? Models are Qwen3-0.6B, Qwen3-4B, KONI-8B and Gemma3-1B; Qwen2.5 is excluded.

## Setup

Use the existing capital/arithmetic/Korean/retrieval cases with 400 background-note repetitions. Compile width8192, retain true token length, run three producer repeats and decode up to64 tokens or EOS. Check exact token IDs and native/imported continuation equality, KV readback, and original/portable importer logits. These fixtures do not establish broad answer quality; shared wrong answers remain documented in the native EOS record.

## Command

Exact accelerator launch commands, using the isolated wrappers from preceding #227 experiments:

```bash
bash run-koni219.sh --width 8192 --partitioned --group 1 --rms-tile 512 --fixtures semantic227-koni-fixtures-eos.json --cases capital-400 arithmetic-400 korean-400 retrieval-400 --output semantic227-koni-rngd8192
bash run-qwen227.sh --model Qwen/Qwen3-0.6B --width 8192 --experimental-long-width --fixtures semantic227-qwen06b-fixtures.json --cases capital-400 arithmetic-400 korean-400 retrieval-400 --output semantic227-qwen06b-rngd8192
bash run-gemma223.sh --width 8192 --fixtures semantic227-gemma-fixtures.json --cases capital-400 arithmetic-400 korean-400 retrieval-400 --output semantic227-gemma-rngd8192
bash run-qwen227.sh --model Qwen/Qwen3-4B --width 8192 --experimental-long-width --fixtures semantic227-qwen4b-fixtures.json --cases capital-400 arithmetic-400 korean-400 retrieval-400 --output semantic227-qwen4b-rngd8192
```

For each completed producer, run check_rngd_gpu.py with the pinned local model, all four numbered exports, --eos-aware --max-output 64. ATOM uses run-atom-handoff227.sh with the matching HF model, --cache-dir atom-artifacts, --kv-blocks 1 for KONI or3 for Qwen, the four exports and the same EOS/output arguments. Gemma requires its corrected global-causal artifact and specialized checker.

## Results

| Model | Actual input length | Producer runs | GPU EOS matches | ATOM EOS matches | Warm prefill/export seconds |
|---|---:|---:|---:|---:|---:|
| KONI-8B | 6137–6143 | 12 | 4/4 | 4/4 | 2.672–2.966 |
| Qwen3-0.6B | 6026–6031 | 12 | 4/4 | 4/4 | 1.308–1.457 |
| Gemma3-1B | 6423–6428 | 12 | 3/4 | 4/4 | 1.588–1.608 |
| Qwen3-4B | 6026–6031 | 12 | 4/4 | 4/4 | 2.744–2.982 |

KONI64 and Qwen56 KV slots read back exactly. ATOM original and portable imports match native tokens and each other's logits. Native GPU and ATOM token lists also agree for these eight cases. Timing ranges are observations, not normalized serving performance or overlap measurements.

Initial queue launch failed because the Pod container was not ready; its terminal failure log was preserved before restarting. Successful queue has completed KONI and Qwen0.6B and is executing Gemma; Qwen4B follows. Do not start duplicate producers while this queue is live. Local log: analysis/serving-matrix-201/semantic227-long-queue.log; failed launch: semantic227-long-queue-launch-failed.log. Per-model producer JSONL, GPU/ATOM checker JSON and raw exports remain in the same local analysis directory. Raw .pt files are not committed.

## Conclusion

Partial result: 16/16 model/question/decoder comparisons match through EOS. The two other models are still pending, and live PD serving, genuine chunk transfer overlap and end-to-end throughput remain unverified. Queue remains active under its owned allocation; no other workloads were changed.

## Gemma completed: one numerical mismatch retained

Gemma12prefill repeats are finite/exact; exact fixture IDs and unpadded exported lengths verified.52KV slots read back on GPU and ATOM, with exact native ATOM state roundtrip. GPU matches3/4; ATOM4/4. The long arithmetic case yields30 on native GPU (BF16 and FP32),25 after RNGD prefill on either GPU or ATOM; native ATOM also yields25. Both answers are wrong for17+25. The first generated token already differs, so the recorded producer logits differ before KV transport/decode. Transport readback is exact, but this does not establish compute equivalence or identify the upstream numerical cause. A GPU execution of the same layer equations is being investigated separately.

Completed long-context comparisons now23/24, not24/24. Qwen3-4B is compiling in the existing queue after Gemma exited0. Exact extra check: `python check_rngd_gpu.py --model "$GEMMA_SNAPSHOT" --exports semantic227-gemma-rngd8192-1.pt --eos-aware --max-output 64 --dtype float32 --output semantic227-gemma-long-arithmetic-fp32.json`. Raw GPU/ATOM result JSON and producer JSONL are retained.

## Layer-equation diagnostic (#233)

The unchanged GemmaLayer equations executed on GPU, at matching padded width8192 and both BF16/FP32, pick first token3, unlike RNGD token2. BF16 first-layer K/V relative RMS versus RNGD is0.00001666/0.000000205, rising to0.07479/0.11380 in the last layer. FP32 logits for3 and2 are33.33634/33.24598 (margin0.09036); BF16 logits32.5/29.625. This supports accumulated execution/numerical differences before transfer; it does not identify the causal operator or certify all equations/precisions. Both generated answers remain wrong. Follow-up issue:https://github.com/ai-computing/aicomp/issues/233.

Reproduction: `CUDA_VISIBLE_DEVICES=0 python check_gemma_equations.py --model "$GEMMA_SNAPSHOT" --export semantic227-gemma-rngd8192-1.pt --width 8192 --output semantic227-gemma-long-equations-final.json`. TF32 disabled, GPU A10080GB, four CPU threads, pinned weights, same gemma_parts.py equations. This is a diagnostic, not a latency benchmark.

## Final Qwen3-4B and matrix completion

The original queue exited0 for all four models. Qwen3-4B12prefill runs are finite/repeat-exact, with4/4 GPU and4/4 ATOM EOS comparisons. True fixture IDs and KV lengths verified (36layers,8KV heads,128head dimension);72slot readbacks exact, both ATOM importers and nativeGPU/ATOM sequences agree. All four models now total48producer runs and31/32 model/question/decoder comparisons. The earlier pending/partial entries above describe intermediate checkpoints; no producer queue remains live. Final summary:semantic227-long-summary.json. The one Gemma mismatch is retained, not waived. Serving performance, real producer integration, cache restoration and overlap remain open.
