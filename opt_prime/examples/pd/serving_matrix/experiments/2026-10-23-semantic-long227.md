# Six-thousand-token semantic handoff (partial results)

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
| Gemma3-1B | pending | pending | pending | pending | pending |
| Qwen3-4B | pending | pending | pending | pending | pending |

KONI64 and Qwen56 KV slots read back exactly. ATOM original and portable imports match native tokens and each other's logits. Native GPU and ATOM token lists also agree for these eight cases. Timing ranges are observations, not normalized serving performance or overlap measurements.

Initial queue launch failed because the Pod container was not ready; its terminal failure log was preserved before restarting. Successful queue has completed KONI and Qwen0.6B and is executing Gemma; Qwen4B follows. Do not start duplicate producers while this queue is live. Local log: analysis/serving-matrix-201/semantic227-long-queue.log; failed launch: semantic227-long-queue-launch-failed.log. Per-model producer JSONL, GPU/ATOM checker JSON and raw exports remain in the same local analysis directory. Raw .pt files are not committed.

## Conclusion

Partial result: 16/16 model/question/decoder comparisons match through EOS. The two other models are still pending, and live PD serving, genuine chunk transfer overlap and end-to-end throughput remain unverified. Queue remains active under its owned allocation; no other workloads were changed.
