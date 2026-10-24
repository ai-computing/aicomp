# ATOM HTTP streaming: three models verified, KONI incomplete

- **Date**: 2026-10-24
- **Task**: RNGD layer streaming (aicomp #265); HTTP reliability (#267)
- **Host / device**: RNGD+ATOM benchmark host; one RNGD and one allocated ATOM
- **Commit**: production 33d6cb08ec806f693dbecfd7d630331a3bd1bb2f; HTTP-only listener candidate uses backlog 128
- **Environment**: Furiosa 2026.4 source environment; ATOM harness Torch 2.11.0+cpu / Transformers 5.8.1

## Setup and command

Source width 8192, BF16, context 9216, decoder B1, concurrency 16,
caches disabled. Inputs 128/512/2048/8192, outputs 32/128. Eight warmups,
128 retained-token correctness responses and 128 timed SSE text responses per
successful mode. One timed batch per condition. Fixed output length ignores EOS.
Use the existing source factories and ATOM configuration from STREAMING.md;
model-specific configs and raw artifacts are retained by the source hashes in
`evidence265/http267-atom-source-hashes.json`.

Recorded research-driver invocation, with site values supplied externally:

```bash
python3 analysis/serving-matrix-201/run_http267_atom.py \
  --host "$ACCELERATOR_SSH_TARGET" --pod "http267-$MODEL_LABEL" \
  --model "$MODEL_LABEL"
```

## Verified results

Fresh independent audit checks whole and stream for Qwen3-0.6B, Qwen3-4B and
Gemma3-1B: 48 conditions, 48 warmups, 768 full-token checks and 768 timed SSE text
checks. Whole/stream prompts and reference arrays match exactly in all eight
conditions per model. Qwen3-4B and Gemma native modes also completed and were
independently audited. Summaries are in evidence265.

At input 8192/output 128, throughput in tokens/s:

| Model | Native ATOM | Whole KV | Layer streaming |
|---|---:|---:|---:|
| Qwen3-0.6B | Not compared in this table | 33.0665 | 43.3496 |
| Qwen3-4B | 12.2876 | 12.6716 | 14.5315 |
| Gemma3-1B | 57.1134 | 39.4864 | 43.5681 |

Qwen3-4B streaming improves whole-transfer throughput by 10.8–27.4% across the
measured conditions; Gemma improves by 2.6–15.6%. Gemma remains slower than
native ATOM in all eight conditions. Native uses one ATOM; PD adds one RNGD,
so these are not equal-resource or cost comparisons. Qwen3-4B native/PD has
pre-EOS differences in two conditions. Same-source transfer parity does not
establish native equivalence or semantic answer quality.

## Retained failure and remaining scope

KONI native ATOM failed the original 180-second socket response wait at
8192/128/c16. Seven earlier correctness conditions (112 responses) and eight
warmups completed; no timed batch completed. Server remained alive at the
failure snapshot, with zero OOM events and zero listener drops/overflows.
This does not establish a queueing-only cause. Whole/stream were not launched.
A distinct 600-second diagnostic is running; it does not erase or fix the
original deadline failure. The failed-run summary is retained in evidence265.

The listener candidate eliminated observed overflow/drop counters in a paired
Qwen3-0.6B run, but the original connection-reset cause is not conclusively
proven. It is diagnostic-only and is not integrated into this PR's runtime.

Separate width-1024 experiments (#268) are also not part of this PR: Qwen3-0.6B
has a heldout answer regression and KONI has width-dependent token divergence.
Production source width/default behavior is unchanged here. Repeated timing,
full KONI ATOM coverage, and direct device-DMA overlap remain outstanding.
