# Four-model GPU / RNGD / ATOM serving matrix — ongoing

- **Date**: 2026-10-23
- **Task**: Four-model serving matrix (aicomp #201)
- **Host / device**: GPU host, two A10080GB PCIe; RNGD+ATOM host, dedicated device-plugin allocation
- **Commit**: exp/201-serving-matrix from main fd89265, working changes; RNGD prefill from d4b6fe6 (#200)

## Question

Test the user-selected Qwen3-0.6B, Qwen3-4B, KONI-8B and Gemma3-1B across GPU/RNGD/ATOM, short/long sequences, output lengths, concurrency and cache conditions; improve evidenced failures and bottlenecks. A passing small case does not complete a route.

## Setup

Checkpoints (same weights copied from the NPU host for GPU comparison):

| Model | Revision |
|---|---|
| Qwen/Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca |
| Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c |
| KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024 | 0172bf7e73977ae213613529f3d2d5400a33f678 |
| unsloth/gemma-3-1b-it | 5b11413a10db4e486ef16a20101fd028f8f2499c |

GPU direct-engine baseline: torch2.10.0+cu128, Transformers5.1.0, BF16/SDPA, no prefix cache, fixed-length outputs ignoringEOS. Inputs are exact128/512/2048/8192 tokens of repeated background plus a question; B1/4/16; output32/128; two repeats. No HTTP scheduler is involved in this baseline, and batch is not equivalent to asynchronous HTTP concurrency. GPU0 andGPU1 run separate models concurrently, so host bandwidth contention is possible. No percentile or quality claims from two repeats.

RNGD: Furiosa2026.4, fusion4/Mesh2, Qwen3 research graph fromPR200 with host stacking. Dedicated Pod requests oneRNGD and oneATOM; other workers are untouched. ATOM artifacts and heterogeneous admission are subsequent phases, not inferred from GPU results.

## Command

```bash
# In the serving_matrix example directory, supply a local checkpoint snapshot and its exact revision:
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 gpu_sweep.py \
  --model "$MODEL_SNAPSHOT" --revision "$REVISION" --output "$RESULT_JSONL"
# Extend the default B1/4, output32 sweep:
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 gpu_sweep.py \
  --model "$MODEL_SNAPSHOT" --revision "$REVISION" --batches 1 4 16 --new-tokens 128 --output "$RESULT_128_JSONL"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 gpu_sweep.py \
  --model "$MODEL_SNAPSHOT" --revision "$REVISION" --batches 16 --output "$RESULT_B16_JSONL"
# Isolated RNGD SDK environment, PR195 launcher + PR200 prefill.py/spec.py:
./furiosa_prefill_python.sh -m rngd_qwen_sweep --model Qwen/Qwen3-4B --width 128 --output rngd-qwen3-4b-128
./furiosa_prefill_python.sh -m rngd_qwen_sweep --model Qwen/Qwen3-4B --width 512 --output rngd-qwen3-4b-512
```

Exact local launch scripts and all raw evidence: analysis/serving-matrix-201/. Hardware .pt files are trusted local diagnostic KV/logit exports and are not committed.

## Results (ongoing)

First GPU sweep:64/64 runs completed with finite logits,32/32 same-configuration repeat pairs token-exact. Width128/8192 B1 TTFT for the second repeat:

| Model |128tokens (ms)|8192tokens (ms)|
|---|---:|---:|
| Qwen3-0.6B |13.0|142.0|
| Qwen3-4B |18.3|516.1|
| KONI-8B |21.5|725.9|
| Gemma3-1B |18.2|212.4|

Cross-batch equality is a separate question: B1 vsB4 lane0 diverges after generated position11/7 for KONI at128/512 inputs and position8/24 for Gemma at512/2048 inputs. Both Qwen3 sizes match across all four lengths in the first sweep. No conclusion yet on the cause of these differences; teacher-forced logit/margin diagnosis is pending. Repetition stability does not establish cross-batch numerical parity or factual quality.

RNGD Qwen3-4B width128 compiled in102.825s, then prefill0.849187s first and0.100832/0.099333s warm. All three KV/logit tensors finite; two repeat comparisons bit-exact. KV shape[36,8,128,128]. GPU-oracle and ATOM-admission checks remain pending. Width512 compilation is running separately.

### Extended GPU and numerical findings

All192 direct GPU runs completed:4models ×4input lengths ×3batch sizes ×2output lengths ×2repeats. NoOOM;96/96 repeat pairs token-exact. This establishes finite-logit/repeat stability only. B1 versusB4/B16 lane0 comparisons across both output lengths: Qwen3-0.6B16/16 exact; Qwen3-4B10/16; KONI5/16; Gemma10/16 raw but16/16 when truncated atEOS. Gemma differences in this sweep are post-EOS stress outputs, not different completed answers.

Teacher forcing reproduces KONI B1/B4 divergence at128/512 input and Qwen3-4B B1/B16 divergence at512/2048 input. BF16 top-two margins include exact ties and0.0625–0.25 differences. FP32 repeats at those same prefixes restore identical argmax, with maximum cross-batch logit differences about1.3e-5–8.8e-5. This supports a numerical-precision explanation for the investigated positions; it does not prove every uninvestigated case has the same cause.

RNGD Qwen3-4B width512 warm prefill222.306/228.404ms; first1204.409ms, compile88.813s. Imported128/512-token KV into GPU decode:2/2 fixed32-token outputs match nativeGPU,144/144 K/V slot readbacks exact. However versusGPU FP32, V relativeRMS11.93%/7.92%, K3.53%/2.24%, logits1.45%/2.14%. Numerical fidelity remains an open issue despite token equality.

Issue202 isolates normalization carrier precision. FP32 RMS gamma weights at128tokens reduce K relativeRMS3.53→0.95%, V11.93→3.37%; logits1.45→1.89% (not improved),32decode tokens still exact. Not adopted as a production fix: more prompts/widths and controlled performance measurement are needed. ATOM compilation was concurrent during this candidate's timings, so its131–178ms warm timings are not an A/B performance comparison.

ATOM Qwen3-4B B1/context9216/3KVblocks compiled successfully into the task-owned cache.9216 includes headroom for8192input+128outputs.16/16 direct-batcher runs completed,8/8 repeat pairs exact. Second-repeat TTFT is63ms at128tokens,248–249ms at512,1023–1085ms at2048 and5021–5048ms at8192. All8 first tokens match GPU; full sequences match3/8 (512/2048/8192 with32outputs), so broad cross-device numerical parity is not established.

RNGD→ATOM Qwen3-4B128/512 checks: original vsportable importers yield bit-exact full decode logits in2/2 cases;288/288 imported slot readbacks exact.512-token source matches nativeATOM for32outputs,128-token source diverges after the shared answer prefix and matches GPU continuation instead. The FP32-RMS candidate gives the same1/2 nativeATOM agreement, with another288 exact readbacks. These comparisons isolate transfer correctness from prefill numerical differences. Import timing includes allocation/order effects and is not an importer speedup claim.

At512tokens, FP32-RMS-gamma does not reduce V error (7.92→8.00%); K2.24→2.17%, logits2.14→2.01%. Thus the precision candidate is not adopted; the128-token improvement does not generalize sufficiently. See the separate issue202 experiment.

All launched benchmark commands completed. The allocation Pod subsequently reached its configured30-minute deadline; this is not a benchmark failure. Its task-owned artifacts persist on the host, and the expired namespace was deleted. A fresh allocation is needed for subsequent hardware phases.

## Conclusion / completion checklist

The full user objective remains active. GPU direct-engine stability is the first layer of evidence only. Pending: extended GPU sweep; batch-divergence diagnosis; fresh standalone ATOM sweep including Qwen3-4B artifact; RNGD numerical oracle at longer inputs; missing long-context/model graphs; RNGD→GPU and RNGD→ATOM admission; real HTTP concurrency/cache/cancellation/recovery; before/after verification of resulting improvements. Gemma RNGD prefill is not implemented and must not be marked supported from ATOM/GPU checks. KONI existing B4/context512 graphs do not establish long-context support.
