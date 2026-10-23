# KONI short-input numerical compatibility

- **Date**: 2026-10-24
- **Task**: aicomp #261; four-model matrix #201
- **Host / device**: GPU+RNGD host; A10080GB, RNGD SDK2026.4; GPU Torch2.13; producer Torch2.10+cpu
- **Commit**: serving runtime32ce5d5, research producer source retained in local201-source-hashes.json

## Question
Why does short128-input128-output RNGD→GPU differ before EOS at index64 when nativeGPU and same-size HF-prefill import agree?

## Setup
KONI revision0172bf7e73977ae213613529f3d2d5400a33f678. Original native chunk256 vs RNGDwidth8192, BF16, GPU4slots/context9216/page64/cacheoff. Prior completed GPU padded-prefill control changes output atindex11 but matches neither native nor actualRNGD; padding alone is insufficient. Long8192input128output can be reproduced by nativeGPU full8192prefill, separately.

## Command
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/tmp/rngd-gpu201-src:/home/gspark/workspace/aicomp-live-pd-234/opt_prime/examples/pd/serving_matrix:/home/gspark/workspace/aicomp-dense-hybrid-258/opt_prime /home/gspark/venvs/pt213/bin/python analysis/serving-matrix-201/export_koni_short261.py > analysis/serving-matrix-201/koni261-short-export.log 2>&1
```
Owned worker uses oneSHMslot, same model/factory/width/context as original. Compute is independent of number of transport slots. Exact original128input IDs supplied; two exports cloned before next request; assert samefirsttoken and bit-identicalK/V. GPU and NPU process lists empty before start. No production sampling modification.

## Results
Export completed: repeat_exact=true, firsttoken578, BF16keys/values[32,8,128,128]. Initialization135.956s, two observed prefill+export calls3.309s/3.025s (diagnostic only, not a matched benchmark). Both tensors cloned before the next call; retained17MB file stays local. Expected retained tensor shape32x8x128x128, then sameGPUdecoder replay must reproduce live128outputs before matched-prefix diagnostics are interpreted. Originalfailedmatrix, long-input controls and paddedGPUcontrol remain unchanged.

## Conclusion
Numerical cause unresolved. Pending live shortKV capture/replay and identical-prefix comparisons; neither lifecycle success nor one alternate GPU geometry establishes general equivalence.

## Identical-prefix diagnostic running
Executed `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/home/gspark/workspace/aicomp-dense-hybrid-258/opt_prime /home/gspark/venvs/pt213/bin/python analysis/serving-matrix-201/diagnose_koni261_logits.py > analysis/serving-matrix-201/koni261-forced-logits.log 2>&1` after GPU process list was empty. Script first requires native128 and retainedKV128-token outputs reproduce their respective original sequences. Then forces common64tokens and probes63/64indices underBF16/FP32 native and retainedBF16KV paths; also measures per-layer KV differences. Production sampling unchanged. Pending results, no numerical claim yet.

## Replay and forced-prefix diagnosis complete
NativeBF16 reproduces originalnative128tokens; retainedshortRNGDKV reproduces original live128tokens exactly. Four forced-prefix runs complete, all probes finite. At index64 (token65), candidate(native3940,live8712) logits:

|Decoder|KV source|Native candidate|Live candidate|Selected|
|---|---|---:|---:|---:|
|BF16|native128|14.875|14.875|3940|
|BF16|retainedRNGD|14.8125|14.9375|8712|
|FP32|native128|14.8171244|14.8624229|8712|
|FP32|retainedBF16RNGD|14.8413544|14.8718882|8712|

Raw koni261-forced-logits.json also includes per-layer KV errors. This establishes sensitivity at a tied BF16 native decision and rules out a live-only transfer explanation for this fixture, not general equivalence or answer quality. FP32 native changes bothprefill anddecode arithmetic; retainedKV staysBF16. Next controlled hypothesis: isolate FP32 output projection from full FP32 decoder before proposing any productionprecision option. Do not change tie-breaking or force fixture tokens in production.

## Output projection precision control
Only the LM output projection is recomputed inFP32; all original model parameters/hidden calculations and importedKV remainBF16. Common64-token forced prefix unchanged. Native logits3940/8712=14.8758430/14.8851137 and retainedRNGDKV=14.8079271/14.9206982; both select8712. This resolves the tied decision at this position only. Free-generation equivalence, broader prompts and performance remain unverified; no production change.

## Reproduction
Portable forced-prefix entry point: `PYTHONPATH=opt_prime python opt_prime/examples/pd/http_matrix/forced_prefix_koni261.py --model /path/to/KONI-8B --export /path/to/koni261-short-kv.pt --live opt_prime/examples/pd/http_matrix/evidence261/local201-koni-matrix-a.json --output /tmp/koni261-forced-logits.json`. CLI adaptation syntax/help checked; original equivalent executed on hardware. The17MB shortKV and1.1GB longKV dumps stay local. Evidence includes exact executed local diagnostic scripts; their environment paths must be adapted when reproducing elsewhere. ShortKV capture uses the published livePD research factories from#235 and installed SDK2026.4.

## Unforced output-head precision control completed
check_head_precision_free.py first reproduces all originalBF16native andretainedKV128token sequences, then computes onlyoutputprojectionFP32 withBF16model/KV. Complete=true. KONI128input native/import diverge atindex14 (earlier than original64); KONI8192input withmatchedfullprefill remains128tokens exact; Qwen4long matchedfullprefill diverges atindex13 (earlier than original107). No forcedtokens used in these runs. Raw head-fp32-free-generation.json. Therefore head-onlyFP32 is rejected as a general cross-backend equivalence fix; no production precision option added. First-divergence index measures equality, not semantic quality. Fixed-prefix agreement at onebranch did not generalize.
