# Qwen3-4B RMS gamma-carrier precision hypothesis (not adopted)

- **Date**: 2026-10-23
- **Task**: Qwen3 RMS precision (aicomp #202)
- **Host / device**: RNGD+ATOM host allocatedRNGD fusion4/Mesh2; GPU reference onA10080GB
- **Commit**: prefill d4b6fe6 plus isolated rms_precision_variant.py in exp/201-serving-matrix

## Question

Does rounding gamma*sqrt(hidden_size) toBF16 explain accumulated Qwen3-4B KV error? Hidden size2560 has a non-exact square-root scale, unlike the earlier1024 case.

## Setup

Same Qwen3-4B revision1cfa9a7208912126459214e8b04321603b3df60c and deterministic128/512token fixtures as the four-model matrix. Change only normalization gamma carriers toFP32 (input/post-attention/Q/K norms); all other math, widths and exports unchanged. Generate the variant from the trusted PR200 source without editing the installed engine. Reference is HF GPU FP32; fixed32-token target decode. ATOM check uses both existing importers against the same compiled graph.

## Command

```bash
./furiosa_prefill_python.sh -m rngd_qwen_sweep --model Qwen/Qwen3-4B \
  --width 128 --fp32-rms-weights --output rngd-qwen3-4b-128-fp32rms
./furiosa_prefill_python.sh -m rngd_qwen_sweep --model Qwen/Qwen3-4B \
  --width 512 --fp32-rms-weights --output rngd-qwen3-4b-512-fp32rms
python3 check_rngd_gpu.py --model "$QWEN3_4B_SNAPSHOT" --dtype float32 \
  --exports rngd-qwen3-4b-128.pt rngd-qwen3-4b-128-fp32rms.pt --output rms-128.json
# Repeat comparison for512; check_rngd_atom.py exercises both imports.
```

## Results

RelativeRMS percentages against GPUFP32:

| Input | K baseline→candidate | V baseline→candidate | logits baseline→candidate |
|---|---:|---:|---:|
|128|3.531→0.954|11.930→3.366|1.448→1.892|
|512|2.237→2.173|7.924→8.000|2.142→2.013|

Both variants match the32-token GPU reference continuation for both inputs. OnATOM, both variants match the standalone route only at512tokens;128tokens diverges after the common answer. Both importers' logits remain exact and all576 aggregate slot readbacks succeed across2widths×2precision variants×2importers×72slots.

Warm candidate times128:178.028/131.622ms;512:163.024/167.772ms. Different CPU contention/compilation conditions prevent a performance conclusion. Do not compare these to baseline timings as a controlled speedup.

Raw evidence: analysis/serving-matrix-201/rngd-rms-precision-comparison.json, rngd-rms-precision-512-comparison.json, rngd-atom-qwen3-4b*.json.

## Conclusion

Hypothesis is only partially supported:128-token KV improves, but logits do not, and512-token V does not improve. Do not adopt this as a general fix. Next: isolate normalization/attention errors with more prompts and intermediate-layer references, and measure performance under matched contention. The full four-model serving goal remains active.
