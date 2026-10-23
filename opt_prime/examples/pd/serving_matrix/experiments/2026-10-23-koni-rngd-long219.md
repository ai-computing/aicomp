# KONI long-prefix RNGD handoff

- **Date**: 2026-10-23
- **Task**: aicomp #219, parent #201
- **Host / device**: RNGD+ATOM host, isolated second RNGD allocation; Furiosa2026.4 fusion4/Mesh2. ATOM standard B1/context9216/k1 artifact planned. Other users' workloads unchanged.
- **Commit**: aicomp base314a43d plus exp/219-koni-long runner/export helper/checker option. rngd_llmd HEADf807da295deead1d2779444ebf263174122bd350; actual dependency file hashes below.

## Question

Can the already-used Llama3.1 RNGD graph produce long KONI KV for the standard GPU and ATOM decoders, rather than only the earlier short external-cache graph?

## Setup

KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024 revision0172bf7e73977ae213613529f3d2d5400a33f678. Strict fixed graph geometry:32layers,hidden4096,intermediate14336,32query/8KVheads,128head width, unbiased SwiGLU, Llama3 RoPE. Reuse FullPrefillEngine with2-layer groups; skip its device lm_head and use CPU FP32 final RMS/head as in the Qwen probe. Export each group to CPU and concatenate/crop into[layer,head,token,dim]. No device-side full-cache padding or assembly, no overlap claim. The exporter refuses incomplete stages.

Exact-length background/capital-question fixture2048tokens, three repeated prefills, finite tensors and bit-exact repeat checks, then planned32-token decoder continuation. This allocation's compiler overlaps the separate Qwen4B8K compiler, so timing is diagnostic, not a controlled performance comparison.

Dependency SHA256:
- engines/prefill_full.py:0ac864695d8c2cedc5fb2ddadbc3d3d782bb21909a83f2b4d82944f32940e8ea
- engines/model.py:e9f89d5bde73015d6f630a14cf498a34cab9c753367d7afe2d0ec93e81fdc547
- kernels/prefill_full.py:4fb91421d5501e6f103fd6f8cf9bd26f481a275ecfd60516f8e9b4d9bf569807

## Command

```bash
python3 -m pytest -q opt_prime/examples/pd/serving_matrix/test_koni_export.py
# PR203 isolated SDK launcher; run-koni219.sh selects rngd_koni_sweep:
bash run-koni219.sh --width 2048 --output rngd-koni-8b-2048
bash run-koni219.sh --width 2048 --rms-tile 512 --output rngd-koni-8b-2048-rms512
# Same code,128GiB allocation after confirmed OOMKilled:
bash run-koni219.sh --width 2048 --rms-tile 512 --output rngd-koni-8b-2048-rms512-mem128
bash run-probe-rms219.sh
python3 -m pytest -q opt_prime/examples/pd/serving_matrix/test_rowwise_rms.py opt_prime/examples/pd/serving_matrix/test_koni_export.py
# After successful export, planned:
# check_rngd_atom.py --model KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024 --cache-dir atom-artifacts --kv-blocks 1 --exports rngd-koni-8b-2048.pt --output rngd-atom-koni-8b-2048.json
```

## Results

CPU export regression tests:3 fail with unimplemented stubs, then3 pass after implementation. Tests cover incompatible geometry/semantics rejection, layer/head/position order and prefix cropping, final-position normalization/head, and incomplete-stage rejection. Ruff passes after converting dict constructors to literals.

Baseline2048 exits1 before any inference: rms_output [2048,4096] fails lowering (ReduceByVe, no populated tactics). An isolated same-SDK probe compiles and runs RMS shapes[512,4096] and[256,4096], producing finite zero-input outputs (1.903/1.348s including compile/setup); it does not establish general numerical parity.

Candidate retains2048-token full attention but tiles only independent RMS rows into512. CPU dense-versus-tiled normalization and incomplete-tile rejection tests fail with stubs then pass; combined exporter/tiler suite5 passed. Candidate with64GiB limit exits137; Kubernetes container termination reason OOMKilled confirms resource exhaustion before any inference. Failed Pod status retained privately. Host available memory was1374GiB; recreate only the terminal owned allocation with128GiB request/limit and retry the same code. The128GiB retry is pending; no fix claim yet. Raw workspace log: analysis/serving-matrix-201/rngd-koni-8b-2048.log.

## Conclusion

The adapter and diagnostic are prepared and CPU-tested. The baseline identifies a compiler shape constraint in RMS, not a transport failure. Candidate full-model compilation, numerical agreement and decoder handoff remain unproven until this run completes. Keep full four-model objective active.
