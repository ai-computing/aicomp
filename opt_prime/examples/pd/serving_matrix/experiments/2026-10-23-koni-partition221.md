# KONI full-context prefill with separate compiled stages

- **Date**: 2026-10-23
- **Task**: aicomp #221, parent #201; follows failed #219
- **Host / device**: RNGD+ATOM host, isolated RNGD allocation,128GiB host-memory request/limit,4requested/16limited CPUs; Furiosa2026.4, fusion4/Mesh2. No ATOM allocated until a valid export exists.
- **Commit**: baseline6787628; exp/221-koni-parts working tree. Dependency hashes inherited from experiments/2026-10-23-koni-rngd-long219.md.

## Question

Can separately compiled RMS, attention and FFN avoid the RMS lowering / fused-layout memory explosion without shortening the2K context or dropping any of the32layers?

## Setup

Pinned KONI8B0172bf7e73977ae213613529f3d2d5400a33f678. Same exact2048-token background/capital fixture and32-token continuation criterion as the existing matrix. Reuse the existing Llama3.1 weight packing/scaled Q projection/RoPE. Compile RMS512 independently; invoke it over every token row outside the attention/FFN graphs and concatenate results on-device. RMS parameters are token-independent and stored as one reusable512-row tile per layer. Attention still computes full2048-token causal context; FFN is a separate full-width graph. Final normalization/output head and KV export use the same CPU helper as #219. No host activation math inside transformer layers, no true token chunking or transfer overlap.

The runtime emits compile_start/compile_done per stage to locate failures. The research engine is stage-only: use begin_device_prefill and export_groups, not the inherited device lm_head path. Production defaults and shared SDK files remain unchanged.

## Command

```bash
python3 -m pytest -q opt_prime/examples/pd/serving_matrix/test_koni_parts.py opt_prime/examples/pd/serving_matrix/test_koni_export.py opt_prime/examples/pd/serving_matrix/test_rowwise_rms.py
# Same isolated SDK launcher as PR220, updated runner/modules:
bash run-koni219.sh --width 2048 --rms-tile 512 --group 1 --partitioned --output rngd-koni-8b-2048-parts221
```

## Results

The initial3 stage tests fail with unimplemented stubs. Implemented CPU stages pass independent Hugging Face LlamaDecoderLayer output and DynamicCache K/V comparisons, including Llama3 RoPE, widths4/8 and GQA ratios2/4. A future-token perturbation test checks prefix causality. Combined suite11 passed with existing dependency deprecation warnings. FP32 tolerances atol2e-6/rtol2e-5; CPU results do not establish NPU BF16 parity.

Initial full2K compilation succeeds: RMS1.023s, attention58.452s, FFN71.406s, total engine initialization189.329s. The first full32-layer run fails the KV shape assertion. A cache-hit diagnostic rerun (initialization59.447s) confirms returned per-layer K and V both have token-major shape[1,2048,8,128], instead of requested[1,8,128,2048]/[1,8,2048,128]. These are failed runs, not valid exports or latency results.

Small SDK probes rule out generic new-axis indexing, CPU transfer, inference_mode, and retaining views across five repeated calls: all shapes/values match for those isolated operations. A32-token actual attention graph reproduces the fault immediately: output shapes[32,4096],[32,8,128],[32,8,128], versus requested[32,4096],[8,128,32],[8,32,128]. Interpreting the actual KV as token-major gives relative RMS differences from independent CPU arithmetic of0.00436% K and0.00471% V; hidden output0.01335%. All finite. The fault is specific to the compiled attention boundary; the exact SDK compiler pass responsible is not yet isolated.

Change: explicitly return token-major K/V from the compiled graph, and perform the canonical permutations outside that graph in the device runtime glue. No shape-based guessing or silent reshape. Updated CPU boundary-contract tests first fail5/6, then the implementation and existing HF semantic checks pass11/11. Full2K rerun succeeds. Compile RMS0.271s, attention57.732s, FFN0.183s (RMS/FFN cache hits); init118.503s. Three prefill+host-export runs0.864897/0.769522/0.769151s; all finite, shape[32,8,2048,128], repeated K/V/logits bit-exact. First token12366. This is stage/export latency, not HTTP TTFT or decode latency. The32-token attention regression now asserts matching shapes and relative RMS<0.1% for all outputs and passes on RNGD.

GPU A100 BF16 continuation:32/32 native versus imported tokens match,64 cache slots read back exactly. Relative RMS versus native GPU: K3.1488%, V5.8658%, logits2.2725%. Token agreement on one fixture is not broad quality equivalence.

ATOM k1 artifact initially fails the checker before generation: `b.blocks` is None for the supported static B1 path. Added a research-only `SingleBlockTarget` that validates static B1/full-attention geometry and reuses the existing AtomKVBlocks staging/import routines for owned block0. It does not add a scheduler, prefix cache or production PD support. Tests first fail6/6 with the stub, then pass6/6. Actual ATOM import succeeds on the static k1 artifact. Both original and portable implementations read back all64 slots exactly and produce bit-exact decode logits with each other. Their32-token outputs match the GPU path, but differ from native ATOM at index27 (28th token): native chooses9257 at20.875 over6366 at20.75; imported logits tie9257/6366 at20.75 and argmax chooses6366. This establishes the immediate argmax divergence, not the upstream numerical cause or general quality equivalence. Import times: original0.4083/0.3935s, portable0.3038/0.2950s on two diagnostic runs; first-use allocation and fixed order prevent a controlled speedup claim.

Broader CPU test command (`PYTHONPATH=opt_prime python -m pytest -q opt_prime/tests opt_prime/examples/pd/serving_matrix/test_*.py`) before the single-block helper:21passed,2failed. Both failures are `test_kv_cache_generation_matches_hf[1]` and `[2]`: unregistered transformers DynamicCache output during AOTAutograd tracing. Both reproduce unchanged on parent6787628; no full-suite green claim.

The same architecture succeeds at8192tokens. RMS0.293s, attention154.434s, FFN182.725s; init403.451s. Three full32-layer prefill+export runs4.047805/3.317492/2.933984s, all finite and bit-exact across repeats, shape[32,8,8192,128]. Timing dispersion needs controlled repetitions before a performance conclusion.

8K GPU BF16 native/imported32tokens match,64slots readback exact. Relative RMS versus GPU: K3.2969%, V5.8546%, logits2.4353%.8K ATOM native/original-import/portable-import32tokens all match;64slots readback per importer and all31 decode-logit vectors exact between importers. ATOM original/portable import1.2517/1.0614s (single diagnostic samples, not controlled timing). Some logits remain tied even when argmax agrees. No numerical identity or broad quality claim.

All experiments terminal; owned RNGD and ATOM allocations released. Draft PR222 contains the implementation and evidence.

Private logs under analysis/serving-matrix-201: rngd-koni-8b-2048-parts221{,-shape,-layout}.log, shape-probe221.log, output-probe221.log, context-probe221.log, retain-probe221.log, attention-probe221.log. The optional stack-output microprobe fails compilation; it was not adopted.

## Conclusion

The partitioned engine now executes full2K KONI on RNGD and passes the selected GPU continuation fixture. ATOM transport/continuation works on the exclusive k1 research path, with the native-output divergence documented.8K execution and both handoff paths pass this fixture, while broad numerical/quality coverage and production serving integration remain open. Keep full four-model goal active.

## Additional commands and validation

```bash
# Same pinned SDK bootstrap/environment as the setup above:
bash run-koni219.sh --width 8192 --rms-tile 512 --group 1 --partitioned --output rngd-koni-8b-8192-parts221-layout
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python check_rngd_gpu.py --model "$KONI_SNAPSHOT" --exports rngd-koni-8b-2048-parts221-layout.pt --output rngd-koni-8b-2048-gpu221.json
bash run-pd-atom201.sh --model KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024 --cache-dir atom-artifacts --kv-blocks 1 --exports rngd-koni-8b-2048-parts221-layout.pt --output rngd-koni-8b-2048-atom221-divergence.json
PYTHONPATH=opt_prime python -m pytest -q opt_prime/tests opt_prime/examples/pd/serving_matrix/test_*.py
```

Latest expanded CPU suite:27passed,2failed (the same two parent-reproduced DynamicCache tracing failures). Scoped Ruff passes. New research code does not modify the core trace/cache implementation responsible for those failures.

Sensitive-data scan: staged change files pass. Full repository scan reports7 unrelated existing findings, reproduced on parent6787628; none occur in the changed files.

## 8K handoff reproduction

Use the GPU/ATOM commands above with `rngd-koni-8b-8192-parts221-layout.pt` and separate output names (`rngd-koni-8b-8192-gpu221.json`, `rngd-koni-8b-8192-atom221.json`). Implementation commit720e8dc; code hashes in evidence221/summary.json. No code changes between2K and8K runs. The executable dependency revisions match the setup and parent #219 record.
