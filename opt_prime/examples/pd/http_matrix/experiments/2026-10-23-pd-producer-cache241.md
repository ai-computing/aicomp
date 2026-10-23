# Explicit Torch PD producer cache and chunk policy

- **Date**: 2026-10-23
- **Task**: aicomp #241
- **Host / device**: CPU factory tests; GPU controls on two A10080GB PCIe
- **Commit**: baseline369c8ff; feat/241-pd-producer-cache candidate

## Question

Can serving explicitly disable only the Torch producer's prefix cache, without changing the existing default or decoder cache policy?

## Setup

The public Torch PD producer previously always used TorchPrefillEngine's prefix_cache=True default. Add --pd-prefill-no-prefix-cache, default false, and --pd-prefill-chunk, default2048. Reject nonpositive chunks before loading the model. The flag affects only --pd-prefill-device; vendor/subprocess producers retain their own configuration. Existing SimpleNamespace callers without this attribute retain the original default.

## Command

```bash
PYTHONPATH=opt_prime python -m pytest -q opt_prime/tests/test_serving_prefill_cache.py opt_prime/tests/test_serve_batched.py opt_prime/tests/test_pd_batching_disagg.py
# From opt_prime/:
PYTHONPATH=. python -m pytest -q tests
# Public serving flag:
python examples/pd/serve_batched.py --model "$MODEL" --device cuda:1 --pd-prefill-device cuda:0 --pd-prefill-no-prefix-cache --pd-prefill-chunk 256
```

## Results

- Baseline:1failed,2passed. Explicit cache-off still reused64tokens on a repeated128-token prompt.
- Candidate focused set:19passed,1existing flex_attention warning. Actual tiny Llama model checkpoint is loaded by the public factory; first prefill has0hits, repeated prefill has64hits by default/explicit false and0hits with opt-out. First token and KV shape remain valid. No mock engine.
- Chunk control red:1failed,3passed (64tokens executed as one chunk instead of two32-token chunks). Final focused suite20passed,1existing warning. Final full suite464passed,15skipped,135subtests passed,1existing flex_attention warning,142.43s. Previous cache-only full suite463passed,15skipped,135subtests.
- Supporting hardware controls under#240: Qwen3-0.6B BF16 concurrent equality8/12 with producer cache on versus12/12 with cache off; KONI7/12 versus12/12. Both FP32 cache-on controls also reach12/12. These controls used a process-local constructor override; cache-off alone leaves native/PD2K differences for Qwen0.6B and Gemma. Matching chunk256 additionally restores both3/3native/PD serial comparisons. Public factory/validator flags now pass all4models:84remote prefills,0producer cache-hit tokens,48/48concurrent equality,12/12native/PD serial equality; all cancel/disconnect/invalid-input checks pass. Inputs128/2048/8192,32outputs ignoring EOS,decode slots4. This uses actual GPU0-prefill/GPU1-decode, BF16, producer cache off/chunk256.

## Conclusion

The option provides a reproducible cache-off producer path without silently trading cache throughput for numerical reproducibility. It does not guarantee equality for all inputs, batches or devices and is not a model-quality or performance improvement claim. GPU public-flag validation is complete for this matrix. Production defaults remain cache-on/chunk2048; reproducible uncached256 execution must be selected explicitly.
