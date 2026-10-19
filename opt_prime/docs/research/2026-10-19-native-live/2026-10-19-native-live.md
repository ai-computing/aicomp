# Live native host PD

- Date: 2026-10-19
- Task: PDDISAG-6
- Host/device: etri-001, RNGD npu3ch0 + ATOM rbln0
- Commit: 795af1d + dirty producer/validator

## Question / setup
Does freshly computed RNGD KV resume in native ATOM without prefill?
Separate Furiosa2026.4 and Rebel0.11.0/Optimum0.11.0.post1 processes.
B1/context512, input32/64/41/37, output32, four prompts twice per model.

## Command
```bash
PYTHONPATH=aicomp-llmd/opt_prime:hetero-npu-p2p/src:rngd_llmd/src experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv/bin/python aicomp-llmd/opt_prime/examples/pd/validate_native_no_primer.py --config experiments/native_baseline_20261019/a1.json --native-artifact experiments/npu_model_expansion_20261019/atom-qwen05-c64-r1/compiled --prefix-dir experiments/native_baseline_20261019/a1 --live-rngd-python /home/gspark/venvs/furiosa-2026.4/bin/python --out experiments/native_live_20261019/q05-r1
PYTHONPATH=aicomp-llmd/opt_prime:hetero-npu-p2p/src:rngd_llmd/src experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv/bin/python aicomp-llmd/opt_prime/examples/pd/validate_native_no_primer.py --config experiments/native_engine_optimization_20261019/q15-r1/config.json --native-artifact experiments/npu_model_expansion_20261019/atom-qwen15-c64-build/compiled --prefix-dir experiments/native_engine_optimization_20261019/q15-r1 --live-rngd-python /home/gspark/venvs/furiosa-2026.4/bin/python --out experiments/native_live_20261019/q15-r1
```

## Results
Qwen0.5B and1.5B each8requests/256tokens, all full logits exact versus same-KV primer reference.
Zero forbidden prefill calls.37CPU tests passed. No failed hardware runs this unit.
Raw logs: experiments/native_live_20261019; manifest and code snapshots preserved.

## Conclusion
Live host admission works for tested scope. NPZ transport, not P2P/HTTP or performance benchmark.
Next: request-bound admission/transport integration into existing runtime.
