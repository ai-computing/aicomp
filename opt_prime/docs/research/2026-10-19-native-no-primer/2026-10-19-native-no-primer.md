# Native no-primer resume

- Date: 2026-10-19
- Task: PDDISAG-6
- Host/device: etri-001 / ATOM rbln0
- Commit: c8df2f8 + dirty new adapter/validator

## Question / setup
Can explicit static block and mask arguments eliminate redundant prefill?
Saved RNGD KV, native Qwen0.5B/1.5B, B1/context512, Rebel0.11.0/Optimum0.11.0.post1.

## Command
```bash
PYTHONPATH=aicomp-llmd/opt_prime:hetero-npu-p2p/src:rngd_llmd/src experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv/bin/python aicomp-llmd/opt_prime/examples/pd/validate_native_no_primer.py --config experiments/native_baseline_20261019/a1.json --native-artifact experiments/npu_model_expansion_20261019/atom-qwen05-c64-r1/compiled --prefix-dir experiments/native_baseline_20261019/a1 --out experiments/native_no_primer_20261019/q05-r2
PYTHONPATH=aicomp-llmd/opt_prime:hetero-npu-p2p/src:rngd_llmd/src experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv/bin/python aicomp-llmd/opt_prime/examples/pd/validate_native_no_primer.py --config experiments/native_engine_optimization_20261019/q15-r1/config.json --native-artifact experiments/npu_model_expansion_20261019/atom-qwen15-c64-build/compiled --prefix-dir experiments/native_engine_optimization_20261019/q15-r1 --out experiments/native_no_primer_20261019/q15-r1
```

## Results / conclusion
Both models 8 requests/256 tokens, all full logits exact, zero guarded prefill calls.
0.5B candidate/reference 1.477171/1.573087s; 1.5B 3.904350/4.077893s.
Diagnostic timings only, not controlled end-to-end performance.
Initial q05-r1 failed NameError before decode (same command as q05-r2 with prior source).
37 regression tests passed. No live RNGD, P2P, HTTP or new quality claim.
See aicomp-llmd/opt_prime/docs/NATIVE_NO_PRIMER.md and raw experiments/native_no_primer_20261019.
Next: live native adapter admission and serving parity.
