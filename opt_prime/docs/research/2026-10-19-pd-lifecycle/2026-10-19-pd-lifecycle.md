# PD capability and cooperative cancellation
- Date:2026-10-19
- Task:PDDISAG-6
- Commit:75d54ae + dirty lifecycle contracts/runtime
- Host/device:etri-001 RNGD npu3ch0(logical2), ATOM rbln0
- SDK:Furiosa2026.4, Rebel0.11.0, driver3.0.0

## Question/setup
Reserve destination first, bind leases to request/generation/lane, validate endpoint
capabilities and drain cooperative cancellation at synchronous boundaries.
Qwen1.5B and0.5B external-cache P2P, B1/context512, input32/64/41/37 output8/32.

## Commands
```bash
PYTHONPATH=aicomp-llmd/opt_prime:hetero-npu-p2p/src:rngd_llmd/src /home/gspark/venvs/furiosa-2026.4/bin/python aicomp-llmd/opt_prime/examples/pd/validate_qwen_pd.py --config experiments/p2p_lifecycle_20261019/q15.json
PYTHONPATH=aicomp-llmd/opt_prime:hetero-npu-p2p/src:rngd_llmd/src /home/gspark/venvs/furiosa-2026.4/bin/python aicomp-llmd/opt_prime/examples/pd/validate_qwen_pd.py --config experiments/p2p_lifecycle_20261019/cancel.json --test-cancel-drain
```

## Results
1.5B16requests320tokens896DMA full-logit exact, sidecar8requests160tokens exact.
0.5B cancellation after48DMA (393216bytes): decode skipped, idle restored;
then16requests320tokens768DMA full-logit exact. Both workers exited0.
47CPU tests and static checks passed. No failed hardware run.

## Conclusion
Safe single-flight synchronous-boundary cancellation; uncertain DMA still poisoned.
Not device abort or HTTP cancellation. No native-cache P2P, universal registry,
async pool or restart recovery claim. Next: connect request cancellation semantics
to serving and expand semantic KV/capability coverage without implicit fallback.
