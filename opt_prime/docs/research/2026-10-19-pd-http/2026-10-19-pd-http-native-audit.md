# HTTP cancellation and native connector audit
- Date:2026-10-19
- Task:PDDISAG-6
- Commit:3b405d8 + dirty HTTP and semantic KV changes
- Host/device:etri-001 RNGD npu3ch0 + ATOM rbln0

## Question/setup
Connect explicit HTTP cancellation to fenced runtime, reject semantic KV mismatch,
and verify manufacturer-native DMA options. Qwen1.5B B1/context512.

## Command
```bash
PYTHONPATH=aicomp-llmd/opt_prime:hetero-npu-p2p/src:rngd_llmd/src /home/gspark/venvs/furiosa-2026.4/bin/python aicomp-llmd/opt_prime/examples/pd/validate_qwen_pd.py --config experiments/pd_http_20261019/q15.json
```
Additional read-only audit: rdma link show; ls /sys/class/infiniband; nvidia-smi -L;
ATOM python importlib.metadata versions; pip index versions nixl-rbln;
pip download --no-deps --dest /tmp/optimus-nixl-audit --index-url https://pypi.rbln.ai/simple nixl-rbln.

## Results
Direct16requests320tokens896DMA logits exact; sidecar8requests160tokens exact.
HTTP cancellation after56DMA:202 accepted,499 completion, decode skipped; same lane reused exact.
52CPU tests pass. Native audit: official latest NIXL D2D documentation exists, but
vllm implementation targets Rebel100; local ATOM optimum is not verified compatible.
Package download failed default-index missing/vendor-index authentication. No install.
RDMA devices absent; NVIDIA driver unavailable. No direct DMA experiment on native cache.

## Conclusion
HTTP and semantic checks implemented. Native ATOM P2P remains unimplemented, not
proven impossible universally. Preserve external-cache P2P and unsupported route rejection.
