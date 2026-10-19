# Model-refactor llm-d regression

Engine143e618 merged into feat/optimus-llmd (previous9ac7b0e). Combined suite:
21tests passed, including120real tiny-model cases. Actual upstream sidecar:
3requests/18tokens exact vs direct optimized Torch engine, owner idle, proxy exit0.
No NPU run in this regression; hardware transport/execution code is unchanged.

Commands from aicomp-llmd:

```bash
PYTHONPATH=opt_prime:../hetero-npu-p2p/src python3 -m unittest discover -s opt_prime/tests -v
PYTHONPATH=opt_prime python3 opt_prime/examples/pd/validate_torch_sidecar.py --sidecar ../experiments/native_pd_2026_4/llmd_integration/pd-sidecar --out /tmp/optimus-expanded-llmd-r1
```
