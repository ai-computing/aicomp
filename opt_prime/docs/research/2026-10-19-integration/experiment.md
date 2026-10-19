# OptimusPrime heterogeneous PD integration foundation

- Date: 2026-10-19
- Task: PDDISAG-6
- Environment: CPU contract tests only; no NPU processes started
- Base: ai-computing/aicomp 62a783114afa3531beeb7955c66da2e6aaf4ba65
- Branch: feat/optimus-heterogeneous-pd (new implementation initially uncommitted)
- Optional transport source: hetero-npu-p2p 5cbce71

## Question
Can OptimusPrime host a vendor-neutral PD interface while preserving the existing
P2P completion/lifetime contract and lazy CUDA initialization?

## Setup and command
Added opt_prime.pd engine/lease/transport contracts, single-flight runtime and an
optional adapter around the real hetero_p2p.CopySession. Existing inference unchanged.

```bash
cd /home/gspark/workspace/01_pd_disag/aicomp-integration
PYTHONPATH=opt_prime:../hetero-npu-p2p/src python3 -m unittest discover -s opt_prime/tests -v
git diff --check
```

## Results
7 CPU tests passed, 0 skipped with optional dependency present; diff check passed.
Includes real CopySession with fake backend/plan, no real DMA. Covers ordering,
request reuse, incompatibility, unsupported routes, error retention and lazy import.
No performance/MFU/power or actual KONI/GPU engine integration claim.

## Conclusion
The integration foundation is executable and separates common contracts from
hardware constraints. Real KONI endpoint adapters, Optimus PP/TP cache export/import,
and llm-d service lifecycle remain to implement and validate. See opt_prime/docs/
HETEROGENEOUS_PD.md for contracts and migration boundaries. No existing results altered.
