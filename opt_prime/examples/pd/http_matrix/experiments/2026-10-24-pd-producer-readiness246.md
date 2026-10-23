# Terminal producer failure in HTTP readiness and admission

- **Date**: 2026-10-24
- **Task**: aicomp #246
- **Host / device**: CPU development host, real Torch subprocess and real HTTP server; no accelerator reset
- **Commit**: ea09aca baseline; fix/246-pd-producer-readiness candidate

## Question

Does the service reject new work and stop advertising readiness when a required subprocess prefill producer has terminally failed?

## Setup

Actual tiny Llama checkpoint, SubprocessPrefillWorker and PDBatcher under BatchedOwner/HTTP. Kill only the test child and await its receiver termination. Check readiness and request admission, then explicitly replace the worker at an idle boundary. Also test a closed producer, a pool with one failed required member, and a per-request embedding error that must not make a live producer unavailable.

## Command

```bash
PYTHONPATH=opt_prime python -m pytest -q opt_prime/tests/test_pd_producer_readiness.py
PYTHONPATH=opt_prime python -m pytest -q opt_prime/tests/test_pd_producer_readiness.py opt_prime/tests/test_subprocess_prefill_failure.py opt_prime/tests/test_serve_batched.py opt_prime/tests/test_pd_llmd.py opt_prime/tests/test_pd_http_cancel.py
# From opt_prime/:
PYTHONPATH=. python -m pytest -q tests
```

Independent exact local probe:/tmp/probe_pd_readiness.py, run with PYTHONPATH pointing to the baseline/candidate package. JSON under analysis/serving-matrix-201/readiness-baseline.json and readiness246-fixed.json; selected copies accompany the change.

## Results

- Baseline: /ready200 before and after confirmed terminal worker failure; new completion500/non-retryable, service state stillidle. Regression set3failed/1passed.
- Candidate: /ready503 after failure, new completion503/retryable before scheduling; explicit replacement restores200 and correct output. Closed worker and failed pool member likewise block admission; request-only error leaves readiness200.
- Focused serving/PD suite30passed,1existing flex_attention warning,27.02s.
- Final full suite468passed,15skipped,135subtests passed,1existing flex_attention warning,151.40s.

## Conclusion

Known subprocess/pool terminal availability is now reflected consistently by readiness and admission. Already-admitted work keeps its existing error/drain handling; the service is not globally poisoned just because a producer can be replaced. Pools conservatively require all configured members because their routing still includes each member. No silent rerouting/retry or automatic restart. Legacy/custom workers without the optional readiness property keep prior behavior. The hardware fault record#244 predates this change and retains its later-request500; the corrected service now rejects new requests with503 instead.
