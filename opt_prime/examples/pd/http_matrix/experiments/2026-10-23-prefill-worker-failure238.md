# Subprocess prefill failure and cancellation recovery

- **Date**: 2026-10-23
- **Task**: aicomp #238
- **Host / device**: CPU subprocess fault tests on development host; no accelerator reset or live hardware injection
- **Commit**: df02223 baseline; fix/238-prefill-worker-failure working tree (final commit recorded in PR)

## Question

Can structured SDK logs, cancellation, malformed replies, worker exit or failed startup strand requests or leak the worker's shared-memory segment?

## Setup

A tiny real subprocess replaces only the expensive model engine. The production worker still owns real stdin/stdout pipes, Futures, receiver thread and shared memory. A file gate ensures cancellation occurs before a response. Faults are startup stall/exit, process exit with queued work, malformed response, ignored SIGTERM and Popen failure. The normal Torch prefill subprocess test continues to exercise actual KV generation and transfer.

Issue: https://github.com/ai-computing/aicomp/issues/238

## Command

Run from the aicomp checkout with a Torch/Transformers test environment:

```bash
# Baseline: run this test file with PYTHONPATH pointing to df02223/opt_prime.
PYTHONPATH=opt_prime python -m pytest -q opt_prime/tests/test_subprocess_prefill_failure.py
# Focused regression set:
PYTHONPATH=opt_prime python -m pytest -q opt_prime/tests/test_subprocess_prefill_failure.py opt_prime/tests/test_pd_batching_disagg.py opt_prime/tests/test_atom_batching.py opt_prime/tests/test_atom_hybrid.py opt_prime/tests/test_pd_recovery.py opt_prime/tests/test_serve_batched.py
# Full project suite, from opt_prime/:
PYTHONPATH=. python -m pytest -q tests
```

## Results

- Baseline df02223: 9 failure cases fail; 3 unhandled receiver-thread warnings and 1 cleanup error. Structured JSON logs trigger KeyError, a cancelled Future triggers InvalidStateError, malformed span length triggers TypeError. Failed startup leaves shared memory or a live child. A post-EOF submission writes a dead pipe instead of returning a failed Future.
- Fixed focused set: 64 passed, 1 existing flex_attention warning, 21.32 s. Includes all 9 fault tests plus actual tiny Torch subprocess prefill.
- Final full suite: 460 passed, 15 skipped, 135 subtests passed, 1 existing flex_attention warning, 134.34 s. No failed tests.
- No automatic request replay. A healthy replacement worker explicitly processes a fresh request after terminal failure.

Local raw logs: /tmp/worker238-baseline.log, /tmp/worker238-targeted.log, /tmp/worker238-final-full.log.

## Conclusion

The receiver now ignores non-protocol JSON logs, settles cancelled responses safely, fails all outstanding Futures after terminal protocol/pipe failure and rejects later work through failed Futures. Spawn/startup failure and bounded terminate/kill cleanup release the owned child and slab. Request/response and shared-memory layouts remain unchanged. This is CPU process-lifecycle evidence, not accelerator recovery, service-level automatic restart, compute/copy overlap or performance evidence. Long four-model hardware validation continues independently under #234.
