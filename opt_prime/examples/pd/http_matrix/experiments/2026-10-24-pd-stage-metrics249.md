# Batched PD stage duration metrics (#249)

- **Date**: 2026-10-24
- **Task**: aicomp #249
- **Host / device**: CPU development host; tiny real Torch model
- **Commit**: d0f6f12 baseline plus feat/249-pd-stage-metrics candidate

## Question

Can the public metrics expose durations already accumulated by PDBatcher, without confusing the batched path with the single-stage transfer histogram?

## Setup

The #248 GPU results have42remote prefills but zero observations in the single-stage transfer histogram. PDBatcher accumulates prefill_s and transfer_s at successful admission; BatchedService previously exported neither. Add a separate counter family, airm_pd_producer_seconds_total, labelled stage=prefill|transfer.

A real tiny-model producer and batcher execute two handoffs. The test controls only the worker-reported durations (0.25/0.125seconds per handoff), verifies public totals0.5/0.25, repeats scraping, and verifies missing timing keys are omitted. It also asserts that the original real producer measured positive compute time.

## Command

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=opt_prime python -m pytest -q opt_prime/tests/test_pd_stage_metrics.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=opt_prime python -m pytest -q opt_prime/tests/test_pd_stage_metrics.py opt_prime/tests/test_pd_batching_disagg.py opt_prime/tests/test_atom_batching.py
```

## Results

Baseline:1failed at missing public timing metric. Candidate:34passed,1existing unfused-flex-attention warning,12.93seconds. Ruff on new test and git diff --check clean. Full CPU suite after the GPU queue terminated:467passed,17skipped,135subtests,1existing warning in145.94seconds. CUDA_VISIBLE_DEVICES was empty; hardware-dependent tests are not covered by this CPU run. Command from opt_prime/: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=. python -m pytest -q tests`.

## Conclusion

The missing export is corrected in the candidate. These are sums of worker-reported durations for admitted handoffs, not request latencies, histograms or critical-path time. Failed/cancelled handoffs that never reach admission are excluded. Torch prefill includes KV gathering; transfer excludes the decoder's later KV installation. Custom/vendor workers can report incomplete or default-zero durations; exported zero alone does not establish zero physical transfer cost. Counter presence follows existing batcher timing keys, not hardware timing availability. No change to scheduling, copies, or hardware overlap.
