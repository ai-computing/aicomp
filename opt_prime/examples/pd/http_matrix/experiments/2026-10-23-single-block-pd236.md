# Static single-block ATOM PD

- **Date**: 2026-10-23
- **Task**: aicomp #236
- **Host / device**: owned RNGD+ATOM allocation, KONI-8B static B1/context9216/k1 artifact; SDK versions in live-http234 record.
- **Commit**: feat/236-atom-single-pd working tree based69bca73; final changes preserve the hardware-tested behavior, with only explanatory text changed after the run.

## Question

Does static B1 cache-off remote PD require a separate idle-lane scratch block?

## Setup

For exactly one block and one non-hybrid lane, with prefix/host caches disabled, allow block0 to be assigned and released per request. No active/idle lane pair exists, so there is no concurrent scratch access. All other scratch/cache requirements remain. Require a full prefix from position0; invalid partial results fail per request.

## Command

```bash
PYTHONPATH=opt_prime python -m pytest -q opt_prime/tests/test_atom_batching.py opt_prime/tests/test_atom_hybrid.py opt_prime/tests/test_pd_recovery.py opt_prime/tests/test_serve_batched.py
```

Hardware command and pinned checkpoint are in live-http234; the KONI row uses this candidate and its existing k1 artifact.

## Results

Initial regression1failed/3passed at the existing PD capacity guard. Final43tests pass (one existing unfused attention warning), including queued request parity/reuse, cancellation while prefill pending and ignoring its late result, malformed partial result recovery, and rejection of cache/multi-lane combinations. Real KONI HTTP:14normal/reuse responses token-identical through EOS,16remote/0local prefills, cancellation/disconnect followed by successful reuse; free lane[0]. No new artifact or extra KV block was required.

## Conclusion

Static B1 cache-off PD is supported with one block. This is not prefix-cache support, hybrid single-block support or hardware batch>1. The broader four-model goal remains active. Evidence: single236-red.log, single236-final.log and live234-koni-http.json in the local serving-matrix-201 analysis directory.
