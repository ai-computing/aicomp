# RNGD layer-progress import toward within-request KV overlap

- **Date**: 2026-10-24
- **Task**: Layer-progress RNGD transfer (aicomp #265, original #47)
- **Host / device**: local unit/regression tests; no new streaming hardware experiment yet
- **Commit**: isolated perf/265-rngd-stream, base098169b plus dense-hybrid fix cherry-picked as8903b73; uncommitted implementation

## Question

Can completed RNGD layer groups be imported before final prefill/logits without publishing incomplete state or corrupting another request? GPU chunk-copy#254 and ATOM next-request prefetch#255 do not implement this source-to-decoder connection.

## Setup

The public research FullPrefillEngine.prefill_stream has per-group host output, but the current subprocess worker sends only whole-request completion. Work is isolated from the live #263 queue. First unit adds a thread-safe coverage/notification validator and partial dense model-layer import into reserved PagedKV pages. It does not yet alter worker messages, GPU/ATOM admission, or source adapters. No overlap or speedup claim.

Coverage validates integer geometry, source span bounds, overlap across every affected layer before any state mutation, complete layer/token coverage on finish, and no messages after closure. Draining notifications is separate from source completion; neither is a destination transfer fence. Cache import validates every affected hybrid group before writing and leaves lengths/cache publication unchanged. It rejects unreserved pages and overwrites of an admitted/shared prefix.

## Command

```bash
cd /home/gspark/workspace/aicomp-rngd-stream-265/opt_prime
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/gspark/venvs/pt213/bin/python -m pytest tests/test_pd_layer_progress.py tests/test_pd_dense_hybrid.py -q
# Full suite with the same environment:
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/gspark/venvs/pt213/bin/python -m pytest -q
```

## Results

- Coverage tests first11failed for missing implementation, then11passed.
- Partial receiver tests6failed before implementation (controlled removal in correct package working directory:6failed/11passed), then combined21passed/1existing flex-attention warning in5.18s.
- Additional admitted-prefix regression:1failed/17passed before guard; final focused22passed/1existing warning in4.93s.
- One initial focused invocation from the repository parent failed collection (`opt_prime.accel` unavailable with backend autoload disabled). Correct package-directory command above resolves it; failure is not an inference result.
- Initial full regression before the last admitted-prefix guard:555passed/15skipped/253subtests in160.70s. Final full run after guard:556passed/15skipped/253subtests in158.09s.13existing pinned-memory deprecation/flex-attention warnings retained.

Raw logs: /tmp/stream265-red.log, /tmp/stream265-import-red-controlled.log, /tmp/stream265-prefix-red.log, /tmp/stream265-focused-final.log, /tmp/stream265-full.log and /tmp/stream265-full-final.log.

## Conclusion

Only validation/import primitives are implemented. Still required: opt-in progress wire messages, source adapters, owner drain/commit/failure integration, ATOM cache-ownership handling, actual overlap traces, same-route exactness and long-context/large-KV performance. Existing #47 export-only0.1–4GB curve showed chunking slower but reduced incremental staging to~67MB; it does not prove RNGD streaming improvement. Preserve that negative control and measure the new path when connected.
