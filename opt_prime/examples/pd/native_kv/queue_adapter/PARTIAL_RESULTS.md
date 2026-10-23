# Partial physical-run coalescing improves mixed HTTP first-token latency

The reverse-only adapter did not improve mixed HTTP serving. Live inspection showed shuffled source addresses and variable segment lengths; sorting physically adjacent sources allows coalescing even when logical token order is arbitrary. This version stages sorted runs and restores the original host destinations at completion.

## Method and reproducibility

Measured base `abaaef4` plus sources identified by `partial_manifest.json`. Hardware: dedicated RNGD device 0, native SDK/driver 2026.4.0; GPU decoder A100 80GB PCIe device 0. Same model revision and FXB hashes as SERVING_RESULTS.md. Native prefix caching off, no debugger or native profiler. Both modes use the same streamed-KV GPU server configuration.

ABBA order: baseline, partial adapter, partial adapter, baseline; restart both servers each run. Six mixed native generations prime allocation state before the export benchmark. Then exact-byte exports at 128/512/1024 input tokens, concurrency 1 and concurrency 4 for 1024 tokens; discard first two repetitions per geometry per run. Each adapter export is checked against the first baseline before HTTP benchmarking proceeds.

HTTP is bench_arrival_http.py with deterministic mixed 128/512/1024 input and 32/128 output, 96 requests per endpoint at 6 and 12 requests/s. Ranges below are two observed runs, not confidence intervals. Fixed offered load is not a maximum-capacity benchmark. Native standalone control is measured separately; heterogeneous serving uses an additional GPU and is not a cost-parity comparison.

See README.md for build, test and run_benchmark.py commands. Source/binary hashes and complete compact measurement summaries are checked in. Root-local research launchers used these same native/GPU/benchmark commands; the portable wrapper only parameterizes paths and adds persisted phase-counter snapshots.

## Results

All 1536 HTTP benchmark requests succeeded, and all 272 complete KV exports matched baseline hashes.

| Heterogeneous load | Baseline TTFT p95 ms | Partial adapter TTFT p95 ms | Baseline output tok/s | Partial adapter output tok/s |
|---|---:|---:|---:|---:|
| 6 requests/s | 243.5–244.8 | 208.0–214.9 | 439.2–439.3 | 439.8–440.0 |
| 12 requests/s | 246.4–248.6 | 208.5–214.8 | 797.0–797.5 | 800.5–800.9 |

The mean of the two per-run TTFT p95 values decreases 13.4% and 14.5%, respectively. Throughput changes are small; the demonstrated result is first-token latency improvement. Native standalone TTFT p95 remained lower (158.8–170.5 ms at 6/s and 187.2–199.8 ms at 12/s across modes); native-performance parity is not achieved.

| KV handoff geometry | Baseline mean ms | Partial mean ms | Baseline p95 ms | Partial p95 ms |
|---|---:|---:|---:|---:|
| 128 tokens / concurrency 1 | 13.10 | 8.41 | 19.88 | 9.57 |
| 512 tokens / concurrency 1 | 35.54 | 22.61 | 52.57 | 23.95 |
| 1024 tokens / concurrency 1 | 70.05 | 42.06 | 99.01 | 46.96 |
| 1024 tokens / concurrency 4 | 87.38 | 42.25 | 141.59 | 49.44 |

Handoff includes D2H, transport and receiver assembly (`kv_ready_ms - ready_ms`). Its values should not be compared directly with the prior reverse-only study: this experiment also primes allocator state with six normal generations.

The two adapter runs rewrote/restored 13608 and 14544 transfers, with zero failed completions, no retained plans/staging after workloads, and no capacity fallback. Combined input segments 14113584 became 5275800 segments on transformed commands, a 62.6% reduction including isolated exports. HTTP-only phase deltas show 9432 and 10368 rewritten commands, reducing 5099760 -> 2767248 and 5169312 -> 2488392 segments, respectively. Other scatter commands remain on the native path when the reduction is below 25%.

## Correctness, cancellation and limits

The shuffled four-segment/two-run test failed on the reverse-only implementation before the fix. Unit tests and ASan/UBSan now pass: variable lengths, byte permutation, unchanged source descriptors, overlap fallback, completion-before-return, failure cleanup and two out-of-order completions after overwriting original descriptor storage.

Quality: all 15 cases had identical native/GPU/streamed text; strict-format grading was 12/15 on each endpoint. Cancellation diagnostic: three waves of four raw HTTP connections were closed after the native adapter counter showed two transformed DMAs in flight in each wave. Those observations precede socket closure and do not prove the native engine processed cancellation before DMA completion. An additional 16 stream disconnects and 16 successful identical recovery responses completed. Afterward, all 3744 transformed transfers were restored; failed completions, retained plans and staging bytes were all zero.

The adapter remains opt-in, build-specific and experimental. Retained staging is capped at 256 MiB with at most 4 MiB per plan and 1024 plans; submitter temporary allocations are additional. Physical holes are not read. Native source/destination lifetimes still belong to the engine. This limited cancellation/recovery coverage does not establish production safety for all shutdown or driver-error races. Larger-load, longer-duration and additional-model validation remain future work.
