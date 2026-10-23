# Experimental native KV coalescing adapter

This opt-in adapter sorts native KV D2H segments by physical source address, merges adjacent runs, reads those runs into owned staging memory, then restores each original segment to its logical host destination before native completion handling. It preserves the engine's original SQEs and descriptors. The native binary is SHA256-guarded by bootstrap.py; installation happens before engine construction. This is not a supported vendor plugin or a default serving configuration.

[Partial-run serving results](PARTIAL_RESULTS.md) show reduced HTTP TTFT. The earlier [reverse-only experiment](SERVING_RESULTS.md) did not improve serving; that implementation is preserved in commit abaaef4.

## Build and test

```bash
gcc -O2 -Wall -Wextra -pthread test_adapter.c -o /tmp/test_partial
/tmp/test_partial
gcc -g -fsanitize=address,undefined -pthread test_adapter.c -o /tmp/test_partial_sanitize
/tmp/test_partial_sanitize
gcc -shared -fPIC -O3 -Wall -Wextra -pthread adapter.c -o /tmp/native_queue_partial.so
```

Launch using the native environment's interpreter:

```bash
NATIVE_QUEUE_ADAPTER=/tmp/native_queue_partial.so python bootstrap.py /path/to/furiosa-llm serve ...
```

## Reproduce the ABBA benchmark

Run with a GPU-capable Python environment and dedicated RNGD device 0/GPU device 0:

```bash
python run_benchmark.py --native-cli /path/to/furiosa-llm --native-python /path/to/native/python --model-dir MODEL --fxb FXB --adapter /tmp/native_queue_partial.so --out RESULTS
```

The wrapper runs baseline/adapter/adapter/baseline, exact-byte export comparisons and mixed HTTP arrivals at 6 and 12 requests/s. The research run used the same commands through a local launcher; the path-parameterized wrapper was syntax/help checked, not separately rerun on hardware.

## Scope and ownership

Linux x86-64, measured native SDK 2026.4.0 build, RNGD device 0 read channels 4–7. Supports nonoverlapping physical source segments of variable lengths divisible by 2048, with contiguous logical host destinations. Unsupported geometry and plans saving fewer than 25% of driver segments use the original submission. No holes between physical runs are read.

Limits: 4 MiB per plan, 1024 retained plans, 256 MiB retained staging. Submitters can temporarily allocate a plan before budget admission. Each descriptor, staging buffer and permutation map remains owned until its matching completion. Failed completions retire metadata without copying. Native source/destination lifetimes remain engine-owned. No on-disk vendor executable is modified.

The stats symbol exposes ten atomic counters: rewritten, restored, failed, in-flight plans, capacity/duplicate fallback, input segments, output segments, in-flight staging bytes, insufficient reduction and unsupported geometry. No KV bytes are logged.

Tests cover partial runs, variable lengths, byte permutation, immutable original metadata, duplicate-source fallback, completion before submit returns, failed completion cleanup and two out-of-order completions after original descriptor storage is overwritten. Hardware cancellation observations are documented in PARTIAL_RESULTS.md; they do not prove all shutdown/cancellation races safe.
