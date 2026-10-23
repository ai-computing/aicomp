# Experimental native KV coalescing adapter

This benchmark-only adapter converts reverse-contiguous native KV D2H scatter reads into one contiguous read, then reverses 2048-byte token blocks before native completion handling. It preserves original SQEs and descriptors. It installs before engine construction, through two build-specific function-pointer slots; `bootstrap.py` refuses any native binary except the measured SHA256. It is not a supported vendor plugin or a default serving configuration.

Build and check from this directory:

```bash
gcc -O2 -Wall -Wextra -pthread test_adapter.c -o /tmp/test_adapter
/tmp/test_adapter
gcc -shared -fPIC -O3 -Wall -Wextra -pthread adapter.c -o /tmp/native_queue_adapter.so
```

Launch with the same interpreter and CLI arguments as the baseline:

```bash
NATIVE_QUEUE_ADAPTER=/tmp/native_queue_adapter.so python bootstrap.py /path/to/furiosa-llm serve ...
```

The supported experiment uses native SDK 2026.4.0 on Linux x86-64, RNGD device 0 read channels 4–7, and 2048-byte token blocks. Nonmatching geometries use the original submission. Up to 1024 rewritten transfers can be in flight; saturation falls back to native transfers. The replacement descriptor lives until the corresponding completion. Negative completion results are passed through without reordering. The engine continues to own and retain source and destination buffers; cancellation/shutdown ownership must be validated before broader use.

The `stats` symbol exposes five atomic counters: rewritten, restored, failed, in-flight, and capacity/duplicate-tag fallback. No KV bytes are logged. Loading is explicit, and the normal CLI remains unchanged. No on-disk vendor executable is modified.

The unit harness checks restoration, preservation of input descriptors, noncontiguous fallback, completion arriving before submit returns, and failed completion cleanup. Hardware experiments must additionally compare full KV hashes and HTTP behavior. Do not infer speedup from debugger timings or isolated DMA measurements.
