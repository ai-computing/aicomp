# In-memory KV path — aicomp #139

The native receiver now has three in-memory assembly modes:

- `legacy`: original Python per-block parser/copies, followed by canonical packing;
  file I/O is already excluded so disk removal cannot inflate the speedup.
- `fast`: vectorized metadata checks and direct copies into canonical BF16 storage.
- `views`: the same validation, retaining payload-backed NumPy views without copying
  the KV body. Their base ownership retains the ZMQ frames. `torch_layers()` exposes
  BF16 CPU tensors with shape [1,8,T-1,128], preserving those owners.

`NativeKVClient` reuses ZMQ sockets, an HTTP session and a single worker thread. It
is single-flight, loopback-only and experimental for the exact build/layout in
RESULTS.md. The views are strided, not a contiguous slab. Consumers must retain
ownership until any asynchronous device upload finishes and must not mutate the
shared cache. GPU upload is not included in these measurements. Failure poisons
the client; restart the dedicated server before constructing another client.

## Reproduce

Start the dedicated server with the README command, then:

```bash
python opt_prime/examples/pd/native_kv/bench_memory.py \
  --http http://127.0.0.1:19101 --prefill-port 19201 --receiver-port 19202 \
  --lengths 128,512,1024 --warmups 2 --repeats 7 \
  --modes native,legacy,fast,views --out /tmp/native-memory-new-run
python3 -m unittest discover -s opt_prime/examples/pd/native_kv -v
```

Seed 139; random token IDs in [1000,30000). Two warmup cycles and seven measured
cycles per length, reversing method order on alternating cycles. KV byte hashes
must match across *all* external methods and repetitions. Hashing is outside the
timing interval. Nothing writes KV files. Timed paths include metadata validation.

## Paired results (2026-10-22)

Same RNGD+ATOM host, npu0, model revision/native binary/FXB identities as RESULTS.md;
SDK/driver 2026.4.0. CPU load average during the primary run: 98.60–100.55.
These are contention-affected measurements, not a quiet-host capacity claim.
Times are medians in milliseconds, seven samples each:

| Input tokens | Native HTTP TTFT | Legacy KV ready | Copied canonical KV ready | View KV ready | Legacy parse/pack | View parse/pack |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 29.11 | 75.47 | 67.68 | 57.91 | 23.96 | 2.86 |
| 512 | 55.75 | 194.21 | 167.84 | 160.58 | 100.76 | 3.93 |
| 1024 | 95.33 | 388.56 | 290.60 | 263.58 | 261.66 | 4.59 |

At 1024 tokens, view assembly reduces parse/pack by approximately 57x and KV-ready
latency by approximately 32%. All 81 external requests (three methods x three
lengths x nine cycles) had identical canonical bytes for their prompt length.
This does **not** achieve native-engine speed.

Native standalone returns one generated token; the external path returns T-1 KV
positions and no generated token. These are explicitly different boundaries.
Native TTFT is a reference point, not a matched-operation performance pass.
The external path still needs final-prompt-token decode and device upload.

## Follow-up timing

A separate run configured the server's optional `poll_timeout_ms=1`, comparing
only native and views with the same inputs/repetitions. The external KV-ready
medians were 53.58/113.50/247.55 ms; native TTFT 25.53/56.61/96.94 ms.
This was not an interleaved server-configuration A/B test, so it does not establish
a polling-parameter improvement. The client does not change server defaults.

Median event times relative to request start:

| Input tokens | PrefillReady | First KV batch | Last KV batch |
|---:|---:|---:|---:|
| 128 | 25.14 | 30.0 | 53.4 |
| 512 | 58.63 | 66.8 | 113.3 |
| 1024 | 93.42 | 107.9 | 247.3 |

The remaining delay is mainly between native ready and final payload arrival,
not the now-small Python parsing time. This identifies export/transport as the
next investigation target; it does not separate native D2H from ZMQ or scheduling.

## Verification and next steps

15 CPU tests pass, including malformed fields, incomplete/duplicate transfers,
legacy/copy/view byte equivalence, buffer lifetime after garbage collection, and
zero-copy BF16 torch views. All hardware test processes were stopped.

Raw repetitions remain in research evidence; the small summaries are included as
`memory_summary.json` and `memory_poll1_summary.json`. No weights/KV payloads,
private addresses or site paths are published.

Next: matching GPU decode baseline; account for inter-host transport before
end-to-end GPU admission; investigate native export or direct custom-prefill
handoff. Full native-speed serving remains unproven. Related custom Llama prefill
work (#137) is independent and was not modified here.
