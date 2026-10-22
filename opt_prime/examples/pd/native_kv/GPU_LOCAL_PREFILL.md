# Experimental GPU-local prefill and HTTP continuation

- Date: 2026-10-22
- Issue: #148; stacked on #147, parent ad2649e; source hashes: gpu_local_manifest.json
- Hardware: A10080GB PCIe on GPU+RNGD host; torch2.10.0+cu128,
  transformers5.1.0, vLLM0.18.0, driver580.178.04.
- Checkpoint: Qwen3-8B-FP8 revision220b46e3b2180893580a4454f21f22d3ebb187d3.

## Background

Native export/import adds first-token delay to the heterogeneous path. A local
BF16-dequantized prefill model can reuse its device cache in Marlin decode.
This is an optional latency candidate, not completion of the heterogeneous goal.

## Implementation

`--prefill-backend gpu` selects a separate local worker. Default remains native.
Packed ragged FA2 prefill resets RoPE positions for each request. Device import
copies full prompt KV; decode consumes the first generated token and requests
one fewer subsequent token. Requests have separate bounded output queues; a full
queue pauses its lane. EOS, count and cancellation retire that lane. Wave admission
uses batch-size and native-batch-wait-ms; continuous admission is not implemented.
Native pipeline/async/adaptive options are rejected for this backend.
BF16 prefill weights are additional to Marlin decode weights. BF16-only diagnostic
allocated15.84GiB including graph/cache; combined serving memory not yet quantified.

## Verification

From this directory with MODEL_DIR pointing to the pinned checkpoint:

```bash
python3 diagnose_local_prefill.py --model-dir "$MODEL_DIR" --out diagnostic.json
python3 check_local_prefill.py --model-dir "$MODEL_DIR" --out check.json
python3 serve_gpu.py --prefill-backend gpu --model-dir "$MODEL_DIR" --batch-size 4 --norm-mode fused --fuse-mlp --fuse-qkv --port 19111
python3 check_local_http.py --out http_smoke.json
python3 -m unittest discover -s . -p 'test_*.py'
```

37 existing tests pass; new live HTTP smoke checks stream completion and exact
128-output usage, plus nonstream output counts1 and2. These are not comprehensive
cancellation/EOS/backpressure coverage of the new worker.
HTTP input512/output128, one warmup then three repeats, fixed GPU batch width4:

| Clients | Median aggregate tok/s | Median mean TTFT ms |
|---:|---:|---:|
| 1 | 96.05 | 50.29 |
| 4 | 344.07 | 170.22 |

No native endpoint measured in these runs; no native-parity claim. Other host work
was uncontrolled. B1 BF16 FA2 prefill microbenchmark ~46ms/512 tokens excludes HTTP.

## Numerical limits

Random prompts32/128/257/512: packed first tokens4/4 equal independent prefill;
KV is not bitexact (maxabs2.0). Packed continuation24/32 matches full-context
recomputation; individual-cache continuation32/32. Exact device cache copy passes.
Changing all neighboring requests while preserving lengths leaves target KV
bitexact, supporting request isolation for the tested case. Independent per-request
SDPA using the same packed projections differs from FA2 (KVmaxabs2.25,
minimum cosine0.999685; prefill logit maxdiff0.125).
At first divergent continuation, full-context candidates785/32313 tie at9.75;
packed-cache decode raises32313 to9.8125. Single-cache and SDPA both tie these
candidates at9.6875. This supports numerical sensitivity rather than demonstrated
cross-request corruption; it does not establish global correctness or parity.
Diagnostics exit0 even when numerical differences are reported; inspect JSON.

## Remaining work

Same-run native HTTP comparisons across input lengths/concurrency, quality cases,
new-worker slow-consumer/cancellation/EOS tests, combined memory measurement,
and full implementation review. No production default change. No weights/KV published.
Owned HTTP process shut down and all diagnostic/test processes exited.

Generated with Codex; maintainer review required.

Follow-up same-run comparisons, graph-width/admission changes and additional worker checks: GPU_LOCAL_PAIRED.md.

Request-group prefill and its early streaming pause tradeoff: GPU_LOCAL_CHUNKS.md.
