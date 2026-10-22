# Experimental native Furiosa KV receiver

Related to [aicomp #136](https://github.com/ai-computing/aicomp/issues/136) and
[#132](https://github.com/ai-computing/aicomp/issues/132).

This standalone experiment receives native prefill KV into Python without running
Furiosa's decoder. It generates its own Handshake, PullRequest and Ack messages;
no packet capture or replay files are required. It is outside the default engine
path and does not register a serving backend.

The observed subset is specific to **furiosa-llm/native 2026.4.0** and the measured
**Qwen3-8B-FP8, context 2048 FXB**. This is an inferred, unsupported wire format,
not a vendor API. Native binaries, weights, FXBs and KV payloads are not included.
The receiver requires an explicit experimental flag and a version declaration;
the declaration does not detect the remote server's build. Use the exact build
identities in [RESULTS.md](RESULTS.md). Model compatibility is not inferred from
an arbitrary `--model` string.

## Reproduce

Use a separate shell with the matching vendor environment. Choose an idle RNGD;
replace `npu:0` if necessary. Keep the server running only for this experiment:

```bash
furiosa-llm serve Qwen/Qwen3-8B-FP8 \
  --fxb "$FXB_PATH" --devices npu:0 --host 127.0.0.1 --port 19101 \
  --no-enable-prefix-caching \
  --experimental-kv-transfer-config \
  '{"kv_connector":"TcpConnector","host":"127.0.0.1","port":19201}'
```

After `/health` returns 200, run the receiver (Python 3.10+, requests and pyzmq;
measured with requests 2.32.2, pyzmq 27.2.0):

```bash
python opt_prime/examples/pd/native_kv/receive.py \
  --experimental --protocol-version 2026.4.0 \
  --prefill-http http://127.0.0.1:19101 --prefill-port 19201 \
  --receiver-port 19202 --lengths 2,4,16,33 --out /tmp/native-kv-new-run
```

Output directory must not exist. The synthetic prompt is token IDs
`1000..1000+length-1`. Every request uses a fresh ID and logical destination block
IDs `0..length-2`. HTTP and ZMQ progress concurrently. The receiver validates IDs,
metadata bounds, payload sizes, block uniqueness, complete coverage and the final
batch before acknowledging. Only loopback is supported by this example. Stop
the server afterward; the receiver does not own or terminate a separately started
server. On a failed transfer it aborts rather than issuing a success Ack; restart
the dedicated test server before retrying (CancelTransfer is not implemented).

Each case contains 72 `slotNN.bf16` files, with little-endian BF16 values in
`[transferred_position, 8, 128]` order. Slots `2*layer` and `2*layer+1` are the
candidate K and V mapping checked by the diagnostics. Each transfer contains the
prompt **excluding its final token**. A decoder must process that final input
token at position `length-1` before sampling. The receiver itself does no decoding.

## CPU reference diagnostics

These require torch, safetensors and transformers. The measured versions are
torch 2.10.0+cpu and transformers 5.19.0. Use the pinned checkpoint in RESULTS.md:

```bash
python opt_prime/examples/pd/native_kv/check_first_layer.py \
  --model-dir "$MODEL_DIR" --slots /tmp/native-kv-new-run/case-03-n33 \
  --positions 32 --out /tmp/first-layer.json
python opt_prime/examples/pd/native_kv/check_resume.py \
  --model-dir "$MODEL_DIR" --capture-dir /tmp/native-kv-new-run \
  --decode-steps 4 --out /tmp/resume-cases.json
```

The reference dequantizes the checkpoint's FP8 weights using its 128x128 scales
and performs BF16 CPU operations. It does **not** emulate native dynamic activation
quantization. `check_resume.py` needs memory for the approximately 16 GB BF16 model
plus loading/workspace overhead. It compares all 36 layer caches, resumes from
native KV, and compares first-step logits and generated tokens with torch's own
cache. A full-prompt torch pass measures the reference's own numerical variation.
These diagnostics do not establish native FP8 numerical parity or ATOM support.

## CPU-only protocol checks

```bash
python3 -m unittest discover -s opt_prime/examples/pd/native_kv -v
```

Tests use small control/header fixtures from synthetic owned requests; no model
payload. They cover observed bytes, variable IDs, truncated headers, invalid
references, unsupported layouts, metadata and payload mismatches.

## Limits and remaining work

- The layout is fixed to one chip, global BF16, 2048-byte blocks, 72 slots and the
  observed pool geometry. No sliding/hybrid attention, auxiliary cache or TP variants.
- Out-of-line request IDs of 8..127 ASCII bytes are supported; inline short-string
  encoding is intentionally excluded. The example generates valid IDs itself.
- Wire parser/serializer supports only observed variants; it is not a general
  implementation of the native serialization library.
- No cancellation, reconnection, concurrent requests, device-memory transport or
  serving lifecycle integration. This is a single-client research tool.
- Validate more prompts, prefix reuse, matching precision and first-step logits
  before admitting a production backend. Native PD itself differed from standalone
  text on one of the initial four synthetic cases; the cause remains unresolved.
- ATOM compiled artifacts and physical cache layout must be validated separately.

The branch starts at `main` and is deliberately standalone: the integrated PD
engine currently lives on `optimus-rm`. Engine integration should be a separate
reviewable change on that line, once its admission criteria are met.

## In-memory performance path

`memory_receiver.py` adds a reusable, single-flight client and a payload-backed
view path without file I/O or per-token copies. See [MEMORY_PERFORMANCE.md](MEMORY_PERFORMANCE.md)
for reproducible old/new comparisons, boundary differences and the remaining gap
against native TTFT. This remains experimental and does not enable a serving backend.


## Experimental A100 continuation

`gpu_decoder.py` loads the pinned Qwen3-8B FP8 checkpoint as BF16 or Marlin
weight-only FP8 and owns a single fixed-capacity CUDA graph lane. `bench_live_gpu.py`
connects the native receiver directly to that lane. Use `--packed-import` for one
pinned contiguous upload. These examples are not registered as a production backend.

`bench_gpu_decode.py` measures isolated decode. `check_gpu_logits.py` compares
teacher-forced logits on a shared BF16 prefix and history. Both require a local
checkpoint via `--model-dir`; see [GPU_PERFORMANCE.md](GPU_PERFORMANCE.md) for versions,
commands, timing boundaries, failed runs and precision limitations.

Norm optimization is opt-in. `--fused-norms --norm-mode torch-stats` preserves the
reference variance/rsqrt operations and passed the recorded bitwise unit checks.
The fully fused default norm mode changes native-KV outputs and is diagnostic only.
No current result establishes full native-speed serving or native FP8 quality parity.

Colocated RNGD+A100 follow-up, including live torch-stats parity and remaining latency gap: [GPU_COLOCATED.md](GPU_COLOCATED.md). `--cpu-threads` controls the client copy worker count; default remains4.


`--streaming-import` uploads validated batches during reception; `--compare-imports`
rotates it against native and packed modes. It is experimental and disabled by default:
latency gains were not stable across lengths or CPU affinity settings. See
[GPU_STREAMING.md](GPU_STREAMING.md) for positive and negative results.


`--fuse-mlp` combines gate/up projections. `--fuse-qkv` combines decode QKV in the
Flash path and retains original prefill weights (approximately0.9GB extra memory).
Both are opt-in. `check_live_quality.py` compares EOS-truncated responses with native;
`--short-only` selects the original12 smoke cases and `--long-only` selects3 retrieval
cases. See [GPU_PROJECTION_FUSION.md](GPU_PROJECTION_FUSION.md):15 native-text matches
are encouraging but not a broad quality guarantee, and current throughput timings do
not include equivalent GPU HTTP streaming or concurrency.


## HTTP streaming experiment

`serve_gpu.py` exposes loopback `/health` and `/v1/completions` with greedy SSE or
nonstream responses, bounded FIFO admission and one dedicated worker. It is not a
batched production backend. `bench_http.py` compares identical requests over HTTP;
`check_http_quality.py` checks EOS/response text. See [GPU_HTTP.md](GPU_HTTP.md) for
commands, supported limits, cancellation checks and the large c4 batching gap.

## Experimental batched HTTP decode

`serve_gpu.py --batch-size 4` enables wave batching with independent KV positions.
See [GPU_BATCH.md](GPU_BATCH.md) for measurements and limitations. Native prefill
remains serial; concurrent throughput and TTFT have not reached native parity.

`--pipeline --batch-size 4` separates the native socket owner from the GPU owner,
admits prepared prefixes into available lanes during decode, and pauses lanes
with stalled event consumers. See [GPU_PIPELINE.md](GPU_PIPELINE.md).

Eight-lane pipeline and prefill-side CPU packing measurements are in [GPU_EIGHT.md](GPU_EIGHT.md).

Concurrent native prefill is opt-in with `--native-batch-size 8`; see [GPU_MULTIPLEX.md](GPU_MULTIPLEX.md) for protocol tests, numerical limits and HTTP results.
