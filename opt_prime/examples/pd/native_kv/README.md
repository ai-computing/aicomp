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
