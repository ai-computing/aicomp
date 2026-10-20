# Live RNGD → native ATOM validation

2026-10-19, parent commit 795af1d with new producer and modified validator.
Physical RNGD npu3ch0 (compute index2), ATOM rbln0. Furiosa2026.4 and
Rebel0.11.0/Optimum0.11.0.post1 run in separate Python processes.

| Model | Requests | Output tokens | Full-logit parity | Forbidden prefill calls |
|---|---:|---:|---|---:|
| Qwen2.5-0.5B-Instruct | 8 | 256 | exact, every step | 0 |
| Qwen2.5-1.5B-Instruct | 8 | 256 | exact, every step | 0 |

For each request a persistent RNGD worker computes a new prefix using the existing
custom Qwen SDK graph (accurate_attention=true, width64). It synchronously copies
KV to host and writes a unique NPZ; ATOM reads it and runs native decode without
prefill. A native-prefill primer followed by replacement with that same RNGD KV
is the full-logit reference. First candidate precedes any native prefill.
Four prompts × two repeats, B1/context512, greedy32. All 37 CPU regression tests pass.

This is an integration/ownership validation, not an end-to-end performance result.
It does not use Furiosa-LLM native serving, DMA, PDRuntime or HTTP/llm-d in this new
path. Existing P2P serving is unchanged. File I/O is deliberate test transport;
no latency or throughput improvement is claimed. Numerical quality differences
between RNGD prefill and native/FP32 prefill remain a separate issue.

Use `examples/pd/validate_native_no_primer.py` with the same config/native artifact
arguments as the saved-KV probe, adding `--live-rngd-python /path/to/furiosa/python`.
Select a new `--out`; each run saves exact argv, source snapshots, bindings and
raw per-request results. Worker startup/response timeout is 300s; parent cleans up
only its own subprocess. NPZ hashes are recorded; weights and KV binaries are not
committed. Evidence: `validation/native-live/`.

Next: separate host admission from decode and implement a request-bound transport
adapter under the existing KVLease contract; only then test llm-d serving and
controlled performance comparisons. Review guide: [code map](CODE_REVIEW_GUIDE_KO.md).
