# Local llm-d integration feature

Branch: `feat/optimus-llmd`, incorporating engine commit `a2740f1` from
`feat/optimus-heterogeneous-pd`. This feature does not change the engine branch.

## Implemented path

```text
client → upstream llm-d shared-storage sidecar
         ├─ prefill HTTP → owner queue → PrefillEngine → retained source KV
         └─ decode HTTP  → owner queue → PDRuntime using prepared continuation
                                        → explicit transport → DecodeEngine
```

`opt_prime.accel.serve.llmd` is optional and uses only Python's standard library plus the
engine contracts. `PDStageService` binds the model's vocabulary/context/output
limits and a rendering callback. It is not tied to KONI's vocabulary. The two
loopback HTTP endpoints share one service and one source lease; KV bytes and DMA
addresses are never serialized over HTTP.

The inspected upstream source is inference-scheduler commit
`5f4e762f341a5196393ce79f8a57c3e1900c4a6b`, particularly
`pkg/sidecar/proxy/connector_shared_storage.go`. Its sidecar uses
`x-prefiller-host-port`, rewrites the prefill output limit to one, and forwards the
original decode limit with `cache_hit_threshold=0`. Our key is `x-request-id` plus
exact model/prompt identity. A changed output count does not trigger a second
prefill. No NIXL or Mooncake wire-protocol implementation is claimed.

## Ownership and API

- `/v1/completions` and `/v1/chat/completions` (#41): token-ID or text `prompt`, or chat
  `messages` through the tokenizer's chat template (`PDStageService(tokenizer=...)`),
  `max_tokens` as an upper bound. `stream=true` returns OpenAI-compatible SSE: one chunk
  per token as the owner produces it (the prefill's token before the KV copy, then each
  decoded token if the decode engine streams, else after decode), a chunk with
  `finish_reason`, a `usage` chunk and `data: [DONE]`; text that could start a `stop`
  string is held back until it cannot. Sampling fields `temperature`, `top_p`, `top_k`, `seed` (#42) are
  accepted when both engines declare `supports_sampling` (the torch engine does);
  greedy-only engines (RNGD/ATOM graphs) reject other values with 400. Generation
  ends at an EOS token (`PDStageService(eos_token_ids=...)`, unless `ignore_eos`) or
  at a `stop` string (up to 4; the text excludes it); `finish_reason` is `stop` or
  `length`. Engines that cannot stop early are cut after the stop token, so the
  output is the same but the device time is not saved. Conflicting output counts,
  unknown fields, incompatible model and over-capacity requests fail explicitly.
- `/health`, `/ready`: reflect owner-loop startup and quarantine state.
- `OwnerExecutor.run()` must execute in the main owning process thread for RNGD
  PDMA. HTTP worker threads only enqueue commands; they never touch NPU buffers.
- One lane: a second outstanding prefill gets 429; a duplicate ready prefill is
  idempotent. Decode requires matching ready state. This is not a batch scheduler.
- `cancel` is an owner command for a completed source prefill only. TTL expiration
  is checked on every request and by the idle owner loop every `sweep` seconds
  (`OwnerExecutor(service, sweep=1.0)`), so an abandoned prepared prefill frees the
  lane without a new request. Neither cancels an in-flight copy/decode or authorizes
  premature destination reuse.
- Failures (#49): an endpoint that raises `RetryableError` guarantees nothing is
  outstanding; the lane returns to idle and the client gets 503 with `Retry-After: 1`
  and `"retryable": true`. Any other engine failure quarantines the service and
  retains the prepared source and runtime owners: 500 `"retryable": false`, then
  readiness 503 and 503 `"retryable": true` for new requests (retry on another
  replica). `POST /v1/admin/reset` reopens the lane only when the prefill engine,
  decode engine and transport each confirm a known-good device with `recover()`;
  a backend without `recover()` answers 501 (restart the stage). No fallback occurs.
- A client that closes its connection while its request is queued or running is
  cancelled through the same fenced path as `POST /v1/cancel`: device work already
  submitted finishes or drains, then the lane is released (499, not delivered).
  Stop is serialized
  behind prior commands and only succeeds with an idle/ready, quiescent owner.

`GET /metrics` exposes the stage in Prometheus text format under vLLM names, labelled `model_name`:
`vllm:num_requests_running`, `vllm:num_requests_waiting` (owner queue), `vllm:kv_cache_usage_perc` and
`vllm:gpu_cache_usage_perc` (occupied fraction of the single lane), `vllm:request_success_total{finished_reason}`,
`vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, histograms `vllm:time_to_first_token_seconds` (prefill
stage), `vllm:time_per_output_token_seconds` (decode stage), `vllm:e2e_request_latency_seconds`, and
`airm_kv_transfer_seconds` / `airm_stage_request_failures_total`. No client library is needed.

**Continuous batching (#44).** `opt_prime.accel.serve.batched` serves a `pd.batching.ContinuousBatcher` (iteration-level
scheduling, chunked prefill, slot or paged KV) through the same `serve_stage` HTTP layer: any number of requests in
flight, admitted into free slots/pages, with streaming, cancellation on disconnect and /metrics. It runs prefill and
decode in one process (not disaggregated); PD decode admission into the batcher is the next stage.

Prefill readiness here means a fenced source KV is available. Destination admission
occurs in the decode stage immediately before compute. This differs from the old
experimental broker's destination-admission-before-prefill-response ordering.
Both preserve the required transfer-before-decode ordering; this feature does not
claim the two have identical latency.

## Reproduction

Torch-only real-model validation with the external pinned sidecar binary:

```bash
PYTHONPATH=opt_prime python3 opt_prime/examples/pd/validate_torch_sidecar.py \
  --sidecar /path/to/pd-sidecar --out /tmp/new-optimus-torch-run
```

KONI hardware validation extends the engine runner on this branch:

```bash
PYTHONPATH=opt_prime:/path/to/hetero-npu-p2p/src:/path/to/rngd-pd/src \
  /path/to/furiosa-python opt_prime/examples/pd/validate_koni.py \
  --config /path/to/koni-config.json --hetero-repo /path/to/hetero-npu-p2p \
  --llmd-sidecar /path/to/pd-sidecar
```

It first measures direct engine parity, then passes the same cases through the
actual sidecar. The harness binds new loopback ports, records binary identity,
checks tokens and completion receipts, and stops only its own proxy after owner
quiescence. The existing NPU worker is stopped through its normal protocol.
The sidecar binary is external and is not redistributed in this repository.

## Validation boundaries

Unit tests cover sidecar limit rewriting, no repeated prefill, request mismatch,
capacity, TTL expiry (on request and by the owner sweep), cancellation on client
disconnect, safe cancellation, quarantine retention, actual HTTP
requests and owner shutdown. Real-model and real-NPU sidecar evidence is kept in
`docs/validation/llmd/` after each run. CPU fixtures alone do not establish a
hardware path.

This is local, non-streaming shared-storage-sidecar integration. Multi-Pod ownership,
FD transfer across namespaces, authentication, EPP/Gateway API routing, deployment
manifests, streaming, sampling and multi-node operation remain separate work.
Unsupported modes are not silently represented as working distributed serving.

Completed run results: [sidecar validation](validation/llmd/RESULTS.md).

## Qwen NPU endpoints

Qwen2.5 uses the same service and owner queue as KONI; the llm-d module does not
need model-specific code. Copy `examples/pd/qwen_pd_config.example.json` and
replace every path with your local checkout, pinned model, compiled artifact,
SDK interpreter and sidecar binary. Device indices in the example are illustrative:
choose an idle physical card and verify its corresponding logical RNGD index.
Keep each run's output directory new. Remove `sidecar_binary` for direct-only runs.

```sh
PYTHONPATH=opt_prime:/path/to/hetero-npu-p2p/src:/path/to/rngd_llmd/src \
 /path/to/furiosa-venv/bin/python opt_prime/examples/pd/validate_qwen_pd.py \
 --config /path/to/local-qwen-config.json
```

This finite runner first checks direct P2P and then the local sidecar with the
same endpoints. It is a validation harness, not a persistent production service.
Model revision/geometry must match the compiled artifact's adjacent `model.json`.
See [backend installation and extension](NPU_BACKEND_EXTENSIONS.md) and
[Qwen results](QWEN_PD_VALIDATION.md).
