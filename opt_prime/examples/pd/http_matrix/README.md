# Four-model GPU and ATOM HTTP validation (#211/#212)

Run `validate.py --model MODEL --output result.json` under the engine environment with a free CUDA device. `--prefix-cache` enables the separate cache case; `--dtype float32` provides the numerical control. Offline immutable model snapshots are required. The default configuration is four paged execution slots, context9216, page64, prefill chunk256, inputs128/2048/8192 and32 fixed output tokens. This exercises the real HTTP service factory and sockets, not a mocked handler.

`passed` covers HTTP lifecycle and completion counts. It does **not** assert equality of concurrent versus serial output: that is stored per response as `baseline_equal`. Cancellation compares warm-state output immediately before/after, and records cold-baseline equality separately. Cache-enabled runs require positive observed cache hits. Metrics are sampled after drain.

`trace_prefix_http.py` uses the same CLI and records actual2K prefill boundaries and initial logits. `diagnose_prefix.py --model MODEL --output result.json` teacher-forces histories in the direct batcher. The direct probe did not reproduce the HTTP argmax change, so it must not replace actual HTTP evidence. An additional uncommitted allocation-mode probe changed only inference_mode to no_grad; its result is retained as a negative diagnostic in evidence.

See experiments/2026-10-23-http-matrix211.md and evidence/summary.json. Evidence includes the initial failed KONI cold-baseline cancellation assertion. Earlier runs preceded the warm-state cancellation control and some metadata additions; their commands/configurations are recorded in the experiment. Public copies normalize local model-directory labels in metrics only; raw measurements remain in the research workspace.

This GPU experiment does not complete ATOM/RNGD HTTP or PD coverage, long-context RNGD support, hybrid NPU cache restoration, broader semantic quality or true compute/transfer overlap.

## ATOM extension

Use `--backend atom` in an isolated allocation exposing exactly one ATOM. Existing artifacts must use B1/context9216 and match `--atom-kv-blocks` (default3; KONI8B requires1). The two Qwen models support `--prefix-cache`; add `--host-kv-gib 2` to exercise the host tier and require positive observed host hits. Artifact precision is fixed; float32 is rejected for ATOM. Set OPT_PRIME_CACHE to the task-owned artifact cache. SDK/device bootstrap is deployment-specific and must map the allocated physical device to runtime-visible0 when required.

Inputs128/2048/8192 and clients4/8 are queued over one ATOM execution lane, not hardware batching. All four cache-off runs pass lifecycle checks and48/48 concurrent/serial comparisons. Both Qwen device/host cache cases pass12/12 each. Gemma requires the causal-prefill SDK graph correction and #210 hybrid local-table/mask fix; its prefix/host/PD restoration remains unsupported. KONI's one-block configuration has no spare prefix block.

See experiments/2026-10-23-atom-http211.md and evidence/atom-http-*.json. Files ending `fixed` include #214's HTTP MSG_NOSIGNAL fix; the Qwen0.6 matched rerun preserves all output sequences and eliminates the observed native disconnect backtraces. This does not establish all RNGD/heterogeneous routes or real transfer overlap.
