# Four-model GPU HTTP validation (#211/#212)

Run `validate.py --model MODEL --output result.json` under the engine environment with a free CUDA device. `--prefix-cache` enables the separate cache case; `--dtype float32` provides the numerical control. Offline immutable model snapshots are required. The default configuration is four paged execution slots, context9216, page64, prefill chunk256, inputs128/2048/8192 and32 fixed output tokens. This exercises the real HTTP service factory and sockets, not a mocked handler.

`passed` covers HTTP lifecycle and completion counts. It does **not** assert equality of concurrent versus serial output: that is stored per response as `baseline_equal`. Cancellation compares warm-state output immediately before/after, and records cold-baseline equality separately. Cache-enabled runs require positive observed cache hits. Metrics are sampled after drain.

`trace_prefix_http.py` uses the same CLI and records actual2K prefill boundaries and initial logits. `diagnose_prefix.py --model MODEL --output result.json` teacher-forces histories in the direct batcher. The direct probe did not reproduce the HTTP argmax change, so it must not replace actual HTTP evidence. An additional uncommitted allocation-mode probe changed only inference_mode to no_grad; its result is retained as a negative diagnostic in evidence.

See experiments/2026-10-23-http-matrix211.md and evidence/summary.json. Evidence includes the initial failed KONI cold-baseline cancellation assertion. Earlier runs preceded the warm-state cancellation control and some metadata additions; their commands/configurations are recorded in the experiment. Public copies normalize local model-directory labels in metrics only; raw measurements remain in the research workspace.

This GPU experiment does not complete ATOM/RNGD HTTP or PD coverage, long-context RNGD support, hybrid NPU cache restoration, broader semantic quality or true compute/transfer overlap.
