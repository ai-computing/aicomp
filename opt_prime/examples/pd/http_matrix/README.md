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

### Torch PD producer reproducibility controls

`serve_batched.py --pd-prefill-device cuda:0` keeps producer prefix caching enabled and uses a 2048-token producer chunk by default. These are independent of decoder `--prefix-cache` and local `--chunk`.

To compare BF16 native and PD under uncached, matching chunk execution:

```bash
python examples/pd/serve_batched.py --model "$MODEL" --device cuda:1 --slots 4 \
  --max-context 9216 --chunk 256 --pd-prefill-device cuda:0 \
  --pd-prefill-no-prefix-cache --pd-prefill-chunk 256
```

The new flags affect the in-process Torch producer only. Nonpositive producer chunks are rejected. Disabling reuse or using smaller chunks can increase prefill work/overhead; these options do not promise a speedup or universal bitwise equality. CPU factory and four-model GPU controls are recorded under issue #241.

### Live GPU PD matrix

The validator accepts `--device`, `--pd-prefill-device` and the producer controls above, records actual producer policy and asserts remote prefill/transfer counters. It also accepts `--http-module` for a trusted local HTTP implementation candidate; the candidate is loaded before its consumers so exception classes retain their identity.

```bash
git show 9e442cd:opt_prime/opt_prime/accel/serve/llmd.py > /tmp/pd-http-safe.py
PYTHONPATH=opt_prime python opt_prime/examples/pd/http_matrix/validate.py \
  --http-module /tmp/pd-http-safe.py --model "$MODEL" --output "$OUTPUT" \
  --device cuda:1 --pd-prefill-device cuda:0 \
  --pd-prefill-no-prefix-cache --pd-prefill-chunk 256
```

Run from the repository root. Omit the producer arguments for native control. `passed` means lifecycle/count assertions passed; concurrent/serial token equality is reported separately, including failed numerical comparisons. See gpu-live-pd240 for the complete default-policy, FP32 and matching-policy results.

### Terminal producer availability

Subprocess producers expose a boolean `ready` property; pools require every configured member that implements it to be ready. Batched `/ready` and `/health` return503 after a known terminal failure/close, and new completions receive retryable503 before admission. A healthy replacement at an idle boundary restores availability without poisoning the decoder. Individual request errors do not make a still-running producer unready.

No pool member is silently removed and no request is automatically retried. Workers without this optional property retain prior behavior. Existing requests keep their existing completion/error handling. See producer-readiness246 for the real subprocess/HTTP reproduction and regression results.

### Matched HTTP performance (#248)

`benchmark.py` starts isolated production server processes, warms each endpoint, checks nonstream token IDs separately, then measures simultaneous SSE request batches. Run from the repository root with a locally cached checkpoint:

```bash
PYTHONPATH=opt_prime python opt_prime/examples/pd/http_matrix/benchmark.py \
  --model "$MODEL" --mode native --lengths 128 8192 --outputs 32 \
  --concurrency 1 4 --repeats 3 --output native.json
PYTHONPATH=opt_prime python opt_prime/examples/pd/http_matrix/benchmark.py \
  --model "$MODEL" --mode pd --lengths 128 8192 --outputs 32 \
  --concurrency 1 4 --repeats 3 --reference native.json --output pd.json
PYTHONPATH=opt_prime python opt_prime/examples/pd/http_matrix/benchmark.py \
  --model "$MODEL" --mode dp2 --lengths 128 8192 --outputs 32 \
  --concurrency 1 4 --repeats 3 --reference native.json --output dp2.json
```

Where the SIGPIPE candidate is needed, pass `--http-module /tmp/pd-http-safe.py` from the earlier command to all three invocations. The module is trusted executable code. Native uses cuda:1; PD uses cuda:0 for prefill and cuda:1 for decode; DP2 uses two independent processes and explicit client round-robin routing. Use only GPUs allocated to the experiment. Each decoder has four slots by default, so DP2 has eight total slots versus four for PD; each server process has four CPU threads. This compares the same GPU count for PD/DP2, not identical decoder capacity.

Both caches are disabled and producer/local chunks are256. Warmup, loading and separate token-ID checks are excluded from timed batches. JSON retains raw per-request SSE events' first-arrival latency, E2E latency, estimated TPOT, reconstructed text equality, whole-batch throughput and final server metrics. SSE events can coalesce tokens: first-event latency and estimated TPOT are not per-token engine timings. `passed` records workflow/count success; `parity_passed` additionally requires reference token/text equality. A parity failure saves evidence and exits unsuccessfully.

Repeat modes in reverse order against the same native reference and do not run CPU regression suites concurrently with final measurements. These Torch PD baselines gather and transfer the whole KV after prefill; they do not implement within-request compute/copy overlap. Existing HTTP transfer histograms do not observe the batched worker timing path; zero observations do not mean zero transfer cost. See issue249 for separate accumulated worker duration counters.

### External backend comparison

`benchmark.py --server-config config.json` uses the same client requests, reference checks and timing loop with explicitly supplied server commands. The file is trusted executable configuration: `servers` is a nonempty list of argv arrays (no shell expansion), each containing exactly one standalone `{port}` argument. `metadata` describes physical resources and execution policies; it replaces GPU-specific defaults rather than implying a GPU configuration for an NPU server. Only supported resource fields are accepted.

`atom_server.py` connects one assigned ATOM to its existing B1/context9216 artifact. Omit `--worker-python`/`--engine-factory` for native ATOM; supply both for the established RNGD subprocess producer. Pass explicit trusted module candidates and artifact root. The allocator/container bootstrap must restrict visible devices; the server also rejects anything other than one ATOM node. Requests in this benchmark use `ignore_eos=true`, so fixed output lengths are comparable. Client concurrency above one is queued work, not physical batch size.

For slow cold initialization, pass `--startup-timeout 1800` explicitly. Default180 remains unchanged; nonpositive values fail before model loading. Startup is excluded from measured latency/throughput. In the first hardware attempt, the shorter outer timeout terminated initialization before the existing subprocess worker's1800second deadline; that failure is retained alongside the passing nativeATOM baseline. A longer deadline is not evidence that initialization or PD succeeds.

The initial Qwen3-0.6B nativeATOM run passed12batches at inputs128/8192,output32,concurrency1/4. RNGD-to-ATOM performance/parity remains under validation. Use the experiment record for pinned runtime modules, environment and commands.
