# Experimental RNGD layer streaming

`serve_batched.py --pd-prefill-stream-layers` enables validated partial KV reception
with a Torch or ATOM decoder. The subprocess must advertise progress support.
ATOM imports only into a reserved lane. B1 next-request prefetch may compute ahead,
but its partial data waits until the current decode releases the destination.
Multi-lane hybrid caches defer layer imports until final decode admission because
idle lanes may still write their local cache. SDK write failures poison the owner;
invalid source geometry fails only the request. Incomplete prompts never enter
the reusable prefix index.

## Code review map

This path uses same-host shared memory for KV payloads and subprocess pipe messages
for progress descriptors. The measured producer is the research factory engine
listed below. It uses the existing vendor SDK; this change does not expose the
closed native furiosa-llm engine's KV protocol or establish cross-host streaming.

| Responsibility | Code | Review invariant |
|---|---|---|
| Source export | [prefill_server.py](../../opt_prime/accel/pd/prefill_server.py), [factories](streaming_factories.py) | Copy each layer/group to SHM before notification; preserve layer arithmetic and head. |
| Transport and coverage | [pd_batching.py](../../opt_prime/accel/pd/pd_batching.py), [layer_progress.py](../../opt_prime/accel/pd/layer_progress.py) | Validate request IDs, nonoverlapping spans and complete coverage; serial writers keep published storage stable. |
| Torch destination | [paged.py](../../opt_prime/accel/pd/paged.py), [PDBatcher](../../opt_prime/accel/pd/pd_batching.py) | Map dense layers to cache groups, fence destination copies before page reuse, admit only complete KV. |
| ATOM destination | [batching.py](../../opt_prime/accel/platforms/atom/batching.py) | Reserve the lane before import; protect active B1/hybrid caches; never publish an incomplete prefix. |
| Serving switch | [serve_batched.py](serve_batched.py) | Explicit opt-in and a compatible subprocess producer; whole transfer remains the default. |

Per-layer source export is separate from destination notification timing. The
three-mode GPU control retains per-layer source/SHM work while delaying progress
notifications until source logits return. Much of the measured Qwen/KONI gain
persists under that control; early notification adds a smaller benefit in the
recorded workload. The control also retains per-layer destination import and
allows cross-request overlap, so it cannot assign the remaining gain to source
assembly alone. See the [control summary](http_matrix/evidence265/stream265-control-summary.json)
and experiment record for samples and limits.

The three factories in `streaming_factories.py` cover the four measured models:

| Factory | Model | Existing producer |
|---|---|---|
| `streaming_factories:qwen` | Qwen3-0.6B, Qwen3-4B | `long_width_variant.make_prefill` |
| `streaming_factories:gemma` | Gemma3-1B | `gemma_prefill.Prefill` |
| `streaming_factories:koni` | KONI-Llama3.1-8B | `koni_prefill.Prefill` |

Set the producer's `PYTHONPATH` to include this package, `examples/pd`, and the
existing `serving_matrix` and `qwen3_rngd` example directories from aicomp commit
`04dab7cb6961103f9404c99e9fcbcaadac14b098`, plus the compatible `rngd_pd` package.
These are explicit research dependencies; the adapters do not bundle the compiler
or model implementations. Exact measured source hashes are in evidence265.

The producer arguments are:

```text
--engine factory --engine-factory streaming_factories:qwen --width 128 --chip 0
```

Pass those through `--pd-prefill-args`, choose the corresponding factory, and use
`--pd-prefill-python` for the isolated Furiosa interpreter/registration wrapper.
Use local safetensors model checkpoints. Factory width is one compiled width and
must fit `--max-context`; the model-specific constructor also checks supported
widths and geometry. KONI's validated partitioned producer requires width512 or
larger in multiples of512.

Keep the same factory and omit only `--pd-prefill-stream-layers` for the whole-KV
control. That path calls the original producer's `prefill` method. Streaming
preserves its layer arithmetic and final head but exports each finished layer or
group before the next one runs. Published spans are copied to shared memory before
notification and must remain immutable until the serial producer finishes the
request. Decoder cache lengths and reusable prefix state are published only after
all requested suffix spans arrive and destination copies complete.

This is blocking per-group CPU export followed by incremental decoder import.
It can allow decoder import during later source computation, but does not imply
asynchronous RNGD download, device DMA overlap, or a serving speedup. Hardware
source parity has been checked at widths128/512 and8192. Recorded GPU timing
improvements apply only to the measured c2/fixed32 and c4/fixed128 workloads. Four-model ATOM timing
and same-route parity are also verified for B1/two queued2K/8K prompts, fixed32
outputs, and prefetch off/on. Actual ATOM streaming fault isolation remains pending.
Notification-timing controls show that total improvement is not solely the effect
of early notification; source/host path changes and cross-request overlap remain. None establishes cross-backend
numerical equivalence. See the experiment record for exact cases,
failed attempts and the current verification scope.
