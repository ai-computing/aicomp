# Experimental RNGD layer streaming

`serve_batched.py --pd-prefill-stream-layers` enables validated partial KV reception
with a Torch or ATOM decoder. The subprocess must advertise progress support.
ATOM imports only into a reserved lane. B1 next-request prefetch may compute ahead,
but its partial data waits until the current decode releases the destination.
Multi-lane hybrid caches defer layer imports until final decode admission because
idle lanes may still write their local cache. SDK write failures poison the owner;
invalid source geometry fails only the request. Incomplete prompts never enter
the reusable prefix index.

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
improvements apply only to the measured c2 workload; ATOM timing and hardware
fault isolation still require verification. None establishes cross-backend
numerical equivalence. See the experiment record for exact cases,
failed attempts and the current verification scope.
