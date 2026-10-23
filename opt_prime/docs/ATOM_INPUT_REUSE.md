# ATOM persistent-input optimization

Date:2026-10-19. RNGD3→ATOM0, Qwen2.5-0.5B, B1/context512.
A/B/B/A order,16requests and320tokens per run. Four prompts, lengths32/64/41/37,
outputs8/32,2repeats. Same artifact, graph, precision, full-logit materialization
and verification. Host-admitted references always use the unoptimized path.
Each variant totals32requests/640tokens; all64requests matched full logits,
all3072DMA commands completed, all workers exited0. Cross-run tokens matched.

| Metric | Original | Persistent inputs | Reduction |
|---|---:|---:|---:|
| Aggregate decode time | 5.722573s | 4.693283s | 17.99% |
| Aggregate PD request time | 9.078880s | 7.965827s | 12.26% |
| Decode time/token (includes verification) | 8.942ms | 7.333ms | 17.99% |
| Median input-transfer host interval/token | 0.691ms | 0.350ms | diagnostic |

All8paired prompt/output cases improved in median decode and PD time. Raw
first-use costs remain included; no warmup rows were silently excluded.
`summary.json` contains every paired case. Timings are fenced stage host-wall
intervals, not isolated kernel profiler measurements. The measured change combines
persistent device buffers, position tables and constant block handling; individual
contributions were not separately ablated. Runtime interval also changed, so do
not attribute the entire gain solely to host allocation.

This is a finite single-lane test, not saturation throughput, p99 SLO, power/MFU
or vendor-native-engine superiority. No confidence interval is claimed from two
process runs per variant. The change preserves the original numerical behavior;
it does not fix3B's previously observed FP32 mismatch.

Enable with `atom_reuse_inputs: true` in the Qwen runner config. It is opt-in;
the default remains false. Use one synchronous owner; input buffers cannot be
shared across concurrent calls. KV/export allocations remain unchanged.

Run `examples/pd/summarize_input_reuse.py --root <experiment-directory>` to
recalculate the A/B/B/A table. Evidence is in `validation/native-engine-optimization/`.
See [native engine integration assessment](NATIVE_ENGINE_INTEGRATION.md) for the
vendor reuse strategy and currently missing connector/hardware prerequisites.

## Additional regression check

Qwen2.5-1.5B with persistent inputs passed16direct requests/320tokens/896DMA
commands plus8actual llm-d sidecar requests/160tokens/448DMA commands. All
full-logit checks used original-path references; all sidecar tokens matched the
direct path. Worker and sidecar exited0. This is compatibility evidence, not a
1.5B speedup measurement. Unit suite32passed. A prior overly long socket-path
run failed before ATOM startup and is retained separately, with an early path
validation fix.
