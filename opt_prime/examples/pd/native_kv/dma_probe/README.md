# Reproduce reverse-contiguous KV read overhead

This isolated hardware probe compares reverse scatter with one contiguous read followed
by CPU token-block restoration. It does not modify or accelerate furiosa-llm itself.
It exclusively opens RNGD chip0 (8 PEs) and allocates its own small DRAM buffer. Run only
when that chip is available. It never reads another process's allocations.

## Reproduction

Requires Rust nightly2026-05-01 and the tested local furiosa-opt-rt0.9.0 source used by
the research engine. Set FURIOSA_OPT_RT_DIR to that source directory. The local runtime
has changes beyond the published crate; do not assume an arbitrary0.9.0 checkout is
identical. The runtime source hashes in the accompanying record identify the measured
implementation. Cargo's patch applies only to this invocation.

```bash
cd opt_prime/examples/pd/native_kv/dma_probe
cargo +nightly-2026-05-01 --config "patch.crates-io.furiosa-opt-rt.path=\"$FURIOSA_OPT_RT_DIR\"" test --offline
cargo +nightly-2026-05-01 --config "patch.crates-io.furiosa-opt-rt.path=\"$FURIOSA_OPT_RT_DIR\"" build --release --offline
timeout 60s target/release/kv-read-probe > results.jsonl
```

The program validates every output byte against an independently indexed pattern,
prints126 JSON records and exits nonzero on mismatch/transfer failure. There are21
repeats per method/size, AB/BA order alternating, with repeat0 marked warmup. Buffers
are allocated before timing. Scatter time includes descriptor construction and DMA;
merged total includes DMA and CPU restoration. Initial H2D and equality checks are
outside timing. Compare warm medians per method/size, including restoration cost.

The only restoration helper requires separate, equally sized slices composed of whole
blocks. It reverses block order while preserving bytes inside a block. The hardware
probe awaits every read before inspecting/restoring/dropping any buffers; it has no
concurrent request cancellation. It is not a cancellation-safe serving adapter.
