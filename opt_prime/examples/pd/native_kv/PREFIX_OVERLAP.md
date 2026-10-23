# Overlapped GPU prefix and native suffix prototype (#162)

Parent565092d; exact source hashes in prefix_overlap_manifest.json. Same pinned
Qwen3-8B-FP8, FXB, SDK and GPU+RNGD hardware as PREFIX_RECOMPUTE.md. Native1RNGD
reference versus RNGD+A100 prototype. Native prefix caching remains disabled.

## Implementation

Extend bounded multiplex/direct receive with a shared source-prefix offset. Full
transfer remains offset0. Validate offset before submitting HTTP and size owned pinned
storage to the selected suffix. The existing rolling serving path still uses offset0.

The single-request prototype starts native full prefill on the receiver's dedicated
owner thread while the GPU LocalPrefiller computes the first fraction of the prompt.
The native exporter sends only remaining KV positions. GPU prefix tensors and native
suffix tensors are concatenated in position order, then imported into the existing
fixed16-lane Marlin decode graph. Ownership is fenced before either producer's buffers
may be reused. Prefix BF16-prefill weights are dequantized from the same FP8 checkpoint;
this arithmetic differs from native, so combined-cache quality must be measured.

Tests:54 pass. Added split-boundary byte equality on CUDA, buffer reuse after import,
other-lane isolation, invalid geometry/capacity/lane checks, multiplex suffix routing
and offset bounds. Existing full-transfer wire fixtures still pass. No serving default
or continuous-admission policy changes.

## Commands and boundaries

Start dedicated native as in GPU_LOCAL_PAIRED.md. Run from native_kv:

```bash
python3 -m unittest discover -s . -p 'test_*.py'
python3 bench_prefix_recompute.py --model-dir "$MODEL_DIR" --mode perf --repeats 3 --out "$OUT/prefix_overlap_perf.json"
python3 bench_prefix_recompute.py --model-dir "$MODEL_DIR" --mode quality --fractions 0,0.75 --out "$OUT/prefix_overlap_quality.json"
python3 bench_prefix_recompute.py --model-dir "$MODEL_DIR" --mode perf --repeats 3 --fractions 0,0.75,0.875 --out "$OUT/prefix_overlap_repeat.json"
python3 bench_prefix_recompute.py --model-dir "$MODEL_DIR" --mode quality --fractions 0.875 --out "$OUT/prefix_overlap_quality875.json"
python3 summarize_prefix_overlap.py --root "$OUT" --out "$OUT/prefix_overlap_summary.json"
```

Performance:one request at a time; deterministic random prompts seed139+length,
128/512/1024 input tokens,32 generated tokens. Each length has one excluded warmup
and three measured repeats. Variant order alternates; native HTTP reference runs first
per repeat. Native remains alive across all four experiments; each prototype invocation
reloads GPU models. No CPU affinity/background-load isolation.

Both heterogeneous variants use direct pinned receive and the same16-lane decoder.
Their time boundary is internal request dispatch through first GPU token. Native uses
HTTP through first nonempty SSE text. The heterogeneous boundary omits outer HTTP/text
processing, so it cannot establish HTTP/native parity. No offered-load/capacity claim.

## Results

Median first-token milliseconds; three measured samples per cell:

| Run/input | Native HTTP | Full KV | GPU prefix50% | GPU prefix75% | GPU prefix87.5% |
|---|---:|---:|---:|---:|---:|
|first128|20.76|48.52|44.28|42.11|—|
|first512|50.93|91.33|99.50|81.70|—|
|first1024|89.40|143.24|172.49|120.92|—|
|repeat128|21.72|87.27|—|48.47|50.85|
|repeat512|51.69|131.69|—|100.88|75.76|
|repeat1024|106.98|353.16|—|182.08|134.76|

75% prefix reduces1024 transfer from150,847,488 to37,601,280 bytes;87.5% transfers
18,726,912 bytes. Peak allocated GPU memory across these runs is about30.01GiB;
the prototype loads both BF16 prefill weights and Marlin decode weights even for its
full-transfer baseline. This is not a baseline-only memory comparison.

Full-transfer latency varies substantially between runs. At1024 native KV-ready
median moves123→327ms while native-ready moves89→105ms. Preserve this drift and all
raw samples; do not attribute it to a specific allocator/NUMA/interference cause or
compress the result into a stable speedup factor.50% is slower at512/1024 and is not
selected.75% and87.5% are promising candidates but need concurrent-serving validation.

## Quality and limits

21 cases:12 arithmetic/format/language prompts plus9 long exact-retrieval prompts.
Native, full-KV GPU,75% and87.5% all pass18/21 strict checks. All fail the same three
strict checks (logic,korean,subtract); do not report21/21 accuracy. Long retrieval9/9
passes in all modes.75% and87.5% final text each matches native in21/21 cases.
GPU runs generate a fixed64-token budget and truncate at first EOS for grading;
native stops at EOS. Quality-run times are not performance comparisons. For full vs75%,
only16/21 entire64-token sequences match, so there is no bit-identical generation claim.
This is a smoke check, not a broad perplexity/general-quality evaluation.

All owned processes stopped. Separate GPU1 test processes were observed after the
performance repeat had finished; they cannot explain that earlier timing drift.
No processes belonging to other work were terminated.

## Next

Integrate an opt-in bounded serving path with GPU-prefill chunking and joint ownership
of prefix/suffix work. Measure active-decode stalls, overload/cancellation, first-token
latency and throughput against rolling full-transfer and native HTTP. Consider a load-
aware prefix fraction because GPU recomputation consumes decode capacity. Native
performance is not achieved by this single-request prototype; keep defaults unchanged.
