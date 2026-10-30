# ATOM short/long HTTP comparison with final HTTP module

- Date: 2026-10-30; related #201, #265 and PR266.
- HTTP source SHA256: `e3449979b88d03edb6d5bf0d402a708e5d98054fdf203e22b480f3d7e6fd759a`.
- Runtime: historical stream265 B1 engine and research RNGD producers with final PR266 HTTP override. This is not whole-runtime current-main or vendor native furiosa-llm qualification.
- One ATOM decode lane; PD adds one RNGD. c4 means queued clients, not hardware batch four.
- Decoder: torch2.9.1+cpu, Transformers4.57.6, optimum-rbln0.10.2, rebel-compiler0.11.0. Producer: torch2.10.0+cpu, Transformers5.19.0, furiosa-torch2026.4.0.

## Coverage and output checks

Four models, three routes (native/whole/stream), input128/8192, output32/128, concurrency1/4: 12 arms,96 conditions,528 responses (48warmups,240full-token checks,240timed SSE-text checks). All within-route checks pass. Independent raw auditing verifies exact warmup/condition identities, actual tokens/text, resource counts and timing arithmetic. All16 whole/stream reference pairs match. Eleven selfchecks include corrupt responses, duplicate warmups/arms, absent artifacts, wrong resources and invalid throughput. All children and coordinator exit0; UID-scoped Pod deletion succeeds and a fresh query confirms absence.

Native/PD equivalence is separate. Qwen0.6B and Qwen4B match4/4 reference arrays each. Gemma matches2/4 full arrays: its two128-input references first differ at index5 after shared EOS index2, with EOS-inclusive prefixes identical. Both8K cases match. KONI's32-output references match, but128-output references differ before EOS at index40(input128) and37(input8192); no configured EOS appears within those128 tokens. These differences remain failures of full native equivalence, not transport-mode regressions or semantic quality passes.

## Timing samples

At input8192/output128/c4 (512 generated tokens in each timed batch):

| Model | Native tok/s | Whole tok/s | Stream tok/s | Native/PD references equal |
|---|---:|---:|---:|---:|
| qwen06b | 50.65 | 33.11 | 44.24 | 4/4 |
| qwen4b | 12.30 | 12.72 | 14.47 | 4/4 |
| gemma | 57.68 | 40.19 | 43.87 | 2/4 |
| koni | 7.78 | 9.32 | 10.44 | 2/4 |

One timed batch per cell, ordered native then whole then stream. Stream is faster than whole in these cells, with identical whole/stream references. Native uses fewer accelerators; KONI additionally has different native/PD outputs, so its native comparison is not an equivalent-output speedup. These timings do not establish stable gains or device DMA overlap.

Single-client8K/128 measurements (native / whole / stream):

| Model | First SSE (s) | Estimated TPOT (ms) |
|---|---:|---:|
| qwen06b | 1.089 / 2.420 / 1.554 | 11.26 / 11.31 / 11.32 |
| qwen4b | 4.970 / 4.601 / 3.219 | 42.76 / 42.74 / 42.85 |
| gemma | 0.806 / 1.778 / 1.516 | 10.94 / 10.56 / 10.78 |
| koni | 7.302 / 4.600 / 2.702 | 72.04 / 71.87 / 71.72 |

Most observed route latency differences occur before first SSE, while estimated TPOT is similar. For Qwen4B,8K first SSE drops4.970→3.219s from native to stream; the earlier2K comparison instead favored native1.022s over stream2.534s. This suggests input length matters for route selection, but repeat measurements and resource cost are required before setting a threshold. SSE may coalesce tokens; these intervals do not isolate producer compute, D2H, packing, transport, import or queuing. The earlier2K and this8K runs used different assigned physical ATOMs; cross-run deltas are not solely attributable to prompt length.

## Reproduction and evidence

Use the archived benchmark267_deadline.py and atom_server265_http.py configurations, context9216, producer width8192, caches off, one decode lane and maximum output128. Qwen and corrected Gemma use3 KV blocks; KONI uses1. Gemma uses the research global225 hybrid artifact, not a generally supported export. Metadata hashes and compiled filenames/sizes are in the existing `../evidence265/atom-mid201-artifact-metadata.json`; they are not full compiled-binary hashes. Exact runtime source hashes, per-arm raw hashes, all condition timings and EOS differences are in `../evidence265/atom-edge201-final-summary.json`.

```bash
"$PYTHON" "$HARNESS" --model "$MODEL" --mode "$MODE" --server-config "$CONFIG" --lengths 128 8192 --outputs 32 128 --concurrency 1 4 --slots 1 --repeats 1 --startup-timeout 1800 --request-timeout 600 --output "$NEW_RESULT"
# MODE=native for ATOM alone, pd for whole/stream.
# Stream config adds --stream-layers; add --reference "$WHOLE_RESULT" to the harness.
python ../evidence265/audit_atom_edge201.py "$ARCHIVED_ARTIFACT_DIRECTORY"
python ../evidence265/selfcheck_atom_edge201.py "$ARCHIVED_ARTIFACT_DIRECTORY"
```

Archive inputs: edge201-atom1030-queue.json; edge201-atom-final-{model}-{native,whole,stream}.json; mid201-atom-final-artifact-metadata.json; mid201-atom-final-koni-eos.json. Historical stream265 c16 raw controls are optional and remain unavailable rather than passing when absent. Model aliases: qwen06b/qwen4b/gemma/koni. Site-local commands, device bindings and full logs remain in the private raw archive.

## Remaining scope

Cache-on, genuinely supported model export paths, semantic quality, repeated performance and current-main integration remain separate. The earlier native numerical/quality counterexamples are not resolved by this fixed-prompt sweep. #201 remains open. After verified cleanup, a separate Qwen ATOM cache experiment was allocated; its results are not included here.
