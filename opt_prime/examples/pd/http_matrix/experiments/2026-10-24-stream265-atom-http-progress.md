# ATOM HTTP streaming: three models verified, KONI incomplete

- **Date**: 2026-10-24
- **Task**: RNGD layer streaming (aicomp #265); HTTP reliability (#267)
- **Host / device**: RNGD+ATOM benchmark host; one RNGD and one allocated ATOM
- **Commit**: production 33d6cb08ec806f693dbecfd7d630331a3bd1bb2f; HTTP-only listener candidate uses backlog 128
- **Environment**: Furiosa 2026.4 source environment; ATOM harness Torch 2.11.0+cpu / Transformers 5.8.1

## Setup and command

Source width 8192, BF16, context 9216, decoder B1, concurrency 16,
caches disabled. Inputs 128/512/2048/8192, outputs 32/128. Eight warmups,
128 retained-token correctness responses and 128 timed SSE text responses per
successful mode. One timed batch per condition. Fixed output length ignores EOS.
Use the existing source factories and ATOM configuration from STREAMING.md;
model-specific configs and raw artifacts are retained by the source hashes in
`evidence265/http267-atom-source-hashes.json`.

Recorded research-driver invocation, with site values supplied externally:

```bash
python3 analysis/serving-matrix-201/run_http267_atom.py \
  --host "$ACCELERATOR_SSH_TARGET" --pod "http267-$MODEL_LABEL" \
  --model "$MODEL_LABEL"
```

## Verified results

Fresh independent audit checks whole and stream for Qwen3-0.6B, Qwen3-4B and
Gemma3-1B: 48 conditions, 48 warmups, 768 full-token checks and 768 timed SSE text
checks. Whole/stream prompts and reference arrays match exactly in all eight
conditions per model. Qwen3-4B and Gemma native modes also completed and were
independently audited. Summaries are in evidence265.

At input 8192/output 128, throughput in tokens/s:

| Model | Native ATOM | Whole KV | Layer streaming |
|---|---:|---:|---:|
| Qwen3-0.6B | Not compared in this table | 33.0665 | 43.3496 |
| Qwen3-4B | 12.2876 | 12.6716 | 14.5315 |
| Gemma3-1B | 57.1134 | 39.4864 | 43.5681 |

Qwen3-4B streaming improves whole-transfer throughput by 10.8–27.4% across the
measured conditions; Gemma improves by 2.6–15.6%. Gemma remains slower than
native ATOM in all eight conditions. Native uses one ATOM; PD adds one RNGD,
so these are not equal-resource or cost comparisons. Qwen3-4B native/PD has
pre-EOS differences in two conditions. Same-source transfer parity does not
establish native equivalence or semantic answer quality.

## Retained failure and remaining scope

KONI native ATOM failed the original 180-second socket response wait at
8192/128/c16. Seven earlier correctness conditions (112 responses) and eight
warmups completed; no timed batch completed. Server remained alive at the
failure snapshot, with zero OOM events and zero listener drops/overflows.
This does not establish a queueing-only cause. Whole/stream were not launched.
A distinct 600-second native diagnostic completed; it does not erase or fix the
original deadline failure. The failed-run summary is retained in evidence265.

The listener candidate eliminated observed overflow/drop counters in a paired
Qwen3-0.6B run, but the original connection-reset cause is not conclusively
proven. It is diagnostic-only and is not integrated into this PR's runtime.

Separate width-1024 experiments (#268) are also not part of this PR: Qwen3-0.6B
has a heldout answer regression and KONI has width-dependent token divergence.
Production source width/default behavior is unchanged here. Repeated timing,
ATOM compiled batch >1, and direct device-DMA overlap remain outstanding.


## KONI native 600-second diagnostic completed

The separate native child and artifact copies exited0. Independent audit passes
all eight conditions: eight warmups,128 full-token checks and128 timed SSE text
checks (264 responses). Prompts and all eight reference token arrays match the
original failed180-second run. End-of-run listener drops/overflows and OOM event
deltas are zero; the server was alive before normal cleanup.

At8K input/128 output, native ATOM throughput is7.7847tok/s, median first SSE
130.6080s and median completion139.7550s. Maximum first SSE is253.9345s and maximum
completion263.0788s; five timed requests wait over180s for first SSE and six finish
after180s. This is a single ATOM execution slot with16 queued requests. It shows
long latency also exists without PD; the evidence does not separate queue wait
from prefill time or prove the original nonstreamed timeout's precise cause.

The600-second value controls socket wait, not a guaranteed total request deadline.
Nonstreamed correctness checks retain token arrays but not individual durations.
One timed batch per condition is diagnostic, not statistical performance evidence.
The original180-second failure remains failed. RNGD-to-ATOM whole subsequently completed as recorded below; streaming also completed as recorded below.

Reproduce the diagnostic with run_http267_deadline_atom.py, --model koni,
--pod http267d-koni and the existing site-specific --host. It uses the same model
and hardware configuration, and benchmark267_deadline.py --request-timeout 600.
See evidence265/http267d-koni-native-complete-summary.json for all eight timing
rows and the raw-result SHA256. Runtime remains33d6cb0; no serving code changed.


## KONI whole KV diagnostic completed

Fresh independent audit confirms native and whole each pass eight conditions,
eight warmups, 128 retained-token checks and 128 timed SSE text checks.
Both children and artifact copies exited zero; raw hashes match the summary.
Runtime remains 33d6cb0, socket wait 600 seconds, B1/c16, caches off.

| Input/output tokens | Native tok/s | Whole KV tok/s |
|---|---:|---:|
| 128/32 | 14.6499 | 5.9430 |
| 128/128 | 14.7951 | 10.8468 |
| 512/32 | 12.8533 | 5.8954 |
| 512/128 | 14.2666 | 10.7350 |
| 2048/32 | 8.5544 | 5.5431 |
| 2048/128 | 12.4293 | 10.3935 |
| 8192/32 | 3.3584 | 4.6693 |
| 8192/128 | 7.7847 | 9.3113 |

At 8K/128, whole improves observed throughput by 19.6%; median completion falls
139.755 to 117.284 seconds, maximum 263.079 to 219.946 seconds. Shorter inputs
remain slower with PD. This is one measured batch per condition; native uses one
ATOM while PD adds one RNGD. Neither statistical nor equal-resource benefit is
established. The original 180-second failure is retained and remains unresolved.

Native/whole references match six of eight conditions. The first mismatch is at
zero-based token 40 for 128/128 and token 37 for 8192/128. EOS-aware comparison
also matches only six: no EOS occurs in these fixed-length references. Passing
same-route response checks does not establish native or semantic equivalence.
See evidence265/http267d-koni-native-whole-summary.json and the companion EOS
summary for hashes, individual timings and comparison details.

Before the next phase, the owned Pod had 2265 seconds remaining versus 2842 seconds
used by the completed whole phase. After child/copy completion and absence of
benchmark/engine processes were checked, only that Pod was recreated. No active
measurement was restarted. Stream now uses a different physical ATOM card of
the same class, a limitation for subsequent timing comparisons. That streaming phase subsequently completed as recorded below.


## Completed KONI streaming and four-model whole/stream coverage

All three KONI children and both artifact copies per child exited zero; outer
queue exited zero. Independent raw audit validates each route's eight conditions,
eight warmups, 128 full-token checks and 128 timed SSE text checks. Whole/stream
reference arrays match all eight conditions. Native/whole remains six of eight;
the two pre-EOS differences remain. Original 180-second failure remains retained.
Completed Pod absence was verified after parent cleanup.

| Input/output tokens | Native tok/s | Whole KV tok/s | Streaming tok/s |
|---|---:|---:|---:|
| 128/32 | 14.6499 | 5.9430 | 6.7293 |
| 128/128 | 14.7951 | 10.8468 | 11.3686 |
| 512/32 | 12.8533 | 5.8954 | 6.8371 |
| 512/128 | 14.2666 | 10.7350 | 11.3826 |
| 2048/32 | 8.5544 | 5.5431 | 6.7273 |
| 2048/128 | 12.4293 | 10.3935 | 11.4099 |
| 8192/32 | 3.3584 | 4.6693 | 6.6058 |
| 8192/128 | 7.7847 | 9.3113 | 10.8298 |

At 8K/128, streaming has 16.3% greater observed throughput than whole transfer,
39.1% greater than native. Median completion is 100.698 seconds versus 117.284
whole and 139.755 native; median first SSE is 91.565 versus 108.164 and 130.608.
Streaming maximum completion is 189.105 seconds, so a 180-second total response
SLO is still not met. Shorter inputs remain slower than native. One timed batch
per condition and different physical ATOM cards limit timing conclusions; PD adds
one RNGD versus native. Stream OOM events and listener drop/overflow deltas are
zero; server alive at final snapshot before normal teardown. No DMA-overlap claim.

Together with the preceding Qwen3-0.6B, Qwen3-4B and Gemma3-1B results, all four
models now have completed B1/c16 ATOM whole/stream HTTP controls, eight exact
reference pairs per model. This does not establish native equivalence, semantic
answer quality, batch-four behavior or equal-resource performance superiority.

Raw three-route JSON and summary with hashes, terminal and copy evidence are in
evidence265/http267d-koni-atom-c16-{native,whole,stream}.json and
http267d-koni-three-route-summary.json. Reproduce with the diagnostic runner above;
stream uses the corresponding whole result as --reference. Runtime stays33d6cb0.
