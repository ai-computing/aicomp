# Native reference, KV resume probe and compact-prefix experiment

Date:2026-10-19. Qwen2.5-0.5B pinned upstream revision
`7ae557604adf67be50417f59c2c2f167def9a775`, B1/context512.
Evidence: `validation/native-baseline/`. No driver or SDK upgrades.

## Native ATOM reference

The vendor Optimum RBLN model executes prefill and decode with an existing
BF16-declared compiled artifact (chunk64). This is the vendor model API, not a
full vLLM scheduling/HTTP benchmark. Explicit physical binding selects ATOM0.
The runner now accepts a shared token-ID matrix, avoiding different prompts
between native and heterogeneous tests. Input32/64/41/37, output8/32; each
case one warmup and two measured repeats. Artifact identity and device-FD
checks remain required.

Sixteen measured requests/320tokens take1.829362s in aggregate. Median request
time is about46ms for8tokens and182–184ms for32tokens; native decode interval
is about5.6ms/token. FP32 teacher-forced top1 is160/160 on unique trajectories.
For comparison, our existing optimized PD path's warmed repeat1 totals3.322838s
for the same16requests/320tokens across two runs. This is a diagnostic gap,
not an equal-resource score: native uses1ATOM; PD uses1RNGD+1ATOM and includes
full-logit transfer checks. Both artifacts declare BF16 but compiled graphs and
accumulation details differ. In this B1 small-model workload, native ATOM is
faster; no claim about concurrent or large-model workloads follows.

## Native KV API resume probe

`validate_native_kv_resume.py` loads that same native model and checks exact
cache names, dimensions and per-layer/total bytes. It uses documented runtime
get/set KV calls with owned aligned host buffers; there is no address guessing,
export, DMA, connector patch or group change.

For four prompts:

1. Run native prefix prefill, save48native cache tensors, and decode32tokens.
2. Restore the saved cache and decode again: all full logits match exactly.
3. Convert saved RNGD prefix KV into the validated Rebel0.11 representation,
   admit via per-layer set calls, and resume the native decoder.

All four cross-vendor host-admitted generations are finite. Three match native
standalone trajectories32/32; the English37-token prompt matches only2/32
because the autoregressive trajectories diverge. The separate FP32
teacher-forced check is126/128 (that prompt30/32), not128/128. Preserve both
metrics; trajectory agreement is not the same as teacher-forced agreement.

This establishes a concrete native-engine integration surface. It is **not** a
production native PD adapter: native prefill is still used to initialize wrapper
page-table/mask state before cache replacement. Removing this redundant work
requires an explicit destination reservation/reset interface. Native KV layout
is version-pinned; host staging is explicit. No direct-device performance gain
or quality equivalence is claimed.

## Compact RNGD prefix ablation

Optional `rngd_compact_cache=true` avoids growing64-token source tensors to512
and then slicing them back down. It uses the same16-token DLFloat encoder,
collects range results into one host materialization, and checks every result
before DMA. Default remains false. Source K+V tensor storage in this geometry
is0.75MiB instead of6MiB (tensor-size calculation, not measured peak memory).

A/B/B/A, each run16requests/320tokens. Both variants use ATOM input reuse.
All64requests/1280tokens pass full-logit checks, all3072DMA commands complete,
workers exit0 and cross-run tokens match.

| Metric | Original full-context source | Compact source |
|---|---:|---:|
| All32requests/640tokens per variant | 8.004686s | 8.087266s |
| Separately reported warmed repeat1,16requests/320tokens | 3.322838s | 3.297864s |
| Warmed source-prefill stage sum | 0.847398s | 0.811922s |

Including first use, compact is1.03% slower; warmed total is only0.75% lower.
This does not establish a worthwhile throughput gain, so it is not promoted to
the default path. All raw first-use timings remain. Recalculate using
`examples/pd/summarize_compact.py --root <experiment-root>`.
The prior ATOM input-reuse improvement remains separate and is not rolled into
this experiment's claim. Offline suite34passed, including byte-identical packing
at head dimensions64/128 and rejection of nonfinite/out-of-capacity input.

## Furiosa native baseline limitation

Installed `fxb.check` found no compatible cached Qwen0.5B bundle. Native
`fxb.download` rejected the official `furiosa-ai/Qwen2.5-0.5B-Instruct` main
revision because it contains no .fxb file. Repo refs show main equalsv2026.4.0b9;
the previously inspected listing contains legacy artifact.json/binary_bundle.zip.
We did not relabel those files or bypass compatibility checks. A compatible FXB
or supported conversion/build path is needed; RNGD native throughput is unmeasured.

## Next integration priorities

- Expose native decoder destination reservation/reset so prefill is unnecessary.
- Validate logical-to-physical KV conversion and numerical differences per layer.
- Reuse vendor compute/batching via that contract; add direct transfer only after
  the native destination export/lifetime contract is established.
- Compare concurrent requests at equal card budgets, including native replicas.
