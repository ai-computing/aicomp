# Native ATOM reference and compact RNGD prefix

- Date:2026-10-19; task:PDDISAG-6
- Source:aicomp-llmd f01f121 plus recorded edits
- Devices:ATOM physical0 and RNGD physical3/logical2; no group changes
- Model:Qwen2.5-0.5B revision7ae557604adf67be50417f59c2c2f167def9a775

## Native reference

Extended validate_atom_checkpoint.py to explicitly select physical ATOM and
accept token-ID matrices. Retained existing artifact identity and legacy device
checks. Existing vendor-compiled B1/context512/chunk64 artifact reused.
Same prompt token IDs32/64/41/37, output8/32. Each case1warmup+2measured repeats.
Raw command/source identity:atom05-r1/invocation.json; artifact and SDK/config
manifests adjacent. Total16measured requests/320tokens in1.829362s;
FP32 teacher-forced160/160 for unique trajectories. This is Optimum RBLN
prefill/decoder APIs, not a full vLLM production batching/HTTP benchmark.
Native uses1ATOM versus PD1RNGD+1ATOM; compiler precision/graph implementations
and verification overhead differ. Do not call this an equal-resource speedup.

Furiosa compatible FXB check returned0cached matches. Official0.5B main
repository discovery explicitly failed: no .fxb file. Refs were inspected;
main is also v2026.4.0b9. Preserve discovery error, don't rename legacy artifacts
or bypass compiler compatibility checks. No native RNGD performance measured.

## Compact prefix experiment

A/B/B/A;16requests320tokens each. Both variants use ATOM input reuse and full
logit reference checks. A=512-token zero-padded source and old per-chunk range
materialization; B=actual64-token source, compact packer and one vector range
materialization. Same16-token encoder graph, precision, metadata, fences and
physical DMA plans. Ranges must still pass before any DMA. Reference admission
always obtains full512-token KV from the same engine via its default path.

Launch with PYTHONPATH=aicomp-llmd/opt_prime:hetero-npu-p2p/src:rngd_llmd/src
/home/gspark/venvs/furiosa-2026.4/bin/python
 aicomp-llmd/opt_prime/examples/pd/validate_qwen_pd.py --config
 experiments/native_baseline_20261019/{a1,b1,b2,a2}.json.
Actual command/source/version snapshots in each provenance.json.
Unit suite34passed, including bit-exact compact/full packing and nonfinite and
capacity rejection. Report all cases and first-use costs; no saturation claim.

## Harness failures

An initial launch used workspace-relative paths from the worktree cwd and
failed before edits/model startup. Corrected working directories. A Furiosa
CLI fxb help invocation was rejected by the installed entrypoint; capability
checks used the installed native fxb module instead. No device mutations occurred.

## Final results

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
