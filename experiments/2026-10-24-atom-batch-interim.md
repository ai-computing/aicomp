# Interim ATOM batch serving evidence (#270)

- Date: 2026-10-24
- Status: incomplete experiment; no default change proposed
- Related: ai-computing/aicomp #201, #225, #265, #270, #271; RBLN-SW/optimum-rbln #827

## Method

Four models: Qwen3-0.6B, Qwen3-4B, KONI-8B, Gemma3-1B. ATOM compiled B1/B4 with five KV blocks, context9216; KONI uses TP2, others TP1. Native ATOM, RNGD prefill with whole KV transfer, and RNGD prefill with layer-stream transfer. Input128/512/2048/8192, output32/128, concurrency16, caches off, one measured batch per condition. Each successful mode has8 warmups,128 full-token checks and128 timed SSE text checks. Loading and correctness checks are excluded from serving timing. EOS is deliberately ignored for the fixed-output workload; full-token and through-EOS comparisons are different criteria.

SDK: rebel-compiler0.11.0, optimum-rbln0.11.0.post1, torch2.11.0+cpu, Transformers5.8.1; RNGD worker uses torch2.10.0+cpu, Transformers5.19.0, furiosa-torch2026.4.0; both interpreters are Python3.10.12. These package versions were queried separately during the corrected Gemma run, rather than assumed to be shared. The corrected-Gemma source manifest records104 on-disk Python/script files, with no missing requested files. This is not proof of every imported dependency or of historical in-memory code. Historical per-run dependency identity, harness packaging and driver inventory remain publication work. Experimental serving modules are on the PR266 dependency stack, not on current main. This document does not make this checkout a runnable serving implementation.

## Original inventory (before the corrected Gemma follow-up)

| Model | Compiled batch | Route | Audit | 8K/128 tok/s |
|---|---:|---|---|---:|
| qwen06b | 1 | native | within_route_pass | 50.3245 |
| qwen06b | 1 | whole | within_route_pass | 33.1562 |
| qwen06b | 1 | stream | within_route_pass | 43.1570 |
| qwen06b | 4 | native | within_route_pass | 63.6370 |
| qwen06b | 4 | whole | within_route_pass | 58.4628 |
| qwen06b | 4 | stream | within_route_pass | 45.4781 |
| qwen4b | 1 | native | within_route_pass | 12.2832 |
| qwen4b | 1 | whole | within_route_pass | 12.6169 |
| qwen4b | 1 | stream | within_route_pass | 14.5038 |
| qwen4b | 4 | native | audit_failed | 18.2422 |
| qwen4b | 4 | whole | not_collected | — |
| qwen4b | 4 | stream | not_collected | — |
| koni | 1 | native | within_route_pass | 15.0183 |
| koni | 1 | whole | within_route_pass | 13.8944 |
| koni | 1 | stream | within_route_pass | 17.2012 |
| koni | 4 | native | within_route_pass | 23.8487 |
| koni | 4 | whole | within_route_pass | 31.8286 |
| koni | 4 | stream | within_route_pass | 38.5542 |
| gemma | 1 | native | within_route_pass; INVALID causal baseline | 57.9914 |
| gemma | 1 | whole | within_route_pass | 40.3069 |
| gemma | 1 | stream | within_route_pass | 43.6196 |
| gemma | 4 | native | not_collected | — |
| gemma | 4 | whole | not_collected | — |
| gemma | 4 | stream | not_collected | — |

The inventory JSON records source result hashes and results of rerunning the HTTP, geometry and lane audit against the local raw files. Within-route success does not establish native/PD parity, answer quality, device utilization, or statistical improvement. Failed native Qwen3-4B B4 blocks its whole/stream sequence; their absence is not a pass. Corrected Gemma B4 native has now passed 264 responses; B4 whole and stream are pending. Its separate corrected-artifact128-token causal test passed all4 lanes×52 slots, including prefix-changing positive controls for every slot. This does not substitute for the B4 HTTP matrix.

## Findings that constrain promotion

- Qwen3-4B B4: repeated same-prompt lane outputs diverge. Direct SDK replay reproduces lane-dependent BF16 argmax ties without custom AtomBatcher or KV transport. Do not relax the parity gate or attribute this to PD overhead.
- Qwen3-0.6B B4: stream is slower than whole for8K/128 (45.4781 vs58.4628tok/s); repeat and stage attribution are needed before any default change.
- Gemma original k5 native: suffix-only prompt changes alter prefix KV in50/52 slots. Old corrected k3 control preserves52/52; a fresh k5 export with the existing sliding/global causal correction also preserves52/52. Original and corrected k5 tests used different assigned ATOMs. Raw per-slot files and patch hashes are included.
- Corrected Gemma B1 native subsequently passed264 HTTP responses, matches old corrected k3 native full8/8, and previous whole PD full6/8 / through-EOS8/8. Corrected whole subsequently passed264 responses: native/whole full6/8 and through-EOS8/8 on the same ATOM and corrected artifact. Whole references remain full8/8 identical to prior whole. At8K/128, native57.5700 vs whole39.9662tok/s; median response completion18.8013 vs27.4025s. Whole startup837.6305s is excluded. Corrected stream also passed264 responses; whole/stream full8/8 and native/stream through-EOS8/8 (full6/8). Stream8K/128 throughput43.2203tok/s, median completion25.1978s: +8.1422% throughput over whole, but below native. One timed batch; corrected B1 controller and cleanup exited0. These completed follow-up comparisons are separate from the original inventory so the invalid baseline is not silently replaced.

## Remaining work

Complete the four-model B4 fault-isolation matrix and operational generalization; diagnose/fix Qwen3-4B B4 numerical differences; repeat Qwen3-0.6B B4 performance with producer/import/decode timing; perform batched cancellation/reuse checks; package exact commands/harness source and SDK/driver identities; obtain engine-owned compile/cache handling under#271. No completion or universal performance claim is made.

Generated with Codex; evidence reviewed against local result files.

## Native B1/B4 output comparability

Direct source-result comparison: Qwen3-0.6B full6/8, Qwen3-4B4/8, KONI1/8, corrected Gemma4/8. At8K/128 only Qwen3-0.6B and corrected Gemma full outputs match across batch sizes. Qwen3-4B differs first at output index12 for both8K lengths, independently of its within-B4 repeated-output failure at other conditions. KONI8K/128 also differs. Do not present all four raw throughput changes as equivalent-output speedups. See atom270-four-model-native-batch-comparison.json.

## Corrected Gemma B4 native follow-up

B4 native passed 264 responses. Successful decode-call histogram is `[0, 664, 32, 32, 5008]`; this establishes actual four-lane calls, not device utilization. B1/B4 references match in 4/8 full sequences and 8/8 through EOS. At 8K input / 128 output the full tokens match: throughput is 57.5700 → 95.1626 tok/s, median completion 18.8013 → 14.7210 s. Estimated TPOT worsens from 11.2129 to 36.0703 ms. B1 used ATOM0 and B4 ATOM2, with one measured batch each, so this is not an isolated same-card batch-size experiment.

## Causal regression reproduction

The included diagnostic scripts require the experimental AtomRblnEngine/AtomBatcher from the PR #266 dependency stack, the installed SDK versions above, one assigned ATOM with a valid existing RSD mapping, and the cached model weights. The fixture must be a torch file containing `ids`: a 128-token Gemma prompt. The original fixture and full HTTP harness packaging remain publication work. Commands below run from the evidence directory in that environment; no shared SDK files are edited.

```bash
# Existing uncorrected B1 artifact: expected causal failure; retain its JSON.
python container_bootstrap270.py check_gemma270_causality.py --cache-dir atom270-artifacts --kv-blocks 5 --exports prompt128.pt --output original-k5.json
# Fresh corrected B1 export, then reload in a separate process.
python container_bootstrap270.py build_gemma270_causal.py
python container_bootstrap270.py check_gemma270_causality.py --cache-dir atom270-artifacts-causal --kv-blocks 5 --exports prompt128.pt --output corrected-k5.json
# B4 export and all four lanes, including prefix-changing positive controls.
python container_bootstrap270.py build_gemma270_causal_b4.py
python container_bootstrap270.py check_gemma270_causality_b4.py --cache-dir atom270-artifacts-causal --kv-blocks 5 --exports prompt128.pt --output corrected-b4-k5.json
```

Export scripts reject existing artifact directories; diagnostics reject output overwrites and never compile missing models. These are research reproductions, not the engine-owned compatibility implementation requested in #271.


## Corrected Gemma B4 whole follow-up: correctness passes, occupancy fails

All 264 responses passed within-route HTTP checks, but the full B4 audit failed: successful decode-call lane histogram `[0, 14313, 3132, 93, 0]` contains no four-lane calls. Native and whole ran on the same physical ATOM2 and corrected artifact. Their full reference sequences match in 6/8 conditions and through EOS in 8/8. At 8K/128, full tokens match; whole reaches 77.0898 tok/s, median first SSE 13.3368 s and completion 15.7735 s, versus native 95.1626 tok/s and completion 14.7210 s. This is one timed batch per condition, with an extra RNGD for whole; no statistical or equal-resource speedup claim is supported.

The strict queue stopped before stream and cleaned up its Pod. The occupancy failure remains a failure; HTTP correctness does not validate B4 utilization. The separate host-timeline pair below supports producer-limited readiness. The same-Pod width ABBA below then tests a short-input correction.

For hybrid Gemma B4, `_import_progress` deliberately defers device import until the producer completes: idle compiled lanes can otherwise overwrite local sliding-window KV. Host layer streaming therefore does not establish early ATOM import overlap. Removing this guard requires an ownership solution and cancellation/reuse validation, not just a performance change. See the two new evidence JSON files.


## Gemma B4 host timeline pair

The diagnostic pair completed on the same physical ATOM2. Both routes passed 264 HTTP responses; all eight full reference sequences match. Neither produced a successful four-active-lane decode call: whole histogram `[0,13952,3299,102,0]`, stream `[0,13676,3452,92,0]`. These occupancy failures remain failures even though HTTP correctness passed.

| Host observation | Whole | Stream |
|---|---:|---:|
| Backlogged KV completion gap, median | 1.5548 s | 1.5085 s |
| Decode call, median | 13.1467 ms | 12.8802 ms |
| Ready prefill left at decode entry | 0 / 17353 | 0 / 17220 |
| 8K/128 output throughput | 76.2890 tok/s | 77.8249 tok/s |
| 8K/128 median completion | 15.9551 s | 15.6698 s |

The observations support producer-limited readiness, rather than a universal scheduler cap below B4. Completion gaps include host delivery and Python timing; they are not pure NPU compute. Queue states are snapshots, and timing sums overlap. The roughly 2% throughput difference is one timed batch per route, not a stable speedup. All-global synthetic CPU controls also admit four lanes when all four results are ready, but do not emulate Gemma sliding-window hardware.

For whole mode, backlogged completion-gap medians remain about 1.55 s at every input length from 128 to 8192. The short-input-only width1024 follow-up and same-Pod ABBA below completed against pinned width8192 outputs. It does not replace the wide path for long inputs or establish heldout quality. Earlier Gemma strict-format quality tests failed in both widths; zero regressions there is not quality acceptance.

Whole initialization took 840.58 s and subsequent stream initialization 18.40 s. This warm-state difference is not isolated. Do not assume every startup takes 14 minutes or attribute the difference to streaming alone.

Raw HTTP results and host-timeline logs are registered as research-hub attachments, identified by `raw-evidence-manifest.json` (research event `578a57666040434b90f5b76a51871973`). Remote synchronization has not been independently checked. Restore those exact files into evidence270 and verify their hashes before recomputing host summaries (use a new output filename; overwrites are rejected):

```bash
python analyze_timeline270.py atom270-timeline-gemma-b4-whole.server0.log --output whole-recomputed.json
python analyze_timeline270.py atom270-timeline-gemma-b4-stream.server0.log --output stream-recomputed.json
python selfcheck_timeline270.py
python selfcheck_analyze_timeline270.py
```

The probe entrypoint depends on the original HTTP harness and PR266 runtime stack; it is not a standalone server in this branch. Raw-result HTTP audits use the included audit_http265.audit_result with backend atom, expected_slots=4, expected_atom_devices=1. Neither raw timing evidence nor the synthetic tests prove device DMA overlap.


## Same-Pod Gemma B4 width ABBA: measured short-input improvement

Four whole-transfer runs used width8192 → 1024 → 1024 → 8192 in one Pod on ATOM3. All 528 responses passed (16 warmups, 256 full-token checks, 256 timed SSE responses). Independent audit rechecked the physical binding, compiled geometry, identical prompts/reference outputs, timing arithmetic and server settings differing only in producer width. Each condition has two timed batches per width.

| Input / output tokens | Width8192 tok/s | Width1024 tok/s | Ratio | Median completion, wide → narrow |
|---|---:|---:|---:|---:|
| 128 / 32 | 20.199 | 40.348 | 1.9975× | 13.661 → 6.958 s |
| 128 / 128 | 76.874 | 145.839 | 1.8971× | 15.004 → 8.398 s |
| 512 / 32 | 19.943 | 40.157 | 2.0136× | 13.824 → 6.960 s |
| 512 / 128 | 76.522 | 144.942 | 1.8941× | 15.119 → 8.496 s |

Values are medians of the two measured batches per width. Backlogged KV completion medians in run order were 1.5651, 0.7661, 0.7695 and 1.5618 s. Successful decode calls reached at most two active lanes in the wide arms and three in the narrow arms. None reached four: the B4 occupancy criterion remains failed, despite the measured serving gain and exact outputs.

This supports selecting a smaller producer shape for short inputs in this configuration. It does not establish a production default, arbitrary-prompt quality, long-input support, GPU/other-model improvement, confidence intervals or device DMA overlap. Inputs above1024 require a separate wider path; the experiments here used only128 and512. Existing strict Gemma output-format quality failures remain unresolved. Qwen3-4B B4 numerical divergence also remains unresolved and is not waived by this Gemma result.

Per-run host summaries and the terminal queue are included. Raw results/logs remain in the separate research archive above, as required by the contribution guide. After restoring the hash-matched raw files, reproduce the full audit from evidence270 with a fresh output filename:

```bash
python selfcheck_atom270_abba.py
python audit_atom270_abba.py --queue atom270-abba-gemma-outer.json --output abba-recomputed.json
```

The self-check uses small synthetic fixtures and rejects eight corruption types; it runs without archived data and is not another hardware measurement. The CPU readiness control requires the PR266 test/runtime stack and tests synthetic all-global KV, not Gemma sliding-window hardware. The four-model hardware cancellation/reuse/producer-death matrix is running separately; completed cases and their limits follow below.


## B4 fault isolation: all four hardware cases complete

Gemma3-1B, Qwen3-0.6B, Qwen3-4B and KONI-8B each completed one direct-batcher control/cancel-reuse/producer-death sequence. Three survivor requests each produced512tokens exactly matching their same-lane controls after either fault. The reused fourth lane produced16tokens matching its control. Owners drained idle/nonpoisoned; all slots were returned. These are hardware fault-isolation results, not HTTP disconnect, semantic-quality or throughput acceptance.

| Model | Control active-lane histogram (0–4) | Cancel/reuse histogram | Producer-death histogram |
|---|---|---|---|
| Gemma3-1B | [0,220,220,276,15] | [0,221,221,275,15] | [0,212,212,299,0] |
| Qwen3-0.6B | [0,168,168,328,15] | [0,158,158,338,15] | [0,167,167,344,0] |
| Qwen3-4B | [0,120,120,376,15] | [0,110,110,386,15] | [0,108,108,403,0] |
| KONI-8B (TP2) | [0,174,174,322,15] | [0,164,164,332,15] | [0,171,171,340,0] |

The longer512-output workload achieved real four-active-lane calls in control/reuse. This does not erase the short-output occupancy failures above. In all four models, both injected faults occurred with host-pending victim KV and **zero victim device-import spans**. At this stage hardware safety after partial victim KV import was unproven; the follow-up below covers the three dense models. Gemma's multilane hybrid path deliberately defers import; its guard has not been removed.

The terminal queue and raw arrays were independently audited: eight fault cases,12,288 survivor-token comparisons and64 replacement-token comparisons match; all four owned Pods were cleaned up successfully. The separate Qwen3-4B B4 lane-dependent numerical divergence remains unresolved despite the same-lane fault test passing. Its three survivor control lanes happened to agree for this fixture on the RNGD-prefill route; that does not waive independent native-prefill failures on other prompts. Compact audits and source driver are included; raw token arrays are archived separately with full hashes and research event IDs in fault-evidence-manifest.json. Remote archive synchronization has not been independently verified. The driver requires the PR266 runtime and original hardware harness; it is not standalone on main.

CPU coverage additionally verifies **actual partial writes** using the real AtomBatcher with the existing fake SDK: three active survivors, fourth-lane cancellation/late completion or producer error, same-lane reuse, and no stale overwrite of replacement KV. Survivor24-token outputs equal a no-fault control. New cases2passed; with related batching/hybrid/stream checks53passed6.19s on Torch2.13/Transformers5.15. The initial new tests failed because the test assumed an admitted attribute exists before admission; the test now follows the runtime's getattr default. No runtime change was made. This is CPU evidence, not NPU hardware or full-suite acceptance.

Run the published test against the recorded PR266 checkout (eaf7daa219a6cea68db4ec5fc08850edc848b271):

```bash
PYTHONPATH="$PR266/opt_prime:$PR266/opt_prime/tests" "$TEST_PYTHON" -m pytest experiments/evidence270/test_atom270_dense_partial.py "$PR266/opt_prime/tests/test_atom_batching.py" "$PR266/opt_prime/tests/test_atom_hybrid.py" "$PR266/opt_prime/tests/test_atom_stream_progress.py" -q
```


## Qwen3-0.6B long-input whole/stream ABBA (2026-10-25)

Repeated whole/stream/stream/whole in one Pod on ATOM2 with width8192,B4/TP1/k5,concurrency16,cacheoff,inputs2048/8192 and outputs32/128. All528responses passed (16warmups,256full-token checks,256timed SSE responses). Independent audit rechecked full prompts/references, config equality except the stream flag, compiled geometry, actual device binding, raw hashes and timing arithmetic. The queue completed and the owned Pod was cleaned up. Two timed batches per route/condition; no confidence interval or arbitrary-prompt quality claim.

| Input/output | Whole tok/s | Stream tok/s | Stream/whole | Median E2E whole/stream s |
|---|---:|---:|---:|---:|
|2048/32|23.579|30.152|1.2788|12.111/9.423|
|2048/128|93.515|113.257|1.2111|12.645/11.222|
|8192/32|20.441|16.196|0.7923|15.826/19.796|
|8192/128|53.113|53.354|1.0045|24.817/26.332|

Values are medians of two batches. Streaming improves2K throughput21–28percent but reduces8K/32throughput20.8percent and increases completion latency25.1percent. Both stream samples are below both whole samples for8K/32. At8K/128throughput is effectively unchanged(+0.45percent), with6.1percent higher completion latency. The earlier single-run22percent8K/128throughput penalty did not reproduce here. This supports condition-specific investigation, not globally enabling/disabling streaming.

Import host totals inABBAorder:76.672,95.628,98.060,78.305s, with132,3696,3696,132calls respectively. Successful4-active-lane calls:639,1143,1065,619. Import includes torch stacking, dtype conversion, staging assignment and SDK calls; these counters do not isolate network/DMA or establish a causal bottleneck. Extra component instrumentation is a separate follow-up with no result claimed here. Startup14.05–14.30s is excluded from steady-state metrics and does not independently establish compiler-cache hits.

The first attempt's whole arm passed132responses but its controller rejected missing binding logs caused by the older bootstrap configuration; it cleaned up its Pod. Those data were preserved separately and excluded from this ABBA. All four accepted arms were rerun after fixing the bootstrap and adding a preallocation guard. No production engine/default changed.

Compact audit: atom270-qwen06-regression-retry1-audit.json. Eight raw results/logs remain outside Git; qwen06-regression-evidence-manifest.json records full hashes and research event ID. Remote archive synchronization has not been independently verified. For each restored raw result, use the included audit_http265.audit_result with backend atom, mode whole/stream, lengths=(2048,8192), expected_slots=4 and expected_atom_devices=1; compare complete reference dictionaries across arms and recompute median summaries from batches. This evidence does not waive the remaining Qwen4 native numerics, Gemma quality or remaining hybrid partial-import limits.


## Dense B4 faults after actual victim import (2026-10-25)

The earlier hardware triggers occurred before device import. The strict follow-up driver waits for a positive victim progress-span count after the real import call, with the producer still pending, three active survivors, and the victim not admitted. No runtime or SDK behavior is patched. A separate driver preserves the original test and evidence. Gemma multilane hybrid import remains deliberately deferred and is excluded from this trigger.

All three dense models passed cancellation/reuse and producer-death on RNGD→ATOM, B4/k5/context9216/producer-width8192. Survivors use512-input/512-output tokens, victim8192/16, replacement512/16. KONI usesTP2; Qwen modelsTP1. Runtime source hashes are included in the compact audit; the PR266 runtime and the same ATOM/RNGD SDK environment documented above are required.

| Model | Victim spans at cancel/death | Active survivor tokens at cancel | At producer death |
|---|---|---|---|
| Qwen3-0.6B | 1 / 1 | 206,109,11 | 192,94,8 |
| Qwen3-4B | 1 / 1 | 115,57,6 | 108,55,5 |
| KONI-8B TP2 | 1 / 1 | 178,91,7 | 182,90,6 |

Independent raw-array audit confirms9216 survivor-token comparisons and48 replacement-token comparisons exact against same-lane controls. Both controllers exited0; all three owned Pods were cleaned up and the task namespace was empty. The audit rejects the previous zero-import result. The test proves isolation for these fault points and fixtures; it does not prove correctness for every import boundary, native-reference parity, semantic quality, or a throughput improvement. Qwen4 native lane-dependent numerics remain unresolved. Hardware import is observed through the real SDK import call returning and the engine progress counter; no separate physical DMA trace is claimed.

Reproduce using the existing PR266 hardware harness, an already compiled B4/k5/context9216 artifact, and the allocated task devices. MODEL/FAMILY/TP are Qwen/Qwen3-0.6B/qwen/1, Qwen/Qwen3-4B/qwen/1 or KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024/koni/2. Supply site paths through local variables:

```bash
"$ATOM_PYTHON" container_bootstrap270.py check_atom270_b4_imported_faults.py --model "$MODEL" --family "$FAMILY" --cache-dir "$CACHE_DIR" --tensor-parallel-size "$TP" --worker-python "$WORKER_PYTHON" --output "$RAW_RESULT"
python3 experiments/evidence270/audit_atom270_imported_faults.py "$RAW_RESULT"
```

Raw token arrays stay outside Git; imported-fault-evidence-manifest.json records hashes and the research event ID. Remote archive synchronization is unverified. The included driver requires the existing bootstrap, atom_server270_http helpers, streaming factory and task worker environment; it is not standalone on main.


## Same-card Qwen3-4B lane diagnosis (2026-10-25)

The same-Pod normal/fixed-prefill/fixed-prefill/normal ABBA completed128 cases. Each arm checks all16 decode-lane/KV-block placements for free32-token and forced-prefix14-token outputs. All four arms have identical tokens and recorded logits; only lane0 matches the saved reference (4/16free cases per arm). At forced index13 lane0 selects369 and lanes1–3 select264. Four matching binding records remove the earlier cross-card confound. Child/copy/cleanup all0; no task Pods remain.

The first attempt failed SDK INIT_INTERNAL when loading a second engine in one process; the retry uses fresh processes in the same Pod. Both failure and accepted retry were recorded. This result plus the earlier direct SDK replay narrows the reproduction to SDK/compiled decode behavior, without identifying the faulty operator or declaring either numerical output correct. The production runtime is unchanged and B4 native parity remains failed.

See evidence270/QWEN4-LANE-REPRO.md for the minimal token fixture, measured diagnostic scripts, prerequisite artifact/SDK versions, exact commands, acceptance criteria and archive hashes. No new hardware run was needed to package this reproduction.


## Follow-up instrumentation changes numerical behavior (2026-10-25)

Supported hidden-state output and first-layer module hooks narrow the first observed difference to attention output on their instrumented graphs; first-layer KV equality passes positive/restore controls. However, exposing raw attention internals makes the first-attention probes lane-equal and removes the512/index50 token split, while2048/index13 still diverges later. Thus these observations do not identify the original faulty operator or constitute a fix. A packed-output attempt failed DEVICE_GRAPH_CONVERSION and is retained as failed.

See evidence270/NUMERICS-DIAGNOSTICS.md for the progression, exact source scripts, SDK prerequisites, replay commands, compact audits, archive hashes and acceptance criteria. No production change or new performance gain is claimed.


## All-layer attention-output diagnostic: lane parity passes, no performance promotion (2026-10-25)

Appending all36 core-attention outputs to the decode graph, while retaining all37 hidden states, makes all292 recorded outputs lane-exact across the four forced-history probe positions. Full-vocabulary logits also match across lanes. This is an isolated diagnostic export using the same Qwen3-4B weights, TP1/B4/k5/context9216 and SDK0.11.0/optimum0.11.0.post1; it adds1,179,648 bytes of attention outputs per decode call. It does not identify the original compiler transformation or establish a production fix.

The free-generation native HTTP follow-up uses unchanged PR266 runtime33d6cb08ec806f693db4ecfd7d630331a3bd1bb2f, one assigned ATOM, inputs128/512/2048/8192, outputs32/128, concurrency16, caches off, and one timed batch per condition after warmup. All8warmups,128token comparisons and128timed-SSE text comparisons pass against the candidate's own per-condition references. The active-lane histogram is [0,664,32,32,5008]. Child, copies and UID-checked cleanup all exit0.

This removes the original within-B4 parity failure on these fixtures (36/128 failed token checks), but historical B1 references match only4/8conditions and B4 references2/8. Those baselines ran on different physical cards. At8192input/128output, candidate throughput is16.438tok/s, median first SSE60.791s and median E2E86.400s; historical B4 was18.242tok/s and77.600s E2E. Thus there is no measured performance improvement to promote. Read-only metrics polling occurred during this exploratory run; same-card repeated controls and semantic quality remain separate gates.

The task-only export/replay sources and compact audits are in evidence270; raw artifact hashes and the research event ID are in allcore-evidence-manifest.json. Run the builder with the established task environment and assigned device, then the SDK replay; use the existing HTTP harness with the all-core cache directory. As with the earlier reproduction, scripts expect the existing local evidence filenames and helper modules and are not standalone on main. Re-audit the collected HTTP result alongside both historical native files and the terminal controller state with `python3 audit_atom270_allcore_http.py`. The first audit correctly rejected a still-running cleanup state and passed after the controller reached completion. No measured response or timing batch was retried.


## Frozen sequential and concurrent quality gates (2026-10-25)

The Qwen3-4B all-core artifact passes the existing36 frozen arithmetic/passkey/Korean-table cases at short/2K/8K input with max128 output, greedy selection, thinking disabled and EOS respected. Sequential and concurrent runs each score36/36 for answers and strict JSON format; all36 token arrays in each run equal the historical native-ATOM B1 baseline. Every request ends with EOS151645 and finish_reason=stop. The explicit EOS set is [151643,151645]; the fixed-length performance entrypoint is unchanged.

The concurrent run sends three cohorts of12 distinct questions with a thread barrier. All12 client intervals overlap in each cohort. Aggregate server occupancy [0,13,5,9,64] proves64decode steps with four active lanes; it is not a per-condition occupancy trace. Sequential occupancy is [0,306,0,0,0]. Both successful runs have child/server/copy/cleanup exit0. The first sequential setup attempt failed before any request because the frozen fixture referenced another host's model path; the retry changes only that model-location field and verifies every original token ID before serving. Failed setup evidence is retained locally. No quality answers were retried or filtered.

The new concurrent client reuses the unchanged strict scorer. Its real localhost HTTP self-check requires12 simultaneous requests before responding, checks all36 preserved outputs and EOS policy, and verifies exclusive output creation. Run `python3 selfcheck_quality270_concurrent.py`. The quality server, client and independent auditors are in evidence270. To reproduce, start atom_server270_quality.py with the same trusted runtime helper paths and all-core cache as the prior native HTTP setup, then run `python3 heldout_quality270_concurrent.py --fixtures "$FROZEN_FIXTURES" --endpoint "$OWNED_ENDPOINT" --output "$RAW_RESULT"`. Fixtures come from the existing quality26336-case export; only map the model location to the installed Qwen/Qwen3-4B and verify input IDs. Raw hashes and research event identifiers are in allcore-quality-evidence-manifest.json.

These small structured-answer gates do not establish general language quality, parity with all historical performance-matrix outputs, or a performance gain. The same-card ABBA cost control is running separately; no diagnostic artifact is promoted to a production default.


## Four-model evidence coverage re-audit (2026-10-25)

Re-reading raw historical records across four models, GPU B4 and ATOM B1/B4, and native/whole/stream routes yields33 within-route HTTP passes,1 failure and2 missing results out of36 combinations. The failure is the original Qwen3-4B ATOM B4 native run (36/128 token checks fail); its dependent B4 whole/stream runs are absent. The later all-core native candidate is separate evidence and does not erase this failure. Gemma B4 stream evidence exists in the timeline-instrumented run, passes all8conditions, and reaches at most3active lanes; it is not a four-lane occupancy pass or an uninstrumented performance control.

All576 historical frozen quality responses were rescored without postprocessing. Strict answer counts, in native-GPU/RNGD-GPU/native-ATOM/RNGD-ATOM order, are Qwen3-0.6B32/34/33/33, Qwen3-4B36/36/36/36, KONI31/31/31/31 and Gemma0/0/0/0, each out of36. Strict format counts are36 for Qwen,34 for KONI and0 for Gemma. These model/task quality limitations are distinct from transport/runtime failure. Native-to-PD full-token discrepancies are recorded separately from within-route acceptance.

`evidence270/audit_four_model_current270.py` recomputes these gates and records SHA256 for all raw files in `four-model-current270-audit.json`; run it in the task evidence directory containing the named raw records and existing audit/scoring helpers. This inventory covers historical experimental runtimes and isolated exports, not current-main support. At the read-only GitHub check, PR206 and PR266 remain unmerged and issue271 is open. Remaining work is candidate Qwen4 B4 PD compatibility, Gemma occupancy/import restrictions and owner integration, with measured performance and unchanged quality gates. No complete four-model support claim.


## Same-card all-core ABBA completes: consistency costs throughput (2026-10-25)

Original/all-core/all-core/original runs each complete66responses at512/8192input,128output,c16, using fresh processes in one Pod on the same physical ATOM3. Same PR266 runtime, TP1/B4/k5/context9216, SDK0.11.0/optimum0.11.0.post1, caches off. Every arm has identical occupancy [0,262,8,8,2020]. A1/A2 each fail12full-token and12timed-text comparisons; B1/B2 have zero mismatches. Same-variant warmup references match, cross-variant references differ at both lengths. Full controller/copy/cleanup exit0; individual arm exits remain1/0/0/1, preserving the parity failures.

| Input/output | Original mean tok/s | All-core mean tok/s | Change | Original mean of median E2E (s) | All-core mean of median E2E (s) |
|---|---:|---:|---:|---:|---:|
|512/128|75.976|62.894|-17.22%|17.195|20.775|
|8192/128|18.253|16.489|-9.66%|77.575|86.071|

Two timed batches per variant/condition, each after warmup and complete token checks. These are means of batch statistics, not pooled-request quantiles or confidence intervals. Read-only metrics polling occurred during the run and is disclosed. The repeatable within-batch consistency improvement incurs serving cost; the candidate is not a speedup or production default. Different cross-variant outputs preclude an equivalent-output performance claim. Pending candidate RNGD→ATOM compatibility is a separate experiment, not evidence of support yet.

The independent auditor recomputes flags and timing summaries, validates metadata, geometry, all four bindings, occupancy, arm exit semantics and terminal cleanup. Its self-check retains the historical failure and rejects altered flags, tokens, throughput and a missing batch. Published source is audit_atom270_allcore_abba.py; run it beside the named raw results/logs and terminal controller records. The child runner documents exact ordering and CLI; site configs, owned-Pod allocation and shell environment remain local prerequisites. Full raw hashes and archive event are in allcore-abba-evidence-manifest.json. No measured arm was retried or discarded.
