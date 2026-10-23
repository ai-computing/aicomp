# Matched serving performance pilot (#248)

- **Date**: 2026-10-24
- **Task**: Matched PD performance (aicomp #248)
- **Host / device**: GPU host; one GPU for native pilot
- **Commit**: d0f6f12 plus uncommitted benchmark harness in aicomp-perf-248

## Question

Measure native, two-GPU data parallel and PD under matched prompt/output and cache/chunk policies. Selected models: Qwen3-0.6B, Qwen3-4B, KONI-8B, Gemma3-1B; Qwen2.5-0.5B excluded per user.

## Setup

Isolated production server subprocess; BF16, cache disabled, local chunk 256, four decoder slots. Input lengths 128/8192, output 32, concurrency 1/4, three measured repetitions. Warmup and nonstream token correctness checks excluded from timing. First SSE event timing is not necessarily first-token timing. Earlier in-process prototype retained separately and excluded from comparisons.

## Command

Run from aicomp-perf-248:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=opt_prime HF_HUB_OFFLINE=1 /home/gspark/venvs/pt213/bin/python opt_prime/examples/pd/http_matrix/benchmark.py --http-module /tmp/llmd240.py --model /home/gspark/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca --mode native --lengths 128 8192 --outputs 32 --concurrency 1 4 --repeats 3 --output /home/gspark/workspace/npu_offload/analysis/serving-matrix-201/perf248-qwen06b-native-a.json
```

## Results

Qwen3-0.6B native pilot: 12 measured batches completed; passed=true, parity_passed=true. No benchmark/server process remained at follow-up inspection.

Raw results: analysis/serving-matrix-201/perf248-qwen06b-native-a.json. Prototype: perf248-prototype-native.json in the same directory.

## Conclusion

The isolated native pilot is usable as a reference. PD and DP comparisons and larger model runs remain outstanding; this pilot establishes no speedup or hardware overlap claim.

## First PD comparison

Same command as native, with `--mode pd --reference /home/gspark/workspace/npu_offload/analysis/serving-matrix-201/perf248-qwen06b-native-a.json` and output stem `perf248-qwen06b-pd-a`. Completed12 measured batches, output parity passed. Metrics confirm42remote prefills (warmup2 + correctness10 + timed30), zero prefix-cache hits. Hardware: two NVIDIA A10080GB PCIe, driver580.178.04; native uses one GPU, PD uses two.

Medians over three measured batches (output32tokens):

| Input | Concurrency | Native tok/s | PD tok/s | Native first SSE s | PD first SSE s |
|---:|---:|---:|---:|---:|---:|
|128|1|50.4926|47.4866|0.0196|0.0232|
|128|4|165.8372|155.1425|0.0751|0.1237|
|8192|1|20.2321|19.0529|0.9741|1.0215|
|8192|4|20.4327|22.3715|3.3089|3.0963|

This initial block is ordered native then PD, not yet balanced across mode order. It cannot establish sustained speedup or actual chunk-copy overlap. DP2 (two decoder replicas, four slots each) is the pending same-GPU-count control. Queued continuation command: `/home/gspark/venvs/pt213/bin/python /tmp/run_perf248_remaining.py` from aicomp-perf-248; it runs Qwen06DP2 then Qwen4/KONI/Gemma native/PD/DP2 sequentially, stopping on failure. Exact expanded commands are saved in analysis/serving-matrix-201/perf248-remaining-a.log.

## DP2 follow-up

DP2 completed12 batches and passed output parity. Each server is an independent process with four decoder slots and four CPU threads; DP2 has eight aggregate slots, native/PD four. This is a two-GPU resource comparison, not an equal-slot comparison.

| Input | Concurrency | DP2 tok/s | First SSE s | E2E s |
|---:|---:|---:|---:|---:|
|128|1|48.8727|0.0197|0.6546|
|128|4|176.0122|0.0395|0.7126|
|8192|1|20.5095|0.9528|1.5601|
|8192|4|40.1538|1.7396|2.7849|

Full three-mode summary: analysis/serving-matrix-201/perf248-qwen06b-summary-a.json. Larger-model queue moved to Qwen3-4B native after DP2 exit0.

## Qwen3-4B interim and overlap audit

Native and PD each completed12batches with parity_passed=true. Median throughput (native/PD): input128,c1=39.66/35.24; input128,c4=129.02/112.73; input8192,c1=12.98/11.89; input8192,c4=12.86/13.25tok/s. DP2 is still pending at this observation. These are ordered-block observations, not final cross-model conclusions.

Source audit at d0f6f12: `TorchPrefillEngine.prefill` in opt_prime/opt_prime/accel/pd/pd_batching.py completes its chunk loop, gathers the entire KV span and synchronizes the producer before `move_kv`. Thus this GPU PD baseline has no within-request compute/copy overlap. `_move_one` uses completed device-to-pinned-host copy before asynchronous host-to-destination copy by default. This is specific to the measured Torch path; it does not establish behavior of other native adapters. Existing issue47 tracks overlap; acceptance includes transfer-size curves, TTFT improvement and token equality.

The HTTP transfer histogram is zero for this batched path despite real42remote prefills: PDBatcher accumulates prefill_s/transfer_s internally, but BatchedService._pd_metrics exports only remote/local counts and transferred tokens. Zero histogram must not be interpreted as zero transfer cost. Stage attribution needs those internal measurements exposed or dedicated profiling before deciding which optimization dominates.

Qwen3-4B DP2 completed with parity; all36batches now pass. Three-mode long-input concurrency4 medians:

|Mode|tok/s|First SSE s|E2E s|
|---|---:|---:|---:|
|native|12.8632|5.3985|7.6994|
|pd|13.2452|5.2524|7.0725|
|dp2|25.1737|2.9627|4.5550|

Queue advanced to KONI-8B native. Exact rawdata and summary in analysis/serving-matrix-201/perf248-qwen4b-*-a.json.

## KONI-8B block A

All36measured batches completed and passed parity. Input8192/concurrency4/output32 medians: native12.3836tok/s,PD14.4674tok/s,DP224.6932tok/s. First SSE medians5.8010/5.0833/3.2144seconds; E2E medians8.2236/5.9755/4.7571seconds. Rawdata: perf248-koni-{native,pd,dp2}-a.json. Queue advanced to Gemma native.

Contamination note: CPU-only focused regression tests for issue249 (OMP_NUM_THREADS=2, ~17seconds total including red/green) ran on the same host during the KONI queue. This block is preliminary and must be repeated without simultaneous test work and with balanced mode order. The upcoming fullCPU suite is deferred until hardware measurement ends.

## Four-model block A complete

Queue exited0; no GPU compute processes remain. All12model/mode combinations pass workflow and parity,144measured batches total. Gemma8192input/c4/output32: native20.2289,PD23.1076,DP239.6639tok/s; firstSSE3.2272/2.8150/1.6396seconds. All four models favor DP2 in this initial long-input concurrency4 condition. Summary: analysis/serving-matrix-201/perf248-summary-a.json.

This block still needs quiet, reverse-order replication and longer-output conditions. FullCPU regression for issue249 started only after this GPU queue terminated. Actual overlap and RNGD-to-ATOM controlled throughput remain separate outstanding requirements.

## Publication and reversed block B

Initial harness/evidence committed aba3ad2, Draft PR251. Duration counters committed273952a, Draft PR250; fullCPU validation467passed/17skipped/135subtests before restarting GPU work. BlockB uses identical inputs128/8192,outputs32,concurrency1/4,repeat3, references from original blockA native. Model order reversed (Gemma,KONI,Qwen4,Qwen06), mode orderDP2,PD,native. Runtime PYTHONPATH points to aicomp-metrics-249/opt_prime (273952a); benchmark/CLI entrypoint from aicomp-perf-248 (aba3ad2), HTTPcandidate9e442cd. Only runtime difference fromA is additive metrics export, no scheduling/compute/transfer modification.

Command from aicomp-perf-248: `/home/gspark/venvs/pt213/bin/python /tmp/run_perf248_reverse.py`. Expanded commands in analysis/serving-matrix-201/perf248-reverse-b.log; copied queue script run_perf248_reverse.py in same directory. GPU compute-process list was empty before launch. Do not run CPU regression suites during this block. Stage counters are cumulative across warmup/checks/timed requests and must not be treated as isolated per-condition timing.

## First reversed-model result: Gemma

All three blockB modes passed parity,36batches. Long-input/c4 throughput comparison:

| Mode | Block | tok/s |
|---|---|---:|
|native|a|20.2289|
|native|b|19.5801|
|pd|a|23.1076|
|pd|b|17.7899|
|dp2|a|39.6639|
|dp2|b|38.4918|

PD blockB exports prefill30.8477465seconds and transfer0.3611378seconds across all42admitted handoffs including warmup/correctness. The transfer total is positive and worker timing is now observable; totals are not per-condition critical-path attribution. The ~23% PD throughput drop requires diagnosis; additive metrics code is outside the compute path but version/order effects must be separated by a paired rerun. Observed host load at follow-up1.43/1.80/1.61, not contemporaneous per-request telemetry. No claim that the initial PD improvement is stable. Queue continues with KONI DP2.

Heterogeneous setup audit: control access works and experiment namespace has no Pods. Existing ATOM runner uses PYTHONUSERBASE and bootstrap to restrict one assigned ATOM; artifacts are B1/context9216. Need a matched native/PD benchmark with fixed-output requests rather than reuse EOS-terminated lifecycle latency. No accelerator allocation created during this audit.

## KONI reversed PD and follow-up preparation

KONI blockB PD and DP2 each completed12batches with parity. At8192input/c4:PD14.0186tok/s versus14.4674inA;DP224.6388 versus24.6932inA. PD accumulated worker-reported prefill43.6859s and transfer1.68334s across42admitted handoffs. KONI native still running at this observation. Unlike Gemma, these two modes show only modest block-to-block differences; do not generalize Gemma variability to every model.

Source audit: current factory uses threaded PrefillWorker, so producer/decoder share a Python process. Existing ProcessPrefillWorker uses spawn plus CUDA IPC and a receiver copy. This is a candidate experiment for launch contention, not a demonstrated cause or performance fix. Its lifecycle differs from hardened SubprocessPrefillWorker and must be validated before exposing a new serving default.

Prepared (not yet launched) /tmp/run_perf248_long128.py, copied to analysis/serving-matrix-201/run_perf248_long128.py. It uses all four models/native-PD-DP2, input128/8192, output128, concurrency1/4/8, three repeats and native-long128 references. Start only after blockB terminal and device availability check. No improvement claim from preparation.

## Qwen3-4B reversed block and heterogeneous harness preparation

Qwen3-4B all36blockB batches passed parity. At8192/c4 native12.6604,PD15.1194,DP224.9782tok/s; blockA12.8632/13.2452/25.1737. PD varies upward here (~14%), unlike Gemma's downward variation, while DP2 stays close. PD worker totals39.4895prefill seconds and2.16473transfer seconds across42handoffs. Keep the variability conclusion, not a deterministic per-model speedup claim.

New isolated worktree aicomp-heteroperf-248, branch exp/248-heterogeneous-http from latestmain plus aba3ad2 dependency, adds --server-config to reuse the same client/timing/correctness workflow with trusted argv templates. Config requires a standalone {port}, nonempty argv arrays and resource metadata restricted to known fields; external metadata replaces GPU-specific assumptions. Red test1failed/1passed before implementation; final2passed in0.03s, Ruff/diff clean. This was a negligible standard-library-only test, not a Torch/CPU regression suite. Hardware external-server integration is not yet verified; no NPU allocation started. Current GPU queue uses untouched benchmark branch.

## BlockB complete and longer-output run started

Reverse queue exited0; all144blockB batches pass parity (288acrossA/B). At8192/c4 blockB native/PD/DP2 tok/s: Qwen06=19.8565/18.4056/39.3672; Qwen4=12.6604/15.1194/24.9782; KONI=12.3868/14.0186/24.6388; Gemma=19.5801/17.7899/38.4918. Same-two-GPU DP2 advantage persists; PD versus one-GPU native is variable. Summary: perf248-summary-b-longc4.json.

After confirming no GPU processes, launched /tmp/run_perf248_long128.py from aicomp-perf-248. Raw queue log:analysis/serving-matrix-201/perf248-long128.log. Output128,inputs128/8192,concurrency1/4/8,repeat3. Separate NPU host benchmark does not share these GPU-host compute resources.

Qwen3-0.6B long128 extension complete:54batches (native/PD/DP2,2input lengths,3concurrencies,3repeats), all parity pass. At8192input/output128/c8 native65.7449,PD74.6555,DP2120.6477tok/s; c4=60.8016/62.2604/98.9765. Summary:perf248-qwen06b-long128-summary.json. Queue advanced to Qwen3-4B native. Reverse-block evidence published in PR251commitb367fdf; body updated to explicitly report PD variability.

Qwen3-4B native-long128 completed18batches with parity.8192input/c1,c4,c8 throughput27.2032,41.5924,44.3198tok/s. Queue advanced toPD; comparative result pending.

Qwen3-4B long128 three-mode comparison complete:54batches pass parity.8192/c8 native44.3198,PD46.7759,DP280.4156tok/s; summary perf248-qwen4b-long128-summary.json. Queue moved toKONI. Producer-side copy-overlap design constraints recorded in analysis/serving-matrix-201/chunk-overlap-plan.md; no implementation/performance claim.

KONI native-long128 completed18batches with parity; GPU queue advanced toPD. Separate perf/47-chunk-copy worktree at273952a contains a pending two-GPU exact-KV/first-token oracle (dense/hybrid,send_from0/7/29/35,cached repeat). Tests are not executed while performance GPUs are occupied; no copy-overlap implementation or passing-test claim. Need observe RED on idle allocated GPUs before implementation and add failure/lifetime/recurrent/timeline checks.

KONI PD-long128 completed18batches with parity.8192input/c1,c4,c8 throughput27.2150,44.6598,50.1311tok/s; native c8=42.8652. DP2 running. Heterogeneous Qwen4producer confirmed alive at7m32s with CPU activity and~31.8GiBRSS; readiness not yet reached, within1800second bound.

KONI complete-long128 matrix:54batches pass parity; summary perf248-koni-long128-summary.json. GPU queue advanced toGemma native.

Stage audit for completedQwen06/Qwen4/KONIPD long128:106remote handoffs each (warmup2+checks26+measured78). Worker prefill/transfer cumulative seconds respectively78.7096/3.3561,114.3728/4.5195,113.7939/3.7680. These are mixed-condition sums includingwarmup/checks, not critical-path percentages; decoder KV installation is not in transfer_s. Do not assume transfer overlap alone closes the DP2 gap. Audit artifact perf248-pd-long128-stage-audit.json.

Gemma native-long128 completed18batches with parity; PD running. Existing GPU and heterogeneous queue handles remain live. No restart or competing device tests.

## Completed four-model output-128 matrix

All 12 model/mode runs completed successfully: 216 measured batches and all recorded token/text parity checks passed. Models: Qwen3-0.6B, Qwen3-4B, KONI-8B, Gemma3-1B. Inputs 128/8192; output 128; concurrency 1/4/8; three repeats. Exact commands are in `analysis/serving-matrix-201/perf248-long128.log`; raw files use `perf248-*-long128.json`; harness/runtime revisions and setup are recorded above and in raw artifacts.

8192 input / 128 output / concurrency 8, median output tok/s:

| Model | Native (1 GPU) | PD (2 GPUs) | DP2 (2 GPUs) |
|---|---:|---:|---:|
| qwen06b | 65.74 | 74.66 | 120.65 |
| qwen4b | 44.32 | 46.78 | 80.42 |
| koni | 42.87 | 50.13 | 80.45 |
| gemma | 64.08 | 73.72 | 112.83 |

PD exceeds one-GPU native in this long-input/high-concurrency block, but remains below two-GPU DP2. This is one workload block, not a general speedup claim. Parity does not establish answer quality. No new chunk-copy overlap implementation was used.
