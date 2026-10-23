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
