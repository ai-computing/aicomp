# Qwen3 RNGD → ATOM: host stacking and experimental HTTP serving

- **Date**: 2026-10-23
- **Task**: Qwen3 serving and export profiling (aicomp #199)
- **Host / device**: RNGD+ATOM host, dedicated single-card device-plugin Pod; Furiosa 2026.4 fusion4/Mesh2; ATOM rebel-compiler 0.11.0, optimum-rbln 0.11.0.post1, torch 2.11.0+cpu
- **Commit**: baseline 353d15090aa7908e0bb2ea5422f047e59ea9273d; experiment branch exp/199-qwen3-serving (56d9056 plus working changes before final commit)

## Question

Is the roughly 719 ms Qwen3 research prefill dominated by model computation or KV materialization, and does its improvement survive the existing batched HTTP service?

## Setup

Checkpoint Qwen/Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca. Three fixed chat prompts, thinking disabled, 19/19/22 tokens, fixed 16 output tokens for parity diagnostics. RNGD width128, 28 layers, 8 KV heads, dimension128. ATOM own B1/context512/3KV-block artifact from the previous experiment. Prefix caching disabled; B1 serializes three concurrent requests through the existing queue.

Baseline stacks each K/V tensor on RNGD before copying. Improvement copies each layer to CPU and stacks there; model math, widths and sampler unchanged. Existing SubprocessPrefillWorker, AtomRblnEngine, BatchedOwner and HTTP handlers are reused through an explicit experimental launcher. No shared SDK files or other workloads changed. Host-mediated KV, no DMA or producer overlap.

## Command

Use PR195 isolated-container environment and bootstrap; working directory contains the Qwen3 example files and the previous trusted fixture. Set HOME to the SDK owner's home, offline HF cache, OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, RBLN_DEVICES=0, RBLN_NPUS_PER_DEVICE=1, and PYTHONPATH to the existing optimus-rm, rngd_pd and hetero_p2p sources. FURIOSA_PYTHON and ATOM_PYTHON select the previously validated interpreters.

```bash
# From the checkout, save the exact pre-change implementation into the scratch directory:
git show 353d15090aa7908e0bb2ea5422f047e59ea9273d:opt_prime/examples/pd/qwen3_rngd/prefill.py > prefill_baseline.py
./furiosa_prefill_python.sh -m bench_export
# Service on a dedicated Pod, loopback only. Baseline adds --local-prefill.
"$ATOM_PYTHON" check_atom_container.py serve.py --cache-dir "$PWD/qwen3-197-atom"
"$ATOM_PYTHON" bench_http.py --name pd
"$ATOM_PYTHON" bench_stream.py --name pd
```

A/B export alternates baseline/optimized order over four rounds and three prompts; round0 excluded from latency means. Whole K,V,logits are compared bit-for-bit each pair. HTTP harness performs twelve sequential requests, three concurrent requests, an over-capacity HTTP400 check and a successful follow-up. Streaming harness performs nine requests plus EOS termination; first round excluded from warm means.

## Results

- Alternating A/B: baseline 728.435 ms mean (688.581–829.348), host stacking 55.823 ms (44.516–67.473), nine warm pairs; 12/12 complete K,V,logit comparisons exact including first round.
- Phase diagnostic: device stack/export roughly 650 ms versus per-layer CPU export/host stack roughly 11 ms. Layer execution plus final hidden materialization roughly 5 ms; CPU output head roughly 26–40 ms. These are diagnostic wall times with fences, not isolated DMA bandwidth.
- Width32/64 reduced the original stack cost but changed KV/logits numerically; not adopted. Width128 stays unchanged.
- Optimized PD HTTP: 16/16 full output sequences equal ATOM-only; 16 remote prefills, zero local fallback. Warm sequential nonstream mean201.439 ms vs ATOM121.831 ms.
- Warm streaming (six requests): optimized PD TTFT69.567 ms/E2E167.535 ms; ATOM TTFT15.883 ms/E2E128.254 ms.
- Oversize129-token prompt rejectedHTTP400 before device work; next request succeeds. EOS termination passes both routes. Fixed-length diagnostics intentionally ignoreEOS and are not quality tests.

Failures retained: first profiling script attempted to iterate safe_open directly (TypeError); changed to .keys(), with targeted SIM118 exemption. Initial experimental worker launch used a relative executable path and failed before worker startup; changed to an absolute sibling path. Parent resource_tracker cleaned the abandoned slab. No shared package modified.

Raw logs/results: analysis/local_prefill/native_dma/general/serving199/. Exact local launch scripts retained with this experiment's artifacts; public reproduction uses role/environment variable names.

### Pre-change HTTP baseline and final comparison

| Route | Warm streaming TTFT (ms) | Warm streaming E2E,16 outputs (ms) | Warm nonstream E2E (ms) |
|---|---:|---:|---:|
| PD, device-stack baseline |755.042|905.176|887.234|
| PD, host-stack improvement |69.567|167.535|201.439|
| ATOM-only |15.883|128.254|121.831|

Streaming means use six requests after excluding the first round; nonstream means use nine requests after excluding the first round. Routes ran sequentially (optimized PD, ATOM, old PD), not a randomized serving trial. No quiet-host guarantee or confidence intervals; the alternating in-process A/B supports the export mechanism independently. All16 nonstream token sequences match across all three routes. All three pass oversized-request rejection/recovery and EOS termination. The pre-change service uses exactly the old prefiller file in the same task-owned scratch directory, restored to the improved file afterwards.

The improved PD first-token gap to ATOM is53.684ms; E2E gap39.281ms in this small streaming sample. Source export and CPU LM head now merit optimization; no producer-side overlap was added. Server drains were successful and the dedicated validation namespace was deleted; other workloads untouched.

## Conclusion

The largest original delay was device-side KV stacking/materialization, not inherent PD disaggregation or transformer-layer compute. Moving the stack to CPU yields about13× prefill speedup without numerical changes and enables the existing HTTP service. ATOM-only remains faster for these short B1 requests; this does not establish native Furiosa parity, throughput at high concurrency, or an advantage for PD. Remaining work: output-head cost, ATOM import/IPC and longer prompts; source-streaming overlap and KONI HTTP integration remain separate validations.
