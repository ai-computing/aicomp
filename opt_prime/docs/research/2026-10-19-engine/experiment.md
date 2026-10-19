# OptimusPrime integrated engine CPU and real KONI validation

- Date: 2026-10-19
- Task: PDDISAG-6
- Source: aicomp feat/optimus-heterogeneous-pd, base9b40907 plus recorded dirty changes
- Hardware: RNGD3/logical2 and ATOM2, existing B4/context512
- SDK: Furiosa2026.4.0; ATOM driver3.0.0/torch-rbln0.2.2/rebel0.11.0

## Question
Does a vendor-neutral Optimus PD runtime preserve generation boundaries, physical
KV compatibility, fences and outputs across Torch and actual RNGD→ATOM endpoints?

## Implementation
Explicit full-prompt vs prefix-only continuation; real Optimus KVCacheManager
Torch adapter; KONI prefix adapter and existing owned ATOM iteration worker;
optional CopySession transport bridge. New code never guesses a device address.

## Commands
```bash
PD_VALIDATION_REPORT=aicomp-integration/opt_prime/docs/validation/torch-parity.json PYTHONPATH=aicomp-integration/opt_prime:hetero-npu-p2p/src python3 -m unittest discover -s aicomp-integration/opt_prime/tests -v
PYTHONPATH=aicomp-integration/opt_prime:hetero-npu-p2p/src:rngd_llmd/src /home/gspark/venvs/furiosa-2026.4/bin/python aicomp-integration/opt_prime/examples/pd/validate_koni.py --config /tmp/optimus-koni-config-r1.json --hetero-repo hetero-npu-p2p
```

## Results
13 CPU tests passed. Real tiny GPT-2/Llama 24 cases: exact output tokens,
maximum logits absolute error1.1920928955078125e-7 vs independent no-cache oracle.
Real KONI24requests,1288outputtokens,1536DMA commands; all CQE0 and full logits exact
vs same-decoder host admission. Four prompts140/167/227/271, outputs1/32/128, repeat2.
Producer and worker exit0; NPU contexts empty afterward. No resets or killed owners.
Wheel installation and SDK-free imports passed. Hardware run source-manifest notes
input-type guards edited during startup; execution/transport unchanged. Current
CPU tests cover guarded inputs. A DynamicCache indexing development probe and a
wrong-relative-directory build failed before corrected runs; no failed hardware runs.

Raw: /tmp/optimus-koni-r1; retained small evidence in aicomp-integration/opt_prime/docs/validation.
Large weights, compiled graphs, KV/logit arrays excluded from Git.

## Conclusion / remaining boundaries
Integration correctness proved for these routes, not arbitrary hardware, quality,
MFU or saturation throughput. One active B4 slot used. CUDA distributed adapter and
llm-d service integration remain separate features. Create feat/optimus-llmd from
the tested engine commit, with lifecycle/API validation gates before implementation.
