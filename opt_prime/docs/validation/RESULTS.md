# Optimus heterogeneous PD validation — 2026-10-19

## Scope

The new `opt_prime.pd.PDRuntime` executes both ordinary Torch and heterogeneous
RNGD→ATOM adapters. The Torch adapter uses OptimusPrime's real KVCacheManager.
The KONI adapter reuses the existing live-allocation owner and checked P2P backend.
This validates a new engine integration, not a new driver implementation.

| Check | Result | Evidence |
|---|---|---|
| CPU contract / model tests | 13 passed | `cpu-tests.log` |
| Actual GPT-2/Llama PD vs no-cache full-sequence oracle | 24 cases, exact output tokens; maximum absolute logits difference 1.1920929e-7 | `torch-parity.json` |
| KONI real RNGD3 → ATOM2 | 24 requests / 1,288 output tokens, full logits exact against same-decoder host admission | `koni-r1/requests.jsonl`, `koni-r1/audit.json` |
| P2P completion | 1,536 / 1,536 commands, CQE result 0 for every request | `koni-r1/requests.jsonl` |
| Owner teardown | Worker exit 0, owner result passed; no remaining device contexts | `koni-r1/owner-result.json`, `post-*.txt` |
| Packaging | Wheel build/install and SDK-free import succeeded | `packaging.log` |

Torch cases: two small randomly initialized architectures (seed 17), full-prompt
and prefix-only prefill, prompt lengths 2/7/19, output lengths 1/6. The independent
oracle recomputes each whole sequence without cache. Acceptance: tokens exact,
all logits `atol=1e-6, rtol=1e-5`. These are real model executions, not task-quality
benchmarks. Torch 2.11.0+cpu, transformers 5.8.1.

KONI: model revision `0172bf7e73977ae213613529f3d2d5400a33f678`, four existing
prompts of lengths 140/167/227/271, output lengths 1/32/128, two repeats. One active
slot in the existing B4/context512 ATOM graph. RNGD physical 3/logical 2,
ATOM physical 2. Furiosa torch/LLM/driver 2026.4.0 (RNGD Python torch 2.10.0+cpu);
ATOM driver 3.0.0, torch-rbln 0.2.2, rebel-compiler 0.11.0, optimum-rbln
0.11.0.post1 (ATOM Python torch 2.11.0+cpu). Model, graph, helper and workload
provenance is in the config/artifact/source manifests.

## Reproduction

From the aicomp checkout:

```bash
PYTHONPATH=opt_prime:/path/to/hetero-npu-p2p/src \
  python3 -m unittest discover -s opt_prime/tests -v
PYTHONPATH=opt_prime python3 opt_prime/examples/pd/torch_demo.py
python3 -m pip wheel --no-deps --no-build-isolation ./opt_prime --wheel-dir /tmp/optimus-wheels
```

Hardware command used from the shared workspace (paths/config captured in the
adjacent evidence; use the correct SDK environments and selected idle devices):

```bash
PYTHONPATH=aicomp-integration/opt_prime:hetero-npu-p2p/src:rngd_llmd/src \
  /home/<user>/venvs/furiosa-2026.4/bin/python \
  aicomp-integration/opt_prime/examples/pd/validate_koni.py \
  --config /tmp/optimus-koni-config-r1.json --hetero-repo hetero-npu-p2p
```

The runner refuses to overwrite an output directory. Set a new `out` for each
rerun. The checked-in evidence config preserves the exact original inputs; it is
not a portable default configuration. Model weights, compiled graphs, host KV
arrays and full-logit binary references are not committed. They remain external
inputs identified by revision/path/hash. All per-request completion/parity reports
are retained, not only the best result.

## Limits and source provenance

The hardware run used dirty source based on commit `9b40907`. Source hashes and
its exact snapshot are in the raw run directory `/tmp/optimus-koni-r1/source`.
During device startup, two input-type guards were added to `KVSpec`/`PDRequest`;
the run's loaded version is reconstructed and identified in `source-manifest.json`.
Final CPU tests cover the current guarded version. Inference/transport code was
unchanged during the hardware run.

No new kernel failures, device resets or forced owner termination occurred.
Initial development probes established that transformers 5.8.1 uses a
non-subscriptable DynamicCache; the adapter uses its layer interface. An initial
wheel build invoked from the workspace parent used a wrong relative directory;
the corrected repository-local build succeeded. Neither was a hardware failure.

Not established: CUDA PP/TP cache handoff, other device pairs/driver versions,
maximum throughput, power/MFU, native-vendor numerical equivalence, task quality,
streaming, llm-d deployment or multi-node support. The generic interface makes
additional adapters possible; it does not automatically confer hardware support.
