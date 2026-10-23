# Experimental Qwen3 RNGD prefill → ATOM validation (#197)

Qwen3-0.6B requires Q/K normalization, unbiased projections, and a2048-wide query
projection despite hidden_size1024. This separate graph implements those semantics;
it does not loosen the existing Qwen2-only DenseModel contract or register a new
production serving backend. Quantized weights, scaled RoPE, sliding attention and
non-Qwen3 configs are rejected. Only Qwen3-0.6B has been hardware-validated here.

## Reproduce

Use the isolated single-card device-plugin Pod setup from PR195's
`ATOM_CONTAINER_EXPERIMENT.md`, including validated Python environments, HOME,
offline model cache, device leases, and the fusion4/Mesh2 Furiosa launcher. Requires
existing `rngd_pd` and optimus-rm modules on PYTHONPATH, plus PR195's
`atom_stream_upload.py` and `check_atom_container.py`. Do not install into or mutate
another workload's environment. Work from a writable scratch directory containing
these files. Use only the allocated devices.

```bash
python3 test_spec.py
# Export FURIOSA_PYTHON to the validated Furiosa2026.4 interpreter.
./furiosa_prefill_python.sh -m run_prefill
"$ATOM_PYTHON" check_atom_container.py check_atom.py
```

`run_prefill.py` writes trusted tensor data to qwen3_197_kv.pt. `check_atom.py`
compiles/loads its own B1/context512/3KV-block artifact under qwen3-197-atom, compares
CPU FP32 prefill, native ATOM prefill and imported RNGD KV, and writes
qwen3_197_atom.json. It uses both the existing and portable importers and reads back
every imported slot. These two scripts run a finite fixture; the experimental HTTP adapter is described below.

## Hardware evidence (2026-10-23)

RNGD: Furiosa2026.4,fusion4/Mesh2,width128,28layers,8KV heads,head_dim128.
ATOM: rebel-compiler0.11.0,optimum-rbln0.11.0.post1,torch2.11.0+cpu,
B1/context512/3KV blocks. Checkpoint Qwen/Qwen3-0.6B; three chat prompts with thinking
disabled. Exact records in validation.json; compilation provenance in the local
research experiment2026-10-23-gemma-fix-qwen3-koni.md.

-3/3first tokens match CPU FP32 and native ATOM.
-3/3fixed16-token PD outputs match native ATOM.
-3/3full decode-logit sequences match between original and portable importers.
-336/336slot readbacks exact:3prompts ×2importers ×56slots.
-RNGD-vs-CPU relative RMS:K0.470–0.491%,V1.757–1.911%,last logits1.164–1.604%.
-Prefill wall time including host KV/logits:1.321s first,0.723/0.714s subsequent.
  This is a correctness bring-up, not vendor-native performance parity.

Fixed-length diagnostic generation continues beyondEOS; service termination semantics
are not tested. The Korean-capital answer is incorrectlyIncheon on both CPU/ATOM in
this fixture; matching tokens does not establish factual quality. No DMA, zero-copy,
streamed source export or overlap claim. Host transfer occurs after complete prefill.

The initial compile failed because epsilon/exponent inputs wereBF16 while rms_dsl
requiresFP32. They now remainFP32; weights/activation carriers remainBF16. Failed
compile log is retained with the experiment. CPU contract tests reject unsupported
semantics; hardware records are essential evidence beyond those tests.

## Limits

Other sizes, longer prompts, concurrent requests, prefix reuse, cancellation and
production recovery/lifetime behavior are unvalidated. The handoff is exposed by the explicit experimental `serve.py` adapter below;
it is not automatically registered in npu_pd_bench/serve_batched. Keep model-specific math in this prefiller;
the portable56-slot import succeeded without communication-library changes.


## Experimental HTTP adapter and export improvement (#199)

`serve.py` reuses the existing optimus-rm `SubprocessPrefillWorker`, ATOM batcher,
`BatchedService` and `BatchedOwner`. `worker_python.sh` redirects only the expected
prefill worker module launch to `worker.py`; model-specific math stays in
`prefill.py`. This is an explicit experiment requiring the existing optimus-rm
runtime on PYTHONPATH, not a new production backend in this checkout.

Place PR195's `furiosa_prefill_python.sh` alongside `worker_python.sh` and use its
single-card allocation/bootstrap instructions. Keep the task's own compiled
artifact cache and validated interpreters. The HTTP listener binds only loopback.
The service is fixed to Qwen3-0.6B, B1/context512/3KV blocks, no prefix caching.
Prompts above `--width` (default128) are rejected before worker submission. Decode
requests may queue; only one lane executes. SIGTERM drains queued work before
closing the worker and slab. Disconnect cancellation and fault recovery have not
been hardware-validated in this experiment.

```bash
python3 -m unittest discover -s . -p 'test_*.py'
# In the writable scratch directory with the validated fixture/artifacts:
"$ATOM_PYTHON" check_atom_container.py serve.py --cache-dir "$PWD/qwen3-197-atom"
# Another process in the same allocated Pod:
"$ATOM_PYTHON" bench_http.py --name pd
"$ATOM_PYTHON" bench_stream.py --name pd
# Stop/drain the service, then restart with --local-prefill for the same ATOM-only baseline.
```

The export fix replaces device-side `torch.stack(...).cpu()` with per-layer
`.cpu()` followed by CPU stacking. The device graph and width remain unchanged.
To reproduce the alternating A/B, save the old implementation in the scratch
working directory (not over the new prefiller):

```bash
git show 353d15090aa7908e0bb2ea5422f047e59ea9273d:opt_prime/examples/pd/qwen3_rngd/prefill.py > prefill_baseline.py
./furiosa_prefill_python.sh -m bench_export
```

Measured warm prefill: 728.435→55.823ms mean (nine pairs); all twelve paired K,V
and full-logit comparisons exact. Detailed serving results and limitations are
in `SERVING_EXPERIMENT.md` and `serving_results/`. The old roughly719ms measurement
above is historical, before this export fix. Width32/64 were also measured but
changed numerical values, so the verified improvement retains width128.

`safe_open` must be iterated through `.keys()` in this SDK environment; the
specific lint exemption preserves that behavior.
