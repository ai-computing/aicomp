# Native GPU short/long low-concurrency HTTP coverage

- Date: 2026-10-30; related #201,#265 and PR266.
- Runtime:162c2f6c969a876318c7723973e21289d302e9da, branch defaultHTTP. Production sources unchanged from the preceding default201 run; source hashes in the evidence summary.
- Hardware:one A10080GB PCIe on the GPU host; driver580.178.04,torch2.13.0+cu130,Transformers5.15.0. No RNGD/ATOM device used.

## Results

All four selected checkpoints complete input128/8192, output32/128 per request, concurrency1/4: 32 conditions, 176 responses (16 warmups, 80 retained-token checks, 80 timed SSE-text checks). Actual saved arrays/text,lengths,metadata and timing arithmetic independently audited. All 16 reference arrays match the corresponding historical c16 native references. All fourchild exits0; coordinator exits0; fresh compute-process query empty after completion.

At8192 input tokens,128 output tokens per request, c4 (512 total output tokens):

|Model|Aggregate tok/s|Median first SSE(s)|Median E2E(s)|
|---|---:|---:|---:|
|qwen06b|60.84|3.348|7.428|
|qwen4b|40.92|5.408|11.274|
|gemma|58.83|3.270|7.542|
|koni|40.46|5.848|11.603|

One timed batch per condition, not stable speedup evidence. These are native GPU controls; they do not certify heterogeneous native equivalence, answer quality or cache-on behavior. Fixed-length output ignores EOS. FirstSSE and estimatedTPOT are client metrics; SSE can coalesce tokens.

## Reproduction

Use benchmark201_native_http.py with each selected frozen checkpoint and the default serve_batched.py/llmd.py modules. Configs are identical to the default201 middle-length controls: BF16, four decoder slots, paged KV/page64, chunk256, context9216, caches off, one GPU. Use an idle allocated GPU; stop if another workload occupies it,never terminate unrelated processes.

```bash
PYTHONPATH="$RUNTIME/opt_prime:$RUNTIME/opt_prime/examples/pd" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON" "$HARNESS" --model "$MODEL" --mode native --server-config "$NATIVE_CONFIG" --lengths 128 8192 --outputs 32 128 --concurrency 1 4 --repeats 1 --request-timeout 180 --output "$NEW_RESULT"
python3 evidence265/audit_gpu_edge201.py "$ARTIFACT_DIR"
python3 evidence265/selfcheck_gpu_edge201.py "$ARTIFACT_DIR"
```

Raw files are edge201-{model}-gpu-native.json;queue edge201-gpu-queue.json preserves exact argv/config/source hashes and exitstatuses. The auditor also requires matching stream265 c16 native controls and default201 configs from the same artifact archive. Full site-local paths/configs/logs are not published. The portable auditor accepts an artifact directory; incomplete queue state is explicitly retained rather than treated as complete.

## Verifier checks

Eight selfchecks pass,including duplicate complete arms, corrupted warmup/token/SSE data, wrong throughput, missing conditions and missing raw results. The duplicate/warmup negative cases initially reproduced verifier defects; fixed the unique planned-model set check and actual warmup comparison. Prior GPU middle/default/PD raw results were re-audited successfully with the same fixes. Request length and aggregate output tokens remain separate fields. No production engine change.
