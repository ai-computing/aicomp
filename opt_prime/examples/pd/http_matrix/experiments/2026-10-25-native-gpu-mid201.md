# Native GPU intermediate-length coverage

- Date: 2026-10-25; related #201 and PR266.
- Runtime: eaf7daa219a6cea68db4ec5fc08850edc848b271, clean before execution; no production changes.
- Hardware: one A10080GB PCIe on the GPU host; no RNGD/ATOM allocated. Four models run sequentially after empty compute-process checks. All four children and coordinator exited0; final compute-process query empty, both GPUs14MiB idle memory.
- Configuration: BF16, four decoder slots, paged KV/page64, local chunk256, context9216, producer/decoder caches off. Same frozen checkpoints/configuration as the earlier c16 native GPU matrix; torch2.13.0+cu130, Transformers5.15.0. Driver identity retained in the local queue artifact.

## Coverage and results

A schema-scoped inventory of177 archived HTTP JSON files found no native one-GPU correctness batch at input512/2048 and concurrency1/4. Added four models × two input lengths × two output lengths(32/128) × two concurrencies(1/4):32conditions. Per model:4warmups,20nonstream token comparisons,20timed SSE text comparisons. Total176responses.

All conditions passed. An independent audit compared actual saved token arrays and text with same-server references, verified prompt identity and counts, and recalculated throughput/median first-SSE/E2E from response rows. All16reference arrays also match the corresponding earlier c16 native references. This is within-native parity, not native/PD equivalence or semantic quality.

|Model|Conditions|Token checks|SSE checks|2K/128 c1 tok/s|2K/128 c4 tok/s|
|---|---:|---:|---:|---:|---:|
|qwen06b|8|20/20|20/20|50.24|143.18|
|qwen4b|8|20/20|20/20|37.44|102.97|
|gemma|8|20/20|20/20|40.41|118.85|
|koni|8|20/20|20/20|44.71|117.01|

One timing batch per condition. Throughput is aggregate client output tokens divided by batch wall time; no repeated or equal-resource speedup claim. Complete per-condition timings, raw-result hashes and source hashes are in `../evidence265/mid201-gpu-summary.json`.

## Reproduction and provenance

Use the unchanged benchmark201_native_http.py at ai-computing/aicomp commit052f84f, experiments/evidence274 (SHA256e1ea87a51343d1af83c40aaa9c41ae2a20e9864aa8b53bd475d047d175c41433), with each original stream265 native server config and its frozen model. Keep the process-local HTTP module from the prior c16 run: SHA2569f97e5a9b270dc5719e339748a6d6bdccbe6dfdf5267c1dd4451bab123687975. It supplies the MSG_NOSIGNAL writer and an older readiness predicate. Thus these numbers do not certify the unmodified default HTTP path or readiness behavior. Exact site-local queue argv/config hashes are retained under mid201r1-gpu-queue.json; full logs/configs remain outside the public repository.

```bash
PYTHONPATH="$RUNTIME/opt_prime:$RUNTIME/opt_prime/examples/pd" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON" "$HARNESS" --model "$MODEL" --mode native --server-config "$NATIVE_CONFIG" --lengths 512 2048 --outputs 32 128 --concurrency 1 4 --repeats 1 --request-timeout 180 --output "$NEW_RESULT"
```

The first coordinator omitted PYTHONPATH and stopped before Qwen0.6 inference with ModuleNotFoundError for opt_prime.accel.pd.contracts. The failed run remains archived under mid201-*; the corrected environment run uses mid201r1-* and did not change runtime, request options, timeout or correctness criteria. This launcher failure is not a model inference failure.

## Limits

Cache-on, heterogeneous-route intermediate c1/c4 combinations, default HTTP route and engine-owned export/cache compatibility remain separate work. Earlier Qwen4/KONI cross-engine numerical differences and Gemma strict-format failures remain. These successful native GPU cells do not close #201.

## Output-length summary correction (2026-10-30)

The original c4 throughput column selected32output tokens per request because aggregate128tokens overwrote a same-named per-request field in the summary. The corrected2K/128c4 values above select128tokens per request (512tokens total). Raw measurements, timing arithmetic and correctness results are unchanged. Summary JSON now retains both `output_tokens_per_request` and aggregate `output_tokens`. Native c1 values and the separate PD GPU/ATOM tables were checked against raw condition fields and remain unchanged.
