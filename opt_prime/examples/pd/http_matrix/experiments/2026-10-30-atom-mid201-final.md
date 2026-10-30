# ATOM intermediate-length HTTP comparison with final HTTP module

- Date: 2026-10-30; related #201, #265 and PR266.
- HTTP source: PR266 a3497e5, SHA256 `e3449979b88d03edb6d5bf0d402a708e5d98054fdf203e22b480f3d7e6fd759a`.
- Engine: historical stream265 B1 engine and research RNGD producer, with the final HTTP module explicitly loaded. This does not qualify the entire current-main runtime or vendor native furiosa-llm.
- Hardware: one ATOM decode lane; PD additionally uses one RNGD on the same host. Native uses fewer accelerators. c4 means four queued clients, not hardware batch four.
- Decoder: torch2.9.1+cpu, Transformers4.57.6, optimum-rbln0.10.2, rebel-compiler0.11.0. Producer: torch2.10.0+cpu, Transformers5.19.0, furiosa-torch2026.4.0.

## Results

All four selected checkpoints complete native, whole and stream routes at input512/2048, output32/128 per request, concurrency1/4: 12 arms, 96 conditions, 528 responses (48 warmups, 240 retained-token comparisons and 240 timed SSE-text comparisons). The independent auditor checks actual saved arrays/text, counts, condition identities, resource metadata and timing arithmetic. All children and the coordinator exit0. UID-scoped Pod cleanup exits0; a fresh cluster query confirms the owned Pod is absent.

Whole/stream references match4/4 for each model. Native equivalence is separate. At input2048, output128 per request, c4 (512 output tokens total):

| Model | Native tok/s | Whole tok/s | Stream tok/s | Native/PD references equal |
|---|---:|---:|---:|---:|
| qwen06b | 110.71 | 47.75 | 63.00 | 2/4 |
| qwen4b | 21.55 | 15.35 | 17.16 | 2/4 |
| gemma | 80.97 | 43.08 | 43.52 | 4/4 |
| koni | 12.41 | 10.45 | 10.86 | 4/4 |

These are single-batch measurements, ordered native then whole then stream. They do not establish a stable speedup or device DMA overlap. Native remains faster in this cell for all four models.

Qwen0.6B differs from native at the second token for both512-input cases, before EOS; both2K cases match. Qwen4B's32-token cases match, but128-token cases first differ at index50(input512) and61(input2048), before either configured EOS. Gemma and KONI references match4/4 across all three routes here. Gemma reaches EOS at index2 in all four references; fixed-length suffix equality is not answer-quality evidence. Prior quality and numerical counterexamples remain unresolved.

The single-client2K/128 comparison localizes most added PD latency before the first SSE. For Qwen4B, native/whole/stream first-SSE times are1.022/3.411/2.534s while estimated TPOT is38.77/38.58/38.60ms. Four-model details are in `../evidence265/atom-mid201-single-client-latency.json`. SSE can coalesce tokens; these timings do not separate producer compute, conversion, transport, import or scheduling.

## Reproduction and verification

Use the archived benchmark267_deadline.py and atom_server265_http.py configurations with context9216, compiled producer width8192, caches off, one ATOM lane, maximum output128. Qwen and corrected Gemma use three KV blocks; KONI uses one. Gemma uses the research global225 corrected artifact, not a generally supported export. Small config hashes and compiled filenames/sizes are retained in `../evidence265/atom-mid201-artifact-metadata.json`; these are not full compiled-binary hashes. Exact source hashes are in the summary.

```bash
"$PYTHON" "$HARNESS" --model "$MODEL" --mode "$MODE" --server-config "$CONFIG" --lengths 512 2048 --outputs 32 128 --concurrency 1 4 --slots 1 --repeats 1 --startup-timeout 1800 --request-timeout 600 --output "$NEW_RESULT"
# MODE=native for ATOM alone, pd for whole/stream.
# Stream config adds --stream-layers; add --reference "$WHOLE_RESULT" to the harness.
python3 evidence265/audit_atom_mid201.py "$ARTIFACT_DIR"
python3 evidence265/selfcheck_atom_mid201.py "$ARTIFACT_DIR"
```

Site-local configs, raw results and the queue retain exact argv, checkpoint paths, source/config hashes, Pod UID and exits. Raw names are mid201-atom-final-{model}-{native,whole,stream}.json; queue is mid201-atom-final1030-queue.json. The auditor also uses retained artifact/EOS metadata and optional historical references. Missing historical controls are recorded as unavailable, never as matching. Reproduction requires access to these research artifacts; site-local paths and logs are not published.

Ten auditor selfchecks cover corrupt warmup/token/SSE data, missing conditions/raw, bad resource/timing metadata, duplicate arms, absent historical controls and request/aggregate output-length separation. Correctness checks use actual arrays rather than success flags alone.

## Limits and next work

This completes this cache-off intermediate-length B1 comparison only. It does not certify cache-on behavior, B4 support, native vendor engine availability or general answer quality. Supported Gemma export and Qwen4 B4 integration remain separate owner work. Short/long input128/8192 at c1/c4 still need matched ATOM controls.

Existing prefill_s includes source computation plus conversion/SHM publication. Subprocess transfer_s is left at zero, so it cannot establish zero transfer cost. Future profiling must retain separate source/publication/receiver/import/first-emit intervals and account for overlap rather than summing concurrent durations.
