# Native RNGD four-model coverage and KONI batch dependence

- Date: 2026-10-25
- Related: #201, #274; numerical research-engine drift is tracked separately in #261.
- Evidence base: main fd89265; task-specific measurement sources and hashes in evidence274/sources.json. No production engine changes.
- Hardware: one allocated RNGD on the RNGD+ATOM host; no ATOM or GPU allocated for native tests. Explicit assigned-device selection; owned Pods cleaned after the completed matrix and quality runs.
- SDK: furiosa-llm2026.4.0 (b1baffa75), compiler2026.4.0 (ae27d44cd8).

## Problem and findings

The installed native FXB registry accepts the exact KONI checkpoint, but rejects the tested Qwen3-0.6B, Qwen3-4B and Gemma3-1B configurations. This is a version/configuration limitation, not a claim that other SDKs or research engines cannot run those models. Exact model revisions/config hashes and preflight exit statuses are in evidence274/native201-four-model-support-summary.json.

KONI's native build completed with105 successful kernel entries and zero failures. Wall time98m44s; summed compiler task time196m28s at concurrency2 is not wall time. FXB223,526,992bytes, SHA256255e0929e98d85cc244846a6e6a1f76dc5d9d78167bc13c4e5e59eb4c6555e41. Metadata: LlamaForCausalLM,32layers,hidden4096,8KVheads,TP8; runtime context explicitly limited9216. Frozen checkpoint0172bf7e73977ae213613529f3d2d5400a33f678.

All24 HTTP conditions executed (input128/512/2048/8192 × output32/128 × client concurrency1/4/16), with344 responses:8warmups,168 nonstream token checks,168 timed SSE responses. Exact parity failed, so this is not a passing validation:

| Client concurrency | Token mismatches / checks | SSE text mismatches / checks |
|---:|---:|---:|
|1|0/8|0/8|
|4|13/32|13/32|
|16|50/128|50/128|

Each output was compared to the same server's single-request warmup for identical input IDs and output length. Temperature0, ignore_eos=True, prefix cache disabled. This standalone native result involves no external PD transport. It does not identify the underlying numerical/compiler cause. Client concurrency is not physical lane occupancy.

Separate sequential frozen synthetic quality yielded31/36 exact answers,34/36 exact JSON formats,36 stop completions and zero request errors. Explicit stop IDs128001/128009 preserve the prior stopping policy. The five strict answer failures were retained unchanged. This does not prove concurrent quality or general task accuracy. Raw hashes and scoped summaries are included; full raw responses/logs remain in the research artifact store by basename/hash rather than including site-specific paths.

## Exploratory timing and limitations

One timed batch per condition; no confidence interval or equivalent-output speedup claim. At output128/c16, throughput for input128/512/2048/8192 was836.849/583.310/270.309/80.455 output tok/s. At8K/c16, median first SSE11.616s and E2E23.947s; at8K/c1,0.862s and3.031s. First SSE may coalesce output and is not a kernel-level TTFT measurement. Initialization and warmup/correctness checks are excluded.

The original measurement client's missing JSON Content-Type produced415 before inference. The failed initial artifact was preserved. A strict local HTTP fixture reproduced415; adding the JSON header passed44 synthetic responses and the readiness/request profile checks. The full matrix above used the corrected header. An initial host-user attempt to hash the root-created FXB failed permission checks; inspection succeeded inside the owned Pod without changing file permissions.

## Reproduction

```bash
# Frozen local snapshot; paths and assigned device supplied by the operator.
"$FURIOSA_PYTHON" -m furiosa_llm.cli.fxb_main build "$KONI_SNAPSHOT" "$FXB" -tp 8 --max-model-len 9216 --concurrency 2 --compiler-cache-dir "$COMPILER_CACHE" --build-report
furiosa-llm serve "$KONI_SNAPSHOT" --served-model-name "$KONI_SNAPSHOT" --fxb "$FXB" --devices "$ASSIGNED_RNGD" --max-model-len 9216 --no-enable-prefix-caching --host 127.0.0.1 --port "$PORT"
# The harness launches/cleans its own server; SERVER_CONFIG contains the argv above,
# with exactly one standalone {port} placeholder and resource metadata.
"$FURIOSA_PYTHON" experiments/evidence274/benchmark201_native_http.py --model "$KONI_SNAPSHOT" --mode native --api-profile furiosa --server-config "$SERVER_CONFIG" --output "$NEW_RESULT" --lengths 128 512 2048 8192 --outputs 32 128 --concurrency 1 4 16 --repeats 1 --startup-timeout 600 --request-timeout 600
# Quality uses a separately owned server and frozen36case fixtures from #263.
"$FURIOSA_PYTHON" experiments/evidence274/native201_quality.py --fixtures "$FROZEN_KONI_FIXTURES" --endpoint "$ENDPOINT" --model "$KONI_SNAPSHOT" --output "$NEW_QUALITY_JSONL" --eos-token-ids 128001 128009
```

Server-config metadata: physical_rngd1, physical_gpus0, physical_atom0, slots_per_decoder=null, producer_cache=false, decoder_cache=false, producer_chunk=null, local_chunk=null, context9216. The environment used HF_HUB_OFFLINE=1, OMP_NUM_THREADS=4, MKL_NUM_THREADS=4 and a task-local XDG_CACHE_HOME.

Local protocol checks (synthetic fixtures, no hardware claim):

```bash
python3 experiments/evidence274/selfcheck_native201_http_profile.py
python3 experiments/evidence274/selfcheck_native201_benchmark.py
python3 experiments/evidence274/selfcheck_native201_quality.py
```

## Next experiment

#274 tests default/profile-guided-decode-batching-off/off/default on the same RNGD with128/8192input,128output,c1/4/16. Each arm has independent c1 references; cross-arm references must be compared separately. The completed result is recorded below. No serialization workaround or production default change is proposed. Four-model cross-backend validation remains incomplete.

## Profile-guided scheduling ABBA completed

Same physical RNGD/Pod, four fresh server processes in default/off/off/default order, same model/FXB/input IDs/cache settings. The only server-config difference was --no-profile-guided-decode-batching. All four arms completed6conditions/86responses each (344total) and failed exact parity:29/42 token checks and29/42 SSE text checks in every arm. The two c1reference records, per-condition token-sequence multisets and per-condition SSE-text multisets were identical across all four arms. Request identities were not assumed to map to the same physical lane. Summary and raw hashes: evidence274/native274-abba-summary.json.

This public setting did not remove the failure on the tested cases. It does not prove the general scheduler or compiler is uninvolved. The two-run mean throughput change with the option disabled ranged+0.14% to+2.56%, while A2 was slower than A1 in all six conditions. Two observations/configuration and this temporal drift do not establish a stable speedup. No default change is recommended from these data. All servers stopped and the owned Pod was UID-checked and deleted (exit0). Original numerical/compiler cause, broader native model registry support and concurrent held-out quality remain open.

## Public log-probability observation completed

168/168 responses completed,84 with top5 log probabilities,zero errors. All four arms have identical condition-level generated-token multisets; this diagnostic option did not change the output distribution in these cases. The58 divergent instrumented responses have identical generated histories to their corresponding c1reference before their first divergent step. Both competing token IDs are present in all58reported candidate pairs; the reference candidate is at least tied highest between the pair for c1, and the other candidate is at least tied highest in the concurrent response.

Examples (B1): short first divergence11: reference token11190 vs4495 has logprob -1.214444/-1.282498 at c1, versus a tie -1.254718/-1.254718 at c4. Long divergence98:18112 vs16924 is -1.169420/-1.296682 at c1 versus a tie -1.201372/-1.201372 at c4. Other observed first-divergence positions14/49(short) and37(long) show changes from a c1tie to a preference for the alternative token.

These observations support batch-dependent changes in reported candidate probabilities, rather than only a different tie-break decision on otherwise identical probabilities. They do not establish identical hidden/KV state, full raw logits, the precision used at each layer, or a particular faulty compiler/kernel transformation. Raw source hash and every compared pair are in native274-logprobs-summary.json; no output forcing or correction was applied. Controller exit0; server terminated by owner after completion (return-15). Pod cleanup confirmation is recorded separately.

Reproduce against an owned, unchanged native server:

```bash
python3 experiments/evidence274/native274_logprobs.py --endpoint "$ENDPOINT" --model "$KONI_SNAPSHOT" --source "$NATIVE_A1_RESULT_JSON" --output "$NEW_LOGPROBS_JSONL"
python3 experiments/evidence274/selfcheck_native274_logprobs.py
```

The client runs128/8192input,128output,c1/4/16 in no-logprobs/logprobs/logprobs/no-logprobs order, uses logprobs=5 and return_tokens_as_token_ids=True, retains raw responses and verifies echoed inputs/token alignment. This diagnoses probability reporting and is not a timing comparison.

## Same-server sequential/concurrent quality completed

All144 responses completed with zero request errors and finish_reason=stop. Same server, fixed model/FXB/inputs/EOS/scorer, Q1sequential→C1concurrent→C2concurrent→Q2sequential. Q1/Q2 both31/36answers and34/36formats, and all36 token sequences match between Q1/Q2. C1/C2 both32/36answers and35/36formats. Compared to Q1, C1 has2different token sequences and C2 has1. passkey-0-8192 changes from a bare correct passkey (strict format/answer failure) to the required JSON answer in both concurrent runs. passkey-0-2048 differs only in JSON whitespace in C1, with strict answer/format still correct.

Thus batch dependence can affect structured-format compliance, while most cases remain unchanged. No general quality improvement, universal degradation, or engine fix follows from these36synthetic cases. Client12-request cohorts overlap in all three strata in both concurrent runs; physical execution lanes are not inferred. Raw hashes/counts/deltas are in native274-quality-summary.json. Server stopped by owner after controller exit0.

Concurrent quality reproduction (same fixed36case fixtures/EOS; launch sequential client before and after two concurrent runs):

```bash
python3 experiments/evidence274/native274_quality_concurrent.py --fixtures "$FROZEN_KONI_FIXTURES" --endpoint "$ENDPOINT" --model "$KONI_SNAPSHOT" --output "$NEW_QUALITY_JSONL" --eos-token-ids 128001 128009
python3 experiments/evidence274/selfcheck_native274_quality_concurrent.py
```
