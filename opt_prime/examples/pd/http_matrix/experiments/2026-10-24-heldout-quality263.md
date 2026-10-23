# Held-out structured answers across four model families

- **Date**: 2026-10-24
- **Task**: Structured answer compatibility (aicomp #263)
- **Host / device**: GPU and RNGD+ATOM roles; hardware results pending
- **Commit**: evaluator 7b2fd19, draft PR264 (base f9ce6e8); production runtime 32ce5d5

## Question

Do token differences change structured-answer correctness across native GPU, RNGD→GPU, native ATOM, and RNGD→ATOM? Models: Qwen3-0.6B, Qwen3-4B, KONI-8B, Gemma3-1B. Qwen2.5 excluded at user request.

## Setup

Preregistered before new inference: seed 26320261024; three tasks (addition, passkey retrieval, Korean table lookup), four variants, short/~2048/~8192 input tokens = 36 cases per model. Same serialized chat-template input IDs for every route, thinking disabled, greedy, normal EOS, output cap 128. Retrieval fact is in the middle of deterministic filler. Context sizes include the chat template. No answer-conditioned prompt edits.

Exact string answer in a parsed JSON object is scored separately from the one-key JSON format requirement. Duplicate JSON keys, non-string answers, trailing prose, and malformed JSON fail. This is structured-answer adherence, not a universal semantic scorer. Retain complete responses and token IDs; compare paired correct→wrong, wrong→correct, and exact tokens independently. Any transport or inference failure is retained and stops the route; do not silently exclude it. Fixture hashes identify paired inputs.

## Command

```bash
# In the isolated aicomp worktree:
python -m pytest opt_prime/tests/test_heldout_quality.py -q
# Create each fixture file once, before inference:
python opt_prime/examples/pd/http_matrix/heldout_quality.py fixtures --model "$MODEL" --output "$FIXTURE"
python opt_prime/examples/pd/http_matrix/heldout_quality.py run --fixtures "$FIXTURE" --endpoint "$ENDPOINT" --route "$ROUTE" --output "$RESULT"
```

Local Python: /home/gspark/venvs/pt213/bin/python. Full-suite environment: OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_DEVICE_BACKEND_AUTOLOAD=0.

## Results

- Test-first: 9 failures for missing evaluator, then 9 passed in 0.43 seconds.
- HTTP boundary test caught missing request ID: 1 failed/9 passed before fix, then 10 passed.
- Full suite collected before that last test: 533 passed, 15 skipped, 253 subtests in 160.17 s; 13 warnings from existing pinned-memory deprecations and unfused flex attention.
- All 144 input cases frozen and SHA256 manifest saved before inference. GPU paired runs started; no new model accuracy or performance claim.
- Existing Qwen3-4B/KONI exact-token failures remain open (#256/#261).

Raw logs: /tmp/quality263-red.log, /tmp/quality263-full.log; fixtures/results under analysis/serving-matrix-201/quality263-*.

## Conclusion

This small synthetic set can expose paired regressions but cannot establish general model quality or statistical equivalence. The compatibility preference question remains unanswered; exact token failures are not waived.

## First completed paired run

Qwen3-0.6B, 36 paired cases: nativeGPU32correct/36validformat; RNGD→GPU34correct/36validformat. Correct→wrong0, wrong→correct2; exacttoken34/36. This is a structured synthetic task result, not a general model quality advantage. Raw: quality263-qwen06b-{native-gpu,rngd-gpu}.jsonl and quality263-qwen06b-gpu-summary.json under analysis/serving-matrix-201. Other models remain in the active sequential queue.

## ATOM execution

Owned quality263 Pod allocated one ATOM and one RNGD through device plugins; no existing Pod modified. All36 stored prompts are re-tokenized with the remote model tokenizer and asserted equal to frozen input IDs before inference. B1, context9216, cacheoff, normalEOS, max128output; nativeATOM and liveRNGD→ATOM run sequentially with existing compiled artifacts. Runtime3e16c33; kv_blocks3 except KONI1; Gemma global225 artifacts. Queue deletes only its own Pod in finally.

Local exact orchestrator commands:
```bash
/home/gspark/venvs/pt213/bin/python analysis/serving-matrix-201/run_quality263_gpu.py > analysis/serving-matrix-201/quality263-gpu-queue.log 2>&1
/home/gspark/venvs/pt213/bin/python analysis/serving-matrix-201/run_quality263_atom_queue.py > analysis/serving-matrix-201/quality263-atom-queue.log 2>&1
```
Per-route argv/environment and raw HTTP outputs are retained; first-stage ATOM tokenizer equality passed. No final ATOM scores yet.

Qwen3-4B GPU pair completed:36/36 answers and36/36 JSON format on both routes; exacttoken36/36; correct→wrong0 and wrong→correct0. Prior long-free-generation mismatch in#256 remains open.

## GPU pair completion and provenance

All288 GPU-route responses completed, no HTTP/inference failures; owned local GPU/RNGD processes exited and devices idle.

| Model | NativeGPU structured score | RNGD→GPU score | Exact token pairs | Correct→wrong / wrong→correct |
|---|---:|---:|---:|---:|
| Qwen3-0.6B |32/36|34/36|34/36|0 / 2|
| Qwen3-4B |36/36|36/36|36/36|0 / 0|
| KONI-8B |31/36|31/36|35/36|0 / 0|
| Gemma3-1B |0/36|0/36|32/36|0 / 0|

Gemma's zero score measures strict requested JSON/string-answer adherence, not zero content accuracy: code fences are emitted by both routes. Manual comparison of all four differing pairs: three differ only in whitespace; one Korean lookup changes native`null` to correct`춘천` on RNGD→GPU. The frozen primary scorer is unchanged. KONI's one changed pair differs only in JSON whitespace. Historical long-output strict-token failures remain open.

ATOM partial completion: Qwen0.6 native33/36 and RNGD33/36 structured answers,36/36 exact token pairs; Qwen4 native36/36 and identical to its GPU outputs. Qwen4 RNGD→ATOM and both KONI/Gemma ATOM routes remain pending.

Both hosts' complete source weight shards, config.json, generation_config.json and tokenizer.json were hashed: all21 files (~27.67GB per host) SHA256-identical across four models. Source revisions and individual hashes are retained in quality263-{local,remote}-model-hashes.json and quality263-model-identity.json. This checks checkpoint source identity, not compiled-artifact internals.

Auditor summarize_quality263.py verifies fixture-file hashes, case/input-hash correspondence, response fields, scores recomputed from raw text, and paired transitions. Pending routes remain explicitly not_collected/starting; they are not treated as passes.

Qwen3-4B allfour routes completed: each36/36 structured answers and36/36 format, every paired token sequence identical acrossGPU/RNGD/ATOM. Remaining ATOM routes: KONI and Gemma. This does not erase the prior different long-free-generation fixture failure#256.

## Full pairwise audit (primary prompts/scorer unchanged)

Compare all six unordered route pairs, not only pairs anchored on nativeGPU. This exposes a Qwen0.6 counterexample: passkey-3-2048 is correct`SBYZEGE7` on RNGD→GPU, but nativeATOM and RNGD→ATOM emit`SYBZEGE7`. Thus each ATOM route has one correct→wrong transition relative to RNGD→GPU, despite no such transition relative to nativeGPU (which also answered this case incorrectly). Do not claim universal cross-route quality equivalence or infer the failing operator without matched-KV controls.

KONI allfour routes completed:31/36 structured answers each; nativeGPU/nativeATOM36/36 exacttokens; RNGD→GPU/RNGD→ATOM36/36 exacttokens. Native vsRNGD pair35/36; the differing text only changes JSON whitespace. Gemma nativeATOM completed with code-fenced responses (strictscore0); RNGD→ATOM remains running.

## Final completion

All576 responses (4models×4routes×36cases) completed with normal EOS and no length-capped outputs or HTTP/inference failures. Frozen fixture hashes and all returned case/input hashes verified; exact parsed-answer scores recomputed from raw text. OwnedPod quality263 deletion independently confirmed absent.

| Model | nativeGPU | RNGD→GPU | nativeATOM | RNGD→ATOM |
|---|---:|---:|---:|---:|
| Qwen3-0.6B |32/36|34/36|33/36|33/36|
| Qwen3-4B |36/36|36/36|36/36|36/36|
| KONI-8B |31/36|31/36|31/36|31/36|
| Gemma3-1B |0/36|0/36|0/36|0/36|

These are strict structured-answer adherence counts, not unrestricted semantic accuracy. Gemma emits Markdown code fences on every route. Its final ATOM native/PD pair is token-exact36/36; GPU→ATOM comparison still changes content: korean_table-0-2048 yields correct`춘천` on RNGD→GPU but`null` on RNGD→ATOM and nativeATOM. This is in addition to the Qwen0.6 passkey counterexample. Neither counterexample can be blamed on a particular operator or transport without matched-KV controls.

NativeATOM/RNGD→ATOM exact pairs: Q06 36/36, Q4 36/36, KONI35/36 (JSON whitespace only), Gemma36/36. All six route-pair comparisons, per-context counts and changed raw texts are in quality263-summary.json. No universal quality-equivalence or performance-speedup conclusion. Numerical issues#233/#256/#261 and actual within-request streaming#265 remain separate work.
