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
