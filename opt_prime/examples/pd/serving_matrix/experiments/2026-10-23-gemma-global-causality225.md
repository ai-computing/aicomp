# Gemma global text-prefill causality and corrected handoff

- **Date**: 2026-10-23
- **Task**: aicomp #225; upstream optimum-rbln #831
- **Host / device**: RNGD+ATOM host, exclusively owned ATOM Pod; Rebel 0.11 / Optimum 0.11.0.post1, B1, context 9216, k3, hybrid cache.
- **Commit**: exp/225-gemma-global-causal based on Gemma producer 90e4a7d, uncommitted diagnostic scripts; upstream fix/gemma-causal-prefill d6492fa plus two global-attention flag edits and tests.

## Question

Does global masked text prefill incorrectly allow future tokens with explicit position IDs, and does correcting it restore RNGD-to-ATOM continuation parity?

## Setup

Pinned unsloth/gemma-3-1b-it revision 5b11413a10db4e486ef16a20101fd028f8f2499c. Existing artifact includes the sliding-attention fix only. Keep the first 64 tokens of a 128-token prompt unchanged and perturb only its suffix; read prefix state from all 52 KV slots. Candidate changes AttentionOp and FlashAttentionOp bidirectionality to phase == image_prefill. Build into a separate artifact directory using a process-local compatibility patch; shared SDK unchanged. Actual prefill_chunk_size is 256 (not 128).

## Command

Commands run in the dedicated experiment scratch directory, through the existing container bootstrap:

```bash
bash run-gemma-causal225.sh --cache-dir atom-artifacts --exports rngd-gemma-1b-128-223-fixed.pt --output gemma-global-causality225-baseline.json
bash run-gemma-causal225.sh --global-causal --cache-dir atom-artifacts-global225 --exports rngd-gemma-1b-128-223-fixed.pt --output gemma-global-causality225-fixed.json
bash run-gemma-atom223.sh --cache-dir atom-artifacts-global225 --exports rngd-gemma-1b-128-223-fixed.pt rngd-gemma-1b-2048-223-fixed.pt rngd-gemma-1b-8192-223.pt --output gemma-global225-handoff.json
```

## Results

- Baseline: slots 0–11 (layers 0–5) preserve prefix state; slots 12–51 change roughly 99% of encoded entries when only future tokens change. Existing native output is not a valid causal baseline.
- Operator-boundary tests: installed global methods 2 failed / 46 passed; exact patched dev methods with installed SDK globals plus sliding tests 60 passed. This is not a full SDK suite.
- Initial fresh compile completed, but the same-process diagnostic failed its dict/float32 metadata assertion. Saved configuration metadata is dict/float32. Reloading the same artifact passed without changing the assertion or recompiling. The in-memory/export representation difference still needs inspection before generalizing the checker.
- Reloaded candidate: all 52 prefix slots remain bit-exact under suffix perturbation.
- Corrected native versus RNGD-imported ATOM decode: 32/32 tokens match at each of 128, 2048 and 8192 input tokens. Each case also passes all 52 raw-state roundtrip/readback checks and exact native-restoration decoder logits.

Evidence: ../evidence225/gemma-global-causality225-baseline.json, gemma-global-causality225-fixed.json, gemma-global-causality225-fixed.log (initial assertion failure), gemma-global-causality225-reload.log, gemma-global225-handoff.json and .log.

## Conclusion

The corrected compiled global-attention path passes the causality fixture and removes the observed native/imported continuation mismatch at all three lengths. This does not establish broad model quality, production hybrid-cache scheduling, transfer/compute overlap, or end-to-end serving speed. Upstream/downstream PR publication and broader prompt coverage remain work in progress. Four-model scope remains Qwen3-0.6B, Qwen3-4B, KONI-8B and Gemma3-1B.
