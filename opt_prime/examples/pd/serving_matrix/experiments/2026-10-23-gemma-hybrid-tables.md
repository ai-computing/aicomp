# Gemma hybrid cache ownership and request reset

- **Date**: 2026-10-23
- **Task**: aicomp #207; four-model matrix #201
- **Host / device**: RNGD+ATOM host, one isolated ATOM; rebel-compiler0.11.0, optimum-rbln0.11.0.post1, torch2.11.0+cpu, Transformers5.8.1
- **Commit**: matrix faba87bb13579fb559465b077e84a74f4282f365 plus this branch's hybrid_tables probe; deployed runtime is the existing optimus-rm snapshot

## Question

Why does the Gemma3 long-input standalone batcher fail, and can explicit local cache ownership and request mask reset restore repeat stability?

## Setup

unsloth/gemma-3-1b-it revision5b11413a10db4e486ef16a20101fd028f8f2499c. Rebuilt artifacts with process-local causal-prefill correction (upstream optimum-rbln PR827). B1/context9216/three global full-context cache blocks, prefix cache disabled, no host cache or PD. Inputs128/512/2048/8192, fixed outputs32/128 ignoring EOS, two repeats each. One local sliding cache per lane. Global blocks and local lane indices are separate ownership domains.

## Command

Using the same allocated runtime/bootstrap and environment as matrix PR203:

```bash
bash run-atom-gemma201.sh --model unsloth/gemma-3-1b-it --cache-dir atom-artifacts --experimental-hybrid-tables --output atom-gemma3-1b-hybrid-mask.jsonl
python -m unittest discover -s . -p 'test*.py'
```

The launcher executes container_bootstrap.py atom_gemma_sweep.py with the arguments above. The Gemma entry point applies the trusted installed-source causal-prefill patch before loading/compiling the model, then runs atom_sweep. No shared SDK installation or other workload is modified.

## Results

1. Original batcher: compilation completes, first request raises `Both block_tables and local_block_tables must be provided or neither of them must be provided.` No successful measured request.
2. Explicit local lane tables alone: first128-token/32-output request completes (TTFT29.52ms); second repeat fails token equality. This partial fix is rejected.
3. Local tables plus per-request decode mask reset: all16 runs complete;8/8 repeat pairs exact. Warm TTFT25.3–25.4ms at128,49.4ms at512,199.9–202.0ms at2048,816.4–822.1ms at8192.

SDK evidence: hybrid `is_external_block_tables` requires both tables. For external tables, chunked prefill skips copying its prompt mask to `dec_attn_mask`. With `use_position_ids`, decode only marks the next position and therefore does not reconstruct the prompt mask or clear previous request positions. The probe initializes the owning lane's prompt mask and leaves other lanes untouched.

Five local tests pass (two ownership/mask/rejection tests plus three input fixture tests). The mask test failed before adding reset and passes afterward. It covers mismatched lane/global-block indices and untouched other-lane mask; it is not multi-lane hardware validation.

Against GPU B1 with equal input hashes: first token8/8 equal, full output0/8 equal, EOS-normalized output0/8 equal. Divergence begins at generated positions1 or2. Both give Paris on these fixtures, but answer similarity does not establish numerical or quality parity. GPU uses Transformers5.1.0/torch2.10.0+cu128, so runtime/version effects remain to isolate.

Raw results and comparison summary: results/atom-gemma3-1b-hybrid-mask.jsonl and results/atom-gemma3-1b-summary.json.

## Conclusion

The standalone probe restores repeat stability and identifies two missing engine responsibilities. It is not general hybrid-cache support: prefix/host caching and PD are explicitly rejected because sliding-state restoration is not implemented. SDK-managed same-artifact comparison, full logit analysis and engine integration remain necessary. Keep #207 open.

## Same-graph SDK comparison

Ran `bash run-hybrid-sdk207.sh --cache-dir atom-artifacts --output gemma-sdk-owned207.json` (launcher substitutes check_hybrid_sdk.py in the same bootstrap environment). Four lengths128/512/2048/8192,32 output tokens each. All full decode-logit vectors match exactly between SDK-owned tables and explicit external tables: max absolute difference0.0 in every case; all output tokens equal. This isolates the remaining GPU discrepancy from the external table/mask bridge for these cases. It does not establish prefix cache or PD correctness. Results: results/gemma-sdk-owned207.json.

Additional GPU FP32 baseline: at8192 the32-token output equals ATOM; at128/512/2048 differences remain at positions1/2/1. Artifact config SHA256:204d77f67d2566125248a0cc3c74b6a40da7bf718632a14eb70726f4ad47b170. Full weight/compiler graph provenance remains separate from this config hash.
