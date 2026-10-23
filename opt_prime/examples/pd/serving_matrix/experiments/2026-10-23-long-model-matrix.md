# Long-input matrix extension

- **Date**: 2026-10-23
- **Task**: aicomp #201, capacity hypothesis #204
- **Host / device**: RNGD+ATOM host, one allocated device each; local reference A10080GB PCIe. Furiosa2026.4, torch2.11 CPU/rebel-compiler0.11.0 on ATOM; GPU torch2.10+cu128 / Transformers5.1.
- **Commit**: faba87bb13579fb559465b077e84a74f4282f365 plus experimental_long_width variant; portable ATOM importer f4ebe93 (PR206)

## Question

Do results extend to longer input and the selected Qwen3 models? Selected full matrix is Qwen3-0.6B, Qwen3-4B, KONI-8B, Gemma3-1B; Qwen2.5 is excluded.

## Setup

Same token fixtures as PR203, prefix cache disabled. Qwen3-4B revision1cfa9a7208912126459214e8b04321603b3df60c. Isolated trusted-source variant raises the experimental width guard from512 to2048, without changing serving defaults. RNGD fusion4/Mesh2; ATOM B1/context9216/three full-context KV blocks. GPU BF16 reference.

## Command

Within the allocated environment and the PR203 wrapper/bootstrap:

```bash
bash run-rngd201.sh --model Qwen/Qwen3-4B --width 2048 --experimental-long-width --output rngd-qwen3-4b-2048
bash run-pd-atom201.sh --exports rngd-qwen3-4b-2048.pt --output rngd-atom-qwen3-4b-2048.json
bash run-atom201.sh --model Qwen/Qwen3-0.6B --cache-dir atom-artifacts --output atom-qwen3-06b.jsonl
bash run-atom-gemma201.sh --model unsloth/gemma-3-1b-it --cache-dir atom-artifacts --output atom-gemma3-1b.jsonl
```

The Gemma launcher applies the process-local causal-prefill fix before compiling new artifacts (upstream PR827); it does not modify the shared SDK. Wrapper paths and resource allocation remain site-local. Exact wrapper environment follows PR203.

## Results

Qwen3-4B RNGD2K compile/init201.993s; first prefill1069.907ms; two warm runs575.295/570.442ms. Repeated KV/logits identical. GPU32-token decode matches GPU-native generation;72 cache slots read back exactly. BF16-reference relative errors K3.41%, V11.15%, logits2.40% remain; matching tokens do not prove numerical equivalence.

RNGD2K to ATOM: original and candidate portable imports have exactly matching decode logits;72 cache slots per importer read back exactly; both32-token outputs match ATOM-native. Single import samples885.86/540.00ms are not the alternating performance benchmark (see #205 record).

Qwen3-0.6B ATOM: all16 fixed-output stability runs complete through8192 input plus128 output tokens. All8 repeat pairs identical. Against matching-input GPU B1 fixtures, first token8/8, full output4/8 exact. Warm TTFT approximately12ms at128,47–53ms at512,194ms at2048,1115–1224ms at8192. Output mismatch at512/2048 requires numerical/quality analysis; do not mark this as cross-backend parity.

Gemma3-1B long-input compilation completed, but the first serving request failed because the batcher supplies global block_tables without local_block_tables required by hybrid attention. Tracked as #207; no Gemma pass claimed. KONI long-input ATOM and full HTTP/cache/concurrency matrix remain outstanding.

Raw evidence: analysis/serving-matrix-201/{rngd-qwen3-4b-2048.jsonl,rngd-gpu-qwen3-4b-2048.json,rngd-atom-qwen3-4b-2048.json,atom-qwen3-06b.jsonl,atom-qwen3-06b-summary.json}.

## Conclusion

The2K RNGD capacity hypothesis works for this4B fixture and transfers correctly to both decoders. There is no8192-token RNGD result, no demonstrated overlap, and no end-to-end serving win yet. Larger prefixes increase both compute and staging costs; model/backend coverage is still incomplete.
