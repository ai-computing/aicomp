# Four-model actual ATOM streaming fault isolation

- **Date**: 2026-10-24
- **Task**: Within-request layer streaming (aicomp #265), PR266
- **Host / device**: RNGD+ATOM host; one allocated RNGD and one ATOM, B1; existing context9216 artifacts. Same SDK environments as the long-ATOM streaming record.
- **Commit**: production runtime33d6cb0; evidence worktree5978c35 before this update. Runtime file hashes retained in each model JSON; driver SHA256 in stream265-atom-fault-driver-manifest.json.

## Question

Can a partial stream be cancelled or its producer killed without publishing incomplete prefixes, poisoning the ATOM owner, or changing a surviving decode?

## Setup

Isolated deployed package `stream265-code`, sourcewidth8192, one source slot, compiled decode context9216/B1. Prompts2048/8192. Cancel/reuse generates16tokens; active-survivor isolation generates512tokens. Qwen models enable prefix caching with3blocks; Gemma uses the corrected global-attention artifact and cacheoff/3blocks; KONI cacheoff/1block. All tests use the same-route no-fault reference. Device allocation is discovered inside the owned Pod, never assumed from another Pod.

## Command

Run inside the allocated Pod from the experiment directory, using the archived wrapper and driver:

```bash
bash run-stream265-fault-atom.sh --label qwen06b --model Qwen/Qwen3-0.6B --family qwen --kv-blocks 3 --cache-dir /home/gspark/workspace/codex-serving-matrix-201/atom-artifacts
bash run-stream265-fault-atom.sh --label qwen4b --model Qwen/Qwen3-4B --family qwen --kv-blocks 3 --cache-dir /home/gspark/workspace/codex-serving-matrix-201/atom-artifacts
bash run-stream265-fault-atom.sh --label gemma --model unsloth/gemma-3-1b-it --family gemma --kv-blocks 3 --cache-dir /home/gspark/workspace/codex-serving-matrix-201/atom-artifacts-global225
bash run-stream265-fault-atom.sh --label koni --model KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024 --family koni --kv-blocks 1 --cache-dir /home/gspark/workspace/codex-serving-matrix-201/atom-artifacts
python3 evidence265/audit_stream265_atom_faults.py evidence265
```

Wrapper records the interpreter and environment paths. Executed driver SHA256 is7ec939b50a94ddce9aee21a6489d2e91a88f11d298e97659bf94df7bf8423c5b. Keep matching model artifacts and source factories from the main streaming experiment.

## Results

| Model | Cases passed | Replacement tokens exact | Survivor tokens exact | Survivor tokens emitted at producer death |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 3 | 16/16 | 512/512 | 13 |
| Qwen3-4B | 3 | 16/16 | 512/512 | 6 |
| Gemma3-1B | 3 | 16/16 | 512/512 | 10 |
| KONI-8B | 3 | 16/16 | 512/512 | 5 |

Cases: cancel after partial import then reuse; kill producer after partial import before admission; kill a prefetched producer while another request decodes. Prefix-index assertions are in the driver. All source-death owners remain unpoisoned and idle after draining. The prefetched victim has no admitted slot. All child/copy exit codes0; full auditor validates retained token arrays, triggers, ownership and completion. Raw `evidence265/stream265-*-fault-atom.json` and matching logs retain results; header output_tokens16 refers to the initial case, while survivor arrays contain512tokens.

## Conclusion

All12 actual fault cases pass on the measured four-model paths. This proves the recorded streaming lifecycle cases, not native-backend numerical equivalence, hybrid prefix/host restoration, device DMA overlap, or arbitrary multi-lane behavior. The owned fault Pod was still present immediately after the last child finished; explicit wait-for-delete then empty namespace listing confirmed cleanup. Queue exited0. A later HTTP run uses a separate owned Pod.
