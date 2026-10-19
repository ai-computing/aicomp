# Qwen RNGD→ATOM P2P validation — 2026-10-19

This extends OptimusPrime `opt_prime.pd` on `feat/optimus-heterogeneous-pd`.
The new adapters are working-tree changes based on ca9b695 at this checkpoint;
they have not yet been merged into the llm-d feature or upstream main.
The common PDRuntime was unchanged by the extension.

## Actual-device results

| Model | Requests | Generated tokens | DMA commands | Same-decoder full logits vs host admission | FP32 teacher-forced top-1 |
|---|---:|---:|---:|---|---|
| Qwen2.5-0.5B | 16 | 320 | 768 | exact | 160/160 |
| Qwen2.5-1.5B | 16 | 320 | 896 | exact | 158/160 |

B1, context512, compiled prefix width64, input lengths32/64/41/37,
output lengths8/32, two repeats. The prompts include Korean and English chat.
FP32 checks deduplicate repeated trajectories; their denominators differ from
all generated tokens. This is a numerical diagnostic, not task accuracy.
All final DMA commands completed successfully and both workers exited0.

Devices: RNGD physical3 (logical2), ATOM physical1 (`/dev/rbln1`, visible1,
Torch logical0). Other users' devices/groups were not modified.
RNGD SDK2026.4; ATOM compiler0.11.0, torch-rbln0.2.2, driver3.0.0.
Local model revisions: 0.5B `7ae557604adf67be50417f59c2c2f167def9a775`,
1.5B `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

## Failures and corrections

Initial RNGD cache outputs exposed a permuted-view layout mismatch; graph-side
materialization and exact shape checks corrected it before any P2P run.
ATOM preflight rejected a moved group and an ambiguous active device ID;
subsequent experiments selected idle physicalATOM1 and verified its open FD.
Initial 1.5B BF16 attention scores failed the predeclared per-layer KV relative-L2
threshold0.1 (maximum0.1426626). Compiled FP32 attention scores/softmax reduced
this to0.0358222; the threshold was unchanged. This setting is now the default.
A preliminary config-generation failure involving tokenizer return type also
remains in the logs; no device experiment ran from that missing config.

Earlier 0.5B `pd05-r1` used the BF16 diagnostic mode:8requests/160tokens,
384DMA commands, exact same-decoder logits and80/80 FP32 top-1. Its original
config omitted the then-default accuracy flag; reproduce it now with
`accurate_attention=false`. Do not combine this run with the final matrix.

## Reproduction and evidence

Workspace evidence: `experiments/qwen_pd_extension_20261019/`.
Final configs: `pd05-config-r2.json`, `pd15-config-r2.json`.
Raw results: `pd05-r2/` and `pd15-r2/`, including `runs.jsonl`, `result.json`
and source/environment provenance. Offline quality: `quality-pd05-r2.json`
and `quality-pd15-r2.json`. Unit suite: `tests-final.log`,22tests passed.

From the workspace root, run the appropriate final config with:

```sh
PYTHONPATH=aicomp-integration/opt_prime:hetero-npu-p2p/src:rngd_llmd/src \
 /home/gspark/venvs/furiosa-2026.4/bin/python \
 aicomp-integration/opt_prime/examples/pd/validate_qwen_pd.py \
 --config experiments/qwen_pd_extension_20261019/pd05-config-r2.json
```

Use a new output directory in copied configs to preserve existing evidence.
Build local artifacts first using `build_qwen_atom.py`; weights and compiled
binaries are not included in Git. See the recorded config/provenance for exact
local artifact paths and arguments.

## Scope

These are finite transfer/numerical validation runs, not saturation, power, MFU
or same-resource single-NPU speedup measurements. GPU hardware is untested.
The external ATOM KV ABI remains experimental and version-pinned. Same-decoder
transfer equality is not FP32 numerical equality. Actual Qwen execution through
the existing llm-d service remains a separate pending validation.

See [extension boundaries and contributor guidance](NPU_BACKEND_EXTENSIONS.md).
