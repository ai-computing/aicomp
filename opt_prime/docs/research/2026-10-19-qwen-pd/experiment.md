# Qwen actual heterogeneous-NPU PD extension

- Date:2026-10-19 UTC
- Task:PDDISAG-6
- Base:aicomp-integration ca9b695, feat/optimus-heterogeneous-pd; dirty implementation recorded on completion.
- Scope: Qwen2.5-0.5B first, then1.5B if device/graph checks pass. Actual RNGD3 and explicitly selected idle ATOM.

## Protocol

1. Geometry/support contract, QKV biases, tied lm_head and RoPE theta.
2. RNGD prefix graph: compare per-layer K/V against FP32 and next-token top1.
   Initial engineering gate max per-layer relative L2<0.1 and top1 equality;
   this is bring-up, not task-quality certification.
3. ATOM resident decode after host-admitting RNGD prefix:32-token teacher-forced
   FP32 agreement>=29/32 before P2P export. Full logits finite. No P2P in this gate.
4. Reuse live-allocation exporter and checked DMA backend; no new driver ABI.
   Head segment alignment depends on head_dim:HD64 requires32 tokens per4KiB,
   HD128 requires16. Never reuse KONI's fixed64cache/8head constants for Qwen.
5. Compare every output logit to same-decoder host admission after P2P, preserve
   owners on uncertain DMA/decode completion. Separately measure FP32 quality.

## Device changes

Initially rbln2 was in group1; rbln3 remained host-visible2. A small fenced buffer
probe confirmed host-visible2 opened /dev/rbln3. Before compilation, rbln3 moved
to group2 and later VLLM PID818191 occupied14.8GiB. No groups or external processes
were changed. Physical rbln0 was conservatively rejected because SMI context ID0
is ambiguous across groups. Selected physical rbln1 in host group0, verified by
actual /proc/self/fd device node. User asked to use available ATOM and retry.

## Progress and failures

- RNGD r1/r2 failed after graph execution: returned permuted views retained their
  base shape. Explicit packed K/V graph output fixed the shape contract in r3.
- RNGD05 r3 passed, per-layer relativeL2 maximum0.05889, next token220 matches
  FP32, logits relativeL2~0.03147; prefix31, compiled width32,24layers.
- ATOM buildr1 refused changed group mapping; buildr2 refused ambiguous active ID;
  buildr3 on physical rbln1 compiled successfully in29.37s.
- Subsequent execution/PD results will be appended; all failures retained under
  experiments/qwen_pd_extension_20261019.

## Commands

Launchers: aicomp-integration/opt_prime/examples/pd/build_qwen_atom.py,
validate_qwen_rngd.py, validate_qwen_atom.py. Local models pinned to snapshots
7ae557604adf67be50417f59c2c2f167def9a775 (0.5B) and
989aa7980e4cf806f80c7fef2b1adb7bc71aa306 (1.5B).
PYTHONPATH=aicomp-integration/opt_prime:hetero-npu-p2p/src:rngd_llmd/src.
RNGD python=/home/gspark/venvs/furiosa-2026.4/bin/python.
ATOM python=experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv/bin/python.

Device groups follow the vendor model documented at
https://docs.rbln.ai/latest/software/system_management/container.html;
this extension does not mutate host groups or add a container workaround.

## Final direct-PD checkpoint

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
