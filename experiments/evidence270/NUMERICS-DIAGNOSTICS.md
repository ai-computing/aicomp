# Qwen3-4B diagnostic progression and limits

The native B4 parity failure remains unresolved. These isolated exports are diagnostic experiments, not serving fixes or recommended defaults. All use the SDK environment and PR266 prerequisites in QWEN4-LANE-REPRO.md. Shared SDK files and engine code were unchanged.

| Experiment | Result | Limit |
|---|---|---|
| Supported output_hidden_states export | Both original token splits reproduce. At all four observed positions, embeddings are lane-equal; first difference at boundary1, after layer0. | Recorded logits differ from the original uninstrumented graph. |
| First-layer physical KV readback | Sixteen K/V snapshots show identical initialized ranges across all four blocks. Changing only block3 token0 changes1024 K and1024 V elements; restoring the prompt restores equality. | Sampled first-layer stored KV only. Does not identify a compute operator. |
| First-layer module hooks | Input normalization stays equal; first difference at attention output, including output projection. Prior instrumented logit records match. | Extra graph outputs may change optimization. MLP receives differing inputs, so no MLP-origin conclusion. |
| Packed attention-internal outputs | Export falls back to jittrace, then DEVICE_GRAPH_CONVERSION failure. | No runtime numerical result; cause of compiler rejection unidentified. |
| Raw attention-internal decode outputs | Compiles; all observed first-attention internals are lane-equal.2048/index13 split remains,512/index50 split disappears. Later retained boundary11 differs. | Instrumentation materially changes numerical behavior. Not a fix and not proof the original attention kernel is correct or incorrect. |

Compile times: supported hidden153.837882s; module hooks154.000048s; raw attention152.587502s. All owned Pods were cleaned up, including the failed compile attempt. No throughput or semantic-quality gain is claimed.

## Interpretation

Both module-hook and raw-attention exports log two torch.export→jittrace fallbacks; the supported hidden-state export logs none. Fallback alone does not distinguish the two instrumented outcomes. Any further attribution needs matched compiler controls or a smaller isolated replay with the same numerical failure. Do not adjust greedy tie-breaking to hide a changed logit order.

Raw attention diagnostic slots0..10 are embedding, Q/K/V projections, Q/K normalization, rotary Q/K, attention V input, core attention output and output projection. Slots11..36 retain the ordinary hidden-state boundaries. Raw native tensor shapes are preserved only in decode; prefill keeps the original fixed-size hidden-output interface. Layer-module diagnostic slots0..5 have separate labels in their result. Neither layout is a general serving hidden-state API.

## Reproduction

Use a fresh task directory, the compact fixture from QWEN4-LANE-REPRO.md, the existing bootstrap/validation helpers and the recorded SDK versions. Allocate only the owned ATOM. Builders create distinct atom270-artifacts-* cache roots and reject overwrites. Run each compiler and replay in separate Python processes:

```bash
"$ATOM_PYTHON" build_atom270_hidden.py
"$ATOM_PYTHON" container_bootstrap270.py diagnose_atom270_hidden.py
"$ATOM_PYTHON" container_bootstrap270.py diagnose_atom270_firstkv_control.py
"$ATOM_PYTHON" build_atom270_layerprobe.py
"$ATOM_PYTHON" container_bootstrap270.py diagnose_atom270_layerprobe.py
"$ATOM_PYTHON" build_atom270_attentionraw.py
"$ATOM_PYTHON" container_bootstrap270.py diagnose_atom270_attentionraw.py
```

Compact audits and source hashes are included. Raw token/probe arrays and failed compile log remain outside Git, linked by full hashes and research event IDs in diagnostic-evidence-manifest.json. Remote archive synchronization is unverified. Packaging does not constitute a new hardware run, nor a full repository test suite.

## Required next evidence

1. Confirm the original numerical failure under a controlled tracing/compiler comparison without changing observation outputs.
2. If using isolated replay, preserve real inputs/KV and establish the original failure before changing an operator.
3. Validate any candidate on both fixtures, all lane placements, free generation and held-out quality, then matched four-model serving and fault tests. Keep B1 defaults and B4 parity failure until these checks pass.
