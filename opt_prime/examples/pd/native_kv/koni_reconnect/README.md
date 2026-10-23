# KONI RNGD → ATOM reconnection (2026-10-23)

A separate existing KONI graph/adapter was found outside the normal AtomRblnEngine
cache. Thus MODEL_FAMILY_VALIDATION.md's missing-artifact result applies only to that
standard whole-model cache; it is not a repository-wide absence of KONI support.

This finite fixture reuses `rngd_pd.engines.prefill_full.FullPrefillEngine` and
`hetero_p2p.adapters.atom_koni_external.ExternalCacheDecoder`, with the existing
B4/context512 ATOM graph. It does not change the Qwen2-only research worker gate.

## Reproduce

Use the validated isolated Pod environment from ATOM_CONTAINER_EXPERIMENT.md;
exactly one allocated ATOM must be visible as index0, and allocated RNGD logical0
runs fusion4/Mesh2. Have the existing rngd_pd/hetero_p2p packages on PYTHONPATH.

```bash
# Export FURIOSA_PYTHON and use the validated Furiosa launcher.
./furiosa_prefill_python.sh -m run_prefill
"$ATOM_PYTHON" check_atom.py --artifact "$KONI_B4_RBLN" --tables "$KONI_TABLES_NPZ"
```

Run from a writable scratch directory. The prefiller resolves cached checkpoint
KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024, writes four trusted KV npz files and
prompt metadata. The decoder imports those tensors through the host and executes
16fixed greedy steps twice. It deliberately continues beyondEOS for logit parity;
this is not a serving API or an EOS-handling test.

## Results

- Four natural prompts, lengths131,131,132,131 including the model's chat template.
- Prefix-only RNGD prefill (last prompt token consumed by ATOM), compiled width160,
  group2,context512,fusion4/Mesh2,Furiosa2026.4.
- Prefill+host materialization/storage:0.493s first,0.422/0.420/0.422s subsequent.
- ATOM rebel0.11.0,torch2.11.0+cpu,existing B4/context512 graph.
-64/64whole batched KV tensors read back exactly; both4×16-step full-logit arrays
  bit-exact after independent reimports.
- Initial answer content:Paris,5,Seoul,Tokyo. Full output/token/logit hashes are in
  validation.json. This is not an independent CPU-model logit comparison.
- Initial width128 capacity assertion failed before weight loading; the chat
  template adds a long default system prompt. Width160 covers the actual tokens.

No DMA,zero-copy,source-export overlap or native-furiosa-llm performance claim.
The portable optimum-rbln AtomUpload adapter is not used by this external-cache
KONI graph; communication-library changes were unnecessary for this host route.
The dedicated Pod was removed. Existing shared packages/artifacts were unchanged.

Artifact identity used for this run:

- Checkpoint revision: `0172bf7e73977ae213613529f3d2d5400a33f678`.
- B4 graph SHA256: `78a2e79bc6d14176e2df01e972f2938381f9ecbc57140bc6ecc000de772cc5e2`.
- Embedding/RoPE tables SHA256: `17504330a990a46dabcd7c4de97daa9eace5f6f2e1728e887e745ade42429041`.
