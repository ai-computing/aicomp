# RNGD → ATOM numerical divergence

2026-10-23, follow-up to issue #194 and ATOM_CONTAINER_EXPERIMENT.md. Code baseline
9122913; research RNGD Qwen2.5-0.5B-Instruct prefill, fusion4/Mesh2 width256;
ATOM B8, context2048, KV blocks17. Same SDK versions as the container experiment.
This is not native furiosa-llm Qwen3-8B validation or a throughput experiment.

## Reproduction

Use the isolated, device-plugin allocated container and environment described in
ATOM_CONTAINER_EXPERIMENT.md. Select the physical ATOM device actually allocated
to the new Pod. No installation or shared workload changes are required.
Run from a writable scratch directory containing these scripts; add it to
PYTHONPATH. Preserve atom_original.json from check_atom_backend.py.

```bash
# FURIOSA_PYTHON names the validated Furiosa interpreter.
./furiosa_prefill_python.sh -m dump_rngd_numerics
"$ATOM_PYTHON" check_atom_container.py diagnose_atom_numerics.py
```

The fixed fixture regenerates seed0's 12 random prompts (length16–96) and selects
zero-based requests1,6,11. It saves rngd_diagnostic.pt (local trusted tensor data,
loaded with weights_only=True), then numerical_diagnostic.json. The latter is
published as atom_numerical_summary.json. The reference supplies forced tokens:
three decode steps per selected request all consume the same ATOM-reference prefix.
The scripts intentionally cover this exact model/shape fixture, not arbitrary models.

Also run check_atom_backend.py with the same 12-request benchmark and original
importer, changing only the prefill interpreter to ATOM_PYTHON and prefill arguments
to `--engine torch --device cpu --dtype float32`, then `bfloat16`.
Use the allocated ATOM physical index, not the prior run's index.

## Results

| Prefill | First-token agreement with ATOM | Whole 16-token sequence agreement |
|---|---:|---:|
| Research RNGD (prior paired run) | 11/12 | 9/12 |
| CPU FP32 | 12/12 | 10/12 |
| CPU BF16 | 12/12 | 9/12 |

CPU experiments all used remote-only PD: 12 remote requests, 0 local. These random
prompts are a sensitive parity fixture, not a language-quality evaluation.

- ATOM-native raw cache export/reimport: all **9/9** teacher-forced decode logit
  vectors bit-exact. This controls for cache API, block/position mapping and replay.
- External imports: all **288/288** slot checks exact (3 prompts × 2 sources × 48).
  Comparison is against the expected BF16-to-DLFloat encoded payload. It does not
  establish that conversion preserves all source precision.
- Request6 diverges before transfer: ATOM picks262 (16.125 vs16.0); RNGD picks286
  (16.109814 vs16.091438), while CPU FP32 picks262 (16.112167 vs16.094362).
  With the same forced first token, all three subsequent argmaxes agree.
- Request1's third output: ATOM prefers17 (12.25 vs12.1875); RNGD-KV decode ties
  16/17 at12.375; CPU-KV decode ties them at12.25. Argmax chooses16.
- Request11's second output: ATOM prefers480 over198 (11.3125 vs11.0625);
  RNGD-KV decode ties them at11.0625 and argmax chooses198. CPU-KV retains480.
- RNGD versus CPU FP32 KV relative RMS: first K slot0.166–0.174%; final V
  slot2.12–4.22%. This localizes numerical drift across the prefill computation,
  but does not identify a particular operator or prove either accelerator is the oracle.

Top-k order is not a tie-breaking contract; JSON explicitly records argmax.
No production change or altered tie-break rule is justified by these fixtures.
Next: per-layer prefill oracle and natural-prompt/task-quality checks. Preserve
transport byte equality separately from numerical and semantic correctness.

## Limits and native 8B path

The installed optimum-rbln exposes RBLNQwen3ForCausalLM; cached Qwen3 artifacts are
0.6B, not8B. Cached Qwen3-8B-FP8 config has36layers,8KV heads,head_dim128 and dynamic
E4M3,128×128 weight blocks. The current fixture has24layers,2KV heads,head_dim64.
An 8B decoder needs a verified compatible weight/quantization configuration and
compiled artifact, then72-slot KV geometry and teacher-forced correctness tests.
Qwen3 class availability alone proves neither FP8 support nor weight equivalence.
No 8B ATOM compilation or native8B handoff was attempted here.

One initial CPU control failed safely at the device lease because it retained the
previous Pod's physical index. Rerunning with the newly allocated device succeeded.
The dedicated namespace was deleted after validation; other workloads were unchanged.
