# Model-family validation: Qwen, KONI, Gemma

Follow-up: Gemma text-prefill causality was repaired and tested in
[optimum-rbln PR827](https://github.com/RBLN-SW/optimum-rbln/pull/827). A separate
Qwen3 RNGD graph and host handoff passed the finite fixture in
[aicomp PR198](https://github.com/ai-computing/aicomp/pull/198). KONI artifacts were
found outside the standard cache and successfully reconnected; see
[koni_reconnect/README.md](koni_reconnect/README.md). The table below records the
earlier standard-worker/cache checks, not these new experimental paths.

2026-10-23, code baseline7ffd71d plus diagnostic scripts. Same isolated Pod and SDK
setup as ATOM_CONTAINER_EXPERIMENT.md; one ATOM and one RNGD, fusion4/Mesh2 for research
prefill. No shared artifact modifications or model compilation. Results are in
model_family_results.json. Names below identify exact checkpoints, not entire families.

| Model | Research RNGD → ATOM PD | ATOM standalone |
|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct |12/12 remote requests completed;9/12 sequences equal to aggregate;11/12 first tokens|3 prompts ×2 runs, repeat-exact|
| Qwen/Qwen3-0.6B |Rejected by current Qwen2-only prefill gate|3/3 CPU FP32 sequence-exact with thinking disabled|
| Qwen/Qwen3-8B-FP8 |Rejected by research worker; native8B handoff still unvalidated|No compiled artifact; not executed|
| KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024 |Rejected by current Qwen2-only prefill gate|No compiled artifact; not executed|
| unsloth/gemma-3-1b-it |Rejected by current Qwen2-only prefill gate|Runs but fails correctness smoke;0/3 first tokens equal CPU; issue196|

The gate is DenseModel.load in the existing optimus-rm worker. It checks actual
cached config.json and raises `only dense full-attention Qwen2 checkpoints are enabled`.
This is not a claim that vendor-native RNGD cannot serve Qwen3 or KONI; it identifies
the specific worker used by the validated heterogeneous PD path.

## Commands

Use the environment and device-plugin Pod setup in ATOM_CONTAINER_EXPERIMENT.md.
Place scripts in a writable scratch directory. ATOM_PYTHON must resolve the validated
Rebel environment. No script bypasses device lease checks. The model-family checker
only loads existing B1/context2048 compiled artifacts; it never implicitly compiles.

```bash
"$ATOM_PYTHON" check_atom_container.py check_model_families.py
"$ATOM_PYTHON" check_gemma_oracle.py
"$ATOM_PYTHON" check_atom_container.py check_qwen3_no_thinking.py
"$ATOM_PYTHON" check_qwen3_oracle.py
```

The checkers use three fixed natural-language prompts (France capital,2+3,Korean
capital), greedy decoding and repeated ATOM runs. Initial check uses max_new16;
Qwen3 follow-up disables thinking and uses max_new32. CPU oracles use the same
checkpoint ID, tokenizer/chat template/input settings and FP32 weights.
`status: passed` in raw standalone JSON means execution completed only, not answer
quality; the conclusions in this document incorporate CPU comparisons.

Repeat the portable-importer npu_pd_bench run from ATOM_CONTAINER_EXPERIMENT.md,
explicitly exporting FURIOSA_PYTHON and using the current Pod's physical ATOM index:
12requests,seed0,prompt16–96,max_new16,batch8,blocks17,remote-only,width256.
All12imports completed;48/48 first-import slots read back exactly. PD66.6tok/s,
meanTTFT1.683s,meanTPOT7.3ms. These instrumented single-run values are not a speedup claim.

## Findings

Gemma cached B1 produces unrelated/repetitive text. For2+3, ATOM repeats
`@homenews`; CPU answers5. For the Korean capital, ATOM outputs unrelated Japanese
text; CPU answersSeoul. All3CPU/ATOM first tokens differ, although ATOM repeats are
3/3exact. This happens without RNGD or portable KV import. Root cause may involve
compiled artifact provenance, SDK compatibility or model lowering; not yet isolated.
Tracked in https://github.com/ai-computing/aicomp/issues/196.

Qwen3 has exact CPU/ATOM outputs on all3non-thinking prompts, including the erroneous
answerIncheon for SouthKorea's capital. Runtime parity therefore does not imply
factual correctness. These three short prompts are not a broad quality benchmark.
Qwen2.5 answersParis/Seoul; its16-token arithmetic response is truncated, so no claim
that every standalone answer is fully validated.

Failed attempts retained in the local experiment: first harness iteration treated
new Transformers chat-template BatchEncoding as a list; explicit return_dict=False
fixed the harness. First Qwen PD launch lacked FURIOSA_PYTHON; the shell guard exited
before RNGD initialization; exporting the interpreter fixed that launch.

## Required work

- Qwen3: model-correct RNGD prefill (including Q/K normalization) or a verified native
  KV exporter; appropriate ATOM paged-KV artifact. Existing0.6Bstandalone cache is not
  evidence of PD support.
- KONI: Llama3.1-aware prefill/RoPE integration and a verified ATOM compiled decoder;
  vendor-native standalone serving evidence does not validate this handoff.
- Gemma: first resolve issue196, then model-specific norms/activation/sliding-window
  cache semantics and a compatible PD graph.
- Keep transport byte tests separate from model arithmetic and semantic quality.

Dedicated namespace removed after experiments. No production code changed.
