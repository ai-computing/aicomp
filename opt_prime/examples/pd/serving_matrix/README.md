# Four-model serving validation (#201, ongoing)

Target checkpoints: Qwen3-0.6B, Qwen3-4B, KONI-Llama3.1-8B-Instruct-20241024 and
unsloth/gemma-3-1b-it. The objective includes GPU, RNGD and ATOM, long prompts,
concurrency, cache behavior and heterogeneous prefill/decode. This directory is an
experimental measurement harness, not a production backend registration.

## Current evidence and limits

- GPU direct-engine sweep:192 completed stability runs across four models,
  input128/512/2048/8192, B1/4/16, output32/128 and two repeats. Finite logits and
  repeat equality hold; noOOM. This does not measure HTTP concurrency or quality.
- Comparing B1 toB4/B16 changes some BF16 outputs. Gemma differences disappear
  when stopping atEOS. Investigated KONI and Qwen3-4B differences reproduce under
  teacher forcing and disappear at the same positions underFP32; retain the
  numerical evidence instead of labeling all batch differences transport bugs.
- RNGD Qwen3-4B width128/512 produces valid KV exports and continues on GPU with
  the same32-token sequences as nativeGPU on the two tested inputs. Numerical
  V error versusFP32 is7.9–11.9%, so this is not proof of broad numerical fidelity.
- ATOM Qwen3-4B has a newly compiled B1/context9216/3KV-block artifact and16
  stable runs through8K inputs/output128. Full output parity againstGPU is not
  universal. The transfer checks separately verify every imported cache slot.
- An isolated FP32-RMS-gamma experiment (#202) reduces KV errors on the128-token
  case but not the512-token case and does not improve all logit metrics. It is
  retained as a rejected general fix, not a production default.

Full measurements, revisions, methodology, failures and pending work are in
`experiments/2026-10-23-four-model-serving-matrix.md` and `results/`. Raw `.pt` exports and compiled hardware artifacts
remain in task-owned scratch storage. Never infer completion of the full matrix
from `status: passed`: that status covers a run's finite-logit/repetition checks.

## Reproduction

GPU runner dependencies: torch and Transformers with the checkpoint architecture,
HF offline cache or explicit local snapshot. The measured GPU environment is
recorded perJSONL. Use a free, authorized device; concurrent jobs on the same card
would invalidate timing. The first sweep uses two repeats, not enough forp95.

```bash
python3 -m unittest discover -s . -p 'test_*.py'
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 gpu_sweep.py \
  --model "$MODEL_SNAPSHOT" --revision "$REVISION" --output gpu-model.jsonl
# Extend output length and batch size, writing a different result file:
CUDA_VISIBLE_DEVICES=0 python3 gpu_sweep.py --model "$MODEL_SNAPSHOT" \
  --revision "$REVISION" --batches 1 4 16 --new-tokens 128 --output gpu-model-out128.jsonl
CUDA_VISIBLE_DEVICES=0 python3 gpu_sweep.py --model "$MODEL_SNAPSHOT" \
  --revision "$REVISION" --batches 16 --output gpu-model-b16.jsonl
```

Inputs are token fixtures with exact lengths and shared question tails; fixed
outputs deliberately ignoreEOS. They are capacity/stability tests, not a chat
quality evaluation. `summarize_gpu.py` separately compares raw and through-EOS
outputs using each checkpoint's documented generation config. Preserve the result
file names shown in the experiment record for that summarizer. Actual production EOS,
streaming cancellation and request admission need separate HTTP tests.

For RNGD/ATOM, reuse the dedicated device-plugin Pod/SDK bootstrap fromPR195 and
PR200; do not modify shared SDKs. `rngd_qwen_sweep.py` requires PR200 `prefill.py`
and `spec.py` alongside this directory onPYTHONPATH, and the validated Furiosa
launcher. `atom_sweep.py` requires the optimus-rm inference package and ATOM
allocation bootstrap; checkpoint artifacts compile into `--cache-dir` only.

```bash
./furiosa_prefill_python.sh -m rngd_qwen_sweep --model Qwen/Qwen3-4B \
  --width 128 --output rngd-qwen3-4b-128
"$ATOM_PYTHON" check_atom_container.py atom_sweep.py --model Qwen/Qwen3-4B \
  --cache-dir "$OWN_ARTIFACT_CACHE" --output atom-qwen3-4b.jsonl
python3 check_rngd_gpu.py --model "$MODEL_SNAPSHOT" \
  --exports rngd-qwen3-4b-128.pt --output rngd-gpu.json
"$ATOM_PYTHON" check_atom_container.py check_rngd_atom.py --model Qwen/Qwen3-4B \
  --cache-dir "$OWN_ARTIFACT_CACHE" --exports rngd-qwen3-4b-128.pt --output rngd-atom.json
```

`check_rngd_atom.py` additionally requires PR195 `atom_stream_upload.py`. Its
original and portable importers share the same target graph and compare full
logits plus slot readbacks. Its timings include first-use allocation and are not
an unbiased importer performance comparison.

`check_batch_numerics.py` accepts a completed GPU sweep to reproduce first
cross-batch divergences under teacher forcing andFP32. `rms_precision_variant.py`
loads only the trusted local PR200 source to isolate the gamma-carrier hypothesis;
it does not patch the installed engine. More inputs and controlled timing are
required before adopting that candidate.

## Remaining scope

Broader EOS-aware long semantic RNGD coverage and production integration of the
validated Qwen/KONI/Gemma producers; heterogeneous HTTP concurrency, cache reuse,
EOS/cancellation/recovery and sustained performance; model-size/length comparisons
under matched resource budgets. Numerical mismatches require diagnosis or quality
checks, not silent acceptance. No native-engine performance parity or source-copy
compute-overlap claim is made here.

## KONI long-prefix probe (#219)

`rngd_koni_sweep.py --width 2048 --output PREFIX` reuses the validated-environment Llama3.1 FullPrefillEngine and pins the KONI checkpoint. The untiled graph fails RMS lowering at[2048,4096]. `--rms-tile 512` selects a process-local graph with row-tiled normalization; full2K attention is unchanged. The group2 candidate hits64GiB OOM, then a compiler memory-explosion error at128GiB. `--group 1` is the next controlled candidate, still executing all32layers. Group1 also fails compiler memory explosion: this is a reproducible failed experiment, not an adopted fix. Further work must partition compiled stages or implement real cache-aware chunking. `probe_rms219.py` records isolated kernel feasibility; `test_koni_export.py` and `test_rowwise_rms.py` check host layout/normalization and rejection paths.

After a successful export, use the GPU checker or `check_rngd_atom.py --kv-blocks 1` with the existing B1/context9216/k1 KONI artifact. The checkpoint's three-block artifact is not available. No installed SDK modification, real transfer overlap, or numerical parity is implied by compiling. See experiments/2026-10-23-koni-rngd-long219.md and evidence219/.

## Partitioned KONI prefill (#221)

`--partitioned --group 1 --rms-tile 512` compiles normalization, full-context
attention and FFN separately. It executes every layer and every token; the RMS
row tiles are not attention chunks. The compiled attention returns token-major
KV explicitly, then the device runtime converts to the existing export layout.
This avoids incorrect permutation metadata observed at the compiled output.
The stage-only engine uses `begin_device_prefill` and `export_groups`; it does not
implement the inherited device output-head path.

```bash
./furiosa_prefill_python.sh -m probe_koni_attention
./furiosa_prefill_python.sh -m rngd_koni_sweep --width 2048 --rms-tile 512 \
  --group 1 --partitioned --output rngd-koni-8b-2048-parts221-layout
python3 check_rngd_gpu.py --model "$KONI_SNAPSHOT" \
  --exports rngd-koni-8b-2048-parts221-layout.pt --output koni-gpu.json
"$ATOM_PYTHON" container_bootstrap.py check_rngd_atom.py \
  --model KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024 --kv-blocks 1 \
  --cache-dir "$OWN_ARTIFACT_CACHE" \
  --exports rngd-koni-8b-2048-parts221-layout.pt --output koni-atom.json
```

The ATOM checker now accepts an exclusively owned static B1/full-attention
single-block artifact. `SingleBlockTarget` reuses the existing staging/import
arithmetic for block zero; it does not enable the production PD scheduler,
prefix caching, hybrid caches or concurrent requests on that artifact.
See the experiment record and `evidence221/` for measured results and failures.

## Gemma3 RNGD producer (#223)

`rngd_gemma_sweep.py` pins Gemma3-1B and runs all26layers with offset RMS
weights, four residual-path norms, Q/K normalization, GELU gating, scaled
embeddings, and the checkpoint's local/global causal masks and RoPE. It uses
full-prefix attention, with embeddings and final norm/head on CPU. Local/global
mask inputs share one compiled layer graph. This is an experimental producer,
not a production-serving or overlap implementation.

```bash
./furiosa_prefill_python.sh -m probe_gemma_layer
./furiosa_prefill_python.sh -m probe_compile_layout
./furiosa_prefill_python.sh -m rngd_gemma_sweep --width 128 --output gemma128
python3 check_rngd_gpu.py --model "$GEMMA_SNAPSHOT" --exports gemma128.pt \
  --output gemma128-gpu.json
```

Compiler samples must match the contiguous runtime layout. Preserving the strides
of transposed checkpoint weights with `zeros_like` silently changed matrix
results in the validated SDK; the layout probe retains that negative control.
The GPU checker validates absolute cache positions and compares only the retained
window for sliding layers; it still requires the full prefix for global layers.
It also checks the cache contents after import, not just the upload tensor.
See the experiment record for failed runs, tolerances and remaining scope.

## EOS-aware shared semantic inputs (#227)

`semantic_sweep.py --backend gpu` saves chat-template input IDs and both generation-config/tokenizer EOS IDs. The ATOM direct runner and RNGD producers reuse that file verbatim. `--cases` must name existing cases and all selected cases must fit the compiled width; the producer never silently truncates or skips inputs. Gemma exports only the true prompt prefix and uses its last real hidden state, excluding right padding. Qwen/KONI retain their existing true-length exports.

```bash
python semantic_sweep.py --backend gpu --model "$MODEL_SNAPSHOT" \
  --fixtures fixtures.json --output native-gpu.jsonl
./furiosa_prefill_python.sh -m rngd_gemma_sweep --width 2048 \
  --fixtures fixtures.json --cases capital-100 arithmetic-100 korean-100 retrieval-100 \
  --output gemma-chat-mid
python check_rngd_gpu.py --model "$MODEL_SNAPSHOT" \
  --exports gemma-chat-mid-0.pt gemma-chat-mid-1.pt gemma-chat-mid-2.pt gemma-chat-mid-3.pt \
  --eos-aware --max-output 64 --output gpu-handoff.json
```

For Qwen use `rngd_qwen_sweep --model MODEL --experimental-long-width`; for KONI use `rngd_koni_sweep --partitioned --group 1 --rms-tile 512`. Pass the same fixture/case/width/output options. Each selected case saves a numbered export containing its exact IDs and EOS set. Both `check_rngd_atom.py` and `check_gemma_atom_handoff.py` accept `--eos-aware --max-output 64`; legacy fixed-length checks remain the default. Raw full-prefix host exports are not streaming overlap. Shared wrong native answers remain wrong after a successful handoff; inspect the semantic records separately from transport parity.
