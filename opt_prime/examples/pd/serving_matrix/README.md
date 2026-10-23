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

Fresh four-model ATOM coverage; long-context RNGD graphs and Gemma support;
RNGD→GPU/ATOM checks beyond isolated inputs; HTTP concurrency1/4/16, cache reuse,
EOS/cancellation/recovery and sustained performance; model-size/length comparisons
under matched resource budgets. Numerical mismatches require diagnosis or quality
checks, not silent acceptance. No native-engine performance parity or source-copy
compute-overlap claim is made here.
