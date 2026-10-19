# Qwen 3B NPU extension and GPU availability audit

- Date: 2026-10-19
- Task: PDDISAG-6
- Host: etri-001; RNGD physical3/logical2 and ATOM physical1
- Source baseline: aicomp-llmd a614402, initially clean
- Model: Qwen/Qwen2.5-3B-Instruct, revision aa8e72537993ba99e69dfaafa59ed015b17504d1

## Question and setup

Test the existing model-sized builders with hidden2048,36layers,16Qheads,
2KVheads,head_dim128. Preserve B1/context512 and earlier error gates:
RNGD maximum per-layer KV relativeL2<0.1 and matching next token;
ATOM at least29/32 teacher-forced FP32 top1 and finite logits;
then same-decoder host/P2P full-logit equality and direct/sidecar tokens.
No new vendor SDK or driver is installed. GPU test unavailable: nvidia-smi
cannot communicate with driver, and lspci lists only ASPEED VGA, no NVIDIA.

## Commands

PYTHONPATH=aicomp-llmd/opt_prime:hetero-npu-p2p/src:rngd_llmd/src.
ATOM Python: experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv/bin/python.
RNGD Python: /home/gspark/venvs/furiosa-2026.4/bin/python.
Run build_qwen_atom.py with --model <pinned local snapshot> --model-id
Qwen/Qwen2.5-3B-Instruct --physical rbln1 --out experiments/qwen3b_pd_20261019/atom-build-r1.
Run validate_qwen_rngd.py with the same model/id, --accurate-attention and
--out experiments/qwen3b_pd_20261019/rngd-r1.
Exact argv, interpreter, source hashes and environment versions are preserved
in each run's provenance.json. Local weights/compiled binaries remain external.

## Results

Pending; include failures without changing gates. This is a finite functional
and numerical validation, not a same-resource performance comparison.

## Bring-up results

- ATOM build:77.7684s, compiled artifact3ecd287b3e26015d.rbln.
- RNGD r1 maximum KV relativeL2=0.04810079, logits relativeL2=0.01620128,
  next token220 matches. RNGD r2 with explicit device arguments and new default
  accurate attention reproduced the same numerical diagnostics.
- ATOM host-admission:31/32 FP32 teacher-forced top1, logits relativeL2=0.06068493.
  Threshold unchanged; one disagreement retained, no claim of FP32 equality.
- Removed hardcoded four-card name limit; canonical device names must still be
  in reported host group0, idle, context-free, and verified by actual open FD.
  Offline tests include higher-index card selection, active context rejection,
  absent/nonhost devices and malformed names. Integrated suite30tests passed.
- New RNGD CLI supports explicit physical path and logical index. The same
  binding now feeds actual graph compilation rather than using an implicit chip.
  Default accurate attention matches engine; explicit negative option preserves
  the old diagnostic path. Legacy run records retain their original semantics.

Subsequent exact commands are in atom-validate-r1/provenance.json and
pd-llmd-r1/provenance.json, with pd-llmd-config-r1.json. Output matrix uses the
same token-ID prompts as the1.5B run, input32/64/41/37, output8/32, two repeats.

P2P16requests/320tokens/1152DMA and sidecar8requests/160tokens/576DMA passed; workers exited0.

## Full-matrix numerical diagnostic

FP32 teacher-forced top1 agreement is **147/160** across
the8unique trajectories. Per(input,output):32/8=8/8,32/32=31/32,
64/8=6/8,64/32=29/32,41/8=8/8,41/32=27/32,37/8=8/8,37/32=30/32.
The earlier single-prompt31/32 gate did not establish quality across these
prompts. In particular the Korean41-token prompt gives27/32. All disagreements
are retained; thresholds were not lowered. **Transfer integration passes, but
this3B path is not cleared for general model quality.** This diagnostic is not
a task-accuracy score. A same-precision standalone baseline and per-stage
error localization are needed before production use or performance claims.

The implementation has now exercised three sizes within Qwen2, not a new model
family. GPU and newer ABI support remain unverified.
