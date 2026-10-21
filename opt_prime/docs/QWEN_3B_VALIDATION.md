# Qwen2.5-3B NPU extension

The same OptimusPrime dense Qwen graph builders were used with hidden2048,
36layers,16query heads and2KV heads of dimension128. No model-specific graph
or common PDRuntime change was needed. This extends model size coverage within
Qwen2, not support for another architecture such as Qwen3, MoE or multimodal.

Model revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
Source baseline: `a614402`; exact source snapshots and dirty state recorded per
run. RNGD physical3/logical2, ATOM physical1, B1/context512. Same SDKs and
external KV ABI as the previous Qwen results; no driver update was performed.

## Numerical gates before P2P

- ATOM compilation completed in77.7684s.
- RNGD maximum per-layer KV relativeL2:0.04810079 (fixed threshold0.1).
  Last-position logits relativeL2:0.01620128; next token220 matched FP32.
- The explicit-device CLI rerun reproduced those numerical diagnostics.
- ATOM host-admission:31/32 teacher-forced FP32 top1 (fixed minimum29/32),
  full logits relativeL2:0.06068493. One token disagreement is retained.

These checks use a single bring-up prompt and do not establish task accuracy.
Transfer equality and full-matrix teacher-forced diagnostics are separate below.

## Portability changes

The RNGD checker accepts physical path/logical index and uses that same binding
for graph compilation. Accurate attention defaults on, matching the engine;
`--no-accurate-attention` retains the previous diagnostic mode.
ATOM names are checked against reported host-group membership without assuming
there are exactly four cards. Higher indices have offline selection tests;
physical card1 is the only ATOM exercised in this experiment.
Integrated unit suite:30passed. Ruff E4/F and whitespace checks passed.

GPU execution remains untested: nvidia-smi failed, and the PCI inventory showed
only ASPEED VGA and no NVIDIA device. This run neither installs a GPU driver nor
claims a GPU/NPU transfer path.

## Reproduction

Use `examples/pd/qwen_pd_config.example.json` with the pinned3B checkpoint,
compiled artifact and idle device paths. Actual local config, every command,
interpreter and source/version manifest are under `validation/qwen3b/`.
Build via `build_qwen_atom.py`, run `validate_qwen_rngd.py`, then
`validate_qwen_atom.py`; enable `validate_qwen_pd.py` only after both numerical
gates pass. Set `sidecar_binary` for the local llm-d check.
Do not reuse output directories. Weights, compiled binaries and KV tensor dumps
are external to Git. Raw failures, if any, must remain in the ledger.

## Actual P2P and local llm-d

Four prompts of32/64/41/37input tokens,8/32output tokens, two direct repeats:
16direct requests/320tokens/1152DMA commands. Then8sidecar requests/160tokens/
576DMA commands. Every command completed; every same-decoder full-logit check
matched host admission, and every sidecar output matched the direct tokens.
Worker and sidecar exited0; the service owner returned idle. These are functional
validation counts, not throughput or saturation measurements.

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
