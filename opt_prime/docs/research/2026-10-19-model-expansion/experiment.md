# Optimus model coverage, CPU optimization and module cleanup

- Date: 2026-10-19
- Task: PDDISAG-6
- Branch: feat/optimus-heterogeneous-pd, base0c54548 plus changes
- Environment: Torch2.11.0+cpu, transformers5.8.1; no NPU acquisition

## Question / protocol
Expand model execution coverage while separating contracts, model policy and KV
transport. Compare full prompt projection vs last-token-only projection and optional
logit capture using identical weights/prompts/precision, two warmups and9 balanced
repeats. Seed23, CPU2threads, five tiny model families, lengths32/128/256→16.

## Commands
See aicomp-integration/opt_prime/docs/validation/model-expansion/RESULTS.md.
benchmark_torch.py --out /tmp/optimus-model-benchmark-r1 --repeats9.
validate_checkpoint.py ran cached Qwen2.5-0.5B/1.5B snapshots, outputs in
/tmp/optimus-qwen05-r1 and /tmp/optimus-qwen15-r1. No weight download.

## Results
16 tests pass. GPT-2/Llama/Qwen2/Qwen3/Mistral dense configurations:120cases tokens
exact, maximum logits error1.7881393432617188e-7. Actual pretrained Qwen0.5B and1.5B:
8cases/64tokens exact; maximum error5.245208740234375e-5 within predefined2e-4 tolerances.
405 measured CPU ablation requests plus90warmups pass token parity. Whole-request
median speedup1.052–1.275x, prefill1.356–2.094x; all15conditions recorded, not best-only.
Qwen3-0.6B pretrained weights absent; not run. No failed inference tests/runs.

## Conclusion
Modules/contracts split and backward imports preserved; developer guide added.
Default Torch path avoids full-prompt vocabulary projection and diagnostic CPU logits
copies. These results do not claim NPU speedup or new NPU model support. Unsupported
sliding/hybrid/cross-attention explicitly rejected. Existing KONI P2P unchanged.
