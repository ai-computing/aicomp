# KONI long-context ATOM capacity investigation

- **Date**: 2026-10-23
- **Task**: aicomp #208, related #201
- **Host / device**: RNGD+ATOM host, one allocated ATOM, RBLN-CA22 target; rebel-compiler0.11.0 / optimum-rbln0.11.0.post1
- **Commit**: matrix fe1a474 plus --kv-blocks diagnostic option

## Question

Does reducing cache reservation resolve KONI8B graph conversion failure at long context?

## Setup

KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024 revision0172bf7e73977ae213613529f3d2d5400a33f678. B1/context9216; compare3 versus1 full-context KV blocks. Same one allocated ATOM, dtype/model unchanged. Inputs128/512/2048/8192 and outputs32/128 with two repeats planned, only after compilation succeeds.

## Command

Using matrix PR203 runtime/bootstrap environment:

```bash
bash run-atom201.sh --model KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024 --cache-dir atom-artifacts --output atom-koni-8b.jsonl
bash run-atom201.sh --model KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024 --cache-dir atom-artifacts --kv-blocks 1 --output atom-koni-8b-k1.jsonl
```

## Results

Three-block attempt exits1 with `RBLNCompileError: Graph Generation: [DEVICE_GRAPH_CONVERSION]`. Zero inference runs; no throughput/latency result. Compiler first warns that three blocks exceed the required one and may cause compilation failure. This is evidence for the next controlled experiment, not proof that capacity is the cause. Local log: analysis/serving-matrix-201/atom-koni-8b.log.

One-block compile succeeds and all16 inference runs finish (128/512/2048/8192 inputs,32/128 outputs, two repeats). All8 repeat pairs are token-exact. GPU B1 equal-input comparison: first token8/8, full output3/8, EOS-normalized3/8 (checkpoint EOS128001). The8K32/128-output cases both match GPU. Warm TTFT:100.9–102.1ms at128;399.9ms at512;1631.2–1633.4ms at2048;7314.0–7321.1ms at8192. Warm TPOT67.5–73.2ms. Local log: analysis/serving-matrix-201/atom-koni-8b-k1.log. Raw results: atom-koni-8b-k1.jsonl and atom-koni-8b-summary.json.

## Conclusion

The one-block B1 long-input KONI standalone path is stable on these fixtures; cross-backend numerical equivalence is incomplete. Prior short external-cache decoder results used a different graph and do not establish this configuration. Reducing reserved blocks resolves compilation in this controlled comparison, supporting a cache-capacity/configuration constraint; the compiler did not expose an exact memory-budget diagnosis. The successful single-block configuration retains the full long-input condition; it does not provide extra cached-prefix blocks or establish PD behavior.
