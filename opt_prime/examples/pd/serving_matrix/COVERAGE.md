# Four-model coverage, 2026-10-23

Selected models: Qwen3-0.6B, Qwen3-4B, KONI-Llama3.1-8B, Gemma3-1B. Completion is not established. Direct-engine stability does not certify model quality or HTTP serving.

| Model | GPU direct | ATOM direct | RNGD prefill connections |
|---|---|---|---|
|Qwen3-0.6B|B1/4/16,128–8192 inputs,32/128 outputs,48 runs|B1/context9216/k3,16 runs;full GPU output4/8 exact|Short128-input research fixture to ATOM;long route remains|
|Qwen3-4B|Same48-run matrix|B1/context9216/k3,16 runs;full GPU output3/8 exact|128/512/2048 inputs to GPU/ATOM;32-token2K fixture equal|
|KONI8B|Same48-run matrix|B1/context9216/k1,16 runs;full/EOS GPU output3/8 exact; k3 compile fails|Prior short B4 external-cache graph only;long route remains|
|Gemma3-1B|Same48-run matrix|B1/context9216/k3 with causal-prefill SDK fix and hybrid table/mask fix;16 stable runs;GPU BF16 full/EOS0/8|RNGD model path not implemented|

All direct runs use fixed output lengths ignoring EOS. Repeats match within each tested configuration. The GPU batch matrix has investigated numerical ties and Gemma post-EOS differences; it does not imply equality across batches. Gemma same-graph ATOM SDK/external-table logits match exactly at four lengths, but GPU numerical parity is unresolved. GPU FP32 Gemma32-token8K output matches ATOM.

## Outstanding acceptance work

- Fresh HTTP serving under concurrent arrivals, cancellation, errors and recovery for all four models and routes.
- Prefix-cache/host-tier correctness for selected checkpoints; Gemma sliding-state restoration is unimplemented, KONI k1 has no spare prefix-cache blocks.
- Long-context RNGD routes for0.6B/KONI,8K RNGD capacity/chunking, and Gemma RNGD support.
- Layer/logit diagnosis and broader semantic fixtures for cross-backend output differences; matching a repeated capital question is not a quality evaluation.
- Actual producer/consumer overlap and end-to-end serving benefit, distinct from faster CPU staging.

Evidence: PR203 matrix results, PR206 staging optimization, PR209 Gemma probe, engine fix tracked in#207, KONI capacity#208. Keep the original full goal active.
