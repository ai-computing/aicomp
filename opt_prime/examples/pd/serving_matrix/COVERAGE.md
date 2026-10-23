# Four-model coverage, 2026-10-23

Selected models: Qwen3-0.6B, Qwen3-4B, KONI-Llama3.1-8B, Gemma3-1B. Completion is not established. Direct-engine stability does not certify model quality or HTTP serving.

| Model | GPU direct | ATOM direct | RNGD prefill connections |
|---|---|---|---|
|Qwen3-0.6B|B1/4/16,128–8192 inputs,32/128 outputs,48 runs|B1/context9216/k3,16 runs;full GPU output4/8 exact|128-input fixture plus2048-input GPU/ATOM transfer;2K GPU BF16/FP32 continuation matches, ATOM-native differs at second token;8K finite/repeat-exact and GPU/ATOM32-token native equality; substantial numeric error remains|
|Qwen3-4B|Same48-run matrix|B1/context9216/k3,16 runs;full GPU output3/8 exact|128/512/2048/8192 inputs to GPU/ATOM;8K32-token native/imported sequences exact on GPU BF16/FP32 and ATOM; substantial KV/logit numeric error remains|
|KONI8B|Same48-run matrix|B1/context9216/k1,16 runs;full/EOS GPU output3/8 exact; k3 compile fails|Partitioned2K/8K full-prefix RNGD route now runs;8K GPU/ATOM32-token fixtures match64slots;2K ATOM native differs at a measured tie (PR222)|
|Gemma3-1B|Same48-run matrix|Earlier sliding-only native results invalidated by global causality #225; rebuilt global+sliding artifact passes prefix-causality and three-length handoff checks|26-layer RNGD producer validated128/2K/8K; GPU fixtures match; corrected native ATOM/imported32tokens match at all three lengths (PR224/226)|

All direct runs use fixed output lengths ignoring EOS. Repeats match within each tested configuration. The GPU batch matrix has investigated numerical ties and Gemma post-EOS differences; it does not imply equality across batches. Gemma same-graph ATOM SDK/external-table logits match exactly at four lengths, but GPU numerical parity is unresolved. GPU FP32 Gemma32-token8K output matches ATOM.

## Outstanding acceptance work

- Standalone GPU HTTP covered by PR213 and ATOM HTTP by PR216: all four lifecycle checks pass through8K. Cache-off concurrent/serial equality48/48 on each backend. Real RNGD->ATOM short-input HTTP now passes56/56 output checks (#235/#237); short/medium/long live checks pass168/168responses; broader conditions remain incomplete.
- Qwen ATOM device/host-cache cases pass12/12 concurrent equality each (PR216). GPU BF16 cache has numerical ties; FP32 controls match48/48 (PR213). Gemma sliding-state restoration is unimplemented, KONI k1 has no spare prefix-cache blocks.
- Short live producer/scheduler integration is verified for all four models. Live long prefixes now pass for all four; genuine cache-aware producer chunking remains open.
- Layer/logit diagnosis and broader semantic fixtures for cross-backend output differences; matching a repeated capital question is not a quality evaluation.
- Actual producer/consumer overlap and end-to-end serving benefit, distinct from faster CPU staging.

Evidence: PR203 matrix results, PR206 staging optimization, PR209 Gemma probe, engine fix tracked in#207, KONI capacity#208; HTTP PR213/216, disconnect fix PR215, long RNGD issue217. Keep the original full goal active.

## EOS-aware native fixtures (#227)

192 GPU/ATOM runs, all EOS-ended;96 repeat pairs exact. Cross-backend exact cases: Qwen0.6B12/12,Qwen4B12/12,KONI12/12,Gemma11/12. Four short questions with0/100/400 repeated context notes give23–6428 input tokens. Shared wrong factual answers are retained, so this is not a general quality pass. Research EOS policy includes both generation-config and tokenizer turn-end tokens; original KONI policy failure preserved. Subsequent RNGD and live HTTP evidence is listed below; matching outputs still does not certify answer quality.

## RNGD semantic handoff follow-up (#227)

All four models now execute the same four short chat cases through RNGD prefill and GPU/ATOM continuation.32/32 decoder comparisons match native tokens through EOS,48prefill runs finite/repeat-exact. Exact saved IDs and true KV lengths verified, excluding right padding; per-model52/56/72/64slots read back exactly. Qwen/KONI original/portable importer logits agree; Gemma raw native-state roundtrip also passes. Inputs23–143 tokens (compiled width128 or512), with subsequent long-context results below. This is host-staged full-prefix handoff, not actual compute/transfer overlap or production PD serving. See semantic-rngd227 record and evidence227.

## Medium semantic handoff (#227)

Inputs1526–1643tokens in width2048 buckets now pass all32 model/case/decoder EOS comparisons (four models,four cases,GPU/ATOM).48prefill runs finite/repeat-exact; true IDs/KV lengths and52/56/72/64slot readbacks verified. Native GPU/ATOM sequences agree for these cases, including shared wrong answers documented earlier. Approximately6K semantic results and short live HTTP are now available below; true overlap and broader serving conditions remain open. See semantic-mid227 experiment and evidence227.

## Long semantic handoff (#227/#233)

Four models,48producer runs,31/32 EOS comparisons at6026–6428input tokens. Both Qwens and KONI match4/4cases on each decoder; Gemma matches3/4GPU and4/4ATOM. All KV readbacks pass. The retained Gemma arithmetic mismatch is30(nativeGPU BF16/FP32) vs25(RNGD/ATOM), both wrong. Same layer equations on GPU choose a different first token from RNGD; operator cause unresolved in#233. See semantic-long227 and evidence227.

## Real RNGD to ATOM HTTP (#234/#236)

All four short-input paths pass14normal/concurrent/reuse responses each:56total,64remote prefills including cancellation requests,0local fallbacks. Requests trigger actual RNGD subprocess computation; no KV replay. Each model passes explicit cancellation, socket disconnection and subsequent reuse. Gemma/Qwen usek3; static B1 KONI usesk1 with#236. SIGPIPE-safe HTTP is#214/#215. These are B1 hardware requests with queued client concurrency, not hardware batch4. Native parity includes shared wrong answers.

Width2048 live checks now pass for all four models:56responses and64remote prefills. Short+medium total112responses/128prefills, with no local fallbacks. Width8192 live checks also pass all four models:56responses/64remote prefills at6026–6428input tokens. Combined short+medium+long:168responses/192remote prefills,0fallback. Pending-prefill hardware cancellation/worker-failure recovery, cross-host RNGD-to-GPU live PD, hybrid prefix/host restore, applicable hardware batch>1, genuine producer/transfer overlap and matched native-vs-PD latency/throughput remain unverified. GPU-to-GPU live HTTP is tested separately under#240: default BF16 concurrent35/48; explicit uncached/chunk256 producer controls#241 recover48/48concurrent and12/12native/PD serial outputs. Gemma numerical#233 remains unresolved. Overall goal remains active.
