# Four-model serving matrix — active

Objective: test and improve Qwen3-0.6B, Qwen3-4B, KONI-8B and Gemma3-1B across GPU/RNGD/ATOM and varied serving conditions.

Completed evidence:192 GPU stability runs;16 Qwen3-4B ATOM B1 runs through8K;128/512-token Qwen3-4B RNGD→GPU/ATOM import checks; investigated BF16 batch differences; rejected a partially beneficial RMS precision hypothesis as a general fix.

Not complete: fresh four-model NPU/PD coverage, long-context RNGD, Gemma RNGD support, broader numerical/quality checks, HTTP concurrency/cache/cancellation/recovery and sustained performance. Full-token cross-device parity is not established.

Next: extend RNGD beyond512tokens and profile ATOM import; use compiled Qwen3-4B context9216 artifact; continue remaining model/backend cells and exact numerical diagnosis. Dedicated validation allocation has ended; no task servers remain.

These are exported research records for review. The local research-hub retains the authoritative tracking log and synchronizes it; no direct Plane calls are made.

## Long-input extension
Qwen3-4B RNGD2048 warm570–575ms;32-token fixture output matches GPU and ATOM native. Qwen3-0.6B ATOM16 runs complete through8K; repeat8/8 but GPU full output4/8 match. Gemma long artifact compiles but external hybrid cache tables fail (#207). See experiments/2026-10-23-long-model-matrix.md.

## KONI capacity and Gemma reference
KONI8B context9216/B1/three blocks fails graph conversion;one block completes16 runs through8K+128,8/8 repeat pairs exact,3/8 full/EOS GPU output matches (including both8K cases). Raw results and capacity record attached. GPU Gemma FP32 reference added;8K32-token output matches ATOM, shorter cases remain divergent. Full four-model objective remains active.

## RNGD long-prefix follow-up (#217)

Qwen3-0.6B2K finite/repeat-exact; GPU BF16/FP32 continuation32 tokens exact. ATOM importer logits match but native continuation differs at second token due to measured18.25 tie versus18.375 imported winner. Experimental8192 guard enabled only in this research variant;0.6B8K hardware compilation/run and GPU/ATOM handoff pass the fixed fixture.4B8K also compiles/runs and matches GPU BF16/FP32 and ATOM32-token native continuation;72 slots read back exactly per importer. Full-matrix, numerical quality and performance completion remain unproven. Full scope remains open.

## KONI long-prefix follow-up (#219)

Untiled2K graph fails RMS lowering before inference;512/256-row isolated RMS probes compile/run finite. Candidate tiles onlyRMS rows, retaining2K attention;5 CPU tests pass. Full-model64GiB candidate OOMKilled/137; 128GiB group2 then fails compiler memory explosion; group1 also fails compiler memory explosion before inference. Owned allocation released. Next: separately compiled stages or actual cache-aware chunking; decoder verification remains missing.

## Partitioned KONI prefill (#221)
KONI8B full2K executes after separating RMS/attention/FFN and explicitly returning token-major KV at the compiled boundary. Three finite/exact repeats; warm prefill+host export769.15–769.52ms. GPU native/imported32tokens match. Static B1 ATOM import:64slots readback, original/portable logits exact, output matches GPU; native ATOM diverges at token28 when a0.125 logit margin becomes a tie. Research single-block target does not enable production PD scheduling. Expanded CPU suite27passed/2parent-reproduced DynamicCache tracing failures.8K subsequently passes3 finite/exact repeats, warm3.317/2.934s. GPU and ATOM native/imported32tokens match,64slots readback each, ATOM importer logits exact. All owned allocations released; DraftPR222. Full goal remains active. See experiments/2026-10-23-koni-partition221.md and evidence221/.

## Gemma RNGD producer (#223)
Implemented all26 Gemma layers with local/global semantics. Actual128 output initially failed because compiler sample strides differed from contiguous runtime weights; a matrix microprobe isolates this (137% vs0% relativeRMS). Fixed128 runs are finite/repeat-exact, warm44.83–45.08ms. GPU native/imported32tokens match,52slots exact; K/V/logit relativeRMS0.826/0.825/0.923%. Corrected2K passes3finite/exact repeats,warm1.894–1.901s; GPU32tokens match,52slots exact with22local511/4global2048 positions. Retained K/V/logit relativeRMS5.06/3.32/1.54%. Owned allocation released. Hybrid GPU cache checker verifies retained state and absolute positions; KONI2K regression passes. CPU38passed/2existing failures. Gemma ATOM import, long inputs and full four-model serving/quality/overlap remain open.

## Gemma global causality and 8K handoff (#225)

Supersedes older Gemma native-reference claims: a sliding-only native artifact allowed future-token effects after the first global attention layer. The global fix is in upstream draft optimum-rbln#827 (ed7934d). A rebuilt artifact preserves52/52 prefix slots under suffix perturbation and matches RNGD-imported ATOM32-token continuations at128/2K/8K; every raw-state restore/readback also passes. Gemma8K RNGD warm prefill+export1.547–1.552s, GPU32-token match and52retained slots exact. These are fixed fixtures, not production PD or broad quality completion. See evidence225 and both accompanying experiment records.

Remaining four-model requirements: diverse prompts and meaningful EOS-aware output; production Gemma hybrid and KONI PD scheduler integration; cache eviction/host restore/cancellation/recovery under heterogeneous HTTP load; genuine producer-compute/transfer overlap; controlled end-to-end latency/throughput comparison and hardware batch coverage. Scope remains Qwen3-0.6B, Qwen3-4B, KONI-8B, Gemma3-1B across GPU/RNGD/ATOM.

## EOS-aware direct semantics (#227)

192 GPU/ATOM runs across four models,12 cases each,2 repeats, all EOS-ended. Cross-backend tokens match12/12 cases for both Qwens and KONI,11/12 for Gemma;96 within-backend repeat pairs exact. Inputs range23–6428 tokens. Token identity coexists with shared wrong answers (Qwen4B arithmetic52; Qwen0.6B Korean capital Incheon; Gemma arithmetic40). Gemma longest arithmetic differs30 GPU vs25 ATOM. Research runner now unions generation-config EOS with tokenizer EOS, preventing KONI continuation past turn-end. See evidence227 and its experiment record. RNGD semantic/production PD integration remains pending.

## RNGD semantic handoff follow-up (#227)

All four models now execute the same four short chat cases through RNGD prefill and GPU/ATOM continuation.32/32 decoder comparisons match native tokens through EOS,48prefill runs finite/repeat-exact. Exact saved IDs and true KV lengths verified, excluding right padding; per-model52/56/72/64slots read back exactly. Qwen/KONI original/portable importer logits agree; Gemma raw native-state roundtrip also passes. Inputs23–143 tokens (compiled width128 or512), so long semantic fixtures remain pending. This is host-staged full-prefix handoff, not actual compute/transfer overlap or production PD serving. See semantic-rngd227 record and evidence227.

## Medium semantic handoff (#227)

Inputs1526–1643tokens in width2048 buckets now pass all32 model/case/decoder EOS comparisons (four models,four cases,GPU/ATOM).48prefill runs finite/repeat-exact; true IDs/KV lengths and52/56/72/64slot readbacks verified. Native GPU/ATOM sequences agree for these cases, including shared wrong answers documented earlier. Approximately6K semantic inputs and production heterogeneous serving/overlap remain open. See semantic-mid227 experiment and evidence227.

## Six-thousand-token partial results (#227)

KONI (6137–6143 tokens) and Qwen3-0.6B (6026–6031 tokens) pass16/16 GPU/ATOM EOS comparisons across eight questions, with24producer repeats. Readbacks64/56slots exact; both ATOM importers agree. Gemma remains live in the owned queue, Qwen3-4B follows. No live PD or overlap claim. See semantic-long227 record and selected evidence227.
