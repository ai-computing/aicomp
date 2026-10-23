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
