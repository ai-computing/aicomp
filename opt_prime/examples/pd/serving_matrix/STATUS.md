# Four-model serving matrix — active

Objective: test and improve Qwen3-0.6B, Qwen3-4B, KONI-8B and Gemma3-1B across GPU/RNGD/ATOM and varied serving conditions.

Completed evidence:192 GPU stability runs;16 Qwen3-4B ATOM B1 runs through8K;128/512-token Qwen3-4B RNGD→GPU/ATOM import checks; investigated BF16 batch differences; rejected a partially beneficial RMS precision hypothesis as a general fix.

Not complete: fresh four-model NPU/PD coverage, long-context RNGD, Gemma RNGD support, broader numerical/quality checks, HTTP concurrency/cache/cancellation/recovery and sustained performance. Full-token cross-device parity is not established.

Next: extend RNGD beyond512tokens and profile ATOM import; use compiled Qwen3-4B context9216 artifact; continue remaining model/backend cells and exact numerical diagnosis. Dedicated validation allocation has ended; no task servers remain.

These are exported research records for review. The local research-hub retains the authoritative tracking log and synchronizes it; no direct Plane calls are made.

## Gemma hybrid table probe (#207)
Original first request fails; lane-local tables alone fail repeat parity; lane tables plus request mask reset complete16 runs through8K with8/8 repeat pairs exact. Same-graph SDK-owned comparison: full32-step logits exact at four lengths. GPU parity remains incomplete. Prefix/host cache and PD restoration unsupported by this probe; engine integration remains open.
## Long-input extension
Qwen3-4B RNGD2048 warm570–575ms;32-token fixture output matches GPU and ATOM native. Qwen3-0.6B ATOM16 runs complete through8K; repeat8/8 but GPU full output4/8 match. Gemma long artifact compiles but external hybrid cache tables fail (#207). See experiments/2026-10-23-long-model-matrix.md.
