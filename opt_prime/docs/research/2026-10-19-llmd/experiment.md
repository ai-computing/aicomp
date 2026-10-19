# Optimus llm-d feature integration validation

- Date: 2026-10-19
- Task: PDDISAG-6
- Branch: feat/optimus-llmd, new worktree aicomp-llmd, based on engine0c54548
- Hardware: RNGD3/logical2 + ATOM2; KONI B4/context512 one active slot
- Runtime/driver: same2026.4.0 RNGD and ATOM3.0.0/torch-rbln0.2.2 as engine validation

## Question
Can the same engine run through the actual upstream llm-d shared-storage sidecar
without recomputing prefill, transporting KV over HTTP, or executing DMA in HTTP threads?

## Implementation
Optional PDStageService/OwnerExecutor/loopback stage servers on a separate feature
branch. Prefill retains a fenced source continuation. Decode reuses it through the
same PDRuntime and checked transport. Main owner loop performs device work. Capacity,
TTL, idempotency, cancellation and poison retention tested. No engine branch edits.

## Commands
See aicomp-llmd/opt_prime/docs/validation/llmd/RESULTS.md for exact commands.
Primary hardware command ran validate_koni.py with --llmd-sidecar, config
/tmp/optimus-llmd-koni-config-r1.json, output /tmp/optimus-llmd-koni-r1.

## Results
18 unit/model/HTTP tests passed. Actual tiny Llama+sidecar3requests/18tokens exact.
KONI direct8requests/132tokens and sidecar8requests/132tokens exact; sidecar512/512
DMA commands CQE0,1024/1024 including direct baseline. All full-logit comparisons exact.
All request stage sequences verified; producer/ATOM worker and own sidecars exit0;
no NPU contexts after completion. Original unrelated sidecar left untouched.
Upstream5f4e762f341a5196393ce79f8a57c3e1900c4a6b diff exactly matches loopback-only patch.
No failed hardware runs, resets, uncertain timeouts or forced NPU process termination.

## Conclusion
Supported local llm-d integration is verified on the separate feature branch.
Kubernetes/EPP, multi-node, sampling/streaming, service throughput and task quality
remain outside this validation. Raw small evidence retained under feature docs;
large weights, graphs and KV/logit arrays excluded from Git.
