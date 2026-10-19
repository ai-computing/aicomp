# Native shared host transport diagnostic
- Date:2026-10-19
- Task:PDDISAG-6
- Commit:ecf9576 + dirty shared slab/validator
- Host:etri-001 RNGD npu3ch0 + ATOM rbln0; same pinned SDKs as native-live

## Question/setup
Remove NPZ I/O from native host experiment. Not heterogeneous P2P improvement.
B1/context512/input32,64,41,37/output32. Warm both paths then A/B/B/A.

## Command
Same commands in experiments/2026-10-19-native-live.md, adding --compare-shared
and --out experiments/native_shared_20261019/q05-r3 or q15-r1.
Exact argv/source snapshots are in each run provenance.json and sources/.

## Results
0.5B 2.252818→1.916502s;1.5B5.964626→5.018920s per8 measured requests.
24requests768tokens/model including warmup, all full logits exact.38CPU tests.
Initial0.5B r1/r2 no dedicated warmup retained separately. No failed runs.

## Conclusion
Host-only diagnostic. User clarified P2P priority; do not promote to serving or
present as P2P improvement. Existing lib and external KV path confirmed in code.
Native host KV API alone lacks verified export ownership needed by existing lib.
Next: optimize existing P2P path; native cache export requires separate validation.
