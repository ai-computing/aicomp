# Qwen3-4B B4 lane-dependent logits

This reproduction is for engine/SDK diagnosis, not a correctness waiver or a serving fix.

## Observed failure

Qwen/Qwen3-4B, ATOM B4/TP1, five KV blocks, context9216. At input2048/output index13, forced identical history gives lane0 token369 with logits(369,264)=(29.75,29.625); lanes1–3 choose264 with(29.625,29.625). KV block permutation and fixing prefill batch_idx0 do not remove the split. Same-Pod ABBA completes128 cases with identical tokens and recorded logits across arms. Four bootstrap records verify the same assigned physical ATOM.

Earlier direct SDK replay bypasses AtomBatcher and HTTP entirely and reproduces this split with identical per-lane decode tokens/positions. It also reproduces the512-input/index50 split (4829 versus1033). Native SDK output is bfloat16. This narrows reproduction to the SDK/compiled decode path; the responsible operator and preferred numerical result remain unidentified.

## Environment and prerequisites

Measured runtime is the PR266 stack (eaf7daa219a6cea68db4ec5fc08850edc848b271), rebel-compiler0.11.0, optimum-rbln0.11.0.post1, torch2.11.0+cpu, Transformers5.8.1. Use the previously compiled Qwen3-4B B4/k5/context9216 artifact. The scripts load from an atom270-artifacts directory alongside themselves. Provide the existing container_bootstrap270.py and atom_server270_http.py validation helper through the task harness/PYTHONPATH. Allocate one ATOM; no RNGD is needed. Do not reuse an unrelated workload or regroup devices. This package is not standalone against main.

The compact fixture contains only the exact synthetic prompt IDs and reference tokens consumed by these scripts. It is an exact subset of the preserved native HTTP result; the source hash is in samecard-evidence-manifest.json. Reference tokens are replay inputs, not an independent quality oracle.

## Run

Copy the source scripts and fixture to a fresh task directory with the prerequisites above. The measured scripts retain their original input filename:

```bash
cp qwen4-lane-fixtures.json atom270-qwen4b-tp1-b4-native.json
python3 selfcheck_atom270_direct.py
"$ATOM_PYTHON" container_bootstrap270.py diagnose_atom270_direct.py
for arm in a1 b1 b2 a2; do
  "$ATOM_PYTHON" container_bootstrap270.py diagnose_atom270_samecard_retry1.py "$arm"
done
```

Each arm must use a fresh Python process within the same allocated Pod. Re-loading engines in one Python process failed with SDK INIT_INTERNAL after the first arm; that failed attempt is preserved in the research record. All output files must be absent before starting. Scripts reject overwrites. Keep the allocation unchanged and retain all four ATOM270_BINDING records.

## Next investigation and acceptance

- Obtain an SDK-supported intermediate-output/profiling path to identify the first divergent decode operator, preserving identical KV and forced history.
- Evaluate any compiler/runtime candidate against both2048/index13 and512/index50 reproductions, all active lanes and KV placements, then free generation and held-out prompts.
- Re-run matched native/whole/stream serving and fault checks before a B4 promotion. Do not change greedy tie-breaking merely to hide differing logits.

Same-card raw arrays are archived outside Git; compact audit and full hashes are included. Remote synchronization is unverified. B1 remains the existing default and B4 parity remains failed; no production engine change is included.
