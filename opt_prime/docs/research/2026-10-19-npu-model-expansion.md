# Additional checkpoints and stage profiling on actual NPUs

- Date: 2026-10-19 UTC
- Task: PDDISAG-6
- Host: etri-001; physical ATOM2, RNGD3 (logical2)
- Code: aicomp-integration feat/optimus-heterogeneous-pd, base143e618 plus recorded source changes.
- Evidence directory: experiments/npu_model_expansion_20261019
- Skills: benchmark-design and experiment-tracking.

## Question and protocol

Validate actual NPU execution of additional pretrained models and measure native
prefill chunk tuning. Independently profile existing KONI heterogeneous P2P PD.
No CPU performance result is included in NPU performance claims.

ATOM: Qwen2.5-0.5B and1.5B local immutable checkpoints. B1, context512, inputs32/128/256,
outputs8/32, greedy fixed length ignoring EOS. Prefill chunks64/128, one warmup
per shape and three measured repetitions. Compile all artifacts before the final
measurement. Run64/128/128/64 (ABBA) per model; preserve every run.
The initial0.5B c64 run overlapped another CPU compilation and is exploratory,
excluded from the controlled ABBA comparison. SDK-declared BF16 is not evidence
of a native BF16 arithmetic peak or equivalence with RNGD arithmetic.

RNGD FXB dry-run rejects both unquantized Qwen checkpoints with no matching model
registry entry. This is specific to the installed native FXB registry, not proof
that the hardware cannot implement Qwen. Existing custom KONI graphs hardcode
hidden/head/layer dimensions and omit Qwen Q/K/V biases; they cannot be safely
reused by changing only model ID. Qwen uses tied embeddings;0.5B has head_dim64 and1.5B has head_dim128.
KONI has a different number of KV caches and KV heads.

KONI: unchanged model/artifact/worker/P2P fences, B4 graph with one live lane,
inputs140/167/227/271, outputs1/32, repeats3. Added host-wall stage timings around
fenced prefill, reserve/validation, transfer, decode and finish calls. Full logits
remain compared against same-decoder host admission. This is a diagnostic
single-lane test, not saturated throughput, native model-quality parity or MFU.

## Commands and provenance

Each native run saves invocation.json (exact argv, Python, start time, SHA/status),
build-config.json, versions.json, artifact hashes, actual physical device ownership,
runs.jsonl and result.json. matrix-commands.json contains the complete final ABBA
commands; run_matrix.py executes them sequentially and stops on failure.
Source hashes, source snapshots and source.diff are in the evidence directory.

RNGD registry probes:
```
/home/gspark/venvs/furiosa-2026.4/bin/fxb build <local-model-snapshot> /tmp/qwen05-native-test.fxb --max-model-len 512 -O O0 --dry-run
/home/gspark/venvs/furiosa-2026.4/bin/fxb build <local-model-snapshot> /tmp/qwen15-native-test.fxb --max-model-len 512 -O O0 --dry-run
```
Snapshots are recorded in native build-config.json. Initial incorrect CLI
`furiosa-llm fxb build` failed argument parsing before compilation; corrected to
standalone `fxb`. Both corrected registry probes exited1; logs retained.

KONI command (cwd aicomp-integration):
```
PYTHONPATH=opt_prime:../hetero-npu-p2p/src:../rngd_llmd/src /home/gspark/venvs/furiosa-2026.4/bin/python opt_prime/examples/pd/validate_koni.py --config /home/gspark/workspace/01_pd_disag/experiments/npu_model_expansion_20261019/koni-profile-config.json --hetero-repo /home/gspark/workspace/01_pd_disag/hetero-npu-p2p
```

## Results

Completed:144 controlled native measurements,48 warmups; initial18 exploratory
measurements retained separately. Both Qwen checkpoints executed on ATOM2.
All chunk/repetition trajectories agree; FP32 teacher-forced top1 agreement is
120/120 per checkpoint (240/240 total). Inputs128/256 benefit from chunk128:
prefill about25–36% lower, total speedup1.011–1.105x. Inputs32 regress with
speedup0.940–0.994x. No universal chunk default was changed.

KONI PD:24 requests,396 output tokens,1536 successful DMA commands, full-logit
exact versus same-decoder host admission. Output32 medians: prefill57.214ms,
transfer16.447ms, decode2348.696ms, total2425.441ms. Single live lane in B4,
including validation overhead; not saturation throughput or kernel-only time.
All own device processes exited, post-run snapshots show no NPU contexts.

Final runtime tests:17 passed, including prevention of stale successful timing
on a failed request. Quality references ran only after NPU timing completed.
Graph and raw matrix: analysis/ under the evidence directory. Full interpretation
and supported/unsupported paths: aicomp-integration/opt_prime/docs/NPU_MODEL_VALIDATION.md.

Qwen native RNGD/heterogeneous PD remains unimplemented: native FXB registry
rejects the two unquantized checkpoints, while custom graphs need geometry,
Q/K/V bias, tied-weight and physical KV format extensions. Do not label native
ATOM tests as successful Qwen PD tests. Next: versioned model-specific graph
builders and cache-layout capability validation before P2P reuse; for KONI,
profile full batches and verification overhead before prioritizing transfer.
