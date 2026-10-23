# Native KV interoperability experiment — 2026-10-22

## Environment and identities

- RNGD+ATOM host, 2 x AMD EPYC 9254, 96 logical CPUs, approximately 1.5 TiB RAM.
- Receiver test: one RNGD (npu0); initial native-PD capture: npu0 + npu1.
- Driver 2026.4.0 (`32ffd4e`); furiosa-llm/native 2026.4.0.
- Native extension SHA256: `ed459cf42ee5359da01093846d3d8bd260e487f2b602f5cdc71dc87743a29761`.
- FXB SHA256: `36d74b754f580bea893c68cc70147c615d1a9c4273a4580dbebaf87287400713`;
  Qwen3-8B-FP8, context 2048, original build `-tp 8` (processing elements on one RNGD).
- Checkpoint: `Qwen/Qwen3-8B-FP8`, revision `220b46e3b2180893580a4454f21f22d3ebb187d3`.
- Python 3.10; torch 2.10.0+cpu, transformers 5.19.0; CPU reference uses 4 threads.
- Branch base: `62a783114afa3531beeb7955c66da2e6aaf4ba65` (`main`).
- Scripts' content hashes at measurement are in `source_manifest.json`. Final review
  code may add validation/docs without changing the measured wire encoding.
- Host was busy (load average approximately 400 at initial capture). No latency or
  throughput conclusion; one run per case, no performance warmup/repetition claim.

## Transport

Synthetic inputs are token IDs 1000 upward, with caching disabled, temperature 0.
The original fixed-message receiver extracted 442368 bytes with no native decoder
and matched a normal native-PD payload byte-for-byte.

The parameterized receiver then generated fresh request IDs and block IDs starting
at zero, without any capture/replay inputs:

| Input tokens | Transferred positions | Batches | Bytes | Prefiller HTTP |
|---:|---:|---:|---:|---:|
| 2 | 1 | 18 | 147456 | 200 |
| 4 | 3 | 18 | 442368 | 200 |
| 16 | 15 | 18 | 2211840 | 200 |
| 33 | 32 | 18 | 4718592 | 200 |

All physical blocks were checked for missing/duplicate IDs and exact payload size
before Ack. Four-token payload, reordered by slot and requested position, was
byte-identical to the original capture despite using different destination IDs.

Handshake declares global BF16, 2048-byte blocks, one chip, axes [272321,8,128],
strides [1024,128,1]. Total bytes are `36*2*8*128*2*(input_length-1)`.
Seventy-two slot indices are separate K/V tensors, not 72 transformer layers.

## Canonical mapping and torch resume

First-layer comparison across all physical slots identifies slot 0 as rotated,
normalized K and slot 1 as V: cosine 0.99999434 and 0.99985731, respectively, for
32 positions. The tested mapping is then K=slot 2L, V=slot 2L+1, each reshaped to
`[1,8,T-1,128]`. All 36 layers are compared in `resume_cases.json`.

Reference: the exact same checkpoint's FP8 weights dequantized with their block
scales to BF16, CPU SDPA. Native dynamic activation quantization is NOT emulated.
We compare:
1. torch resume using imported native KV and the final input token;
2. torch resume using its own prefix KV and the same final input token;
3. torch full-prompt pass, to measure the reference's own numerical variation.

| Input tokens | Native-KV/torch-KV generated token equality | First-step logits RMSE | Logits cosine |
|---:|---:|---:|---:|
| 2 | 4/4 | 0.028071 | 0.999952 |
| 4 | 4/4 | 0.140514 | 0.999740 |
| 16 | 4/4 | 0.034373 | 0.999969 |
| 33 | 4/4 | 0.055005 | 0.999953 |

All four first-step argmaxes and all 16 generated tokens match this BF16 reference.
Across these cases the lowest layer K/V cosine is approximately 0.994781. These
are measured diagnostics, **not a declared tolerance or native-FP8 parity pass**.
The four-token input has a larger logit deviation than the other cases. Additional
real prompts and a matching arithmetic reference are necessary before serving.

Initial native-PD vs native-standalone text comparison matched only 3/4 cases
(an earlier run with prompt lengths 1, 2, 4 and 16, not the receiver's 2/4/16/33; the length-2 case differed). That is a separate comparison from
the successful torch-resume tests and remains unexplained.

## Verification and failures

- Ten CPU protocol tests pass, including malformed/truncated frame rejection.
- The generated Pull header matches a captured header byte for byte (`test_protocol.py`); Handshake and Ack were compared against captures by hand and have no fixture test.
- Initial handwritten fixture hex accidentally included extra bytes; corrected by
  copying the actual capture headers, then all ten tests passed.
- First full-model reference attempt failed because transformers 5.19.0's
  Qwen3RotaryEmbedding constructor does not accept `device=`. Removed the argument;
  the single-case and four-case reruns completed.
- All owned hardware processes terminated; final device process listing was empty.
- Sensitive-data scan: seven pre-existing findings in unchanged base files, same
  findings on this branch; zero newly introduced findings. Public files contain
  no site paths, private addresses, binaries, weights or KV payloads.

## Remaining

Native FP8 parity, longer/real prompts, prefix caching, other models/builds,
error recovery, sustained concurrency, and ATOM cache import remain unvalidated.
This experiment does not activate an Optimus backend or change default serving.
