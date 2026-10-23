# Eight GPU lanes and prefill-side KV packing

- **Date**: 2026-10-22
- **Task**: aicomp #146 / PR #147
- **Host / device**: GPU+RNGD host; A10080GB PCIe + RNGD; NVIDIA580.178.04 / Furiosa2026.4.0
- **Commit**: 10b1729 parent plus source hashes in gpu_eight_manifest.json

## Question

Can eight execution lanes and CPU packing outside the decode owner reduce the
concurrent-serving gap, and does a controlled width comparison explain numerical divergence?

## Setup

Same Qwen3-8B-FP8 pinned checkpoint/FXB as GPU_BATCH.md, Torch2.10.0+cu128,
vLLM0.18.0 and transformers5.1.0 on GPU. Native baseline uses one RNGD; our path
uses RNGD prefill plus an A100, Marlin W8A16 and fused norms/MLP/QKV. Different
hardware budgets and arithmetic; no cost-normalized claim.

Prefill owner packs validated CPU KV into an owned pinned packet before handing it
to the GPU owner. GPU owner performs only the upload/cache copy and completion
fence. Default adaptive-single remains disabled. Eight GPU lanes are explicit CLI
configuration. No default serving change.

## Command

From native_kv with MODEL_DIR set; native server as in GPU_HTTP.md:

```bash
python3 check_batch_width.py --model-dir "$MODEL_DIR" --out teacherforced.json
python3 serve_gpu.py --model-dir "$MODEL_DIR" --norm-mode fused --fuse-mlp --fuse-qkv --pipeline --batch-size 8
python3 bench_http.py --lengths 512 --concurrency 4 --repeats 3 --out packed_c4.json
python3 bench_http.py --lengths 512 --concurrency 8 --repeats 3 --out packed_c8.json
python3 check_batch_http.py --model-dir "$MODEL_DIR" --concurrency 8 --out quality.json
```

Stop GPU HTTP before the worker test (it owns the same native receiver socket):

```bash
python3 check_pipeline_backpressure.py --model-dir "$MODEL_DIR" --batch-size 8 --out backpressure.json
python3 -m unittest discover -s . -p 'test_*.py'
```

## Results

Three measured alternating HTTP SSE runs after one warmup, input512/output128,
same prompt IDs in each native/GPU comparison:

| Concurrency | Native tok/s | GPU tok/s | Native mean TTFT | GPU mean TTFT |
|---|---:|---:|---:|---:|
| 4 | 330.57 | 276.64 | 162.02 ms | 342.45 ms |
| 8 | 580.18 | 416.54 | 260.38 ms | 607.59 ms |

C8 GPU TPOT10.91ms versus native11.70ms. Prior four-lane pipeline C8 was311.30tok/s
and1073.99ms TTFT, from a separate run. This experiment changes both lane count and
packing ownership; their individual effects are not isolated. Host contention is uncontrolled.

Fifteen mixed question/retrieval cases match native, solo GPU and concurrent8-client
text. This remains a small quality check, not general numerical equivalence.

Teacher-forced diagnostic uses the same GPU-created127-token prefix and forces the
same128 subsequent input tokens into three independent caches: width4 reference,
width1 and switching1/4. Original prefix KV stays bit-identical; positions always
advance as expected. Width1 argmax matches126/128; switching matches125/128.
Reference top-two margins at mismatches are0/0 for width1 and0/0/0.125 for switching.
First observed new-KV/logit difference is step3 (maximum absolute0.125). Numerical
differences exist even with constant width and identical forced inputs, supporting a
width-dependent numerical explanation. This does not prove the original live adaptive
failure has no additional state/scheduling cause; adaptive stays opt-in.

Failure records:
- First eight-lane request aborted on a false pinned-memory validation failure.
  PyTorch's registered default accelerator is RNGD: `is_pinned()` is false while
  `is_pinned('cuda')` is true for this CUDA-pinned packet. Validation now names CUDA
  explicitly; that argument is deprecated upstream but necessary for this installed
  accelerator combination. Initial HTTP failure is in c4.log.
- An intermediate diagnostic error string had an f-string syntax error; startup and
  unit import failed (packed_test.log), fixed before measured runs.
- Packed-memory ownership test passes: source frame reuse and packet reuse after
  upload preserve exact KV; changing/reusing lane0 leaves lane1 sentinel bytes intact.

Final validation:30 tests pass. Eight-lane live worker backpressure test holds32 events while its peer completes128 tokens; resumed text matches, and cancellation followup completes128 tokens. Owned servers stopped.

## Conclusion

Eight-lane serving reaches about72% of native C8 throughput and reduces first-token
waiting relative to the prior four-lane run. It still does not meet native performance.
Serial native prefill/export and import admission remain candidates for profiling;
more lanes alone do not close the gap. Next measure stage costs and examine bounded
multi-request native KV prefill, while preserving isolation and testing broader quality.

Local evidence: analysis/native_kv_eight and analysis/native_kv_width/teacherforced.json.
The teacher-forced diagnostic ran before CPU packing moved to the producer; decode graph arithmetic was unchanged. Final HTTP runs and ownership/backpressure checks use the final implementation.
