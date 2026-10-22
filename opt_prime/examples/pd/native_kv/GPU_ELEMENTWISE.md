# Elementwise fusion and HTTP delivery experiments

- **Date**: 2026-10-22
- **Task**: aicomp #146 / PR #147
- **Host / device**: GPU+RNGD host; A10080GB PCIe + RNGD
- **Commit**: 3b3d09d parent plus gpu_activation_manifest.json source hashes

## Question

Can reducing GPU elementwise kernels and delivery work close the remaining native
serving gap without changing the tested numerical behavior?

## Setup

Same pinned Qwen3-8B-FP8 checkpoint/FXB and SDKs as GPU_MULTIPLEX.md. Actual config:
intermediate_size12288, hidden4096,36layers. GPU uses Marlin W8A16, fully fused norms,
fused gate/up and QKV; native uses one RNGD. Our path adds an A100, so hardware budgets differ.

--fused-elementwise combines SiLU+up multiplication, preserving intermediate BF16
rounding, and post-attention residual addition+the existing fused RMSNorm. It requires
--fuse-mlp and --norm-mode fused; default off. The residual/norm kernel does not claim
equivalence to the original unfused HF norm reduction.

GPUHTTPServer's TCP connection backlog increases from5 to128; the worker queue limit
remains16. Optional --async-text sends raw token events to request handlers, moving
incremental detokenization off the GPU owner. Unicode stabilization, EOS accounting,
SSE termination and output queue backpressure are retained; default off.

## Command

From native_kv, MODEL_DIR is the pinned local checkpoint:

```bash
python3 -m unittest discover -s . -p test_gpu_activation.py
python3 bench_activation.py --model-dir "$MODEL_DIR" --out paired_v2.json
python3 bench_activation.py --model-dir "$MODEL_DIR" --residual-norm --out combined.json
```

With native server as in GPU_HTTP.md:

```bash
python3 serve_gpu.py --model-dir "$MODEL_DIR" --norm-mode fused --fuse-mlp --fuse-qkv --pipeline --batch-size 8 --native-batch-size 8 --native-batch-wait-ms 8 --fused-elementwise --profile-stages stages_backlog.jsonl
python3 bench_http.py --lengths 512 --concurrency 8 --repeats 5 --out backlog_c8.json
python3 check_batch_http.py --model-dir "$MODEL_DIR" --concurrency 8 --out backlog_quality.json
```

Add --async-text to the server for async_text_*.json runs. Streaming/nonstream quality:

```bash
python3 check_http_quality.py --model-dir "$MODEL_DIR" --out async_text_quality.json
```

After stopping GPU HTTP, keep native for the worker check:

```bash
python3 check_pipeline_backpressure.py --model-dir "$MODEL_DIR" --batch-size 8 --native-batch-size 8 --fused-elementwise --async-text --out backpressure.json
python3 -m unittest discover -s . -p 'test_*.py'
```

## Results

Paired isolated GPU decode, eight lanes sharing one512-token synthetic prompt,128
outputs each, one warmup plus five alternating measured repeats. Excludes native,
import and HTTP:

| Variant | Baseline ms/step | Candidate ms/step |
|---|---:|---:|
| SwiGLU fusion |10.5151|10.2602|
| SwiGLU + residual/norm |10.5184|10.2219|

All1024 token positions agree in every paired run. SwiGLU tests cover random gate/up
values at several scales and every BF16 gate bit pattern with unit up; finite results
match, NaNs compared as NaN. Residual/output norm match existing fused norm tests.
Initial graph creation FAILED because the prototype assumed14336 intermediate width;
the installed checkpoint config is12288. Fixed before paired_v2/combined measurements;
paired.log retains the failure. Earlier14336-width tests are not checkpoint validation.

Actual HTTP SSE input512/output128, eight clients/eight lanes, five alternating
measured runs after warmup (all use native batch8 and8ms gather):

| GPU configuration | Native tok/s | GPU tok/s | Native TTFT | GPU TTFT | GPU TPOT |
|---|---:|---:|---:|---:|---:|
| Elementwise fusion, backlog5 |577.49|535.25|264.45ms|393.25ms|10.59ms|
| Elementwise fusion, backlog128 |580.80|538.72|260.62ms|393.74ms|10.59ms|
| Same + handler detokenization |568.50|525.34|269.69ms|444.21ms|10.46ms|

Backlog128 admits all40 measured requests in full native groups of8; warmup split6+2.
It removes the observed measured admission split, but does not prove a throughput gain.
Other compiler CPU work was present and was not changed. Separate runs and native
variation prevent attributing aggregate throughput differences purely to the changes.

Fifteen mixed questions match native/solo/concurrent text after the backlog change.
Fifteen native/nonstream/SSE texts match with handler detokenization. Unit HTTP checks
cover partial Unicode, usage/EOS accounting and nonmonotonic-detokenization errors
(500 JSON or SSE error, no false DONE).

Final validation:37 tests pass. Live fused-elementwise/raw-token worker check preserves slow-consumer resume text, peer128-token completion and cancellation followup. Owned servers stopped.

## Conclusion

GPU elementwise fusion saves about2.8% of isolated decode-step time while retaining
tested outputs. End-to-end throughput has NOT improved beyond the previous541tok/s
measurement: native prefill/export/admission latency still dominates the remaining gap.
Async text processing reduces measured TPOT but worsens overall throughput in this run;
it remains default off. Fusion also stays explicit opt-in pending broader serving data.

Next: profile actual GPU kernels and native KV delivery; assess GPU-local prefill as a
latency-routing candidate without dropping the heterogeneous-serving objective. Wider
length/concurrency and quality validation remain required before claiming native parity.

Evidence: analysis/native_kv_activation/. Final source includes a later detokenization
error-response fix and expanded test CLI; successful runtime paths measured here are
unchanged. No model weights or raw KV published.
