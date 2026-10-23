# Live RNGD-to-GPU cancellation and producer replacement

- **Date**: 2026-10-24
- **Task**: Four-model serving matrix (aicomp #201)
- **Host / device**: GPU+RNGD host; A10080GB GPU and one RNGD; Furiosa Torch/kernel2026.4; GPU Torch2.13.0+cu130
- **Commit**: runtime32ce5d5; HTTP9e442cd plus readinessd0f6f12

## Question
Can live GPU PD cancel an actual RNGD prefill, fail safely on producer death, release its shared memory and recover with a replacement producer?

## Setup
Four models Qwen3-0.6B/Qwen3-4B/KONI-8B/Gemma3-1B. BF16 GPU decode,4slots/paged64/context9216/cacheoff; live RNGDwidth8192. Tests run after performance workloads finish; empty GPU/NPU process lists checked between models. Observer writes atomic actual-prefill start/done markers; no replay/delay gate. Before/after output comparison uses128-input32-output PD control, checked against nativeGPU. Actual fault victims use8192-input32-output. Kill only owned producer.

## Command
```bash
/home/gspark/venvs/pt213/bin/python analysis/serving-matrix-201/run_gpu258_faults_after_matrix.py 1211034 > analysis/serving-matrix-201/local258-faults-queue.log 2>&1
```
This sequentially runs check_gpu201_faults.py; exact per-model argv retained in queue log. Environment/runtime paths are recorded in the launcher.

## Results
Qwen3-0.6B and Qwen3-4B completed: lifecycle_passed=true,passed=true,native_tokens_equal=true. Cancel during actual prefill returns499 and then identical recovery tokens. Producer termination yields pending500, later admission503,/ready503; old shared-memory slab is unlinked. Replacement makes readiness200 and returns identical32tokens. Both raw local201-{qwen06b,qwen4b}-faults.json retained.
KONI now loading; Gemma pending. No total-four-model claim yet.

## Conclusion
Two models pass cancellation/drain/death/readiness/replacement checks. Qwen4 long-output numerical failure remains independently failed. This suite does not test another already-decoding GPU request surviving producer death (covered only by separate ATOM-prefetch suite). Performance is not measured by these fault tests.

## Portable invocation
```bash
export PYTHONPATH=opt_prime:opt_prime/examples/pd/serving_matrix:/path/to/research_producer_factories:/path/to/rngd_research_src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python opt_prime/examples/pd/http_matrix/live_gpu_faults.py --model /path/to/model --factory long_width_variant:make_prefill --worker-python /path/to/furiosa-worker-python --reference opt_prime/examples/pd/http_matrix/evidence201/local201-qwen06b-matrix-a-reference.json --http-module opt_prime/examples/pd/http_matrix/evidence201/llmd_readiness.py --markers /tmp/fault-markers --output /tmp/fault-result.json
```
Worker interpreter must expose Furiosa Torch2026.4, research producer modules/factories from the live PD work (#235/#245), and the bundled observer_factory. Its model/factory must match the supplied native reference. The proprietary SDK and model weights are not bundled. Evidence contains completed first two models only at this stage. Long-output numerical failures remain separate. Markers neither modify KV nor impose an artificial execution gate.
