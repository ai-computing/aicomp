# Fixed-output ATOM and RNGD-to-ATOM HTTP comparison

- **Date**: 2026-10-24
- **Task**: aicomp #248
- **Host / device**: RNGD+ATOM host, one scheduler-assigned RNGD and ATOM; ATOM B1
- **Commit**: aba3ad2 plus uncommitted external-server config/ATOM launcher in aicomp-heteroperf-248; runtime core candidates pd_batching238/atom_batching236/llmd214

## Question

Compare nativeATOM and real RNGD-prefill/ATOM-decode using identical token-ID prompts, fixed output lengths and the same HTTP measurement client.

## Setup

Isolated experiment Podperf248 in codex-serving-matrix-201, deadline7200seconds,128GiB memory. Reuse validated PYTHONUSERBASE/ATOM bootstrap and cached context9216artifacts. Server asserts exactly one exposed ATOM and bootstrap checks actual opened descriptors. Qwen3-0.6B first, input128/8192,output32,client concurrency1/4,three repeats. Physical batch1; concurrent clients queue. Prefix caches off. Producer factorylong_width_variant:make_prefill,width8192whenPD. Do not infer HWbatch4.

## Command

From the allocated container's scratch directory:

```bash
bash run-benchmark248.sh --model Qwen/Qwen3-0.6B --mode native --server-config atom248-qwen06b-native-config.json --lengths 128 8192 --outputs 32 --concurrency 1 4 --repeats 3 --slots 1 --output atom248-qwen06b-native-a.json
```

The external-server config uses atom_server248.py through the existing container_bootstrap.py and ATOM interpreter; scripts are copied from aicomp-heteroperf-248. Configs preserved in /tmp/perf248-configs locally and experiment scratch remotely. PD command additionally selectsPDconfig and uses the native JSON as--reference. Current native launch is live; no performance result yet.

## Results

External-config red/green validation:1failed/1passed before implementation;2passed after, Ruff clean. ATOM server started and bootstrap confirmed only its assigned device opened. Full native benchmark result pending. This startup is not a successful throughput/parity claim.

## Conclusion

Fixed-output heterogeneous benchmark is connected to actual assigned hardware. Finish native, compare PD, retain any parity failure, then expand to other models. Release only the owned Pod when the runs end.

NativeATOM Qwen3-0.6B completed12batches,passed=true,parity_passed=true. RawJSON copied to analysis/serving-matrix-201/atom248-qwen06b-native-a.json. Started identical PDcommand with --mode pd --server-config atom248-qwen06b-pd-config.json --reference atom248-qwen06b-native-a.json --output atom248-qwen06b-pd-a.json. PD result pending.

## PD startup failure and corrected benchmark deadline

Initial PD run exited1 with TimeoutError("server startup") at the benchmark's180second readiness deadline; zero timing batches completed. ATOM was bound correctly; producer was still starting. SubprocessPrefillWorker's existing startup_timeout is1800seconds, so the outer benchmark terminated initialization earlier than the worker's intended bound. The reason for slow producer initialization is not yet proven. Resource-tracker warning reported cleanup of one shared-memory object after group termination; subsequent process inspection found no worker/controller from that run. Failed JSON retained as analysis/serving-matrix-201/atom248-qwen06b-pd-a.json and server log on the accelerator scratch directory.

Added --startup-timeout (default180 unchanged forGPU), positive-value validation, and recorded configured deadline inJSON. Red test1failed/2passed; fixed3passed in0.08s; Ruff clean after explicitcheck=False in the CLI negative test. RetriedPD with --startup-timeout1800 and output atom248-qwen06b-pd-b.json, retaining the original failure. Loading remains excluded from measured throughput. The owned Pod remains allocated with its original deadline; retry is live and no successful PD claim is made yet.

Live retry observation: producer process still active after3m05s, averageCPU770% andRSS~12.7GiB, server waiting for producer readiness. This demonstrates a live CPU-active initialization, not successful completion or proof of the exact slow stage. Worker source loads weights toFP32 then constructs the compiled width8192engine before sendingready. Source also pads shorter prompts to compiledwidth; benchmark must not generalize fixed8192short-prompt costs to a width-dispatched implementation. No restart performed.
