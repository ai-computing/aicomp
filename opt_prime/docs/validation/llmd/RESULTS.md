# llm-d feature validation — 2026-10-19

Base engine: `0c54548`; feature branch: `feat/optimus-llmd`.

| Validation | Outcome | Evidence |
|---|---|---|
| Unit/model/HTTP tests | 18 passed | `cpu-tests.log` |
| Actual tiny Llama + upstream sidecar | 3 requests, 18 tokens, exact direct-engine tokens | `torch/responses.json`, `torch/result.json` |
| Actual KONI direct baseline | 8 requests, 132 tokens, full-logit host-admission parity | `koni/direct-requests.jsonl` |
| Actual KONI + upstream sidecar | 8 requests, 132 tokens, direct tokens exact and full logits exact | `koni/responses.json` |
| Sidecar-route P2P | 512/512 commands, every CQE result 0 | `koni/audit.json` |
| Direct + sidecar P2P total | 1,024/1,024 commands | direct and sidecar raw receipts |
| Lifecycle | Correct prefill-ready → decode-start → decode-complete ordering for all 8 IDs; worker exit0; device contexts empty | `koni/events.json`, `owner-result.json`, `post-*.txt` |

Real KONI settings are the same model revision, SDKs, physical RNGD3/logical2 and
ATOM2 as the engine validation, with four prompts140/167/227/271, outputs1/32,
one repeat, one active B4/context512 slot. The direct comparison used identical
prompts and output counts immediately before sidecar requests. Inference/transport
code in the engine base commit was unchanged. Source hashes are recorded in
`koni/source-manifest.json`; its snapshot is in `/tmp/optimus-llmd-koni-r1/source`.

The upstream source is commit `5f4e762f341a5196393ce79f8a57c3e1900c4a6b`.
The source diff was compared byte-for-byte with the recorded loopback-only patch.
The binary SHA256 and command line are captured per run. Existing unrelated
sidecar processes were left untouched; this harness allocated new loopback ports.
Own sidecars exited0 after normal SIGTERM once device work had completed; the
ATOM owner stopped through its explicit stop command and exited0.

Commands (from `aicomp-llmd`):

```bash
PYTHONPATH=opt_prime:../hetero-npu-p2p/src python3 -m unittest discover -s opt_prime/tests -v
PYTHONPATH=opt_prime python3 opt_prime/examples/pd/validate_torch_sidecar.py \
  --sidecar ../experiments/native_pd_2026_4/llmd_integration/pd-sidecar \
  --out /tmp/optimus-llmd-torch-r1
PYTHONPATH=opt_prime:../hetero-npu-p2p/src:../rngd_llmd/src \
  /home/gspark/venvs/furiosa-2026.4/bin/python opt_prime/examples/pd/validate_koni.py \
  --config /tmp/optimus-llmd-koni-config-r1.json --hetero-repo ../hetero-npu-p2p \
  --llmd-sidecar ../experiments/native_pd_2026_4/llmd_integration/pd-sidecar
```

Use fresh output paths to reproduce; the actual config is retained beside this
report. Weights, compiled binaries and large host KV/full-logit arrays are external,
not committed. All requests and raw receipts are retained.

This proves local sidecar integration and supported-path correctness. It does not
prove Kubernetes/EPP deployment, cross-node transfer, streaming, sampling, maximum
throughput, MFU/power or numerical equivalence to vendor-native serving.
