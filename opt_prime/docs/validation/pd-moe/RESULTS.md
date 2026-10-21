# PD on real MoE checkpoints, with and without expert offloading (#56)

`examples/pd/validate_moe_pd.py` compares two paths on 5 standard prompts (English, Korean, code):
- the PD path: `TorchCausalLMEngine` prefill → `TorchCacheTransport` → `TorchCausalLMEngine` decode, both engines sharing one model object;
- the single-engine reference: Hugging Face `generate` (greedy, no EOS stop) on the same model.

With `--offload-gpu-gb`, the experts are offloaded first (`apply_moe_offload`: hot experts on the GPU within the
budget, the rest on the host CPU), so both paths run the offloaded model.

- **Environment:** bf16, torch 2.13.0+cu130, transformers 5.15.0, 1× A100 80GB, `optimus-rm` 408ecb1.
- **Result:** PD on an offloaded model needs no engine change: the offloaded experts module is part of the model's
  forward.

| model | offload (tier: experts) | new tokens | prompts token-exact | KV transfer |
|---|---|---:|---:|---:|
| Qwen3-30B-A3B-Instruct-2507, whole model on 1 A100 | none | 32 | 5 / 5 | ≤ 1.5 ms |
| Solar-Open-100B, experts offloaded (GPU 40 GiB) | cuda: 1365, cpu: 4779 | 16 | 5 / 5 | ≤ 1.5 ms |

Per-prompt rows, timings and decoded text are in the JSON files next to this one. `reference_s` and `pd_s` are
single runs on a shared host, not a benchmark.

Commands:

```bash
python examples/pd/validate_moe_pd.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --out <dir>
python examples/pd/validate_moe_pd.py --model upstage/Solar-Open-100B --offload-gpu-gb 40 --max-new-tokens 16 --out <dir>
```
