# MoE training through Optimus_p: probe results (#61)

`examples/train_probe.py` checks three things for a tiny random model of a transformers family, trained 5 steps on a fixed batch through `Optimus_p`:
1. The first step's loss equals the plain Hugging Face model's loss on the same batch (fp32).
2. The loss decreases.
3. With LoRA, the base weights stay unchanged.

- **Environment:** torch 2.13.0+cu130, transformers 5.15.0, `--dynamo-capture`, A100 80GB.
- **Commit:** `optimus-rm` 408ecb1.
- **Command:**

  ```bash
  torchrun --nproc_per_node=<pp> examples/train_probe.py --family <family> --pp-size <pp> --dynamo-capture [--lora]
  ```

| family | pp | mode | HF loss | first step | last step | base unchanged | verdict |
|---|---:|---|---:|---:|---:|---|---|
| qwen3_moe | 1 | full | 4.85283 | 4.85283 | 4.23921 | True | PASS |
| qwen3_moe | 1 | LoRA | 4.85283 | 4.85283 | 4.84743 | True | PASS |
| qwen3_moe | 2 | full | 4.85283 | 4.85283 | 4.23921 | True | PASS |
| qwen3_moe | 2 | LoRA | 4.85283 | 4.85283 | 4.84699 | True | PASS |
| solar_open | 1 | full | 4.85167 | 4.85167 | 4.28898 | True | PASS |
| solar_open | 1 | LoRA | 4.85167 | 4.85167 | 4.84679 | True | PASS |
| solar_open | 2 | full | 4.85167 | 4.85167 | 4.28898 | True | PASS |
| solar_open | 2 | LoRA | 4.85167 | 4.85167 | 4.84694 | True | PASS |
| gemma4_text | 1 | full | 4.82818 | 4.82818 | 4.36535 | True | PASS |
| gemma4_text | 1 | LoRA | 4.82818 | 4.82818 | 4.78679 | True | PASS |
| gemma4_text | 2 | full | 4.82818 | 4.82818 | 4.41205 | True | PASS |
| gemma4_text | 2 | LoRA | 4.82818 | 4.82818 | 4.78612 | True | PASS |

All 12 runs pass. The first-step loss matches the HF loss to all printed digits (5 decimals). `trainable` (in the logs) is the last stage's count for pp=2.

The real-checkpoint part of #61 (a real MoE checkpoint trained with experts offloaded) is recorded in `docs/PR_SERIES_KO.md` (verification summary): Qwen3-30B-A3B 3.21 → 0.38, gemma-4-26B 6.42 → 0.72, Solar-Open-100B 3.53 → 0.60.
