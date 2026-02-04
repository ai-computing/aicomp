# ProfileDB Generation

Generate FX graph node-level ProfileDB (`.npz` files) for FASOP optimizer.

## Directory Structure

```
make_profiledb/
├── profile_fx.py           # Main profiling script (FX Interpreter-based)
├── make_profiledb.sh       # Profile all TP configurations (TP=1,2,4,8)
├── run_distributed.sh      # Single configuration execution (torchrun)
├── make_npz.py             # JSON to NPZ converter
├── optimus_p/              # Optimus-Prime core modules (local copy)
│   ├── opti_pri.py         # Optimus_p class
│   ├── IR.py               # FX graph analysis, LayerProfileInterpreter
│   ├── comm.py             # Communication utilities
│   ├── schedule.py         # Pipeline scheduling (GPipe, 1F1B)
│   └── utils.py            # Logging utilities
└── results/                # Generated ProfileDB files (output)
```

## Profiling Large Models on Limited Hardware

When profiling Llama-3.3-70B on a single node, OOM (Out of Memory) errors occur. To resolve this, we reduce the number of layers from 80 to 8 for profiling.

**Why this approach is valid:** Transformer-based models use an architecture that stacks identical layers repeatedly. Since the computational cost of each layer (attention, MLP, etc.) is independent of layer position, profiling only a subset of layers can accurately estimate the computational cost of the entire model.

Modify `num_hidden_layers` in `profile_fx.py` as follows:

```python
MODEL_CONFIGS = {
    # Llama-3.3-70B-Instruct (https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
    "meta-llama/Llama-3.3-70B-Instruct": {
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 8,  # Full model has 80 layers
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "head_dim": 128,  # hidden_size / num_attention_heads = 8192/64
    },
    ...
}
```

> **Key change:** `'num_hidden_layers': 8` (reduced from 80 to 8 layers)

## Usage

### Generate All TP Configurations (Recommended)

Runs Docker container and sweeps MBS (micro batch size) from 1 to 128 (powers of 2) until OOM for each TP configuration.

**Requires `LLAMA_ACCESS_TOKEN` environment variable** (HuggingFace token for Llama models):

```bash
# Single node only (TP=1,2,4)
LLAMA_ACCESS_TOKEN=hf_xxxxx ./make_profiledb.sh 70B

# With multi-node support (TP=1,2,4,8) - auto SSH to node1
LLAMA_ACCESS_TOKEN=hf_xxxxx ./make_profiledb.sh 70B 10.0.0.1 node1 /home/user/aicomp/fasop
```

> **Note:** Get your token from https://huggingface.co/settings/tokens

**Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| MODEL_SIZE | No (default: 70B) | Model size: `1B`, `3B`, `70B` or full HuggingFace name |
| MASTER_ADDR | No (default: 127.0.0.1) | Master node IP address |
| NODE1_HOSTNAME | No | SSH hostname for node1 (enables TP=8) |
| REMOTE_DIR | No (default: parent dir) | fasop directory path on node1 |

**MBS Sweep Behavior:**
- Tries MBS = 1, 2, 4, 8, 16, 32, 64, 128
- BATCH_SIZE = MBS × 8
- Stops when OOM is detected
- Records results to CSV: `results/<model>_all_tp_<timestamp>.csv`

This runs profiling for all predefined configurations:

| TP | PP | DP | GPUs/node | Nodes | Description |
|----|----|----|-----------|-------|-------------|
| 1  | 8  | 1  | 8         | 1     | Single node |
| 2  | 4  | 1  | 8         | 1     | Single node |
| 4  | 2  | 1  | 8         | 1     | Single node |
| 8  | 2  | 1  | 8         | 2     | Multi-node (requires NODE1_HOSTNAME) |

> **Note:** TP=8 requires SSH access from node0 to node1. If NODE1_HOSTNAME is not provided, TP=8 is skipped.

### With Docker

```bash
cd ../docker
./run_docker.sh 70B 0 127.0.0.1 1 8 True 8 1 1
```

### Single Configuration (Debug/Test)

```bash
# Single node, 8 GPUs, TP=1
./run_distributed.sh meta-llama/Llama-3.3-70B-Instruct 0 127.0.0.1 1 8 True 8 1 1

# Arguments: MODEL_NAME NODE_RANK MASTER_ADDR NNODES NPROC USE_CACHE PP TP DP
```

## Output Format

Generated `.npz` files in `results/` directory:
- Shape: `(M, 10)` where M = number of mbs values
- Columns: `[embed, attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_act_fn, mlp_up, mlp_down, lm_head]`
- Unit: milliseconds (ms)

Copy generated files to `known_cost/` for FASOP:
```bash
cp results/llama70b_A40_*.npz ../known_cost/
```

## make_npz.py

Convert JSON profile files to NPZ format.

### Usage

```bash
# Auto-detect all parameters from JSON files
python make_npz.py

# Specify model size and GPU type
python make_npz.py --model-size 70b --gpu-type A40 --tp 2

# Custom input/output directories
python make_npz.py --input-dir ./results --output-dir ../known_cost/
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | `results/` | Directory containing JSON profile files |
| `--output-dir` | `../known_cost/` | Directory for NPZ output files |
| `--model-size` | auto-detect | Model size string (e.g., "70b", "1b") |
| `--gpu-type` | auto-detect | GPU type (e.g., "A40", "A100") |
| `--tp` | auto-detect | Tensor parallel degree |

## Dependencies

- PyTorch >= 2.5.0
- Transformers == 4.46.0
- CUDA-capable GPU(s)
- HuggingFace access token for Llama models

## optimus_p Module

The `optimus_p/` directory contains a local copy of Optimus-Prime core modules.
This ensures FASOP can generate ProfileDB independently without external dependencies.

Key classes:
- `Optimus_p`: Main pipeline parallelism optimizer
- `IR_Anal`: FX graph IR analysis
- `LayerProfileInterpreter`: FX node-level profiling interpreter
