# FASOP Docker Environment

Docker environment for running FX graph node-level profiling to generate ProfileDB files.

## Build Docker Image

```bash
cd docker/
docker build -t make_profiledb:latest .
```

## Usage

### Generate All TP Configurations (Recommended)

Use `make_profiledb.sh` in the `make_profiledb/` directory to automatically profile all TP configurations with MBS sweep.

```bash
cd ../make_profiledb/

# Single node
./make_profiledb.sh 70B

# Multi-node with SSH to node1
./make_profiledb.sh 70B 10.0.0.1 node1 /home/user/aicomp/fasop
```

This script:
- Runs Docker container automatically
- Sweeps MBS (1, 2, 4, 8, ..., 128) until OOM for each TP config
- BATCH_SIZE = MBS × 8
- Outputs results to `make_profiledb/results/`

See `make_profiledb/README.md` for details.

### Single Configuration (Debug/Test)

```bash
# command
./run_docker.sh <MODEL_SIZE> <NODE_RANK> <MASTER_ADDR> <NNODES> <NPROC> <USE_CACHE> <PP> <TP> <DP>

# example
./run_docker.sh 70B 0 127.0.0.1 1 8 True 8 1 1

# Arguments:
#   1. MODEL_SIZE: 1B, 3B, 70B or full model name
#   2. NODE_RANK: Current node rank (0 for single node)
#   3. MASTER_ADDR: Master node IP address
#   4. NNODES: Number of nodes
#   5. NPROC_PER_NODE: Number of GPUs per node
#   6. USE_CACHE: Use cached model (True/False)
#   7. PP: Pipeline parallelism degree
#   8. TP: Tensor parallelism degree
#   9. DP: Data parallelism degree
```

### Environment Variables

- `LLAMA_ACCESS_TOKEN` or `HF_TOKEN`: HuggingFace access token for Llama models

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Docker image definition (PyTorch 2.5.1, CUDA 12.4) |
| `run_docker.sh` | Docker container management and profiling execution |

## Docker Configuration

| Item | Value |
|------|-------|
| Base Image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |
| Container Name | `make_profiledb` |

## Output

Profiling results are saved to `make_profiledb/results/` directory:
- `llama{model_size}_{gpu_type}_{tp}.npz` - ProfileDB files
- `*.log` - Execution logs
- `*_gpustats.log` - GPU utilization logs
