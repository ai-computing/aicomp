# OptimusPrime: Highly-efficient 3D parallelization framework for training LLMs

OptimusPrime is a 3D parallelization framework designed for the efficient training of large-scale DNN models.
It analyzes the model structure in the form of a PyTorch FX graph within a deep learning cluster environment composed of multiple GPUs and nodes.
Based on this analysis, OptimusPrime derives optimized parallelization policies tailored to the hardware environment of the cluster,
enabling efficient parallel training.

## Supported models

OptimusPrime supports major HuggingFace models, and the following models have been tested:

* gpt2
* gpt2-medium
* gpt2-large
* gpt2-xl
* EleutherAI/gpt-neo-2.7B
* EleutherAI/gpt-j-6b
* bert-base-cased
* facebook/opt-6.7b
* facebook/opt-13.7b
* meta-llama/Llama-2-13b-chat-hf
* meta-llama/Meta-Llama-3-8B
* meta-llama/Llama-3.2-1B
* meta-llama/Llama-3.3-70B-Instruct
* openai/whisper-small
* google/electra-base-generator
* google/vit-base-patch16-224-in21k

## Features

* Currently supported
  * **Pipeline parallelism (PP)**: In PP, GPipe/1F1B scheduling algorithms are supported
  * **Data Parallelism (DP)**
  * **Tensor Parallelism (TP)**: For now, TP support is provided only for the Llama model

## Installation

To install OptimusPrime:

    # Make sure PyTorch >= 2.0.1 is installed (Officially tested with version 2.5.0)
    # CUDA and cuDNN libraries compatible with the PyTorch version must be installed as well (Officially tested with cuda12.4 and cudnn9-devel)
    git clone https://github.com/ai-computing/aicomp.git

## Authentication for gated models

Some models (e.g., LLaMA) require HuggingFace authentication. OptimusPrime supports three methods — use whichever is most convenient:

**Method 1: Command-line argument** — pass the token directly when running a script:

    torchrun --nproc_per_node=4 --master_port=29500 pp_train_llama5.py <your_hf_token>

**Method 2: Environment variable** — set once, then run without the token argument:

    export LLAMA_ACCESS_TOKEN=<your_hf_token>
    torchrun --nproc_per_node=4 --master_port=29500 pp_train_llama5.py

**Method 3: HuggingFace CLI login** — log in once, then run without any token:

    huggingface-cli login    # enter your token when prompted (saved to ~/.cache/huggingface/token)
    # or equivalently:
    hf auth login
    torchrun --nproc_per_node=4 --master_port=29500 pp_train_llama5.py

All three methods work for both training (`pp_train_llama*.py`) and inference (`pp_inference_llama*.py`, `pp_generation_llama.py`) scripts. Non-gated models (GPT-2, BERT, OPT, etc.) do not require authentication.

---

## Training (Fine-tuning)

### Running (Example: gpt2)

#### Single-node environment

    cd opt_prime/examples
    torchrun --nproc_per_node=<# of GPUs per node> --nnodes=<# of nodes> --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 pp_train_gpt2.py

#### Multi-node environment

Run the following command for every node:

    cd opt_prime/examples
    torchrun --nproc_per_node=<# of GPUs per node> --nnodes=<# of nodes> --node_rank=<current node rank> --master_addr=<IP of rank 0> --master_port=29500 pp_train_gpt2.py

### Configuring parallelism options

#### Configuring PP only

The most basic parallelism option is 'pp_size'. If the 'pp_size' option is omitted, its value is automatically determined based on the number of available GPUs. You may also specify 'pp_size' explicitly if desired.

    # Example of a 4-GPU setup with pipeline parallel size=4
    optimus_p = Optimus_p(model, num_mb, use_gpu=True)

    # Example of a 4-GPU setup with pipeline parallel size=4
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, pp_size=4)

#### Changing scheduling policy for PP

Pipeline parallel in OptimusPrime supports both 'gpipe' and '1f1b' scheduling options. To use either mode, open the desired script in 'opt_prime/examples' and set the 'mode' option in optimus_p.run() as shown below.

    # Pipeline parallelism uses the 'gpipe' scheduler by default
    optimus_p.run(data, labels)


    # Example of explicitly setting the gpipe scheduler
    optimus_p.run(data, labels, mode="gpipe")

    # Example of explicitly setting the 1f1b scheduler
    optimus_p.run(data, labels, mode="1f1b")

#### Configuring PP+DP

To apply 2D parallelism with PP+DP, use the 'dp_size' option when instantiating the Optimus_p class. The 'pp_size' option is applied by default even if not specified, but it can also be explicitly set.


    # Example of an 8-GPU setup with pipeline parallel size=4 and data parallel size=2
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, dp_size=2)

    # Example of an 8-GPU setup with pipeline parallel size=4 and data parallel size=2
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, pp_size=4, dp_size=2)

Configuration diagram of 2D parallelism with pp_size=4 and dp_size=2 applied simultaneously

<p align="center">
  <img src="https://github.com/ai-computing/aicomp/assets/42994087/9b3546a0-a22a-4014-95a2-420cf742e8be">
</p>

#### Configuring PP+TP
To apply 2D parallelism with PP+TP, use the 'tp_size' option when instantiating the Optimus_p class. 'tp_size' is applicable to the Llama model.

    # Example of an 16-GPU setup with pipeline parallel size=8 and tensor parallel size=2
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, tp_size=2)

    # Example of an 16-GPU setup with pipeline parallel size=8 and tensor parallel size=2
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, pp_size=8, tp_size=2)


#### Configuring PP+TP+DP

To apply 3D parallelism with PP+TP+DP, use the 'pp_size', 'tp_size' and 'dp_size' options. 'tp_size' is applicable to the Llama model.

    # Example of a 16-GPU setup with pipeline parallel size=4, tensor parallel size=2, and data parallel size=2
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, tp_size=2, dp_size=2)

    # Example of a 16-GPU setup with pipeline parallel size=4, tensor parallel size=2, and data parallel size=2
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, pp_size=4, tp_size=2, dp_size=2)

### Configuring memory optimization options

#### Offloading optimizer state to host memory during forward and backward passes
When the 'swap_opt_in_fwdbwd' option is set to True, the optimizer state is offloaded to host memory during the forward and backward passes to reduce GPU memory usage. This helps avoid GPU OOM and enables training of larger models.

    # Example of an 4-GPU setup with pipeline parallel size=4 and swap_opt_in_fwdbwd option is set to True
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, swap_opt_in_fwdbwd=True)

#### Executing the optimizer step on the CPU
When the 'swap_model_in_optstep' option is set to True, the optimizer's step() phase is executed on the CPU. This option must be used together with 'swap_opt_in_fwdbwd', and by avoiding the GPU OOM that can occur when using 'swap_opt_in_fwdbwd' alone, it enables the operation of even larger models.

    # Example of an 4-GPU setup with pipeline parallel size=4 and swap_opt_in_fwdbwd option is set to True
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True)

### Graph capture mode: `--dynamo-capture`

By default, OptimusPrime uses HuggingFace's HFTracer (built on torch.fx.symbolic_trace) to extract the model's FX graph. For enhanced model compatibility, the --dynamo-capture option enables the use of torch.export.export(). This leverages TorchDynamo to capture a comprehensive ATen-level IR, which is then reconstructed into a module-level FX graph, ensuring a more robust extraction for complex architectures.

#### Why use `--dynamo-capture`?

| | HFTracer (default) | `--dynamo-capture` |
|---|---|---|
| Capture engine | HuggingFace `HFTracer` | TorchDynamo (`torch.export`) |
| Graph completeness | May fail on some models | Full graph, no graph breaks |
| IR level | Module-level directly | ATen → Module-level reconstruction |
| PyTorch alignment | HuggingFace-specific | PyTorch official export path |

#### Usage

Add `--dynamo-capture` to any training or inference script:

    # Training (PP=4)
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_train_gpt2.py --dynamo-capture

    # Training with 3D parallelism (PP=2, DP=2, TP=2)
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_train_llama5.py --dynamo-capture <llama access token>

    # Inference (PP=4, TP=2)
    torchrun --nproc_per_node=8 --master_port=29500 pp_inference_llama.py --pp-size 4 --tp-size 2 --use-kv-manager --dynamo-capture

In Python, pass `dynamo_capture=True` to the constructor:

```python
optimus_p = Optimus_p(model, num_mb, use_gpu=True, dynamo_capture=True)
```

#### Notes

* When using `--dynamo-capture` with `torch.amp.autocast` (mixed-precision training), a lower learning rate (e.g., 1e-5 instead of 3e-5) may be needed for stable convergence.

---

## Inference

OptimusPrime provides a pipeline/tensor parallel inference engine (`Optimus_Inference`) that reuses the same FX-based IR transformation and communication infrastructure used for training. The inference engine supports autoregressive text generation with three KV cache modes.

### KV Cache Modes

| Mode | Option | Decode Complexity | Description |
|------|--------|:-----------------:|-------------|
| No KV cache | (default) | O(n²) | Full-sequence recomputation at every decode step. Simple but slow for long sequences. |
| CachedSDPA (internal) | `--use-kv-cache` | O(n) | FX graph surgery replaces SDPA with `CachedScaledDotProductAttention`, which lazily allocates and manages K,V cache tensors internally. |
| KVCacheManager (external) | `--use-kv-manager` | O(n) | `CachedSDPA` delegates K,V storage to a centralized `KVCacheManager` with pre-allocated cache. Useful for advanced memory management scenarios. |

### Quick Start — Single GPU (no torchrun)

    cd opt_prime/examples

    # Mode 1: No KV cache (full-sequence recomputation)
    python3 single_gpu_inference_llama.py

    # Mode 2: CachedSDPA internal cache
    python3 single_gpu_inference_llama.py --use-kv-cache

    # Mode 3: KVCacheManager external backend
    python3 single_gpu_inference_llama.py --use-kv-cache --use-kv-manager

### Pipeline Parallel Inference

    cd opt_prime/examples

    # PP=4, no KV cache
    torchrun --nproc_per_node=4 pp_inference_llama.py

    # PP=4, CachedSDPA internal cache
    torchrun --nproc_per_node=4 pp_inference_llama.py --use-kv-cache

    # PP=4, KVCacheManager external backend
    torchrun --nproc_per_node=4 pp_inference_llama.py --use-kv-manager

### Tensor Parallel Inference

    cd opt_prime/examples

    # TP=2, CachedSDPA internal cache
    torchrun --nproc_per_node=2 pp_inference_llama.py --tp-size 2 --use-kv-cache

    # TP=2, KVCacheManager external backend
    torchrun --nproc_per_node=2 pp_inference_llama.py --tp-size 2 --use-kv-manager

    # PP=2 x TP=2, KVCacheManager external backend
    torchrun --nproc_per_node=4 pp_inference_llama.py --pp-size 2 --tp-size 2 --use-kv-manager

### Explicit Prefill/Decode API (ScheduleGeneration)

For fine-grained control over the generation loop, `pp_generation_llama.py` demonstrates the explicit two-phase API:

1. `scheduler.prefill(input_ids)` — forward entire prompt, build KV cache
2. `scheduler.decode_step(token, position)` — forward one token per step

Example:

    # CachedSDPA internal cache
    python3 pp_generation_llama.py

    # KVCacheManager external backend
    python3 pp_generation_llama.py --use-kv-manager

    # PP=2 x TP=2
    torchrun --nproc_per_node=4 pp_generation_llama.py --pp-size 2 --tp-size 2 --use-kv-manager

### Serving Mode vs Batch Mode

When using KV cache (`--use-kv-cache` or `--use-kv-manager`), an additional `--serving-mode` flag controls cache lifecycle:

- **Batch mode** (default): Cache memory is freed after each `generate()` call. Each request pays the allocation cost.
- **Serving mode** (`--serving-mode`): Cache stays allocated between requests. Only the position counter is reset, avoiding re-allocation overhead.

Example:

    torchrun --nproc_per_node=1 serving_vs_batch_demo.py

    torchrun --nproc_per_node=4 serving_vs_batch_demo.py --pp-size 4 --num-requests 10 --max-new-tokens 50

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from opt_prime.inference import Optimus_Inference

config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B", use_cache=False)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", config=config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# --- Mode 1: No KV cache ---
engine = Optimus_Inference(model, use_gpu=True, pp_size=4)

# --- Mode 2: CachedSDPA internal cache ---
engine = Optimus_Inference(model, use_gpu=True, pp_size=4, use_kv_cache=True)

# --- Mode 3: KVCacheManager external backend ---
engine = Optimus_Inference(model, use_gpu=True, pp_size=4, use_kv_manager=True)
engine.init_kv_cache(
    num_layers=config.num_hidden_layers,
    num_heads=config.num_attention_heads,
    head_dim=config.hidden_size // config.num_attention_heads,
    batch_size=1,
)
engine._attach_kv_manager()

# Generate (same API for all modes)
engine.eval()
input_ids = tokenizer("Hello", return_tensors="pt").input_ids.cuda()
output_ids = engine.generate(input_ids, max_new_tokens=50)
```

---

## HuggingFace-compatible Checkpoint

When training with multiple GPUs using pipeline parallelism (PP) and/or tensor parallelism (TP), model parameters are distributed across stages and shards. The HuggingFace-compatible checkpoint pipeline allows you to save these distributed parameters, merge them into a single standard HuggingFace model on CPU, and then use the merged model for inference or continued training with any parallelism configuration.

### Overview

```
[Step 1] Parallel Training          [Step 2] CPU Merge Utility       [Step 3] Use Merged Model
 torchrun (N GPUs)                   python3 (CPU, one-shot)          HuggingFace compatible

 Stage 0 → stage0_tp0.pt            merge_hf_ckpt.py                 from_pretrained()
 Stage 1 → stage1_tp0.pt    ───→    Read stage files          ───→   • Inference (any PP/TP)
 Stage 2 → stage2_tp0.pt            Restore keys + TP merge          • Continued training
 Stage 3 → stage3_tp0.pt            Save single model                • Single GPU usage
```

### Step 1: Save stage checkpoints during training

Use `save_hf_ckpt()` in your training script to save per-stage state dictionaries. DDP prefixes are stripped and DTensor parameters are converted to local tensors automatically.

    cd opt_prime/examples

    # PP=4 training with checkpoint save (20 steps for quick test)
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_train_llama_into_hf_ckpt.py --max-steps 20 <llama_access_token>

This produces:

```
hf_ckpt/
├── stage0_tp0.pt
├── stage1_tp0.pt
├── stage2_tp0.pt
└── stage3_tp0.pt
```

With TP=2, each stage produces two files (e.g., `stage0_tp0.pt`, `stage0_tp1.pt`).

#### Multi-node training

In a multi-node setup, each server saves only the checkpoint files for the stages assigned to its local ranks. For example, with 2 servers (8 GPUs each), PP=8, TP=2:

```
Server 0 (rank 0~7):   hf_ckpt/stage0_tp0.pt ~ stage3_tp1.pt  (8 files)
Server 1 (rank 8~15):  hf_ckpt/stage4_tp0.pt ~ stage7_tp1.pt  (8 files)
```

If the servers share a filesystem (e.g., NFS), all files are automatically in the same directory. If not, gather the files from all servers into a single directory before running the merge utility:

    # Copy from Server 1 to Server 0
    scp server1:./hf_ckpt/stage*.pt ./hf_ckpt/

### Step 2: Merge into a single HuggingFace model (CPU)

Run the merge utility on CPU. No GPU required. It restores mangled state_dict keys to their original HuggingFace fully-qualified names and reassembles TP-sharded parameters. All stage checkpoint files must be present in a single `--ckpt-dir` directory.

    cd opt_prime/examples

    python3 merge_hf_ckpt.py --model meta-llama/Llama-3.2-1B --ckpt-dir ./hf_ckpt --output ./merged_model

    # With HuggingFace access token (for gated models)
    python3 merge_hf_ckpt.py --model meta-llama/Llama-3.2-1B --ckpt-dir ./hf_ckpt --output ./merged_model --token <hf_access_token>

The output directory is a standard HuggingFace model directory:

```
merged_model/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

### Step 3: Use the merged model

#### Inference with any parallelism configuration

The merged model can be loaded with `from_pretrained()` and used with any PP/TP configuration, independent of the original training setup.

    cd opt_prime/examples

    # PP=4, with KV cache
    torchrun --nproc_per_node=4 --master_port=29500 pp_inference_from_hf_ckpt.py --model-dir ./merged_model --use-kv-cache

    # PP=4, TP=2 (8 GPUs) — different configuration from training
    torchrun --nproc_per_node=8 --master_port=29500 pp_inference_from_hf_ckpt.py --model-dir ./merged_model --pp-size 4 --tp-size 2 --use-kv-cache

    # Greedy decoding
    torchrun --nproc_per_node=4 --master_port=29500 pp_inference_from_hf_ckpt.py --model-dir ./merged_model --use-kv-cache --no-sample

    # Custom prompt
    torchrun --nproc_per_node=4 --master_port=29500 pp_inference_from_hf_ckpt.py --model-dir ./merged_model --use-kv-cache --prompt "Who are you"

#### Continued training (fine-tuning) from the merged model

    cd opt_prime/examples

    # Continue training with PP=4 (4 GPUs)
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_train_from_hf_ckpt.py --model-dir ./merged_model

    # Continue training with PP=8 (8 GPUs, different configuration)
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_train_from_hf_ckpt.py --model-dir ./merged_model

After continued training, merge again to produce an updated HuggingFace model (in multi-node setups, gather stage files from all servers first, as described in the Multi-node training section above):

    python3 merge_hf_ckpt.py --model meta-llama/Llama-3.2-1B --ckpt-dir ./hf_ckpt_trained --output ./merged_model_v2

### End-to-end example

```bash
# 1. Train with PP=4 and save checkpoints
torchrun --nproc_per_node=4 pp_train_llama_into_hf_ckpt.py --max-steps 20 <token>

# 2. Merge into HuggingFace model on CPU
python3 merge_hf_ckpt.py --model meta-llama/Llama-3.2-1B --ckpt-dir ./hf_ckpt --output ./merged_model

# 3. Inference with merged model (PP=4, KV cache)
torchrun --nproc_per_node=4 pp_inference_from_hf_ckpt.py --model-dir ./merged_model --use-kv-cache

# 4. Continue training from merged model (PP=8, 8 GPUs)
torchrun --nproc_per_node=8 pp_train_from_hf_ckpt.py --model-dir ./merged_model

# 5. Merge continued training result
python3 merge_hf_ckpt.py --model meta-llama/Llama-3.2-1B --ckpt-dir ./hf_ckpt_trained --output ./merged_model_v2

# 6. Inference with updated model
torchrun --nproc_per_node=4 pp_inference_from_hf_ckpt.py --model-dir ./merged_model_v2 --use-kv-cache
```

### Python API

```python
# Save checkpoints during training
optimus_p.save_hf_ckpt("./hf_ckpt", step=100, epoch=1)

# Load merged model for inference
model = AutoModelForCausalLM.from_pretrained("./merged_model", use_cache=False)
engine = Optimus_Inference(model, use_gpu=True, pp_size=4, use_kv_cache=True)
output = engine.generate(input_ids, max_new_tokens=50)

# Load merged model for continued training
model = AutoModelForCausalLM.from_pretrained("./merged_model", use_cache=False)
optimus_p = Optimus_p(model, num_mb, use_gpu=True, activation_ckpt=True)
```

---

## LoRA Fine-tuning

OptimusPrime supports LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning. Only ~0.5% of parameters are trained, significantly reducing memory usage and training time compared to full fine-tuning. LoRA is compatible with PP, DP, TP, and `--dynamo-capture`.

### How it works

LoRA adapters are applied **after** pipeline partitioning. Each target `nn.Linear` module (e.g., Q/K/V/O projections) is replaced with a `LoRALinear` wrapper that freezes the base weight and trains only small `lora_A` and `lora_B` matrices. When TP is active, LoRA adapters are automatically parallelized to match the base linear's sharding plan.

```python
from opt_prime.opti_pri import Optimus_p
from opt_prime.lora import LoRAConfig

# 1. Create Optimus_p (PP split + TP + DDP)
optimus_p = Optimus_p(model, num_mb, use_gpu=True, ...)

# 2. Apply LoRA AFTER init, BEFORE optimizer
lora_config = LoRAConfig(r=8, alpha=16.0, dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"])
optimus_p.apply_lora(lora_config)

# 3. Optimizer sees only LoRA params (~0.5%)
optimus_p.optimizer = torch.optim.AdamW(optimus_p.parameters(), lr=2e-4)

# 4. Training loop — causal LM requires label shift
input_ids = tokens.input_ids
shifted_labels = input_ids.clone()
shifted_labels[:, :-1] = input_ids[:, 1:]
shifted_labels[:, -1] = -100  # no target for last position
optimus_p.run(input_ids, shifted_labels, mode="1f1b")

# 5. Save LoRA adapter weights only (very small)
optimus_p.save_lora_ckpt(step=50, epoch=1)
```

### Step 1: LoRA training

    cd opt_prime/examples

    # PP=4, 50 steps
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_train_llama_lora.py --max-steps 50 <llama_access_token>

    # With custom model (e.g., previously merged model)
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_train_llama_lora.py --model-dir ./lora_merged_model --max-steps 30 <llama_access_token>

    # With dynamo-capture
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_train_llama_lora.py --dynamo-capture --max-steps 50 <llama_access_token>

    # Custom LoRA hyperparameters
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_train_llama_lora.py --lora-r 16 --lora-alpha 32 --max-steps 50 <llama_access_token>

This produces per-stage LoRA checkpoints:

```
lora_checkpoint_stage_0/lora_step_50_epoch_1.pt  (~0.5% of model size)
lora_checkpoint_stage_1/lora_step_50_epoch_1.pt
lora_checkpoint_stage_2/lora_step_50_epoch_1.pt
lora_checkpoint_stage_3/lora_step_50_epoch_1.pt
```

### Step 2: Inference — Merge mode (default)

Merge LoRA weights into base model, then run inference with zero LoRA overhead. Requires the same PP configuration as training.

    cd opt_prime/examples

    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_inference_llama_lora.py --lora-step 50 --lora-epoch 1

### Step 3: Inference — Adapter mode

Keep LoRA adapters active during inference. Allows swapping different adapters without reloading the base model, at a small performance cost.

    cd opt_prime/examples

    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_inference_llama_lora.py --lora-step 50 --lora-epoch 1 --no-merge

### Step 4: Export as HuggingFace model

Merge LoRA into base weights and save as stage checkpoint files, then use `merge_hf_ckpt.py` to create a standard HuggingFace model. The resulting model can be used with **any PP/TP configuration**, independent of the original training setup.

    cd opt_prime/examples

    # 4a. Merge LoRA and save stage checkpoints
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_inference_llama_lora.py --lora-step 50 --lora-epoch 1 --save-merged ./lora_merged_hf_ckpt

    # 4b. Merge stage files into single HF model (CPU, no GPU required)
    python3 merge_hf_ckpt.py --model meta-llama/Llama-3.2-1B --ckpt-dir ./lora_merged_hf_ckpt --output ./lora_merged_model

    # 4c. Inference with any PP/TP configuration
    torchrun --nproc_per_node=4 --master_port=29500 pp_inference_from_hf_ckpt.py --model-dir ./lora_merged_model --use-kv-cache

    # 4d. Or with different GPU count
    torchrun --nproc_per_node=8 --master_port=29500 pp_inference_from_hf_ckpt.py --model-dir ./lora_merged_model --pp-size 4 --tp-size 2 --use-kv-cache

### Step 5: Continued LoRA training from merged model

Use the exported HF model as a base for further LoRA fine-tuning. Use `--lora-dir` to keep checkpoints separate from previous rounds.

    cd opt_prime/examples

    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=29500 pp_train_llama_lora.py --model-dir ./lora_merged_model --max-steps 30 --lora-dir lora_v2_checkpoint <llama_access_token>

### End-to-end example

```bash
# 1. LoRA training (PP=4, 50 steps)
torchrun --nproc_per_node=4 pp_train_llama_lora.py --max-steps 50 <token>

# 2. Quick inference — merge mode (same PP=4 required)
torchrun --nproc_per_node=4 pp_inference_llama_lora.py --lora-step 50 --lora-epoch 1

# 3. Quick inference — adapter mode (same PP=4 required)
torchrun --nproc_per_node=4 pp_inference_llama_lora.py --lora-step 50 --lora-epoch 1 --no-merge

# 4. Export as HF model (for any PP/TP configuration)
torchrun --nproc_per_node=4 pp_inference_llama_lora.py --lora-step 50 --lora-epoch 1 --save-merged ./lora_merged_hf_ckpt
python3 merge_hf_ckpt.py --model meta-llama/Llama-3.2-1B --ckpt-dir ./lora_merged_hf_ckpt --output ./lora_merged_model

# 5. Inference with merged HF model (any GPU count)
torchrun --nproc_per_node=4 pp_inference_from_hf_ckpt.py --model-dir ./lora_merged_model --use-kv-cache

# 6. Continued LoRA training from merged model
torchrun --nproc_per_node=4 pp_train_llama_lora.py --model-dir ./lora_merged_model --max-steps 30 --lora-dir lora_v2_checkpoint <token>
```

---

## License

The results of the AIcomp project are distributed under the 3-clause BSD license.
