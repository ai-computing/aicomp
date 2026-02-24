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

## Running (Example: gpt2)

### Single-node environment

    cd opt_prime/examples
    torchrun --nproc_per_node=<# of GPUs per node> --nnodes=<# of nodes> --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 pp_train_gpt2.py

### Multi-node environment

Run the following command for every node:

    cd opt_prime/examples
    torchrun --nproc_per_node=<# of GPUs per node> --nnodes=<# of nodes> --node_rank=<current node rank> --master_addr=<IP of rank 0> --master_port=29500 pp_train_gpt2.py

## Configuring parallelism options

### Configuring PP only

The most basic parallelism option is 'pp_size'. If the 'pp_size' option is omitted, its value is automatically determined based on the number of available GPUs. You may also specify 'pp_size' explicitly if desired.

    # Example of a 4-GPU setup with pipeline parallel size=4
    optimus_p = Optimus_p(model, num_mb, use_gpu=True)

    # Example of a 4-GPU setup with pipeline parallel size=4
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, pp_size=4)

### Changing scheduling policy for PP

Pipeline parallel in OptimusPrime supports both 'gpipe' and '1f1b' scheduling options. To use either mode, open the desired script in 'opt_prime/examples' and set the 'mode' option in optimus_p.run() as shown below.

    # Pipeline parallelism uses the 'gpipe' scheduler by default
    optimus_p.run(data, labels)


    # Example of explicitly setting the gpipe scheduler
    optimus_p.run(data, labels, mode="gpipe")

    # Example of explicitly setting the 1f1b scheduler
    optimus_p.run(data, labels, mode="1f1b")

### Configuring PP+DP 

To apply 2D parallelism with PP+DP, use the 'dp_size' option when instantiating the Optimus_p class. The 'pp_size' option is applied by default even if not specified, but it can also be explicitly set.


    # Example of an 8-GPU setup with pipeline parallel size=4 and data parallel size=2
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, dp_size=2)

    # Example of an 8-GPU setup with pipeline parallel size=4 and data parallel size=2
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, pp_size=4, dp_size=2)

Configuration diagram of 2D parallelism with pp_size=4 and dp_size=2 applied simultaneously

<p align="center">
  <img src="https://github.com/ai-computing/aicomp/assets/42994087/9b3546a0-a22a-4014-95a2-420cf742e8be">
</p>

### Configuring PP+TP
To apply 2D parallelism with PP+TP, use the 'tp_size' option when instantiating the Optimus_p class. 'tp_size' is applicable to the Llama model.

    # Example of an 16-GPU setup with pipeline parallel size=8 and tensor parallel size=2
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, tp_size=2)

    # Example of an 16-GPU setup with pipeline parallel size=8 and tensor parallel size=2
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, pp_size=8, tp_size=2)


### Configuring PP+TP+DP

To apply 3D parallelism with PP+TP+DP, use the 'pp_size', 'tp_size' and 'dp_size' options. 'tp_size' is applicable to the Llama model.

    # Example of a 16-GPU setup with pipeline parallel size=4, tensor parallel size=2, and data parallel size=2
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, tp_size=2, dp_size=2)

    # Example of a 16-GPU setup with pipeline parallel size=4, tensor parallel size=2, and data parallel size=2
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, pp_size=4, tp_size=2, dp_size=2)

## Configuring memory optimization options

### Offloading optimizer state to host memory during forward and backward passes
When the 'swap_opt_in_fwdbwd' option is set to True, the optimizer state is offloaded to host memory during the forward and backward passes to reduce GPU memory usage. This helps avoid GPU OOM and enables training of larger models.

    # Example of an 4-GPU setup with pipeline parallel size=4 and swap_opt_in_fwdbwd option is set to True
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, swap_opt_in_fwdbwd=True)

### Executing the optimizer step on the CPU 
When the 'swap_model_in_optstep' option is set to True, the optimizer’s step() phase is executed on the CPU. This option must be used together with 'swap_opt_in_fwdbwd', and by avoiding the GPU OOM that can occur when using 'swap_opt_in_fwdbwd' alone, it enables the operation of even larger models.
	
    # Example of an 4-GPU setup with pipeline parallel size=4 and swap_opt_in_fwdbwd option is set to True
    optimus_p = Optimus_p(model, num_mb, use_gpu=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True)

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
    torchrun --nproc_per_node=2 tp_inference_llama.py --tp-size 2 --use-kv-cache

    # TP=2, KVCacheManager external backend
    torchrun --nproc_per_node=2 tp_inference_llama.py --tp-size 2 --use-kv-manager

    # PP=2 x TP=2, KVCacheManager external backend
    torchrun --nproc_per_node=4 tp_inference_llama.py --pp-size 2 --tp-size 2 --use-kv-manager

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

    # Serving mode demo — compares memory behavior across multiple requests
    torchrun --nproc_per_node=1 serving_vs_batch_demo.py

    # PP=4, 10 requests, 50 tokens each
    torchrun --nproc_per_node=4 serving_vs_batch_demo.py --pp-size 4 --num-requests 10 --max-new-tokens 50

Example output (PP=4, Llama-3.2-1B, 4x NVIDIA GeForce RTX 3090):

```
======================================================================
  [BATCH MODE] — 10 consecutive requests
  Cache behavior: freed after each request
======================================================================
   Request   Alloc(MB)   Reserv(MB)   Delta(MB)   Time(s)  Status
  ----------------------------------------------------------------
         1      1006.5       1102.0        +8.4      5.68  cache freed
         2      1006.5       1102.0        +8.4      4.05  cache freed
         3      1006.5       1102.0        +8.4      1.45  cache freed
         4      1006.5       1102.0        +8.4      4.86  cache freed
         5      1006.5       1102.0        +8.4      1.43  cache freed
         6      1006.5       1102.0        +8.4      1.43  cache freed
         7      1006.5       1102.0        +8.4      4.87  cache freed
         8      1006.5       1102.0        +8.4      1.45  cache freed
         9      1006.5       1102.0        +8.4      1.44  cache freed
        10      1006.5       1102.0        +8.4      1.44  cache freed
  ----------------------------------------------------------------
  Summary (BATCH):
    Baseline memory    : 998.0 MB
    Peak memory        : 1077.6 MB
    Avg alloc after gen: 1006.5 MB
    Memory delta range : +8.4 ~ +8.4 MB
    Avg time/request   : 2.81s
    Total time         : 28.10s

  [SERVING MODE] — 10 consecutive requests
  Cache behavior: kept between requests
======================================================================
   Request   Alloc(MB)   Reserv(MB)   Delta(MB)   Time(s)  Status
  ----------------------------------------------------------------
         1      2068.5       2100.0       +64.3      1.44  cache kept
         2      2068.5       2100.0       +64.3      1.44  cache kept
         3      2068.5       2100.0       +64.3      1.44  cache kept
         4      2068.5       2100.0       +64.3      1.44  cache kept
         5      2068.5       2100.0       +64.3      1.45  cache kept
         6      2068.5       2100.0       +64.3      1.44  cache kept
         7      2068.5       2100.0       +64.3      1.44  cache kept
         8      2068.5       2100.0       +64.3      1.45  cache kept
         9      2068.5       2100.0       +64.3      1.43  cache kept
        10      2068.5       2100.0       +64.3      1.44  cache kept
  ----------------------------------------------------------------
  Summary (SERVING):
    Baseline memory    : 2004.2 MB
    Peak memory        : 2075.8 MB
    Avg alloc after gen: 2068.5 MB
    Memory delta range : +64.3 ~ +64.3 MB
    Avg time/request   : 1.44s
    Total time         : 14.42s

======================================================================
  COMPARISON: Batch Mode vs Serving Mode
======================================================================

  After release_kv_cache(): 2004.5 MB allocated
  Metric                                 Batch       Serving
  ----------------------------------------------------------
  Avg memory (MB)                        942.2        1940.2
  Memory std dev (MB)                      0.0           0.0
  1st request alloc (MB)                 942.2        1940.2
  Last request alloc (MB)                942.2        1940.2
  Avg time/request (s)                   2.809         1.442
  Avg time (2nd+ req) (s)                2.491         1.442
  Total time (s)                        28.086        14.419
======================================================================

  Key observations:
    - Batch mode: memory allocation fluctuates as cache is
      allocated and freed on each request.
    - Serving mode: memory stays stable after the first request
      because the cache is only cleared (position reset), not freed.
    - Serving mode was 1.73x faster on 2nd+ requests
      (no cache re-allocation overhead).
```

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

## License

The results of the AIcomp project are distributed under the 3-clause BSD license.
