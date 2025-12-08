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

## License

The results of the AIcomp project are distributed under the 3-clause BSD license.
