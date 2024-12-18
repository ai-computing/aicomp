# Llama3 8B inference examples

### Basic version

    python3 llama3_inference_basic.py

### Memory offload version (to free up GPU memory space, some layers are swapped with host memory)

    python3 llama3_inference_memory_offload.py

### Required python packages

    pip3 install torch huggingface_hub transformers datasets bitsandbytes gradio pypdf accelerate

A Gradio-based web UI is provided, and the default configuration allows access at 127.0.0.1:7860.

We recommend using a GPU with more than 8GB of memory.

## License

The results of the AIcomp project are distributed under the 3-clause BSD license.
