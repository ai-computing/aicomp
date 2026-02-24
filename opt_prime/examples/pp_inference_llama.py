#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#

"""
Pipeline Parallel Inference Example for LLaMA Models

This example demonstrates how to run inference on LLaMA models using
pipeline parallelism with the Optimus_Inference engine.

Usage:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 --nnodes=1 --master_port=29500 pp_inference_llama.py

    # Multi-node (run on each node)
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
             --master_addr=<IP> --master_port=29500 pp_inference_llama.py

    # Use KVCacheManager as external backend (instead of CachedSDPA internal cache)
    torchrun --nproc_per_node=4 pp_inference_llama.py --use-kv-manager

Environment:
    - Requires: PyTorch >= 2.0, transformers, CUDA >= 12.1
    - Model: meta-llama/Llama-3.2-1B (or specify via --model)
    - HuggingFace token may be required for gated models
"""

import torch
import torch.distributed as dist
import argparse
import os
import sys
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextStreamer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.inference import Optimus_Inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline Parallel Inference for LLaMA"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="Pipeline parallel size (default: auto)"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for inference"
    )
    parser.add_argument(
        "--use-kv-cache",
        action="store_true",
        help="Use KV cache for O(n) decode (default: full-sequence O(n^2))"
    )
    parser.add_argument(
        "--serving-mode",
        action="store_true",
        help="Keep KV cache allocated between requests (requires --use-kv-cache)"
    )
    parser.add_argument(
        "--use-kv-manager",
        action="store_true",
        help="Use KVCacheManager as external backend for CachedSDPA "
             "(implies --use-kv-cache)"
    )
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if rank == 0:
        print("=" * 60)
        print("Pipeline Parallel Inference Example")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"World Size: {world_size}")
        print(f"PP Size: {args.pp_size if args.pp_size > 1 else 'auto'}")
        print(f"TP Size: {args.tp_size}")
        print(f"Dtype: {args.dtype}")
        print(f"Max New Tokens: {args.max_new_tokens}")
        kv_enabled = args.use_kv_cache or args.use_kv_manager
        print(f"KV Cache: {'enabled' if kv_enabled else 'disabled (full-sequence)'}")
        if kv_enabled:
            print(f"Cache Mode: {'serving (cache kept)' if args.serving_mode else 'batch (cache freed)'}")
            backend = "KVCacheManager (external)" if args.use_kv_manager else "CachedSDPA (internal)"
            print(f"Cache Backend: {backend}")
        print("=" * 60)

    # Load model configuration
    # use_cache=False is required for FX tracing compatibility.
    # When --use-kv-cache is set, KV caching is handled externally by
    # CachedScaledDotProductAttention via FX graph surgery.
    config = AutoConfig.from_pretrained(args.model, use_cache=False)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    dtype = get_dtype(args.dtype)
    if rank == 0:
        print(f"Loading model: {args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype,
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

    # Create inference engine
    engine = Optimus_Inference(
        model,
        use_gpu=True,
        pp_size=args.pp_size,
        tp_size=args.tp_size,
        dtype=dtype,
        use_kv_cache=args.use_kv_cache,
        serving_mode=args.serving_mode,
        use_kv_manager=args.use_kv_manager,
    )

    # Set to evaluation mode
    engine.eval()

    # Setup KVCacheManager backend if requested
    if args.use_kv_manager:
        # Use num_attention_heads (not num_key_value_heads) because the K,V
        # tensors arriving at SDPA have already been expanded by repeat_kv.
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        engine.init_kv_cache(
            num_layers=config.num_hidden_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=1,
        )
        engine._attach_kv_manager()

    # Prepare input (only needed on first stage)
    if engine.is_first_stage():
        tokens = tokenizer(
            args.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = tokens.input_ids.cuda()
        input_seq_len = input_ids.size(1)
        if engine.is_input_rank():
            print(f"\n[Rank {rank}] Prompt: {args.prompt}")
            print(f"[Rank {rank}] Input shape: {input_ids.shape}")
    else:
        input_ids = None
        input_seq_len = 0

    # Set up streaming output on the output rank (first TP rank of last stage)
    if engine.is_output_rank():
        kv_enabled = args.use_kv_cache or args.use_kv_manager
        method = "KV cache, O(n)" if kv_enabled else "full-sequence recomputation, O(n^2)"
        print(f"\nGenerating {args.max_new_tokens} tokens ({method})...")
        print("\n" + "=" * 60)
        # Print prompt, then stream generated tokens right after it
        print(args.prompt, end="", flush=True)
        streamer = TextStreamer(tokenizer, skip_special_tokens=True)
    else:
        streamer = None

    start_time = time.time()

    output_ids = engine.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
        verbose=False,  # disable progress logs during streaming
    )

    generation_time = time.time() - start_time

    # Print statistics (output rank only)
    if engine.is_output_rank() and output_ids is not None:
        prompt_token_count = len(tokenizer(args.prompt).input_ids)
        num_generated = output_ids.size(1) - prompt_token_count

        print("=" * 60)
        print(f"\nGeneration Statistics:")
        print(f"  - Tokens generated: {num_generated}")
        print(f"  - Total time: {generation_time:.2f}s")
        if num_generated > 0:
            print(f"  - Tokens/second: {num_generated / generation_time:.2f}")
            print(f"  - Avg time/token: {generation_time / num_generated:.2f}s")
        if args.use_kv_cache or args.use_kv_manager:
            mode = "serving" if args.serving_mode else "batch"
            backend = "KVCacheManager" if args.use_kv_manager else "CachedSDPA internal"
            print(f"  - Method: KV cache (O(n), {mode} mode, {backend})")
        else:
            print(f"  - Method: full-sequence recomputation (O(n^2))")
        print("=" * 60)

    # Synchronize at the end
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print(f"\n[Rank {rank}] Inference completed successfully!")


if __name__ == "__main__":
    main()
