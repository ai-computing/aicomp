#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#

"""
Single-GPU Inference Example for LLaMA Models (No torchrun Required)

This example demonstrates single-process inference on a single GPU.
Unlike the PP/TP examples, this script can be run directly with `python`
without needing `torchrun` or any distributed launcher.

Usage:
    # Direct execution (no torchrun)
    python single_gpu_inference_llama.py

    # With custom model and prompt
    python single_gpu_inference_llama.py --model meta-llama/Llama-3.2-1B \
        --prompt "The meaning of life is" --max-new-tokens 100

    # With KV cache for faster decode
    python single_gpu_inference_llama.py --use-kv-cache

    # Greedy decoding
    python single_gpu_inference_llama.py --no-sample

Environment:
    - Requires: PyTorch >= 2.0, transformers, CUDA >= 12.1
    - Runs on a single GPU without distributed setup
    - Model: meta-llama/Llama-3.2-1B (or specify via --model)
"""

import torch
import argparse
import os
import sys
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextStreamer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.inference import Optimus_Inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-GPU Inference for LLaMA (no torchrun required)"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--prompt", type=str,
        default="The future of artificial intelligence is",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--no-sample", action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    parser.add_argument(
        "--use-kv-cache", action="store_true",
        help="Use KV cache for O(n) decode (default: full-sequence O(n^2))"
    )
    parser.add_argument(
        "--serving-mode", action="store_true",
        help="Keep KV cache allocated between requests (requires --use-kv-cache)"
    )
    parser.add_argument(
        "--use-kv-manager", action="store_true",
        help="Use KVCacheManager as external backend for CachedSDPA "
             "(implies --use-kv-cache)"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for inference"
    )
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[dtype_str]


def main():
    args = parse_args()

    print("=" * 60)
    print("Single-GPU Inference Example (no torchrun)")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Dtype:          {args.dtype}")
    print(f"  Max New Tokens: {args.max_new_tokens}")
    kv_enabled = args.use_kv_cache or args.use_kv_manager
    print(f"  KV Cache:       {'enabled' if kv_enabled else 'disabled'}")
    if kv_enabled:
        print(f"  Cache Mode:     {'serving' if args.serving_mode else 'batch'}")
        backend = "KVCacheManager (external)" if args.use_kv_manager else "CachedSDPA (internal)"
        print(f"  Cache Backend:  {backend}")
    print(f"  Sampling:       {'greedy' if args.no_sample else f'temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}'}")
    print("=" * 60)

    # Load model configuration
    config = AutoConfig.from_pretrained(args.model, use_cache=False)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    dtype = get_dtype(args.dtype)
    print(f"\nLoading model: {args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, torch_dtype=dtype,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create inference engine — single process, no PP/TP/DP
    engine = Optimus_Inference(
        model,
        use_gpu=True,
        pp_size=1,
        tp_size=1,
        dtype=dtype,
        use_kv_cache=args.use_kv_cache,
        serving_mode=args.serving_mode,
        use_kv_manager=args.use_kv_manager,
    )
    engine.eval()

    # Setup KVCacheManager backend if requested
    if args.use_kv_manager:
        # Use num_attention_heads (not num_key_value_heads) because the K,V
        # tensors arriving at SDPA have already been expanded by repeat_kv
        # from num_kv_heads to num_attention_heads.
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        engine.init_kv_cache(
            num_layers=config.num_hidden_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=1,
        )
        engine._attach_kv_manager()

    # Prepare input
    tokens = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    input_ids = tokens.input_ids.cuda()
    print(f"\nPrompt: {args.prompt}")
    print(f"Input shape: {input_ids.shape}")

    # Set up streaming
    method = "KV cache, O(n)" if args.use_kv_cache else "full-seq recompute, O(n^2)"
    print(f"\nGenerating {args.max_new_tokens} tokens ({method})...")
    print("\n" + "=" * 60)
    print(args.prompt, end="", flush=True)
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)

    # Generate
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
        verbose=False,
    )

    generation_time = time.time() - start_time

    # Statistics
    if output_ids is not None:
        prompt_token_count = len(tokenizer(args.prompt).input_ids)
        num_generated = output_ids.size(1) - prompt_token_count

        print("=" * 60)
        print(f"\nGeneration Statistics:")
        print(f"  Tokens generated : {num_generated}")
        print(f"  Total time       : {generation_time:.2f}s")
        if num_generated > 0:
            print(f"  Tokens/second    : {num_generated / generation_time:.2f}")
            print(f"  Avg time/token   : {generation_time / num_generated * 1000:.1f}ms")
        if args.use_kv_cache or args.use_kv_manager:
            mode = "serving" if args.serving_mode else "batch"
            backend = "KVCacheManager" if args.use_kv_manager else "CachedSDPA internal"
            print(f"  Method           : KV cache (O(n) decode, {mode} mode, {backend})")
        else:
            print(f"  Method           : full-sequence recomputation (O(n^2) decode)")
        print("=" * 60)

    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
