#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#

"""
Explicit Prefill/Decode Generation Example using ScheduleGeneration

This example demonstrates the explicit two-phase generation API:
  1. prefill()     — forward entire prompt, CachedSDPA stores all K,V
  2. decode_step() — forward one token per step, CachedSDPA appends K,V

This gives fine-grained control over the generation loop, unlike
engine.generate() which wraps both phases internally.

Usage:
    # Single GPU (no torchrun)
    python pp_generation_llama.py

    # Pipeline parallel
    torchrun --nproc_per_node=2 pp_generation_llama.py --pp-size 2

    # Tensor parallel
    torchrun --nproc_per_node=2 pp_generation_llama.py --tp-size 2

    # PP=2 x TP=2
    torchrun --nproc_per_node=4 pp_generation_llama.py --pp-size 2 --tp-size 2

    # Use KVCacheManager as external backend (instead of CachedSDPA internal cache)
    python pp_generation_llama.py --use-kv-manager

Environment:
    - Requires: PyTorch >= 2.0, transformers, CUDA >= 12.1
    - Model: meta-llama/Llama-3.2-1B (or specify via --model)
"""

import torch
import torch.distributed as dist
import argparse
import os
import sys
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.inference import Optimus_Inference
from opt_prime.schedule_infer import ScheduleGeneration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explicit Prefill/Decode Generation for LLaMA"
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
        "--pp-size", type=int, default=1,
        help="Pipeline parallel size"
    )
    parser.add_argument(
        "--tp-size", type=int, default=1,
        help="Tensor parallel size"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for inference"
    )
    parser.add_argument(
        "--use-kv-manager", action="store_true",
        help="Use KVCacheManager as external backend for CachedSDPA "
             "(default: CachedSDPA internal cache)"
    )
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[dtype_str]


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    dtype = get_dtype(args.dtype)

    if rank == 0:
        print("=" * 60)
        print("Explicit Prefill/Decode Generation Example")
        print("  (using ScheduleGeneration.prefill + decode_step)")
        print("=" * 60)
        print(f"  Model:          {args.model}")
        print(f"  PP Size:        {args.pp_size}")
        print(f"  TP Size:        {args.tp_size}")
        print(f"  Dtype:          {args.dtype}")
        print(f"  Max New Tokens: {args.max_new_tokens}")
        cache_mode = "KVCacheManager (external)" if args.use_kv_manager else "CachedSDPA (internal)"
        print(f"  KV Cache Mode:  {cache_mode}")
        print("=" * 60)

    # ----- Load model & tokenizer -----
    config = AutoConfig.from_pretrained(args.model, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if rank == 0:
        print(f"\nLoading model: {args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, torch_dtype=dtype,
    )

    # ----- Create engine with KV cache -----
    engine = Optimus_Inference(
        model,
        use_gpu=True,
        pp_size=args.pp_size,
        tp_size=args.tp_size,
        dtype=dtype,
        use_kv_cache=True,     # must be True for CachedSDPA injection
        use_kv_manager=args.use_kv_manager,
    )
    engine.eval()

    # ----- Setup KVCacheManager backend (if requested) -----
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

    # ----- Create ScheduleGeneration -----
    scheduler = ScheduleGeneration(engine)

    # ----- Enable CachedSDPA modules -----
    for m in engine._cached_sdpa_modules:
        m.enable()

    # ----- Prepare input -----
    if engine.is_first_stage():
        tokens = tokenizer(
            args.prompt, return_tensors="pt",
            padding=True, truncation=True, max_length=2048,
        )
        input_ids = tokens.input_ids.cuda()
        input_seq_len = input_ids.size(1)
    else:
        input_ids = None
        input_seq_len = 0

    # Send dims from first stage to last stage (for output allocation)
    if engine.comm.world_size > 1:
        if engine.tpl.is_first_stage() and not engine.tpl.is_last_stage():
            engine.comm.send_data(input_seq_len, engine.tpl.get_last_rank(), engine.device)
            engine.comm.send_data(input_ids, engine.tpl.get_last_rank(), engine.device)
        if engine.tpl.is_last_stage() and not engine.tpl.is_first_stage():
            input_seq_len = engine.comm.receive_data(engine.tpl.get_first_rank(), engine.device)
            input_ids = engine.comm.receive_data(engine.tpl.get_first_rank(), engine.device)

    # Output storage on last stage
    if engine.tpl.is_last_stage():
        generated_ids = torch.zeros(
            (1, input_seq_len + args.max_new_tokens),
            dtype=torch.long, device=engine.device,
        )
        generated_ids[:, :input_seq_len] = input_ids
    else:
        generated_ids = None

    if engine.is_output_rank():
        print(f"\nPrompt: {args.prompt}")
        print(f"Input tokens: {input_seq_len}")
        print(f"\n{'=' * 60}")
        print(f"[Prefill] Processing {input_seq_len} prompt tokens...")

    # =========================================================
    #  Phase 1: PREFILL — forward entire prompt in one call
    # =========================================================
    start_time = time.time()

    logits = scheduler.prefill(input_ids)

    prefill_time = time.time() - start_time

    # Sample first token from prefill output
    if engine.tpl.is_last_stage():
        next_token = engine._sample_token(
            logits, args.temperature, args.top_k, args.top_p,
            do_sample=not args.no_sample,
        )
        generated_ids[:, input_seq_len] = next_token
        if engine.is_output_rank():
            token_str = tokenizer.decode(next_token.item())
            print(f"[Prefill] Done in {prefill_time:.3f}s → first token: '{token_str}'")
    else:
        next_token = None

    # Broadcast first token to all stages for decode loop
    if engine.comm.world_size > 1:
        if engine.tpl.is_last_stage() and not engine.tpl.is_first_stage():
            engine.comm.send_data(
                next_token, engine.tpl.get_first_rank(), engine.device
            )
        if engine.tpl.is_first_stage() and not engine.tpl.is_last_stage():
            next_token = engine.comm.receive_data(
                engine.tpl.get_last_rank(), engine.device
            )

    # TP sync: broadcast sampled token within TP group
    if engine.tpl.tp_size > 1:
        if next_token is None:
            next_token = torch.zeros(1, dtype=torch.long, device=engine.device)
        dist.broadcast(
            next_token,
            src=engine.tpl.stage2rank[engine.tpl.stage][0],
            group=engine.tpl.tp_group,
        )

    # =========================================================
    #  Phase 2: DECODE — one token at a time via decode_step()
    # =========================================================
    if engine.is_output_rank():
        print(f"\n[Decode] Generating up to {args.max_new_tokens - 1} more tokens...")
        # Print prompt + first token
        print(f"\n--- Output ---")
        first_output = tokenizer.decode(
            generated_ids[0, :input_seq_len + 1].tolist(),
            skip_special_tokens=True,
        )
        print(first_output, end="", flush=True)

    decode_start = time.time()
    num_generated = 1  # already have the first token from prefill

    for step in range(1, args.max_new_tokens):
        position = input_seq_len + step - 1  # absolute position of new token

        # Prepare single-token input on first stage
        if engine.tpl.is_first_stage():
            token_input = next_token.unsqueeze(0) if next_token.dim() == 1 else next_token
            if token_input.dim() == 1:
                token_input = token_input.unsqueeze(0)  # [1, 1]
        else:
            token_input = None

        # === decode_step: forward one token, CachedSDPA appends K,V ===
        logits = scheduler.decode_step(token_input, position)

        # Sample next token on last stage
        if engine.tpl.is_last_stage():
            next_token = engine._sample_token(
                logits, args.temperature, args.top_k, args.top_p,
                do_sample=not args.no_sample,
            )
            generated_ids[:, input_seq_len + num_generated] = next_token
            num_generated += 1

            # Stream output
            if engine.is_output_rank():
                token_str = tokenizer.decode(next_token.item())
                print(token_str, end="", flush=True)

            # EOS check
            if tokenizer.eos_token_id is not None and (next_token == tokenizer.eos_token_id).all():
                break
        else:
            next_token = None

        # Send token from last stage → first stage for next decode step
        if engine.comm.world_size > 1:
            if engine.tpl.is_last_stage() and not engine.tpl.is_first_stage():
                engine.comm.send_data(
                    next_token, engine.tpl.get_first_rank(), engine.device
                )
            if engine.tpl.is_first_stage() and not engine.tpl.is_last_stage():
                next_token = engine.comm.receive_data(
                    engine.tpl.get_last_rank(), engine.device
                )

        # TP sync
        if engine.tpl.tp_size > 1:
            if next_token is None:
                next_token = torch.zeros(1, dtype=torch.long, device=engine.device)
            dist.broadcast(
                next_token,
                src=engine.tpl.stage2rank[engine.tpl.stage][0],
                group=engine.tpl.tp_group,
            )

    decode_time = time.time() - decode_start
    total_time = time.time() - start_time

    # =========================================================
    #  Disable CachedSDPA and print statistics
    # =========================================================
    if args.use_kv_manager:
        engine.release_kv_cache()  # frees both CachedSDPA and KVCacheManager
    else:
        for m in engine._cached_sdpa_modules:
            m.disable()

    if engine.is_output_rank():
        print(f"\n--- End ---\n")
        print(f"{'=' * 60}")
        print(f"Generation Statistics (ScheduleGeneration API):")
        print(f"  Prefill time     : {prefill_time:.3f}s ({input_seq_len} tokens)")
        print(f"  Decode time      : {decode_time:.3f}s ({num_generated - 1} steps)")
        print(f"  Total time       : {total_time:.3f}s")
        print(f"  Tokens generated : {num_generated}")
        if num_generated > 1:
            print(f"  Decode tok/s     : {(num_generated - 1) / decode_time:.2f}")
        cache_mode = "KVCacheManager (external)" if args.use_kv_manager else "CachedSDPA (internal)"
        print(f"  KV Cache backend : {cache_mode}")
        print(f"  PP={args.pp_size}, TP={args.tp_size}")
        print(f"{'=' * 60}")

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print("\nCompleted.")


if __name__ == "__main__":
    main()
