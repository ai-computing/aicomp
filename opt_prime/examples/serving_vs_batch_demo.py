#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#

"""
Serving Mode vs Batch Mode — KV Cache Memory Demo

This example demonstrates the key difference between serving mode and batch mode
by running multiple generate() calls and reporting GPU memory after each one.

- Serving mode: KV cache stays allocated between requests → no re-allocation
  overhead, stable memory usage after the first request.
- Batch mode: KV cache is freed after each request → memory drops back down,
  but every request pays the allocation cost.

Usage:
    # Single GPU (PP=1, TP=1)
    torchrun --nproc_per_node=1 serving_vs_batch_demo.py

    # TP=2
    torchrun --nproc_per_node=2 serving_vs_batch_demo.py --tp-size 2

    # PP=2
    torchrun --nproc_per_node=2 serving_vs_batch_demo.py --pp-size 2

    # 10 requests, 50 tokens
    torchrun --nproc_per_node=4 serving_vs_batch_demo.py --pp-size 4 --num-requests 10 --max-new-tokens 50

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serving vs Batch Mode KV Cache Memory Demo"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--num-requests", type=int, default=5,
        help="Number of generate() calls per mode"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=30,
        help="Tokens to generate per request"
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
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[dtype_str]


def gpu_mem_mb() -> dict:
    """Return current GPU memory stats in MB."""
    return {
        "allocated": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved":  torch.cuda.memory_reserved()  / 1024 / 1024,
    }


PROMPTS = [
    "The future of artificial intelligence is",
    "In a distant galaxy, an ancient civilization",
    "The recipe for a perfect chocolate cake includes",
    "Quantum computing will revolutionize the world by",
    "Once upon a time in a land of dragons",
    "The most important scientific discovery of the century",
    "Deep in the ocean, researchers found",
    "A breakthrough in renewable energy was achieved when",
]


def run_benchmark(engine, tokenizer, args, serving_mode: bool, rank: int):
    """Run multiple generate() calls and record memory at each step."""
    mode_name = "SERVING" if serving_mode else "BATCH"

    # Synchronize and clear GPU cache before benchmark
    dist.barrier()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    baseline = gpu_mem_mb()

    if engine.is_output_rank():
        print(f"\n{'=' * 70}")
        print(f"  [{mode_name} MODE] — {args.num_requests} consecutive requests")
        print(f"  Cache behavior: {'kept between requests' if serving_mode else 'freed after each request'}")
        print(f"{'=' * 70}")
        print(f"  {'Request':>8}  {'Alloc(MB)':>10}  {'Reserv(MB)':>11}  "
              f"{'Delta(MB)':>10}  {'Time(s)':>8}  Status")
        print(f"  {'-' * 64}")

    records = []
    for i in range(args.num_requests):
        prompt = PROMPTS[i % len(PROMPTS)]

        # Prepare input on first stage
        if engine.is_first_stage():
            tokens = tokenizer(
                prompt, return_tensors="pt", padding=True,
                truncation=True, max_length=2048,
            )
            input_ids = tokens.input_ids.cuda()
        else:
            input_ids = None

        mem_before = gpu_mem_mb()
        t0 = time.time()

        output_ids = engine.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7, top_k=50, top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            verbose=False,
        )

        elapsed = time.time() - t0
        mem_after = gpu_mem_mb()
        delta = mem_after["allocated"] - baseline["allocated"]

        # Decode output for display (output rank only)
        generated_text = ""
        if engine.is_output_rank() and output_ids is not None:
            generated_text = tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            # Truncate for display
            if len(generated_text) > 60:
                generated_text = generated_text[:57] + "..."

        cache_status = "cache kept" if serving_mode else "cache freed"
        records.append({
            "alloc": mem_after["allocated"],
            "reserved": mem_after["reserved"],
            "delta": delta,
            "elapsed": elapsed,
        })

        if engine.is_output_rank():
            print(f"  {i + 1:>8}  {mem_after['allocated']:>10.1f}  "
                  f"{mem_after['reserved']:>11.1f}  {delta:>+10.1f}  "
                  f"{elapsed:>8.2f}  {cache_status}")

        dist.barrier()

    # Summary
    if engine.is_output_rank() and records:
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        allocs = [r["alloc"] for r in records]
        deltas = [r["delta"] for r in records]
        times = [r["elapsed"] for r in records]

        print(f"  {'-' * 64}")
        print(f"  Summary ({mode_name}):")
        print(f"    Baseline memory    : {baseline['allocated']:.1f} MB")
        print(f"    Peak memory        : {peak:.1f} MB")
        print(f"    Avg alloc after gen: {sum(allocs) / len(allocs):.1f} MB")
        print(f"    Memory delta range : {min(deltas):+.1f} ~ {max(deltas):+.1f} MB")
        print(f"    Avg time/request   : {sum(times) / len(times):.2f}s")
        print(f"    Total time         : {sum(times):.2f}s")

    return records


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dtype = get_dtype(args.dtype)

    # Load model and tokenizer
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

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")
        print(f"PP={args.pp_size}, TP={args.tp_size}, dtype={args.dtype}")
        print(f"Requests per mode: {args.num_requests}, "
              f"max_new_tokens: {args.max_new_tokens}")

    # ----------------------------------------------------------------
    # Phase 1: BATCH MODE (cache freed after each request)
    # ----------------------------------------------------------------
    engine_batch = Optimus_Inference(
        model,
        use_gpu=True,
        pp_size=args.pp_size,
        tp_size=args.tp_size,
        dtype=dtype,
        use_kv_cache=True,
        serving_mode=False,  # batch mode
    )
    engine_batch.eval()

    batch_records = run_benchmark(
        engine_batch, tokenizer, args, serving_mode=False, rank=rank,
    )

    # Clean up batch engine to get a fair baseline for serving mode
    del engine_batch
    torch.cuda.empty_cache()
    dist.barrier()

    # ----------------------------------------------------------------
    # Phase 2: SERVING MODE (cache kept between requests)
    # ----------------------------------------------------------------
    model2 = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, torch_dtype=dtype,
    )

    engine_serving = Optimus_Inference(
        model2,
        use_gpu=True,
        pp_size=args.pp_size,
        tp_size=args.tp_size,
        dtype=dtype,
        use_kv_cache=True,
        serving_mode=True,  # serving mode
    )
    engine_serving.eval()

    serving_records = run_benchmark(
        engine_serving, tokenizer, args, serving_mode=True, rank=rank,
    )

    # Explicitly release KV cache (serving mode teardown)
    engine_serving.release_kv_cache()
    mem_after_release = gpu_mem_mb()

    if engine_serving.is_output_rank():
        print(f"\n  After release_kv_cache(): {mem_after_release['allocated']:.1f} MB allocated")

    # ----------------------------------------------------------------
    # Comparison
    # ----------------------------------------------------------------
    if rank == 0 and batch_records and serving_records:
        print(f"\n{'=' * 70}")
        print(f"  COMPARISON: Batch Mode vs Serving Mode")
        print(f"{'=' * 70}")

        batch_allocs = [r["alloc"] for r in batch_records]
        serving_allocs = [r["alloc"] for r in serving_records]
        batch_times = [r["elapsed"] for r in batch_records]
        serving_times = [r["elapsed"] for r in serving_records]

        # Memory stability: std dev of allocations (lower = more stable)
        import statistics
        batch_std = statistics.stdev(batch_allocs) if len(batch_allocs) > 1 else 0
        serving_std = statistics.stdev(serving_allocs) if len(serving_allocs) > 1 else 0

        print(f"  {'Metric':<30}  {'Batch':>12}  {'Serving':>12}")
        print(f"  {'-' * 58}")
        print(f"  {'Avg memory (MB)':<30}  "
              f"{sum(batch_allocs)/len(batch_allocs):>12.1f}  "
              f"{sum(serving_allocs)/len(serving_allocs):>12.1f}")
        print(f"  {'Memory std dev (MB)':<30}  "
              f"{batch_std:>12.1f}  {serving_std:>12.1f}")
        print(f"  {'1st request alloc (MB)':<30}  "
              f"{batch_allocs[0]:>12.1f}  {serving_allocs[0]:>12.1f}")
        print(f"  {'Last request alloc (MB)':<30}  "
              f"{batch_allocs[-1]:>12.1f}  {serving_allocs[-1]:>12.1f}")
        print(f"  {'Avg time/request (s)':<30}  "
              f"{sum(batch_times)/len(batch_times):>12.3f}  "
              f"{sum(serving_times)/len(serving_times):>12.3f}")
        if len(batch_times) > 1 and len(serving_times) > 1:
            # Skip first request (cold start) for 2nd+ request comparison
            print(f"  {'Avg time (2nd+ req) (s)':<30}  "
                  f"{sum(batch_times[1:])/len(batch_times[1:]):>12.3f}  "
                  f"{sum(serving_times[1:])/len(serving_times[1:]):>12.3f}")
        print(f"  {'Total time (s)':<30}  "
              f"{sum(batch_times):>12.3f}  {sum(serving_times):>12.3f}")
        print(f"{'=' * 70}")

        print(f"\n  Key observations:")
        print(f"    - Batch mode: memory allocation fluctuates as cache is")
        print(f"      allocated and freed on each request.")
        print(f"    - Serving mode: memory stays stable after the first request")
        print(f"      because the cache is only cleared (position reset), not freed.")
        if sum(serving_times[1:]) < sum(batch_times[1:]):
            speedup = sum(batch_times[1:]) / max(sum(serving_times[1:]), 1e-6)
            print(f"    - Serving mode was {speedup:.2f}x faster on 2nd+ requests")
            print(f"      (no cache re-allocation overhead).")
        print()

    dist.barrier()

    if rank == 0:
        print("Demo completed.")


if __name__ == "__main__":
    main()
