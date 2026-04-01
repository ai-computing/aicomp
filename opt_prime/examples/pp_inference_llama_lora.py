#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# Pipeline parallel inference with LoRA adapters.
#
# Three modes:
#   1. Merge mode (default): merge LoRA into base weights, then infer
#      with zero LoRA overhead via Optimus_Inference + KV Cache.
#   2. Adapter mode (--no-merge): keep LoRA adapters active during inference.
#      Allows swapping adapters without reloading the base model.
#   3. Save-merged mode (--save-merged DIR): merge LoRA into base weights,
#      save as stage checkpoint files, then exit. Use merge_hf_ckpt.py to
#      create a HuggingFace-compatible model for any PP/TP configuration.
#
# Usage:
#   # Merge mode (default)
#   torchrun --nproc_per_node=4 --master_port=29500
#     pp_inference_llama_lora.py --lora-step 50 --lora-epoch 1
#
#   # Adapter mode
#   torchrun --nproc_per_node=4 --master_port=29500
#     pp_inference_llama_lora.py --lora-step 50 --lora-epoch 1 --no-merge
#
#   # Save merged weights for HF conversion (no inference)
#   torchrun --nproc_per_node=4 --master_port=29500
#     pp_inference_llama_lora.py --lora-step 50 --lora-epoch 1 --save-merged ./lora_merged_hf_ckpt
#   python3 merge_hf_ckpt.py --model meta-llama/Llama-3.2-1B --ckpt-dir ./lora_merged_hf_ckpt --output ./lora_merged_model
#
#   # With custom model (e.g., previously merged model)
#   torchrun --nproc_per_node=4 --master_port=29500
#     pp_inference_llama_lora.py --model-dir ./lora_merged_model --lora-step 30 --lora-epoch 1
#
#   # With dynamo-capture
#   torchrun --nproc_per_node=4 --master_port=29500
#     pp_inference_llama_lora.py --lora-step 50 --lora-epoch 1 --dynamo-capture
#

import torch
import torch.distributed as dist
import argparse
import os
import sys
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextStreamer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.inference import Optimus_Inference
from opt_prime.opti_pri import Optimus_p
from opt_prime.IR import IR_Anal
from opt_prime.lora import LoRAConfig

import logging
logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(description="PP Inference with LoRA adapters")
parser.add_argument('token', nargs='?', default=None, help='LLaMA access token')
parser.add_argument('--dynamo-capture', action='store_true', default=False,
                    help='Use torch.export instead of HFTracer')
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                    help='HuggingFace model name (default: meta-llama/Llama-3.2-1B)')
parser.add_argument('--model-dir', type=str, default=None,
                    help='Local model directory path (overrides --model)')
parser.add_argument('--pp-size', type=int, default=1,
                    help='Pipeline Parallel size (default: auto-calculated from world_size/tp_size)')
parser.add_argument('--tp-size', type=int, default=1,
                    help='Tensor Parallel size (default: 1, LLaMA only)')
parser.add_argument('--prompt', type=str, default='The future of artificial intelligence is',
                    help='Input prompt for generation')
parser.add_argument('--max-new-tokens', type=int, default=50,
                    help='Maximum tokens to generate')
parser.add_argument('--no-merge', action='store_true', default=False,
                    help='Keep LoRA adapters active (do not merge into base weights)')
parser.add_argument('--no-kv-cache', action='store_true', default=False,
                    help='Disable KV cache (use full-sequence recomputation)')
parser.add_argument('--no-sample', action='store_true', default=False,
                    help='Use greedy decoding instead of sampling')
parser.add_argument('--save-merged', type=str, default=None,
                    help='Save merged (base+LoRA) stage checkpoints to this dir, then exit. '
                         'Use merge_hf_ckpt.py afterwards to create HF model.')

# LoRA checkpoint location
parser.add_argument('--lora-step', type=int, required=True, help='LoRA checkpoint step')
parser.add_argument('--lora-epoch', type=int, required=True, help='LoRA checkpoint epoch')
parser.add_argument('--lora-dir-prefix', type=str, default='lora_checkpoint_stage_',
                    help='LoRA checkpoint directory prefix (default: lora_checkpoint_stage_)')

# LoRA config (must match training)
parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
parser.add_argument('--lora-alpha', type=float, default=16.0, help='LoRA alpha')
parser.add_argument('--lora-targets', nargs='+',
                    default=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'gate_proj', 'up_proj', 'down_proj'],
                    help='LoRA target module names')
args = parser.parse_args()

if args.token:
    os.environ['LLAMA_ACCESS_TOKEN'] = args.token
access_token = os.getenv('LLAMA_ACCESS_TOKEN')

model_name_or_path = args.model_dir if args.model_dir else args.model

rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

lora_config = LoRAConfig(
    r=args.lora_r,
    alpha=args.lora_alpha,
    dropout=0.0,
    target_modules=args.lora_targets,
)


# ================================================================
# --save-merged mode: use Optimus_p for LoRA merge + save_hf_ckpt
# ================================================================
if args.save_merged:
    if rank == 0:
        print(f"LoRA Save-Merged: step={args.lora_step}, epoch={args.lora_epoch}")
        print(f"Output dir: {args.save_merged}")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                  token=access_token, use_cache=False)
    num_mb = world_size // 2

    # Use Optimus_p (has save_hf_ckpt)
    optimus_p = Optimus_p(model, num_mb, use_gpu=True,
                          pp_size=args.pp_size, tp_size=args.tp_size,
                          ir_analyze=IR_Anal.SEQUENTIAL,
                          dynamo_capture=args.dynamo_capture)

    # Apply LoRA → load → merge
    optimus_p.apply_lora(lora_config)
    # Optimus_p.load_lora_ckpt appends "_stage_{N}" internally,
    # so extract the base name from lora_dir_prefix (e.g., "lora_checkpoint_stage_" → "lora_checkpoint")
    lora_base = args.lora_dir_prefix.rstrip('_')
    if lora_base.endswith('_stage'):
        lora_base = lora_base[:-6]  # strip "_stage" suffix
    else:
        lora_base = None  # non-standard prefix, fall back to default
    optimus_p.load_lora_ckpt(step=args.lora_step, epoch=args.lora_epoch, lora_dir=lora_base)
    optimus_p.merge_lora()

    # Save merged weights as stage files
    optimus_p.save_hf_ckpt(args.save_merged)

    if rank == 0:
        print(f"\nMerged (base+LoRA) stage checkpoints saved to: {args.save_merged}/")
        print(f"To create HF model:")
        print(f"  python3 merge_hf_ckpt.py --model {args.model} "
              f"--ckpt-dir {args.save_merged} --output ./lora_merged_model")

    print(f"[rank:{rank}] save-merged done.")
    sys.exit(0)


# ================================================================
# Normal inference mode: use Optimus_Inference
# ================================================================
if rank == 0:
    mode = "adapter (--no-merge)" if args.no_merge else "merge"
    print(f"LoRA Inference: mode={mode}, step={args.lora_step}, epoch={args.lora_epoch}")

# Load model
config = AutoConfig.from_pretrained(model_name_or_path, token=access_token, use_cache=False)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token=access_token,
                                              config=config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=access_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Create inference engine
use_kv = not args.no_kv_cache
engine = Optimus_Inference(
    model,
    use_gpu=True,
    pp_size=args.pp_size,
    tp_size=args.tp_size,
    use_kv_cache=use_kv,
    dynamo_capture=args.dynamo_capture,
)

# Apply LoRA adapters
engine.apply_lora(lora_config)

# Load trained LoRA weights for this stage
lora_dir = f"{args.lora_dir_prefix}{engine.tpl.stage}"
engine.load_lora_ckpt(step=args.lora_step, epoch=args.lora_epoch, lora_dir=lora_dir)

# Merge or keep adapter
if not args.no_merge:
    engine.merge_lora()

engine.eval()

if rank == 0:
    kv_str = "KV cache" if use_kv else "full-sequence"
    print(f"PP={engine.tpl.num_stage}, {kv_str}, prompt='{args.prompt}'")

# Prepare input
if engine.is_first_stage():
    tokens = tokenizer(args.prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    input_ids = tokens.input_ids.cuda()
else:
    input_ids = None

# Streaming output
if engine.is_output_rank():
    print("\n" + "=" * 60)
    print(args.prompt, end="", flush=True)
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)
else:
    streamer = None

start_time = time.time()

output_ids = engine.generate(
    input_ids,
    max_new_tokens=args.max_new_tokens,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=not args.no_sample,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    streamer=streamer,
    verbose=False,
)

gen_time = time.time() - start_time

if engine.is_output_rank() and output_ids is not None:
    print("=" * 60)
    prompt_tokens = len(tokenizer(args.prompt).input_ids)
    num_gen = output_ids.size(1) - prompt_tokens
    mode_str = "adapter" if args.no_merge else "merged"
    print(f"\nLoRA mode: {mode_str}")
    print(f"Tokens: {num_gen}, Time: {gen_time:.2f}s, Speed: {num_gen/gen_time:.1f} tok/s")

if dist.is_initialized():
    dist.barrier()

print(f"[rank:{rank}] done.")
