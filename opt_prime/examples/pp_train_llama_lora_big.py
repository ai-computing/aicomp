#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# LoRA fine-tuning of large LLaMA models (e.g., 70B) using OptimusPrime
# pipeline parallelism with sequential model loading to avoid host memory OOM.
#
# This script combines the memory-efficient sequential loading from
# pp_train_llama7.py with the LoRA training flow from pp_train_llama_lora.py.
#
# Key features for large models:
#   - Sequential model loading per local_rank (prevents host OOM when multiple
#     GPUs on the same node try to load 70B simultaneously)
#   - gloo barrier synchronization via pre_barrier
#   - CPU offload during forward/backward (swap_opt_in_fwdbwd)
#   - CPU offload during optimizer step (swap_model_in_optstep)
#   - LoRA: only adapter weights (~0.03% of 70B) are trained
#
# Usage:
#   torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0
#            --master_addr=xxx.xxx.xxx.xxx --master_port=29500
#            pp_train_llama_lora_big.py [llama_access_token]
#
#   torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0
#            --master_addr=xxx.xxx.xxx.xxx --master_port=29500
#            pp_train_llama_lora_big.py --dynamo-capture [llama_access_token]
#

import torch
import torch.nn as nn
import torch.distributed as dist
import logging
import os
import sys
import math
import time
from packaging import version

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers
import argparse

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p
from opt_prime.IR import IR_Anal
from opt_prime.lora import LoRAConfig

logging.basicConfig(level=logging.ERROR)

#
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
master_addr = os.getenv("MASTER_ADDR")
master_port = os.getenv("MASTER_PORT")

init_method = "tcp://" + str(master_addr) + ":" + str(master_port)
print(f"rank:{rank}, world_size:{world_size}, init_method:{init_method}, local_world_size:{local_world_size}, local_rank:{local_rank}")

dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=init_method)

###
group_gloo = dist.new_group(backend="gloo")


parser = argparse.ArgumentParser()
parser.add_argument('token', nargs='?', default=None, help='LLaMA access token')
parser.add_argument('--dynamo-capture', action='store_true', default=False,
                    help='Use TorchDynamo capture (torch.export) instead of HFTracer')
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.3-70B-Instruct',
                    help='HuggingFace model name (default: meta-llama/Llama-3.3-70B-Instruct)')
parser.add_argument('--model-dir', type=str, default=None,
                    help='Local model directory path (overrides --model)')
parser.add_argument('--pp-size', type=int, default=1,
                    help='Pipeline Parallel size (default: auto from world_size/tp_size)')
parser.add_argument('--tp-size', type=int, default=1,
                    help='Tensor Parallel size (default: 1)')
parser.add_argument('--max-steps', type=int, default=50,
                    help='Stop training after N steps (default: 50, 0=full epoch)')
parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
parser.add_argument('--lora-alpha', type=float, default=16.0, help='LoRA alpha')
parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
parser.add_argument('--lora-targets', nargs='+',
                    default=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'gate_proj', 'up_proj', 'down_proj'],
                    help='LoRA target module names')
parser.add_argument('--lora-dir', type=str, default=None,
                    help='Directory to save LoRA checkpoints')
parser.add_argument('--activation-ckpt', action='store_true', default=False,
                    help='Enable activation checkpointing')
args = parser.parse_args()
if args.token:
    os.environ['LLAMA_ACCESS_TOKEN'] = args.token

access_token = os.getenv('LLAMA_ACCESS_TOKEN')
# access_token=None is OK — HuggingFace will use cached token from `huggingface-cli login`

#
# Version checks
#
required_version = "2.3.1"
current_version = torch.__version__

if version.parse(current_version) >= version.parse(required_version):
    print(f"[rank:{rank}] torch version {required_version} or higher --> OK")
else:
    print(f"[rank:{rank}] current torch version is {current_version}.")
    raise ValueError(f'This program needs torch version {required_version} or higher.')

required_tf_version = "4.46.2"
current_tf_version = transformers.__version__

if version.parse(current_tf_version) >= version.parse(required_tf_version):
    print(f"[rank:{rank}] transformers version {required_tf_version} or higher --> OK")
else:
    print(f"[rank:{rank}] current transformers version is {current_tf_version}.")
    raise ValueError(f'This program needs transformers version {required_tf_version} or higher.')


model_name_or_path = args.model_dir if args.model_dir else args.model

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


#batch_size = 32
#batch_size = 16
batch_size = 2
#num_mb = world_size // args.tp_size
num_mb = 2

if rank == 0:
    print(f"total process count: {world_size}")
    print(f"batch size: {batch_size}")
    print(f"num of mbatch: {num_mb}")


#
# Sequential model loading to avoid host memory OOM on large models.
# Each local_rank loads the full model one at a time. After Optimus_p
# partitions and frees the original model (force_free_mem=True), the
# next local_rank proceeds via gloo barrier synchronization.
#
for i in range(local_world_size):
    if local_rank == i:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, token=access_token, use_cache=False)

        if rank == 0:
            print(f'Model: {model_name_or_path}')
            print(f'Total parameters in model: {get_total_params(model):,}')

        optimus_p = Optimus_p(
            model, num_mb, use_gpu=True,
            pp_size=args.pp_size, tp_size=args.tp_size,
            activation_ckpt=args.activation_ckpt,
            force_free_mem=True, display_mem=True,
            swap_opt_in_fwdbwd=True, swap_model_in_optstep=True,
            ir_analyze=IR_Anal.PARALLEL,
            pre_barrier=group_gloo,
            dynamo_capture=args.dynamo_capture)
        print(f" rank={optimus_p.get_rank()} ...")

    if local_rank > i:
        print(f"..[local_rank:{local_rank}, i:{i}] Before barrier()...")
        dist.barrier(group=group_gloo)
        print(f"..[local_rank:{local_rank}, i:{i}] After barrier()...................................")


# Apply LoRA AFTER pipeline init, BEFORE optimizer
lora_config = LoRAConfig(
    r=args.lora_r,
    alpha=args.lora_alpha,
    dropout=args.lora_dropout,
    target_modules=args.lora_targets,
)
optimus_p.apply_lora(lora_config)

optimus_p.train()

# Optimizer (only LoRA params) — foreach=False required for TP compatibility
if optimus_p.tpl.tp_size > 1:
    optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=2e-4, foreach=False)
else:
    optimus_p.optimizer = torch.optim.AdamW(optimus_p.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimus_p.optimizer, T_max=10)


# Dataset: Q&A format to prevent catastrophic forgetting
squad = load_dataset("squad")
train_data = squad["train"]

def format_squad_qa(examples):
    formatted = []
    for ctx, q, ans in zip(examples["context"], examples["question"], examples["answers"]):
        answer_text = ans["text"][0] if ans["text"] else ""
        text = (f"### Context:\n{ctx}\n\n"
                f"### Question:\n{q}\n\n"
                f"### Answer:\n{answer_text}")
        formatted.append(text)
    return formatted

max_examples = 3000
train_data = train_data.select(range(min(max_examples, len(train_data))))
datasets_formatted = format_squad_qa(train_data)

dataloader = optimus_p.prepare_dataloader(datasets_formatted, batch_size)
data_size = len(dataloader.dataset)
nbatches = len(dataloader)

if rank == 0:
    print(f"data_size={data_size}, nbatches={nbatches}")


epochs = 1

def train():
    optimus_p.train()
    total_loss = 0
    start_time = time.time()
    global_step = 0

    for i, batch in enumerate(dataloader):
        data, labels = None, None

        if optimus_p.is_first_stage():
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            input_ids = tokens.input_ids
            # Causal LM: logits[t] should predict input_ids[t+1]
            shifted_labels = input_ids.clone()
            shifted_labels[:, :-1] = input_ids[:, 1:]
            shifted_labels[:, -1] = -100  # no target for last position
            data, labels = input_ids, shifted_labels

        labels = optimus_p.move_labels2last_stage(labels)

        optimus_p.optimizer.zero_grad()
        optimus_p.run(data, labels, mode="1f1b")

        if optimus_p.is_last_stage():
            loss = optimus_p.get_loss()
        else:
            loss = None

        # clip_grad_norm_ incompatible with DTensor (TP) — skip when tp_size > 1
        if optimus_p.tpl.tp_size <= 1:
            torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 1.0)
        optimus_p.optimizer.step()
        scheduler.step()
        global_step += 1

        if optimus_p.is_last_stage():
            loss = sum(loss) / optimus_p.num_mb
            total_loss += loss
            log_interval = 10
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                cur_lr = scheduler.get_last_lr()[0]
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:.2e} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, i, nbatches, cur_lr,
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

        if args.max_steps > 0 and global_step >= args.max_steps:
            if optimus_p.is_last_stage():
                print(f"| Stopping at step {global_step} (--max-steps={args.max_steps})")
            break

    # Save LoRA checkpoint at end of training
    optimus_p.save_lora_ckpt(step=global_step, epoch=epoch, lora_dir=args.lora_dir)


if optimus_p.get_rank() == 0:
    tick = time.time()

for epoch in range(1, epochs + 1):
    train()

if optimus_p.get_rank() == 0:
    tock = time.time()
    print('Time elapsed: %.3f sec ' % (tock - tick))
    total_gpus = world_size
    print(f"\nLoRA checkpoints saved to: lora_checkpoint_stage_*/")
    print(f"To run inference:")
    tp_flag = f" --tp-size {args.tp_size}" if args.tp_size > 1 else ""
    print(f"  torchrun --nproc_per_node={total_gpus} pp_inference_llama_lora.py"
          f"{tp_flag} --lora-step {args.max_steps} --lora-epoch 1")

if dist.is_initialized():
    try:
        dist.barrier()
        print(f"[rank:{optimus_p.get_rank()} >> barrier ...")
        torch.cuda.synchronize()
        print(f"[rank:{optimus_p.get_rank()} >> synchronize...")
        dist.destroy_process_group()
    except Exception as e:
        print(f"Cleanup on rank {optimus_p.get_rank()}: {e}")

print(f"[rank:{optimus_p.get_rank()}, run completed ...")
