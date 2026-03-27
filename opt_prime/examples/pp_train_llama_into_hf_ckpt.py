#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#
# Usage: torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank>
#                 --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama_into_hf_ckpt.py <llama_access_token>
#
#   PP only (4 GPUs):
#     torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0
#              --master_addr=xxx.xxx.xxx.xxx --master_port=29500
#              pp_train_llama_into_hf_ckpt.py <llama_access_token>
#
# After training, merge the stage checkpoints into a single HuggingFace model:
#   python3 merge_hf_ckpt.py --model meta-llama/Llama-3.2-1B --ckpt-dir ./hf_ckpt --output ./merged_model
#

import torch
import torch.nn as nn
import torch.distributed as dist
import datetime
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

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('token', nargs='?', default=None, help='LLaMA access token')
parser.add_argument('--dynamo-capture', action='store_true', default=False,
                    help='Use TorchDynamo capture (torch.export) instead of HFTracer')
parser.add_argument('--save-dir', type=str, default='./hf_ckpt',
                    help='Directory to save stage checkpoints (default: ./hf_ckpt)')
parser.add_argument('--save-interval', type=int, default=0,
                    help='Save checkpoint every N steps (0 = save at epoch end only)')
parser.add_argument('--max-steps', type=int, default=30,
                    help='Stop training after N steps to prevent overfitting (default: 30)')
args = parser.parse_args()
if args.token:
    os.environ['LLAMA_ACCESS_TOKEN'] = args.token

access_token = os.getenv('LLAMA_ACCESS_TOKEN')
# access_token=None is OK — HuggingFace will use cached token from `huggingface-cli login`

required_version = "2.3.1"
current_version = torch.__version__
if version.parse(current_version) < version.parse(required_version):
    raise ValueError(f'This program needs torch version {required_version} or higher. Current: {current_version}')

required_tf_version = "4.46.2"
current_tf_version = transformers.__version__
if version.parse(current_tf_version) < version.parse(required_tf_version):
    raise ValueError(f'This program needs transformers version {required_tf_version} or higher. Current: {current_tf_version}')


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token, use_cache=False)

def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params

if int(os.environ["RANK"]) == 0:
    print('Total parameters in model: {:,}'.format(get_total_params(model)))


batch_size = 32
num_mb = int(os.environ["WORLD_SIZE"]) // 2

if int(os.environ["RANK"]) == 0:
    print(f"total process count: {os.environ['WORLD_SIZE']}")
    print(f"batch size: {batch_size}")
    print(f"num of mbatch: {num_mb}")

optimus_p = Optimus_p(model, num_mb, use_gpu=True, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=False, ir_analyze=IR_Anal.SEQUENTIAL, dynamo_capture=args.dynamo_capture)
print(f" rank={optimus_p.get_rank()} ...")

optimus_p.train()

# Use Adam with foreach=False for TP compatibility (foreach ops fail with mixed DTensor/Tensor).
# When tp_size=1, AdamW with weight_decay can be used for better regularization.
if optimus_p.tpl.tp_size > 1:
    optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=5e-6, foreach=False)
else:
    optimus_p.optimizer = torch.optim.AdamW(optimus_p.parameters(), lr=5e-6, weight_decay=0.01)

# Format squad data as Q&A pairs for instruction-style fine-tuning.
# Using only raw context for causal LM causes catastrophic forgetting because
# the model overfits to short, homogeneous Wikipedia-style text.
squad = load_dataset("squad")
train_data = squad["train"]

def format_squad_qa(examples):
    """Format squad examples as instruction-following Q&A text."""
    formatted = []
    for ctx, q, ans in zip(examples["context"], examples["question"], examples["answers"]):
        answer_text = ans["text"][0] if ans["text"] else ""
        text = (f"### Context:\n{ctx}\n\n"
                f"### Question:\n{q}\n\n"
                f"### Answer:\n{answer_text}")
        formatted.append(text)
    return formatted

# Build formatted dataset (list of strings)
# Use first 3000 examples for quick experimentation (~100 batches, ~7 min)
max_examples = 3000
train_data = train_data.select(range(min(max_examples, len(train_data))))
datasets_formatted = format_squad_qa(train_data)

dataloader = optimus_p.prepare_dataloader(datasets_formatted, batch_size)
data_size = len(dataloader.dataset)
print(f"data_size={data_size}")
nbatches = len(dataloader)
print(f"nbatches={nbatches}")

# Warmup + Cosine Decay scheduler to prevent aggressive early updates
num_warmup_steps = min(100, nbatches // 10)
total_steps = nbatches  # 1 epoch

def lr_lambda(current_step):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimus_p.optimizer, lr_lambda)

epochs = 1

def train():
    optimus_p.train()
    total_loss = 0
    start_time = time.time()
    global_step = 0

    for i, batch in enumerate(dataloader):
        data, labels = None, None

        if optimus_p.is_first_stage():
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            data, labels = tokens.input_ids, tokens.input_ids

        labels = optimus_p.move_labels2last_stage(labels)

        optimus_p.optimizer.zero_grad()
        optimus_p.run(data, labels, mode="1f1b")

        if optimus_p.is_last_stage():
            loss = optimus_p.get_loss()
        else:
            loss = None

        # clip_grad_norm_ is incompatible with DTensor (TP) — skip when tp_size > 1
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

        # Periodic save
        if args.save_interval > 0 and (i + 1) % args.save_interval == 0:
            optimus_p.save_hf_ckpt(args.save_dir, step=i+1, epoch=epoch)

        # Early stop to prevent overfitting
        if args.max_steps > 0 and global_step >= args.max_steps:
            if optimus_p.is_last_stage():
                print(f"| Stopping at step {global_step} (--max-steps={args.max_steps})")
            break

    # End-of-training save
    optimus_p.save_hf_ckpt(args.save_dir, step=global_step, epoch=epoch)


if optimus_p.get_rank() == 0:
    tick = time.time()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    scheduler.step()

if optimus_p.get_rank() == 0:
    tock = time.time()
    elapsed_time = tock - tick
    print('Time elapsed: %.3f sec ' % (elapsed_time))
    print(f"\nStage checkpoints saved to: {args.save_dir}/")
    print(f"To merge into HuggingFace format, run:")
    print(f"  python3 merge_hf_ckpt.py --model meta-llama/Llama-3.2-1B "
          f"--ckpt-dir {args.save_dir} --output ./merged_model")

print(f"[rank:{optimus_p.get_rank()}, run completed ...")
