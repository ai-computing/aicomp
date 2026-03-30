#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# LoRA fine-tuning example using OptimusPrime pipeline parallelism.
# Only LoRA adapter weights (~0.5% of total) are trained, significantly
# reducing memory usage and training time compared to full fine-tuning.
#
# Usage:
#   torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0
#            --master_addr=xxx.xxx.xxx.xxx --master_port=29500
#            pp_train_llama_lora.py <llama_access_token>
#
#   torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0
#            --master_addr=xxx.xxx.xxx.xxx --master_port=29500
#            pp_train_llama_lora.py --dynamo-capture <llama_access_token>
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

parser = argparse.ArgumentParser()
parser.add_argument('token', nargs='?', default=None, help='LLaMA access token')
parser.add_argument('--dynamo-capture', action='store_true', default=False,
                    help='Use TorchDynamo capture (torch.export) instead of HFTracer')
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                    help='HuggingFace model name or local path (default: meta-llama/Llama-3.2-1B)')
parser.add_argument('--pp-size', type=int, default=1,
                    help='Pipeline Parallel size (default: auto-calculated from world_size/tp_size)')
parser.add_argument('--tp-size', type=int, default=1,
                    help='Tensor Parallel size (default: 1, LLaMA only)')
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
                    help='Directory to save LoRA checkpoints (default: lora_checkpoint_stage_{N})')
args = parser.parse_args()
if args.token:
    os.environ['LLAMA_ACCESS_TOKEN'] = args.token

access_token = os.getenv('LLAMA_ACCESS_TOKEN')
# access_token=None is OK — HuggingFace will use cached token from `huggingface-cli login`

required_version = "2.3.1"
if version.parse(torch.__version__) < version.parse(required_version):
    raise ValueError(f'This program needs torch version {required_version} or higher.')

required_tf_version = "4.46.2"
if version.parse(transformers.__version__) < version.parse(required_tf_version):
    raise ValueError(f'This program needs transformers version {required_tf_version} or higher.')


tokenizer = AutoTokenizer.from_pretrained(args.model, token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token, use_cache=False)

if int(os.environ["RANK"]) == 0:
    print(f"Model: {args.model}")

if int(os.environ["RANK"]) == 0:
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters in model: {total_params:,}')


batch_size = 32
num_mb = int(os.environ["WORLD_SIZE"]) // 2

if int(os.environ["RANK"]) == 0:
    print(f"total process count: {os.environ['WORLD_SIZE']}")
    print(f"batch size: {batch_size}")
    print(f"num of mbatch: {num_mb}")

# Step 1: Create Optimus_p (IR extraction → split → TP → DDP)
optimus_p = Optimus_p(model, num_mb, use_gpu=True,
                      pp_size=args.pp_size, tp_size=args.tp_size,
                      activation_ckpt=False, force_free_mem=True, display_mem=True,
                      swap_opt_in_fwdbwd=False, swap_model_in_optstep=False,
                      ir_analyze=IR_Anal.SEQUENTIAL,
                      dynamo_capture=args.dynamo_capture)
print(f" rank={optimus_p.get_rank()} ...")

# Step 2: Apply LoRA AFTER init, BEFORE optimizer
lora_config = LoRAConfig(
    r=args.lora_r,
    alpha=args.lora_alpha,
    dropout=args.lora_dropout,
    target_modules=args.lora_targets,
)
optimus_p.apply_lora(lora_config)

optimus_p.train()

# Step 3: Optimizer (only LoRA params) — use foreach=False for TP compatibility
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

if int(os.environ["RANK"]) == 0:
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
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
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
    total_gpus = int(os.environ.get('WORLD_SIZE', '1'))
    print(f"\nLoRA checkpoints saved to: lora_checkpoint_stage_*/")
    print(f"To run inference:")
    tp_flag = f" --tp-size {args.tp_size}" if args.tp_size > 1 else ""
    print(f"  torchrun --nproc_per_node={total_gpus} pp_inference_llama_lora.py"
          f"{tp_flag} --lora-step {args.max_steps} --lora-epoch 1")

print(f"[rank:{optimus_p.get_rank()}, run completed ...")
