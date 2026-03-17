#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#
# Train (fine-tune) a merged HuggingFace checkpoint using OptimusPrime pipeline parallelism.
#
# Usage:
#   torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0
#            --master_addr=xxx.xxx.xxx.xxx --master_port=29500
#            pp_train_from_hf_ckpt.py --model-dir ./merged_model
#
#   # With max-steps limit (default: 20)
#   torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0
#            --master_addr=xxx.xxx.xxx.xxx --master_port=29500
#            pp_train_from_hf_ckpt.py --model-dir ./merged_model --max-steps 30
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

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str, required=True,
                    help='Path to merged HuggingFace model directory')
parser.add_argument('--dynamo-capture', action='store_true', default=False,
                    help='Use TorchDynamo capture (torch.export) instead of HFTracer')
parser.add_argument('--save-dir', type=str, default='./hf_ckpt_trained',
                    help='Directory to save stage checkpoints')
parser.add_argument('--max-steps', type=int, default=20,
                    help='Stop training after N steps to prevent overfitting (default: 20, 0=full epoch)')
args = parser.parse_args()

required_version = "2.3.1"
if version.parse(torch.__version__) < version.parse(required_version):
    raise ValueError(f'This program needs torch version {required_version} or higher.')

required_tf_version = "4.46.2"
if version.parse(transformers.__version__) < version.parse(required_tf_version):
    raise ValueError(f'This program needs transformers version {required_tf_version} or higher.')


if int(os.environ["RANK"]) == 0:
    print(f"Loading model from: {args.model_dir}")

tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(args.model_dir, use_cache=False)

if int(os.environ["RANK"]) == 0:
    total_params = sum(p.numel() for p in model.parameters())
    print('Total parameters: {:,}'.format(total_params))


batch_size = 32
num_mb = int(os.environ["WORLD_SIZE"]) // 2

if int(os.environ["RANK"]) == 0:
    print(f"total process count: {os.environ['WORLD_SIZE']}")
    print(f"batch size: {batch_size}")
    print(f"num of mbatch: {num_mb}")

optimus_p = Optimus_p(model, num_mb, use_gpu=True,
                      activation_ckpt=True, force_free_mem=True, display_mem=True,
                      swap_opt_in_fwdbwd=True, swap_model_in_optstep=False,
                      ir_analyze=IR_Anal.SEQUENTIAL,
                      dynamo_capture=args.dynamo_capture)
print(f" rank={optimus_p.get_rank()} ...")

optimus_p.train()

# AdamW with low LR and weight decay for regularization
optimus_p.optimizer = torch.optim.AdamW(optimus_p.parameters(), lr=5e-6, weight_decay=0.01)

# Format squad data as Q&A pairs to prevent catastrophic forgetting
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

# Warmup + Cosine Decay scheduler
num_warmup_steps = min(10, nbatches // 5)
total_steps = args.max_steps if args.max_steps > 0 else nbatches

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
    train()

if optimus_p.get_rank() == 0:
    tock = time.time()
    print('Time elapsed: %.3f sec ' % (tock - tick))
    print(f"\nTrained checkpoints saved to: {args.save_dir}/")
    print(f"To merge: python3 merge_hf_ckpt.py --model {args.model_dir} "
          f"--ckpt-dir {args.save_dir} --output ./trained_model")

print(f"[rank:{optimus_p.get_rank()}, run completed ...")
