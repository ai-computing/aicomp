#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#
# Usage: torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> 
#                 --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama.py <llama_access_token>
#
#        Memory options: [--dtype {auto,fp32,bf16}] [--activation-ckpt] [--swap-opt] [--cpu-opt-step] [--low-mem]
#        With the defaults and fp32 weights (transformers 4.x), Llama-2-13B needs ~53 GB per GPU at PP=4,
#        so 4x 48GB GPUs run out of memory; use --low-mem (or --dtype bf16) there.
#        See "Memory needed for Llama-2-13B" in opt_prime/README.md.
#
# *** This program was tested with torch 2.3.1 and transformers 4.42.4.
#     The version of transformers used must be consistent across all machines used for testing ***
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

import argparse

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p
from opt_prime.IR import IR_Anal

from torch.amp import autocast

logging.basicConfig(level=logging.ERROR)


#
# This program needs 'access token' for Llama. First, obtain your access token for Llama !!!
#
parser = argparse.ArgumentParser()
parser.add_argument('token', nargs='?', default=None, help='LLaMA access token')
parser.add_argument('--dynamo-capture', action='store_true', default=False,
                    help='Use TorchDynamo capture (torch.export) instead of HFTracer')
# Memory options (issue #21).  Defaults keep the original behavior.  With the
# defaults, fp32 weights (transformers 4.x) need ~16 B/param for weights+grads+Adam
# state, i.e. ~53 GB per GPU at PP=4, which does not fit a 48 GB A40.  See
# "Memory needed for Llama-2-13B" in opt_prime/README.md for the estimates.
parser.add_argument('--batch-size', type=int, default=64, help='global batch size (default: 64)')
parser.add_argument('--num-mb', type=int, default=16, help='number of micro-batches (default: 16)')
parser.add_argument('--max-length', type=int, default=1024,
                    help='tokenizer truncation length; batches are padded to their longest sample '
                         '(the SQuAD contexts used here are < 500 chars, ~ <= 220 tokens)')
parser.add_argument('--dtype', choices=['auto', 'fp32', 'bf16'], default='auto',
                    help='weight/grad/optimizer-state dtype. auto (default): checkpoint dtype, '
                         'fp16 cast to bf16 (fp32 with transformers 4.x, bf16 with 5.x)')
parser.add_argument('--activation-ckpt', action='store_true', default=False,
                    help='drop stage activations after forward and recompute them in backward '
                         '(all stages except the last)')
parser.add_argument('--swap-opt', action='store_true', default=False,
                    help='swap_opt_in_fwdbwd: offload optimizer state to host memory during fwd/bwd')
parser.add_argument('--cpu-opt-step', action='store_true', default=False,
                    help='swap_model_in_optstep: run optimizer.step() on the CPU (weights, grads and '
                         'optimizer state are moved to host memory); implies --swap-opt')
parser.add_argument('--low-mem', action='store_true', default=False,
                    help='shortcut for --activation-ckpt --swap-opt --cpu-opt-step '
                         '(estimated to fit Llama-2-13B on 4x 48GB GPUs in fp32)')
args = parser.parse_args()
if args.low_mem:
    args.activation_ckpt = args.swap_opt = args.cpu_opt_step = True
if args.cpu_opt_step:
    args.swap_opt = True   # swap_model_in_optstep must be used together with swap_opt_in_fwdbwd
if args.token:
    os.environ['LLAMA_ACCESS_TOKEN'] = args.token

access_token = os.getenv('LLAMA_ACCESS_TOKEN')
# access_token=None is OK — HuggingFace will use cached token from `huggingface-cli login`


#
# This program needs torch version 2.3.1 or higher !!!
#
required_version = "2.3.1"
current_version = torch.__version__

if version.parse(current_version) >= version.parse(required_version):
    print(f"[rank:{int(os.environ['RANK'])}] torch version 2.3.1 or higher --> OK")
else:
    raise ValueError('This program needs torch version 2.3.1 or higher. But current torch version is {current_version}.')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token=access_token, use_cache=False)
# Only fp16 needs the cast: transformers 5.x keeps the checkpoint dtype (fp16 here)
# and training fp16 weights with Adam gives NaN on the first optimizer.step()
# (Adam eps=1e-8 < smallest fp16 subnormal).  transformers 4.x loads fp32, and
# the guard makes this a no-op there, so the old behavior is preserved exactly.
# bfloat16 keeps the 13B model at 16-bit (fp32 would need ~52GB of host RAM per
# rank); .float() works too since the fp32 RMSNorm alias issue was fixed
# (see CLAUDE.md 3-G), it just costs twice the memory and compute.
if args.dtype == 'fp32':
    model = model.float()
elif args.dtype == 'bf16' or next(model.parameters()).dtype == torch.float16:
    model = model.bfloat16()

def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


if int(os.environ["RANK"]) == 0:
    print ('Total parameters in model: {:,}'.format(get_total_params(model)))


batch_size = args.batch_size
num_mb = args.num_mb

if int(os.environ["RANK"]) == 0:
    print(f"total process count: {os.environ['WORLD_SIZE']}")
    print(f"batch size: {batch_size}")
    print(f"num of mbatch: {num_mb}")
    print(f"dtype: {next(model.parameters()).dtype}, activation_ckpt: {args.activation_ckpt}, "
          f"swap_opt_in_fwdbwd: {args.swap_opt}, swap_model_in_optstep: {args.cpu_opt_step}")

#optimus_p = Optimus_p(model, num_mb, use_gpu=True, dynamo_capture=args.dynamo_capture)
#optimus_p = Optimus_p(model, num_mb, use_gpu=True, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=False, ir_analyze=IR_Anal.SEQUENTIAL, dynamo_capture=args.dynamo_capture)
optimus_p = Optimus_p(model, num_mb, use_gpu=True, activation_ckpt=args.activation_ckpt, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=args.swap_opt, swap_model_in_optstep=args.cpu_opt_step, ir_analyze=IR_Anal.SEQUENTIAL, dynamo_capture=args.dynamo_capture)
print(f" rank={optimus_p.get_rank()} ...")

optimus_p.train()

#optimus_p.optimizer = torch.optim.SGD(optimus_p.parameters(), lr=5.0)
optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimus_p.optimizer, 1.0, gamma=0.95)

datasets = load_dataset("rajpurkar/squad").data["train"]["context"]
datasets = [str(record) for record in datasets if len(str(record)) < 500]
dataloader = DataLoader(datasets, batch_size=batch_size, num_workers=4)
data_size=len(dataloader.dataset)
print(f"data_size={data_size}")
nbatches = len(dataloader)
print(f"nbatches={nbatches}")


epochs = 1 # The number of epochs

def train():

    optimus_p.train() # turn on the train mode

    total_loss = 0
    start_time = time.time()

    for i, batch in enumerate(dataloader):

        data, labels = None, None

        # prepare input and label
        if optimus_p.is_first_stage():
            tokens =  tokenizer(batch, padding=True, truncation=True, max_length=args.max_length,return_tensors="pt")
            input_ids = tokens.input_ids
            # Causal LM: logits[t] should predict input_ids[t+1]
            shifted_labels = input_ids.clone()
            shifted_labels[:, :-1] = input_ids[:, 1:]
            shifted_labels[:, -1] = -100  # no target for last position
            data, labels = input_ids, shifted_labels

        labels = optimus_p.move_labels2last_stage(labels)

        optimus_p.optimizer.zero_grad()

        #optimus_p.run(data, labels)
        #optimus_p.run(data, labels, mode="gpipe")
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            optimus_p.run(data, labels, mode="1f1b")

        if optimus_p.is_last_stage():
            loss = optimus_p.get_loss() 
        else:
            loss = None

        torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)
        optimus_p.optimizer.step()

        if optimus_p.is_last_stage():
            loss = sum(loss) / optimus_p.num_mb
            total_loss += loss
            log_interval = 10
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, i, nbatches, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))

                total_loss = 0
                start_time = time.time()

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

print(f"[rank:{optimus_p.get_rank()}] peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"[rank:{optimus_p.get_rank()}, run completed ...")

