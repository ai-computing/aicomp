#
# Copyright (c) 2024-present, ETRI, All rights reserved.
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

from transformers import GPT2Tokenizer, GPTJForCausalLM, GPTJConfig
from datasets import load_dataset
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p

logging.basicConfig(level=logging.ERROR)

#
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
cache_dir="mycache_dir1"

dist.init_process_group("nccl", rank=rank, world_size=world_size)
        

#tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6b")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6b", cache_dir=cache_dir) 
#tokenizer = GPT2Tokenizer.from_pretrained("anton-l/gpt-j-tiny-random")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

config = GPTJConfig(use_cache=False)
model = GPTJForCausalLM(config)

#model = model.from_pretrained("EleutherAI/gpt-j-6b")
if local_rank == 0:
    model = model.from_pretrained("EleutherAI/gpt-j-6b", cache_dir=cache_dir)
    dist.barrier()
else:
    dist.barrier()
    model = model.from_pretrained("EleutherAI/gpt-j-6b", cache_dir=cache_dir)

#model = model.from_pretrained("anton-l/gpt-j-tiny-random")


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


if int(os.environ["RANK"]) == 0:
    print ('Total parameters in model: {:,}'.format(get_total_params(model)))

print(f"Begin [{local_rank}]: optimus_p ..")

batch_size = 32
micro_batch_size = int(os.environ["WORLD_SIZE"]) // 2 # TODO

if int(os.environ["RANK"]) == 0:
    print(f"total process count: {os.environ['WORLD_SIZE']}")
    print(f"batch size: {batch_size}")
    print(f"micro batch size: {micro_batch_size}")

#optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True)
optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, activation_ckpt=True, force_free_mem=True, display_mem=True, optimizer_offload=True)
print(f" local_rank={optimus_p.get_local_rank()} ...")

optimus_p.train()
#optimus_p.optimizer = torch.optim.SGD(optimus_p.parameters(), lr=5.0)
optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimus_p.optimizer, 1.0, gamma=0.95)

datasets = load_dataset("squad").data["train"]["context"]
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
            tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
            data, labels = tokens.input_ids, tokens.input_ids

        labels = optimus_p.move_labels2last_stage(labels)

        optimus_p.optimizer.zero_grad()

        optimus_p.run(data, labels)

        if optimus_p.is_last_stage():
            loss = optimus_p.get_loss()
        else:
            loss = None

        torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)
        optimus_p.optimizer.step()

        if optimus_p.is_last_stage():
            loss = sum(loss) / optimus_p.mbsize
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

if optimus_p.get_rank == optimus_p.get_world_size() - 1:
    print(f"###################################")
print(f"[rank:{optimus_p.get_rank()}, run completed ...")
