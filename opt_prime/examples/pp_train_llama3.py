#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#
# Usage: torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> 
#                 --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama.py <llama_access_token>
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

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p
from opt_prime.IR import IR_Anal

logging.basicConfig(level=logging.ERROR)

#
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
master_addr = os.getenv("MASTER_ADDR")
master_port = os.getenv("MASTER_PORT")

cache_dir="mycache_dir2"

init_method = "tcp://" + str(master_addr) + ":" + str(master_port)
#print(f"rank:{rank}, world_size:{world_size}, init_method:{init_method}")
print(f"rank:{rank}, world_size:{world_size}, init_method:{init_method}, local_world_size:{local_world_size}, local_rank:{local_rank}")

dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=init_method)

###
group_gloo = dist.new_group(backend="gloo")

batch_size = 32
micro_batch_size = 8 # TODO

if int(os.environ["RANK"]) == 0:
    print(f"total process count: {os.environ['WORLD_SIZE']}")
    print(f"batch size: {batch_size}")
    print(f"micro batch size: {micro_batch_size}")



#
# This program needs 'access token' for Llama. First, obtain your access token for Llama !!!
#
if len(sys.argv) > 1:
    os.environ['LLAMA_ACCESS_TOKEN'] = sys.argv[1]

access_token = os.getenv('LLAMA_ACCESS_TOKEN')
if access_token is None:
    raise ValueError("LLAMA_ACCESS_TOKEN environment variable is not set."
                    "       [Usage:] torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama.py <llama_access_token>")


#
# This program needs torch version 2.3.1 or higher !!!
#
required_version = "2.3.1"
current_version = torch.__version__

if version.parse(current_version) >= version.parse(required_version):
    print(f"[rank:{int(os.environ['RANK'])}] torch version 2.3.1 or higher --> OK")
else:
    raise ValueError('This program needs torch version 2.3.1 or higher. But current torch version is {current_version}.')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", token=access_token, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params

for i in range(local_world_size):
    if local_rank == i:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", token=access_token, use_cache=False, cache_dir=cache_dir)

        if int(os.environ["RANK"]) == 0:
            print('Total parameters in model: {:,}'.format(get_total_params(model)))

        optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.PARALLEL) ## IR_Anal.PARALLEL 
        print(f" rank={optimus_p.get_rank()} ...")

    print(f"..[local_rank:{local_rank}, i:{i}] Before barrier()...")
    dist.barrier(group=group_gloo)
    print(f"..[local_rank:{local_rank}, i:{i}] After barrier()...................................")

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

        #optimus_p.run(data, labels)
        #optimus_p.run(data, labels, mode="gpipe")
        optimus_p.run(data, labels, mode="1f1b")

        if optimus_p.is_last_stage():
            loss = optimus_p.get_loss() 
        else:
            loss = None

        torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)
        optimus_p.optimizer.step()
        #
        print(f">> [Rank:{int(os.environ['RANK'])}] ({i}) After optimizer.step() ..") # TO DELETE

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

print(f"[rank:{optimus_p.get_rank()}, run completed ...")

