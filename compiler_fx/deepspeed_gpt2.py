#
# Copyright (c) 2023-present, ETRI, All rights reserved.
#
#
# This is a test program for running DeepSpeed with huggingface GPT2 model.
#
#
# Sample Usage for GPT2 fine-tuning using DeepSpeed:
# 
#     # deepspeed --hostfile="host_file_name" --num_gpus 2 --num_nodes 2 --master_addr "X.X.X.X" --master_port 29500 deepspeed_gpt2.py --nnodes 2 --nproc_per_node 2
#
#

#!/usr/bin/env python
import os
import argparse

import torch
import torch.distributed as dist

import torch.multiprocessing as mp

import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
#from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import time

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import PreTrainedModel
from transformers.models.gpt2 import GPT2PreTrainedModel
import transformers.utils.fx as hf_fx

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from torch.utils.data.distributed import DistributedSampler

import deepspeed


#batch_size = 64
batch_size = 8

torch.manual_seed(42)

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#tokenizer.pad_token = tokenizer.eos_token
#tokenizer.pad_token_id = tokenizer.eos_token_id

#gpt_config = GPT2Config(use_cache=False)

#model = GPT2LMHeadModel(gpt_config)
#model = model.from_pretrained("gpt2")

config = {
    "train_batch_size": 32,
    #"fp16": {
    #    "enabled": True,
    #},
    "zero_optimization": {
        "stage": 2,
        #"allgather_partitions": True,
        #"allgather_bucket_size": 2e8,
        #"overlap_comm": True,
        #"reduce_scatter": True,
        #"reduce_bucket_size": 2e8,
        #"contiguous_gradients": True,
        #"offload_optimizer": {
        #    "device": "cpu",
        #    "pin_memory": True,
        #    "fast_init": True
        #},
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 3e-05,
            "betas": [
                0.9,
                0.999
            ],
            "eps": 1e-8
        }
    },
    #"zero_force_ds_cpu_optimizer": False,
}


def demo_basic():
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.getenv("MASTER_ADDR")
    master_port = os.getenv("MASTER_PORT")

    #sample_output = torch.rand(batch_size, out_features)
    #sample_input = torch.rand(batch_size, in_features)

    #print("----------------------------------------------------")
    #print("sample_input:", sample_input)
    #print("----------------------------------------------------")
    #print("sample_output:", sample_output)
    #print("----------------------------------------------------")

    init_method = "tcp://" + str(master_addr) + ":" + str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=init_method)
    deepspeed.init_distributed("nccl")
    #dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #if rank == 1:
    #    rank = 4
    rank = dist.get_rank()
    #torch.cuda.set_device(rank)
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    #dp_group = [0,1,2,3]
    #if rank < 2:
    #    dp_group = [0,1]
    #else:
    #    dp_group = [2,3]
    #dp_group = dist.new_group(dp_group)
    #dp_group = None
    device_id = rank % torch.cuda.device_count()
    #device_id = 0

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    gpt_config = GPT2Config(use_cache=False)
    
    model = GPT2LMHeadModel(gpt_config)
    model = model.from_pretrained("gpt2").to(device_id)
    #model = ToyModel().to(device_id)
    #model = ToyModel()
    model, optimizer, _, _ = deepspeed.initialize(model=model,
                                                    config_params=config,
                                                    model_parameters=model.parameters())
    #ddp_model = DDP(model, device_ids=[device_id])
    #ddp_model = DDP(model, process_group=dp_group)
    #ddp_model = DDP(model)
    #ddp_model = FSDP(model)
    #ddp_model = model

    #loss_fn = nn.MSELoss()
    #loss_fn = nn.CrossEntropyLoss()

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(record) for record in datasets if len(str(record)) < 500]
    #dataloader = DataLoader(datasets, batch_size=batch_size, num_workers=4)
    #if rank == 0:
    #    dataloader = DataLoader(dataset=datasets, batch_size=batch_size, pin_memory=True, shuffle=True)
    #else:
    #    dataloader = DataLoader(dataset=datasets, batch_size=batch_size, pin_memory=True, shuffle=False)
  
    #train_samples = DistributedSampler(datasets, shuffle=False)
    #train_samples.set_epoch(0)
    dataloader = DataLoader(dataset=datasets, 
                            batch_size=batch_size,
                            pin_memory=True,
                            shuffle=False,
                            sampler=DistributedSampler(datasets, shuffle=False))
    #print("kkd", train_samples.num_replicas, train_samples.seed)
    data_size=len(dataloader.dataset)   # total count of data
    print(f"data_size={data_size}")
    nbatches = len(dataloader)      # total count of data / batch size
    print(f"nbatches={nbatches}")

    if rank == 0:
        tick = time.time()

    #print("kkd")
    #for i in range(50):
    for i, batch in enumerate(dataloader):
        data = None
        labels = None

        #if i == 0:
        #    print(rank, batch)

        tokens = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        data, labels = tokens.input_ids, tokens.input_ids
        if i == 0:
            print(data, labels)
        #data = batch
        #labels = batch
        
        #if rank == 0:
        #    tick = time.time()
        #optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        #optimizer = optim.Adam(ddp_model.parameters(), lr=3e-5)
        #optimizer = optim.Adam(model.parameters(), lr=3e-5)
        #lr = 5.0
        #optimizer = optim.SGD(ddp_model.parameters(), lr=lr)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        optimizer.zero_grad()
        outputs = model(data.to(device_id), labels=labels.to(device_id))
        #outputs = ddp_model(data, labels=labels)
        #outputs = ddp_model(data.to(device_id))
        #outputs = ddp_model(sample_input.to(device_id))
        #outputs = ddp_model(sample_input)
        #labels = sample_output.to(device_id)
        #labels = sample_output
        #labels = labels.to(device_id)
        #loss = loss_fn(outputs, labels)
        loss = outputs[0]
        #loss.backward()
        model.backward(loss)
        #optimizer.step()
        model.step()

        #if rank == 0:
        #    tock = time.time()
        #    elapsed_time = tock - tick
        #if rank == 0 or rank == 1:
        #if rank == 0:
            #print("\n")
        print('rank:', rank, 'loss:', loss)
            #for param in ddp_model.parameters():
            #    print('rank', rank, 'parameters', param.data.tolist())
        #    print('Time elapsed: %.3f sec ' % (elapsed_time))
        #time.sleep(1)

    if rank == 0:
        tock = time.time()
        elapsed_time = tock - tick
        print('Time elapsed: %.3f sec ' % (elapsed_time))

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='PyTorch DDP Example')
    #args = parser.parse_args()

    #WORLD_SIZE = torch.cuda.device_count()
    #WORLD_SIZE = 4
    #rank = WORLD_SIZE
    
    demo_basic()

    '''
    mp.spawn(demo_basic,
            args=(WORLD_SIZE, args),
            nprocs=WORLD_SIZE,
            join=True)'''

    #demo_basic()
