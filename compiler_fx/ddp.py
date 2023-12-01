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

batch_size = 64

torch.manual_seed(42)


in_features = 5120
out_features = 5120
hidden = 5120


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.ModuleList()
        for i in range(2):
            self.linear2.append(nn.Linear(hidden, hidden))

        self.linear3 = nn.ModuleList()
        for i in range(2):
            self.linear3.append(nn.Linear(hidden, hidden))

        self.linear4 = nn.ModuleList()
        for i in range(2):
            self.linear4.append(nn.Linear(hidden, hidden))

        self.linear5 = nn.ModuleList()
        for i in range(2):
            self.linear5.append(nn.Linear(hidden, hidden))
        self.linear6 = nn.Linear(hidden, out_features)
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        for m in self.linear2:
            x = self.relu(m(x))
        for m in self.linear3:
            x = self.relu(m(x))
        for m in self.linear4:
            x = self.relu(m(x))
        for m in self.linear5:
            x = self.relu(m(x))
        x = self.linear6(x)
        x = self.relu(x)
        return x




def demo_basic():
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    sample_output = torch.rand(batch_size//world_size, out_features)
    sample_input = torch.rand(batch_size//world_size, in_features)

    #print("----------------------------------------------------")
    #print("sample_input:", sample_input)
    #print("----------------------------------------------------")
    #print("sample_output:", sample_output)
    #print("----------------------------------------------------")

    dist.init_process_group("nccl")
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
    
    
    model = TestModel().to(device_id)
    #model = ToyModel().to(device_id)
    #model = ToyModel()
    ddp_model = DDP(model, device_ids=[device_id])
    #ddp_model = DDP(model, process_group=dp_group)
    #ddp_model = DDP(model)
    #ddp_model = FSDP(model)
    #ddp_model = model

    loss_fn = nn.MSELoss()
    #loss_fn = nn.CrossEntropyLoss()


    if rank == 0:
        tick = time.time()
    #print("kkd")
    for i in range(100):
        
        #if rank == 0:
        #    tick = time.time()
        #optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        optimizer = optim.Adam(ddp_model.parameters(), lr=3e-5)
        #lr = 5.0
        #optimizer = optim.SGD(ddp_model.parameters(), lr=lr)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        optimizer.zero_grad()
        #outputs = ddp_model(data, labels=labels)
        #outputs = ddp_model(data.to(device_id))
        outputs = ddp_model(sample_input.to(device_id))
        #outputs = ddp_model(sample_input)
        labels = sample_output.to(device_id)
        #labels = sample_output
        #labels = labels.to(device_id)
        loss = loss_fn(outputs, labels)
        #loss = outputs[0]
        loss.backward()
        optimizer.step()

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
