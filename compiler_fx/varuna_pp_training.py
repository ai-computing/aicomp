#
# Copyright (c) 2023-present, ETRI, All rights reserved.
#
#
#  This is a test program for running Varuna with pipeline-parallel training.
#
#
#  Sample Usage for Pipeline-Parallel execution:
#
#      # python3 -m varuna.run_varuna --machine_list ip_list --gpus_per_node 8 --batch_size 64 --nstages 8 --chunk_size 2 varuna_pp_training.py
#
#



import torch
import torch.distributed.rpc as rpc

import torch.nn as nn
import time

import torch.distributed as dist
import datetime

from torch.optim import Adam


import psutil
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from varuna import CutPoint, Varuna

import argparse

parser = argparse.ArgumentParser(description='TestModel Training')
parser.add_argument("--batch-size", type=int, default=64, help = "per-process batch size given by varuna")
parser.add_argument("--gpus_per_node", type=int, default=-1)
parser.add_argument("--world_size", type=int, default=-1)
parser.add_argument("--rank", type=int, default=-1)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--stage_to_rank_map", type=str, default=None, help = "stage to rank map of Varuna model")
parser.add_argument("--chunk_size", type=int,default=None, help = "number of microbatches for pipeline")
parser.add_argument("--profiling", action='store_true', help="whether to run profiling for Varuna")

args = parser.parse_args()


torch.manual_seed(42)

num_cutpoints = 7 # for 8 processes

in_features = 5120
out_features = 5120
hidden = 5120

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cutpoints = nn.ModuleList([CutPoint() for _ in range(num_cutpoints)])

        self.linear1 = nn.Linear(in_features, hidden)
        self.relu1 = nn.ReLU(inplace = False)

        self.linear2 = nn.ModuleList()
        for i in range(2):
            self.linear2.append(nn.Linear(hidden, hidden))
            self.linear2.append(nn.ReLU(inplace = False))

        self.linear3 = nn.ModuleList()
        for i in range(2):
            self.linear3.append(nn.Linear(hidden, hidden))
            self.linear3.append(nn.ReLU(inplace = False))

        self.linear4 = nn.ModuleList()
        for i in range(2):
            self.linear4.append(nn.Linear(hidden, hidden))
            self.linear4.append(nn.ReLU(inplace = False))

        self.linear5 = nn.ModuleList()
        for i in range(2):
            self.linear5.append(nn.Linear(hidden, hidden))
            self.linear5.append(nn.ReLU(inplace = False))

        self.linear6 = nn.Linear(hidden, out_features)
        self.relu6= nn.ReLU(inplace = False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)

        for idx, m in enumerate(self.linear2, start=0):
            x = m(x)

            i = idx // 2
            j = idx % 2
            if j == 1 and i < num_cutpoints:
                x = self.cutpoints[i](x)

        for idx, m in enumerate(self.linear3, start=4):
            x = m(x)

            i = idx // 2
            j = idx % 2
            if j == 1 and i < num_cutpoints:
                x = self.cutpoints[i](x)

        for idx, m in enumerate(self.linear4, start=8):
            x = m(x)

            i = idx // 2
            j = idx % 2
            if j == 1 and i < num_cutpoints:
                x = self.cutpoints[i](x)

        for idx, m in enumerate(self.linear5, start=12):
            x = m(x)

            i = idx // 2
            j = idx % 2
            if j == 1 and i < num_cutpoints:
                x = self.cutpoints[i](x)

        x = self.linear6(x)
        x = self.relu6(x)

        return x


class LossWrapper(torch.nn.Module):
    def __init__(self, module, loss_fn):
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        raise NotImplementedError("LossWrapper: no forward implementation")


class SimpleLossWrapper(LossWrapper):
    #def forward(self, x, labels):
    def forward(self, *args, **kwargs):

        x = kwargs['inputs']
        labels = kwargs['labels']

        out1 = self.module(x)
        return self.loss_fn(out1, labels)


model = TestModel()
loss_fn = torch.nn.MSELoss()
wrapper = SimpleLossWrapper(model, loss_fn)


print(f"(B) gpus_per_node: {args.gpus_per_node}")
if args.gpus_per_node == -1:
    args.gpus_per_node = torch.cuda.device_count()
    print(f"(A) gpus_per_node: {args.gpus_per_node}")

print(f"batch size: {args.batch_size} ")
print(f"(B) rank: {args.rank} ")
if args.rank == -1:
    args.rank = int(os.environ["RANK"])
    print(f"(A) rank: {args.rank}")

print(f"(B) micro batch size: {args.chunk_size}")
if args.chunk_size == None:
    args.chunk_size = 2
    print(f"(A) micro batch size: {args.chunk_size}")


print(f"(B) local_rank: {args.local_rank}")
if args.local_rank == -1:
    args.local_rank = int(os.environ["LOCAL_RANK"])
    print(f"(A) local_rank: {args.local_rank}")

print(f"(B) world_size: {args.world_size}")
if args.world_size == -1:
    args.world_size = int(os.environ["WORLD_SIZE"])
    print(f"(A) world_size: {args.world_size}")

print(f"stage_to_rank_map: {args.stage_to_rank_map}")


this_device = torch.device(f"cuda:{args.local_rank}")


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


print('Total parameters in model: {:,}'.format(get_total_params(wrapper)))

master_addr = os.getenv("MASTER_ADDR")
master_port = os.getenv("MASTER_PORT")


#backend = "nccl"
backend = "gloo" # for varuna

init_method = "tcp://" + str(master_addr) + ":" + str(master_port)
print(f"backend ==> {backend}")

dist.init_process_group(backend=backend, init_method=init_method, world_size=args.world_size, rank=args.rank)


sample_output = torch.rand(args.batch_size, out_features)
sample_input = torch.rand(args.batch_size, in_features)

def get_batch_fn(size, device=None):
    input, label =  sample_input[:size,:], sample_output[:size,:]
    inputs = {"inputs": input, "labels":label}
    if device is not None:
        inputs["inputs"] = inputs["inputs"].to(device)
        inputs["labels"] = inputs["labels"].to(device)
    return inputs


def get_batch_fn2(size, device=None):
    input, label =  sample_input[:size,:], sample_output[:size,:]
    inputs = {"inputs": input, "labels":label}

    if args.rank == 0:
        inputs = {"inputs": input, "labels":None}
        if device is not None:
            inputs["inputs"] = inputs["inputs"].to(device)

    elif args.rank == args.world_size -1:
        inputs = {"inputs": None, "labels":label}
        if device is not None:
            inputs["labels"] = inputs["labels"].to(device)

    return inputs

wrapper = Varuna(wrapper, args.stage_to_rank_map, get_batch_fn, args.batch_size, args.chunk_size, fp16=False, local_rank=args.local_rank, device=args.local_rank, from_cache=False)


for param in wrapper.parameters():
    print(f"rank:{args.rank}, {type(param)}, {param.size()}")


wrapper.train()

optimizer1 = Adam(wrapper.parameters(), lr=3e-5)
wrapper.set_optimizer(optimizer1)

if args.rank == 0:
    tick = time.time()

for i in range(100):
#for i in range(10):

    batch = get_batch_fn2(args.batch_size, this_device)
    loss, overflow, grad_norm = wrapper.step(batch)

    if not overflow:
        optimizer1.step()

    if args.rank == 0:
        print(f'Step {i}, Loss: {loss}')

    optimizer1.zero_grad()


if args.rank == 0:
    tock=time.time()
    elapsed_time = tock - tick

    print('Time elapsed: %.3f sec ' % (elapsed_time))
