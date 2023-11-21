#
# Copyright (c) 2023-present, ETRI, All rights reserved.
#
#
#  This is a test program for running PiPPy with pipeline-parallel and data-parallel training.
#
#
#  Sample Usage for PP and DP:
#      <machine #0>
#            torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0
#                  --master_addr="X.X.X.X" --master_port=29501 pippy_pp_dp_training.py
#      <machine #1>
#            torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1
#                  --master_addr="X.X.X.X" --master_port=29501 pippy_pp_dp_training.py
#



import torch
import torch.distributed.rpc as rpc

import torch.nn as nn
import time

import torch.distributed as dist
import datetime
import torch.distributed.rpc as rpc


import psutil
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


from pippy.IR import Pipe, pipe_split, annotate_split_points, PipeSplitWrapper
from pippy import split_into_equal_size
from pippy.PipelineDriver import PipelineDriverFillDrain
from pippy.microbatch import TensorChunkSpec
from pippy.IR import LossWrapper



torch.manual_seed(42)

use_gpu = True
#use_gpu = False

world_size=int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

device = None

# PP+DP
backend="nccl"
#backend="gloo"
master_addr=os.getenv("MASTER_ADDR")
master_port = os.getenv("MASTER_PORT")
init_method = "tcp://" + str(master_addr) + ":" + str(master_port)
dist.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method=init_method)
#
dp_group_size = 2
pp_group_size = world_size // dp_group_size
print(f"[rank:{rank}], world_size:{world_size}, pp_group_size:{pp_group_size}, dp_group_size:{dp_group_size}\n")
#
pp_ranks = [[] for _ in range(dp_group_size)]
def get_pp_ranks():
    for i in range(0, world_size):
        pp_grp_idx = i % dp_group_size
        pp_ranks[pp_grp_idx].append(i)
#
get_pp_ranks()
#
if rank == 0:
    print(f"pp_ranks:{pp_ranks}\n")
#
def get_stage(rank):
    return rank // dp_group_size



if use_gpu == True:
    #device = torch.device(f"cuda:{rank}")
    #print(f"Using GPU ... cuda:{rank}")
    device = torch.device(f"cuda:{local_rank}")
    print(f"Using GPU ... cuda:{local_rank}")

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=30)
    n_devs = torch.cuda.device_count()
    dev_id = rank % n_devs
    #dev_id = local_rank % n_devs
    for i in range(world_size):
        options.set_device_map(f"worker{i}", {dev_id: i % n_devs})
        #options.set_device_map(f"worker{i}", {dev_id: rank})

    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)
else:
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

batch_size = 64

#micro_batch_size = world_size // 2 # TODO
micro_batch_size = world_size # TODO
CHUNKS = micro_batch_size

#if int(os.environ["RANK"]) == 0:
if rank == 0:
    print(f"total process count: {world_size}")
    print(f"batch size: {batch_size}")
    print(f"micro batch size: {micro_batch_size}")

in_features = 5120
out_features = 5120
hidden = 5120

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.ModuleList()
        for i in range(2):
        #for i in range(10):
            self.linear2.append(nn.Linear(hidden, hidden))

        self.linear3 = nn.ModuleList()
        for i in range(2):
        #for i in range(10):
            self.linear3.append(nn.Linear(hidden, hidden))

        self.linear4 = nn.ModuleList()
        for i in range(2):
        #for i in range(10):
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


class SimpleLossWrapper(LossWrapper):
    def forward(self, x, labels):
        out1 = self.module(x)
        return self.loss_fn(out1, labels)


# PP+DP
#if rank == 0:
stage = get_stage(rank)
if stage == 0:
    model = TestModel()
    loss_fn = torch.nn.MSELoss()
    wrapper = SimpleLossWrapper(model, loss_fn)

    # PP+DP
    #split_policy = split_into_equal_size(world_size)
    split_policy = split_into_equal_size(pp_group_size)

    #if use_gpu == True:
    #    wrapper.to(device)
    #    print(f">> Moving to: {device}")

    wrapper.train()

    pipe = Pipe.from_tracing(wrapper, output_loss_value_spec=True, split_policy=split_policy)
    print(pipe)

    if use_gpu == True:
        pipe.to(device)
        print(f">> Moving to: {device}")

    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec = {}
    #output_chunk_spec = TensorChunkSpec(0)
    from pippy.microbatch import LossReducer
    output_chunk_spec = LossReducer(0.0, lambda a, b: a+b)

    
    #driver = PipelineDriverFillDrain(
    #        pipe, CHUNKS, args_chunk_spec=args_chunk_spec, kwargs_chunk_spec=kwargs_chunk_spec,
    #        output_chunk_spec=output_chunk_spec, world_size=world_size)
    #
    # PP+DP
    driver = PipelineDriverFillDrain(
            pipe, CHUNKS, args_chunk_spec=args_chunk_spec, kwargs_chunk_spec=kwargs_chunk_spec,
            output_chunk_spec=output_chunk_spec, world_size=pp_group_size, all_ranks=pp_ranks[rank])

    # PP+DP
    driver.init_data_parallel(dp_group_size)

    optimizer1 = driver.instantiate_optimizer(torch.optim.Adam)
    lr_scheduler = driver.instantiate_lr_scheduler(torch.optim.lr_scheduler.LinearLR, total_iters=100)


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


# PP+DP
#if rank == 0:
if stage == 0:
    print('Total parameters in model: {:,}'.format(get_total_params(wrapper)))

    tick = time.time()

    if use_gpu == True:
        sample_output = torch.rand(batch_size, out_features, device=device)
        sample_input = torch.rand(batch_size, in_features, device=device)

        print(f">> Moving input/output to: {device}")
    else:
        sample_output = torch.rand(batch_size, out_features)
        sample_input = torch.rand(batch_size, in_features)

    for i in range(100):
    #for i in range(50):
    #for i in range(10):

        optimizer1.zero_grad()

        pipe_loss = driver(sample_input, sample_output)

        print(f'Step {i}, Loss: {pipe_loss}')

        optimizer1.step()
        lr_scheduler.step()


    tock=time.time()
    elapsed_time = tock - tick

    print('Time elapsed: %.3f sec ' % (elapsed_time))



rpc.shutdown()
print(f"[rank:{rank}, run completed ...")

