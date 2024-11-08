#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#
#
#  This is a test program for running pipeline-parallel training on pytorch pipelining (torch/distributed/pipelining).
#
#
#  Sample Usage:
#      <machine #0>
#            torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0
#                  --master_addr="X.X.X.X" --master_port=29501 pippy-2_pp_training.py
#      <machine #1>
#            torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1
#                  --master_addr="X.X.X.X" --master_port=29501 pippy-2_pp_training.py
#


import torch
import torch.distributed as dist
import torch.nn as nn
import time

import os
import sys
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

#from pippy import ScheduleGPipe, PipelineStage
#from pippy import split_into_equal_size
#from pippy import pipeline

from torch.distributed.pipelining import ScheduleGPipe, PipelineStage
from torch.distributed.pipelining import pipeline
from torch.distributed.pipelining import SplitPoint
from typing import Callable, Dict, List, Tuple
import torch.fx as fx
aten_pipe_split_alias = torch.ops.pippy._pipe_split.default


torch.manual_seed(42)

use_gpu = True

world_size=int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

device = None

if use_gpu == True:
    device = torch.device(f"cuda:{local_rank}")
    print(f"Using GPU ... cuda:{local_rank}")

else:
    device = torch.device("cpu")
    print(f"Using CPU ...")


batch_size = 64

micro_batch_size = world_size // 2 # TODO
CHUNKS = micro_batch_size

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

model = TestModel()

### For pytorch pipelinging
#
#  cited from PiPPy to be used on pytorch pipelining
#     -  split_into_equal_size, _split_on_size_threshold_with_max_stages, _analyze_node_size
### 
def split_into_equal_size(
    nstages: int = 1,
) -> Callable[[fx.GraphModule], fx.GraphModule]:
    def _split_into_nstages_equal_size(
        gm: fx.GraphModule,
    ) -> fx.GraphModule:
        param_size = 0
        for param in gm.parameters():
            param_size += param.numel()
        buffer_size = 0
        for buffer in gm.buffers():
            buffer_size += buffer.numel()

        total_size = param_size + buffer_size
        per_stage_size = total_size // nstages
        print(
            f"Total model size: {total_size}, "
            f"per stage size: {per_stage_size}"
        )

        gm, rv_nstages = _split_on_size_threshold_with_max_stages(
            gm, per_stage_size, nstages
        )
        assert rv_nstages == nstages
        return gm

    return _split_into_nstages_equal_size

def _split_on_size_threshold_with_max_stages(
    gm: fx.GraphModule,
    threshold: int,
    max_stages: int = -1,
) -> Tuple[fx.GraphModule, int]:
    # Analyze size of parameters/buffers used by each node in the graph
    node_param_sizes = _analyze_node_size(gm)

    # Record split positions
    insert_before_nodes: List[fx.Node] = []

    def new_stage_before(node):
        insert_before_nodes.append(node)

    # Track the parameters we have seen in the current bucket and their total size
    accumulate_size = 0
    accumulate_params: Dict = {}

    for node in gm.graph.nodes:
        if node not in node_param_sizes:
            # The callsite of this node does not involve parameters or buffers
            continue

        # Track the new parameters we see at this node as well as parameters that we have seen in current bucket
        new_size = 0
        new_params: Dict = {}
        repeated_size = 0
        repeated_params: Dict = {}
        param_sizes = node_param_sizes[node]
        if node.op == "call_function":
            # For function, the parameter it uses can be shared with other functions seen previously
            for param_name, size in param_sizes.items():
                if param_name not in accumulate_params:  # new parameter
                    new_params.setdefault(param_name)
                    new_size += size
                else:  # repeated parameter; mark down; use later
                    repeated_params.setdefault(param_name)
                    repeated_size += size
        elif node.op == "call_module":
            # For module, we count its paramters as a single whole
            for param_name, size in param_sizes.items():
                new_size += size

        if (
            accumulate_size + new_size <= threshold
        ):  # can accommodate this node in current bucket
            accumulate_size += new_size
            accumulate_params.update(new_params)
        elif (
            accumulate_size == 0 and new_size > threshold
        ):  # this node becomes a stage
            new_stage_before(node.next)
        else:  # cannot accommodate this node
            new_stage_before(node)
            accumulate_size = repeated_size + new_size
            accumulate_params.clear()
            accumulate_params.update(repeated_params)
            accumulate_params.update(new_params)

    # Insert pipe_split nodes at the recorded positions
    nstages = 1
    for node in insert_before_nodes:
        if nstages == max_stages:
            break
        with gm.graph.inserting_before(node):
            gm.graph.call_function(aten_pipe_split_alias, (), {})
        nstages += 1

    # Since we transformed the graph, we need to recompile the module
    gm.recompile()

    return gm, nstages

def _analyze_node_size(
    gm: fx.GraphModule,
) -> Dict[fx.Node, Dict[str, int]]:
    # state_dict helps us to get parameter sizes
    state_dict = gm.state_dict()

    # Function Parameter Usage
    node_param_sizes: Dict[fx.Node, Dict[str, int]] = {}
    for node in gm.graph.nodes:
        if node.op == "get_attr":  # a parameter node
            param_name = node.target
            if param_name not in state_dict:
                # In some cases, attr node is not a parameter or buffer, we just skip it
                continue
            param = state_dict[param_name]
            # Find use site of this parameter
            for user in node.users:
                func_param_sizes = node_param_sizes.setdefault(user, {})
                func_param_sizes.setdefault(param_name, param.numel())

    # Module Parameter Usage
    for node in gm.graph.nodes:
        # We calcuate size of a user-defined submodule as a whole
        if node.op == "call_module":
            mod_param_sizes: Dict[str, int] = {}
            submod: torch.nn.Module = gm.get_submodule(node.target)
            for param_name, param in submod.named_parameters():
                mod_param_sizes.setdefault(param_name, param.numel())
            if mod_param_sizes:
                node_param_sizes.setdefault(node, mod_param_sizes)

    for node, param_sizes in node_param_sizes.items():
        print(f"{node} has params: {param_sizes}")

    return node_param_sizes

###

#####
tensor_type2id = {
    torch.float32: 0,
    torch.float64: 1,
    torch.complex64: 2,
    torch.complex128: 3,
    torch.float16: 4,
    torch.bfloat16: 5,
    torch.uint8: 6,
    torch.int8: 7,
    torch.int16: 8,
    torch.int32: 9,
    torch.int64: 10,
    torch.bool: 11,
    }

tensor_id2type = {v:k for k,v in tensor_type2id.items()}

def send_tensor(obj, to_rank, device):
    if isinstance(obj, torch.Tensor):
        obj_size = obj.size()
        dimension = torch.tensor(len(obj_size), dtype=torch.long, device=device)
    dist.send(dimension, to_rank)

    if isinstance(obj, torch.Tensor):
        shape = torch.tensor(list(obj_size), dtype=torch.long, device=device)
    dist.send(shape, to_rank)

    ttype = tensor_type2id[obj.dtype]
    ttype = torch.tensor(ttype, dtype=torch.long, device=device)
    dist.send(ttype, to_rank)

    if not obj.is_contiguous():
        obj = obj.contiguous()

    obj = obj.to(device)
    dist.send(obj, to_rank)

def recv_tensor(from_rank, device):
    dimension = torch.tensor([0], dtype=torch.long, device=device)
    dist.recv(dimension, from_rank)

    shape = torch.tensor([0] * dimension.item(), dtype=torch.long, device=device)
    dist.recv(shape, from_rank)
    shape = tuple(shape.tolist())

    ttype = torch.tensor([0], dtype=torch.long, device=device)
    dist.recv(ttype, from_rank)

    ttype = tensor_id2type[ttype.item()]

    obj = torch.zeros(size=shape, dtype=ttype, device=device)
    dist.recv(obj, from_rank)

    return obj

def prepare_labels(__labels):
    if rank == 0: # first stage
        mbatches = torch.chunk(__labels, CHUNKS)

        for j in range(CHUNKS):
            send_tensor(mbatches[j], world_size - 1, device)

def ready_labels():
    m_batches = []
    if rank == world_size - 1: # last stage
        for j in range(CHUNKS):
            m_batches.append(recv_tensor(0, device))

        ___labels = torch.cat(m_batches)
        return ___labels
    return None

def move_labels2last_stage(_labels):
    prepare_labels(_labels)
    return ready_labels()
#####



def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


if rank == 0:
    print('Total parameters in model: {:,}'.format(get_total_params(model)))

model = model.to(device)
split_policy = split_into_equal_size(world_size)

sample_input = torch.rand(batch_size // CHUNKS, in_features, device=device)

# PiPPy
#pipe = pipeline(model, CHUNKS, example_args=(sample_input,), split_policy=split_policy)

# pytorch pipelining
pipe = pipeline(model, mb_args=(sample_input,), split_policy=split_policy)

if rank == 0:
    print("### pipe ### ".center(80, "*"))
    print(pipe)


dist.init_process_group(rank=rank, world_size=world_size)

# PiPPy
#stage = PipelineStage(pipe, rank, device)

# pytorch pipelining
#smod = pipe.get_stage_module(rank)
stage = pipe.build_stage(rank, device)

loss_fn = torch.nn.MSELoss()

schedule = ScheduleGPipe(stage, CHUNKS, loss_fn=loss_fn)

#optimizer = optim.SGD(stage.submod.parameters(), lr=1e-3, momentum=0.9)
optimizer = torch.optim.Adam(stage.submod.parameters(), lr=3e-5)


if rank == 0:
    tick = time.time()

for i in range(100):

    data, labels = None, None

    if rank == 0:
        data =  torch.randn(batch_size, in_features, device=device)
        labels =  torch.randn(batch_size, out_features, device=device)

    labels = move_labels2last_stage(labels)

    optimizer.zero_grad()

    if rank == 0:
        schedule.step(data)

    elif rank == world_size - 1:
        losses = []
        #out = schedule.step(target=target, losses=losses)
        out = schedule.step(target=labels, losses=losses)
        print(f"Step {i}, Loss:{sum(losses) / CHUNKS}")
    else:
        schedule.step()

    optimizer.step()

if rank == 0:
    tock = time.time()
    elapsed_time = tock - tick

    print('Time elapsed: %.3f sec ' % (elapsed_time))


dist.barrier()
#dist.destroy_process_group()
print(f"[rank:{rank}, run completed ...")

