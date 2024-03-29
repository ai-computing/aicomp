#
# Copyright (c) 2023-present, ETRI, All rights reserved.
#
#
#  This is a PoC that broadcasts the FX IR generated by FX compile 
#        to another machine, and then inferences in a pipeline style using N hosts.
#
#   In this PoC, whole FX IR is transferred to all hosts, and 
#       distributed inference is performed based on range metadata that describes the area each host is responsible for.
#
#
#
#  Sample Usage:
#      <machine #0>
#            torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0
#                  --master_addr="X.X.X.X" --master_port=29500 fx_dist_inference_type-A.py
#      <machine #1>
#            torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1
#                  --master_addr="X.X.X.X" --master_port=29500 fx_dist_inference_type-A.py
#


import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
import torch.nn as nn
from torch.optim import Adam
from torch import fx
from torch.fx.node import Node
import time

import torch.distributed as dist
import datetime
import torch.distributed.rpc as rpc

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

torch.manual_seed(42)

#
# Total host count
#
#num_host=N
num_host=4  

batch_size = 64
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
        #self.relu = nn.ReLU(inplace = True)
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


class Simple_split_test(object):
    def __init__(self):
        self.initialize_comm()
        self.model_ir = []
        self.range_metadata = []

    def initialize_comm(self):

        if dist.is_initialized():
            print(f"Communication already initialized")
            return


        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.master_addr = os.getenv("MASTER_ADDR")
        self.master_port = os.getenv("MASTER_PORT")

        #
        print(f" --- rank:{self.rank}, world_size:{self.world_size}, master:{self.master_addr}, port:{self.master_port}")

        self.backend = "gloo"
        init_method = "tcp://" + str(self.master_addr) + ":" + str(self.master_port)

        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size, init_method=init_method)

        #
        print(f" --- rank:{dist.get_rank()}, world_size:{dist.get_world_size()}")

        options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=10, rpc_timeout=30)

        rpc.init_rpc(f"worker{self.rank}", rank=self.rank, world_size=self.world_size, rpc_backend_options=options,)

        # rpc.shutdown()


    def simple_split(self, g: fx.Graph):

        length = g.nodes.__len__()
        segment = length // num_host
        print(f"segment ==> {segment}")

        k, cnt = 0, 0
        for n in g.nodes:
            if n.op == 'call_module':
                cnt = cnt + 1

            if cnt == segment:
                self.range_metadata.append((k, n.name))
                k = k + 1
                cnt = 0

        print(self.range_metadata)

        
    def metadata_transfer(self):

        if self.rank == 0:

            t1 = TestModel()
            gm = fx.symbolic_trace(t1)

            self.device = torch.device("cpu")

            self.model_ir.append(gm)

            # blocking call
            dist.broadcast_object_list(self.model_ir, src=0, device=self.device)

            print(f" >> worker:{self.rank} ==> FX IR transfer to all other workers")

            self.simple_split(gm.graph)

            dist.broadcast_object_list(self.range_metadata, src=0, device=self.device)

            print(f" >> worker:{self.rank} ==> range metadata {self.range_metadata} transfer to all other workers")

        else:
            self.device = torch.device("cpu")

            self.model_ir.append(None)

            # blocking call
            dist.broadcast_object_list(self.model_ir, src=0, device=self.device)

            gm = self.model_ir[0]

            if gm is None:
                print(f"FX IR sync failed")
            else:
                print(f" worker: {self.rank} <==  FX IR transferred")
                for node in gm.graph.nodes:
                    print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.all_input_nodes:{node.all_input_nodes}")
                print(f" ---------------------------------")

            for i in range(num_host):
                self.range_metadata.append(None)

            dist.broadcast_object_list(self.range_metadata, src=0, device=self.device)

            print(f" worker: {self.rank} <==  range metadata:{self.range_metadata} transferred")
            print(f" ---------------------------------")


class FXRun2:

    def __init__(self, split_info: Simple_split_test, device):

        gm = split_info.model_ir[0]
        self.mod = gm
        self.graph = gm.graph
        self.modules = dict(self.mod.named_modules())
        self.env: Dict[str, Node] = {}
        self.range_metadata = split_info.range_metadata
        self.rank = split_info.rank
        self.world_size = split_info.world_size
        self.device = device

    def receive_activation(self, split_node_name, from_rank):

        dimension = torch.tensor([0], dtype=torch.long)
        dist.recv(dimension, from_rank)

        shape = torch.tensor([0] * dimension, dtype=torch.long)
        dist.recv(shape, from_rank)
        shape = tuple(shape.tolist())

        obj = torch.zeros(size=shape)
        dist.recv(obj, from_rank)

        return obj

    def send_activation(self, split_node_name, to_rank):

        obj = self.env[split_node_name]

        dimension = torch.tensor(len(obj.size()), dtype=torch.long) # ex. 2
        dist.send(dimension, to_rank)

        shape = torch.tensor(list(obj.size()), dtype=torch.long) # ex. [54, 5120]
        dist.send(shape, to_rank)

        dist.send(obj, to_rank)

    def get_range(self, rank, g: fx.Graph) -> (Node, Node):
        #
        print(f"range_metadata: {self.range_metadata}")

        if rank == 0:
            from_node_name = "-1"
            for n in g.nodes:
                if n.op == 'placeholder':
                    from_node_name = n.name
                else:
                    break
            # not found --> "-1"
        else:
            from_node_name = self.range_metadata[rank-1][1]

        to_node_name = self.range_metadata[rank][1]

        for n in g.nodes:
            if from_node_name == "-1":
                from_node = n
                break

            if n.name == from_node_name:
                from_node = n
                break

        for n in reversed(g.nodes):
            if n.name == to_node_name :
                to_node = n
                break

        return (from_node._next, to_node)


    def print_range(self):
        from_, to_ = self.get_range(self.rank, self.graph)

        print(f"## rank = {self.rank}, from_:{from_.name}, to_:{to_.name}")

        cur = from_ # first node assigned to the host#{rank} in metadata_range
        while cur != to_:
             print(f" ---- node:{cur.name}")
             cur = cur._next
        print(f" ---- node:{cur.name}")  # last node assigned to the host#{rank} in metadata_range
        print(f" -------------------------------")


    def fx_forward2(self, *args):
        self.args_iter = iter(args)

        from_, to_ = self.get_range(self.rank, self.graph)
        print(f"## rank = {self.rank}, world_size={self.world_size}, from_:{from_.name}, to_:{to_.name}")


        if self.rank  > 0:
            split_node_name = from_._prev.name
            pre_split_rank = self.rank - 1
            #print(f"## rank:{self.rank}, receive activation from {pre_split_rank}, split_node_name:{split_node_name}")
            self.env[split_node_name] = self.receive_activation(split_node_name, pre_split_rank)

        if self.rank == 0:
            for n in self.graph.nodes:
                cur = n
                break
        else:
            cur = from_
        while cur != to_:
            self.fx_ir_run_node(cur)
            cur = cur._next

        result = self.fx_ir_run_node(cur)

        #print(f" rank:{self.rank}, cur.node name{cur.name}, split_node_name:{to_.name}")

        if self.rank < self.world_size - 1:
            split_node_name = to_.name
            next_split_rank = self.rank + 1
            #print(f"### rank:{self.rank} send activation to {next_split_rank}, split_node_name:{split_node_name}")
            self.send_activation(split_node_name, next_split_rank)

        return result

    def restore_env(self, node: Node) -> Tuple[Tuple, Dict]:
        #print(f"## before restore_env, node:{node}, node.args:{node.args}, node.kwargs:{node.kwargs}")

        args = fx.graph.map_arg(node.args, lambda n: self.env[n.name])
        assert isinstance(args, tuple)

        kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name])
        assert isinstance(kwargs, dict)

        #print(f">>> after restore_env, node:{node}, node.name:{node.name}, args:{args}, kwargs:{kwargs}")

        return args, kwargs
        

    def fx_ir_run_node(self, node):

        args, kwargs = self.restore_env(node)

        if node.op == 'placeholder':
            result = next(self.args_iter)

        elif node.op == 'get_attr':
            target_atoms = node.target.split('.')
            attr_itr = self.mod
            for i , atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(\
                            f"Node referenced nonexistant target{'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            result = attr_itr

        elif node.op == 'call_function':
            #result = node.target(\
            #        *fx.graph.map_arg(node.args, lambda n: self.env[n.name]), \
            #        **fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name]))
            result = node.target(*args, **kwargs)

        elif node.op == 'call_method':
            #self_obj, *args = fx.graph.map_arg(node.args, lambda n: self.env[n.name])
            #kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name])
            #result = getattr(self_obj, node.target)(*args, **kwargs)

            self_obj = args[0]
            args = args[1:]
            result = getattr(self_obj, node.target)(*args, **kwargs)

        elif node.op == 'call_module':
            #result = self.modules[node.target](\
            #        *fx.graph.map_arg(node.args, lambda n: self.env[n.name]),\
            #        **fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name]))
            result = self.modules[node.target](*args, **kwargs)

        elif node.op == 'output':
            #result = fx.graph.map_arg(node.args[0], lambda n: self.env[n.name])
            result =  args[0]

        #
        print(f" ## [rank:{sim_split.rank}], run - node:{node.name}, node.op:{node.op}")

        self.env[node.name] = result

        return result



sim_split = Simple_split_test()
sim_split.metadata_transfer()


if sim_split.rank == 0:
    sample_input = torch.rand(batch_size, in_features)
else:
    sample_input = None

fx_run2 = FXRun2(sim_split, sim_split.device)

fx_run2.print_range()

output1 = fx_run2.fx_forward2(sample_input)

if sim_split.rank == sim_split.world_size - 1:
    print(output1)

print(f"[rank:{sim_split.rank}], run completed ...")

rpc.shutdown()

