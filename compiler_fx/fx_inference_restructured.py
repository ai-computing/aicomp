#
# Copyright (c) 2023-present, ETRI, All rights reserved.
#
#
#  This is a PoC that inferences a model using IR generated by FX compile.
#  The IR based inference execution logic is restructured inspired by the FX code.
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

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

torch.manual_seed(42)

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


class FXRun2:

    def __init__(self, mod):

        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.env: Dict[str, Node] = {}

    def fx_forward2(self, *args):
        self.args_iter = iter(args)

        for n in self.graph.nodes: 
            result = self.fx_ir_run_node(n)

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
            result = node.target(\
                    *fx.graph.map_arg(node.args, lambda n: self.env[n.name]), \
                    **fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name]))

        elif node.op == 'call_method':
            self_obj, *args = fx.graph.map_arg(node.args, lambda n: self.env[n.name])
            kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name])
            result = getattr(self_obj, node.target)(*args, **kwargs)

        elif node.op == 'call_module':
            result = self.modules[node.target](\
                    *fx.graph.map_arg(node.args, lambda n: self.env[n.name]),\
                    **fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name]))

        elif node.op == 'output':
            result = fx.graph.map_arg(node.args[0], lambda n: self.env[n.name])

        #
        print(f" ## run - node:{node.name}, node.op:{node.op}")

        self.env[node.name] = result

        return result

t1 = TestModel()
print(t1)
gm = fx.symbolic_trace(t1)

for node in gm.graph.nodes:
    print(f"node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.args:{node.args}, node.all_input_nodes:{node.all_input_nodes}")

print("-----------------------------------")


sample_input = torch.rand(batch_size, in_features)

tick = time.time()

fx_run2 = FXRun2(gm)

output1 = fx_run2.fx_forward2(sample_input) # actual

output2 = t1(sample_input) # expected

torch.testing.assert_close(output1, output2)

tock = time.time()
elapsed_time = tock - tick
print('Time elapsed: %.3f sec ' % (elapsed_time))
print(output1)
print("############")
print(output2)


