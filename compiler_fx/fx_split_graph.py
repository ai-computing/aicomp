#
# Copyright (c) 2023-present, ETRI, All rights reserved.
#
#
# This PoC actually splits FX IR, then handles the IR partitions at GraphModule level
#
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


#from fx.symbolic_trace import symbolic_trace
from torch.fx.graph_module import GraphModule
from torch.fx.passes.split_module import split_module


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


t1 = TestModel()

gm = fx.symbolic_trace(t1)

print(gm)

#
for n in gm.graph.nodes:
    print(f"n.op:{n.op}, n.name:{n.name}, n.args:{n.args}")

metadata_range = []

num_host = 4   # 4 hosts
#num_host = 6   # 6 hosts
#num_host = 8   # 8 hosts

# CASE:  num_host == 4
#metadata_range = [(0, 'linear2_1'), (1, 'relu_4'), (2, 'linear5_0'), (3, 'relu_9')]

last_flag = False

def part_fn(node):

    global last_flag
    last_idx, last_name = metadata_range[-1]

    if last_flag == True:
        idx = last_idx
        print(f" part_fn:  node.name:{node.name}, --> {idx}")
        return idx

    idx = 0

    cur = node
    while cur.name != last_name:
        for i, m_name in metadata_range:
            if cur.name == m_name:
                idx = i
                #print(f" part_fn:  node.name:{node.name}, m_name:{m_name}, --> {idx}")
                print(f" part_fn:  node.name:{node.name}, cur.name:{cur.name} m_name:{m_name}, --> {idx}")
                return idx

        cur = cur._next

    if cur.name == last_name:
        idx = last_idx
        last_flag = True
    print(f" part_fn:  node.name:{node.name}, --> {idx}")
    return idx


def simple_split(gm, t1, metadata_range):
    length = gm.graph.nodes.__len__()
    segment = length // num_host
    print(f"length: {length}, segment ==> {segment}")
    
    k, cnt = 0, 0
    for n in gm.graph.nodes:
        if n.op == 'call_module':
            cnt = cnt + 1

        if cnt == segment:
            metadata_range.append((k, n.name))
            k = k + 1
            cnt = 0

        if k > num_host - 1:
            break

    if len(metadata_range) <  num_host:
        metadata_range.append((k, n.name))

    print(metadata_range)

    submodules = split_module(gm, t1, part_fn, keep_original_order=True)
    #print(submodules)

    return submodules

def print_range(submod):

    for m in submod.graph.nodes:
        print(f"m:{m}, m.name:{m.name}, m.target:{m.target}, m.all_input_nodes:{m.all_input_nodes}")
    print(f" ------------------------------------")

def fake_transfer(submods):

    skip = False
    for submod in submods.modules():
        if skip == False and isinstance(submod, fx.GraphModule):
            skip = True
            continue
        if skip == True and isinstance(submod, fx.GraphModule):
            print(f"submod:{submod._get_name()}")
            #
            print_range(submod)



submods = simple_split(gm, t1, metadata_range)

print(f"--- After split_module ---")

fake_transfer(submods)

        

