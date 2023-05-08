#
# Copyright (c) 2023-present, ETRI, All rights reserved.
#
#
#  Assuming that there are N hosts, this PoC traverse only the specific range 
#      simply assigned to each host within IR



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


t1 = TestModel()

gm = fx.symbolic_trace(t1)

print(gm)

#
for n in gm.graph.nodes:
    print(f"n.op:{n.op}, n.name:{n.name}, n.args:{n.args}")

metadata_range = []

N = 4   # 4 hosts

#metadata_range = [(0, 'linear2_1'), (1, 'relu_4'), (2, 'linear5_0'), (3, 'relu_9')]

def simple_split(metadata_range):

    length = gm.graph.nodes.__len__()
    segment = length // N
    print(f"segment ==> {segment}")
    
    k, cnt = 0, 0
    for n in gm.graph.nodes:
        if n.op == 'call_module':
            cnt = cnt + 1

        if cnt == segment:
            metadata_range.append((k, n.name))
            k = k + 1
            cnt = 0

    print(metadata_range)

simple_split(metadata_range)
        
print(f" #####################################")
print(metadata_range)


def get_range(rank, metadata_range) -> (Node, Node):
    print(f"metadata_range: {metadata_range}")

    if rank == 0:
        from_node_name = "-1"
        for n in gm.graph.nodes:
            if n.op == 'placeholder':
                from_node_name = n.name
            else:
                break
        # not found --> "-1"
    else:
        from_node_name = metadata_range[rank-1][1]

    to_node_name = metadata_range[rank][1]

    for n in gm.graph.nodes:
        if from_node_name == "-1":
            from_node = n
            break

        if n.name == from_node_name:
            from_node = n
            break
        
    for n in reversed(gm.graph.nodes):
        if n.name == to_node_name :
            to_node = n
            break

    return (from_node._next, to_node)


for rank in range(N):
    from_, to_ = get_range(rank, metadata_range)
    #print(f"rank = {rank}, from_:{from_}, to_:{to_}")
    print(f"## rank = {rank}, from_:{from_.name}, to_:{to_.name}")

    cur = from_  # first node assigned to the host#{rank} in metadata_range
    while cur != to_:
        print(f" ---- cur node:{cur.name}")

        cur = cur._next
    print(f" ---- cur node:{cur.name}")  # last node assigned to the host#{rank} in metadata_range 

    




