#
# Copyright (c) 2024-present, ETRI, All rights reserved.
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


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p

logging.basicConfig(level=logging.ERROR)



batch_size = 64

in_features = 5120
out_features = 5120
hidden = 5120

class SyntheticModel(nn.Module):
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


model = SyntheticModel()

def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


if int(os.environ["RANK"]) == 0:
    print ('Total parameters in model: {:,}'.format(get_total_params(model)))


micro_batch_size = int(os.environ["WORLD_SIZE"]) // 2 if int(os.environ["WORLD_SIZE"]) > 1 else 1 # TODO

if int(os.environ["RANK"]) == 0:
    print(f"total process count: {os.environ['WORLD_SIZE']}")
    print(f"batch size: {batch_size}")
    print(f"micro batch size: {micro_batch_size}")

optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True)
#optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, dp_size=2, preserve_output=True)
#optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, dp_size=2)
print(f" rank={optimus_p.get_rank()} ...")

optimus_p.train()
optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5)


optimus_p.train() # turn on the train mode

if optimus_p.get_rank() == 0:
    tick = time.time()

for i in range(100):

    data, labels = None, None

    # prepare input and label
    if optimus_p.is_first_stage():
        data = torch.rand(batch_size, in_features)
        labels = torch.rand(batch_size, out_features)

    labels = optimus_p.move_labels2last_stage(labels)

    optimizer.zero_grad()

    #optimus_p.run(data, labels)
    #optimus_p.run(data, labels, mode="gpipe")
    optimus_p.run(data, labels, mode="1f1b")

    if optimus_p.is_last_stage():
        loss = optimus_p.get_loss() 
        print(f'Step {i}, Loss1:{sum(loss) / micro_batch_size}')
    else:
        loss = None

    optimizer.step()


if optimus_p.get_rank() == 0:
    tock = time.time()
    elapsed_time = tock - tick

    print('Time elapsed: %.3f sec ' % (elapsed_time))

#if optimus_p.get_rank() == optimus_p.get_world_size() - 1: 
if optimus_p.is_last_stage(): 
    output = optimus_p.get_output()
    if output != None:
        print(f">> [RANK:{optimus_p.get_rank()} ###################### output: {output} #############")
print(f"[rank:{optimus_p.get_rank()}, run completed ...")
