#
# Copyright (c) 2022-present, ETRI, All rights reserved.
#

# 
# thin, synthetic model on torchgpipe
#

import torch
import torch.nn as nn
from torch.optim import Adam
import time

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torchgpipe import GPipe
from torchgpipe.gpipe import verify_module

torch.manual_seed(42)

batch_size = 64
in_features = 64
out_features = 64
hidden = 64

# Comment this line when measuring
torch.autograd.set_detect_anomaly(True)

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.ModuleList()
        for i in range(20):
            self.linear2.append(nn.Linear(hidden, hidden))

        self.linear3 = nn.ModuleList()
        for i in range(20):
            self.linear3.append(nn.Linear(hidden, hidden))

        self.linear4 = nn.ModuleList()
        for i in range(20):
            self.linear4.append(nn.Linear(hidden, hidden))

        self.linear5 = nn.ModuleList()
        for i in range(20):
            self.linear5.append(nn.Linear(hidden, hidden))
        self.linear6 = nn.Linear(hidden, out_features)
        self.relu = nn.ReLU(inplace = True)

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
t2 = TestModel()
t3 = TestModel()
t4 = TestModel()

model = nn.Sequential(t1, t2, t3, t4)
model = GPipe(model, balance=[1,1,1,1], devices=['cpu', 'cpu', 'cpu', 'cpu'])

print(f'model len = {len(model)}')

model.train()
optimizer = Adam(model.parameters(), lr=3e-5)

tick = time.time()

for i in range(10):
    sample_input = torch.rand(batch_size, in_features)
    sample_output = torch.rand(batch_size, out_features)

    optimizer.zero_grad()
    output = model(sample_input)
    loss = torch.nn.MSELoss()(output, sample_output)
    loss.backward()
    optimizer.step()
    print(f'step {i}')

tock = time.time()
elapsed_time = tock - tick
print('Time elapsed: %.3f sec' % (elapsed_time))

