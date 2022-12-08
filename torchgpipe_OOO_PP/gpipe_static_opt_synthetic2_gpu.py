#
# Copyright (c) 2022-present, ETRI, All rights reserved.
#

#
# thin, synthetic model on torchgpipe (GPU version)
#
#    OPTIMIZE_FLAG = True --> This model is composed of the OutGradOnlyLinear instead of nn.Linear
#    OPTIMIZE_FLAG = False --> This model is composed of the original nn.Linear
#
#     *** In this models, the Out Of Order technique does not work well.
#


import math
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
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


OPTIMIZE_FLAG = True
#OPTIMIZE_FLAG = False


class Hooking():
    def __init__(self, module, name):
        self.forwrad_hook = module.register_forward_hook(self.fwd_h)
        self.backward_hook = module.register_backward_hook(self.bwd_h)
        self.weight = module.weight
        self.name = name
        self.fwd_inputs = []
        self.grad_outputs = []
        self.counter = 0

    def fwd_h(self, module, input, output):
        self.fwd_inputs.append(input[0])

        self.counter = self.counter + 1
        if self.counter < 2:
            print(f'forward hook called {self.counter}')

    def bwd_h(self, module, input, output):
        self.grad_outputs.append(output[0])

    def compute_weight_grad(self):
        grad_output = self.grad_outputs.pop(0)
        #total_input = self.fwd_inputs.pop(0)
        total_input = self.fwd_inputs.pop()

        #grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        #total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])
        grad_weight = grad_output.t().matmul(total_input)
        # ?
        self.weight.grad = grad_weight

    def check_consistency(self):
        size = len(self.grad_outputs)
        assert size == 0, f"grad output size is not zero SIZE : {size} name {self.name}"
        size = len(self.fwd_inputs)
        assert size == 0, f"forward inputs size is not zero SIZE : {size} name {self.name}"

_HOOKING_LIST = []

def run_for_hooking(model):
    global _HOOKING_LIST
    
    modules = model.named_modules()
    for name, module in modules:
        #print(f'module name = { module.__class__.__name__}')
        #if module.__class__.__name__ == "Linear":
        if module.__class__.__name__ == "OutGradOnlyLinear":
            h = Hooking(module, name)
            _HOOKING_LIST.append(h)


def print_modules(model):
    modules = model.named_modules()
    for name, module in modules:
        print(name, module)

def compute_weight_grad_all():
    global _HOOKING_LIST
    for h in _HOOKING_LIST:
        h.compute_weight_grad()
        # DEBUG
        h.check_consistency()


class OutGradOnlyMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, weight, bias):
        ctx.save_for_backward(data, weight)
        ctx.use_bias = bias is not None

        output = torch.matmul(data, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        grad_input = grad_output.matmul(weight)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        return grad_input, None, grad_bias

class OutGradOnlyLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(OutGradOnlyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        output = OutGradOnlyMatMul.apply(x, self.weight, self.bias)
        return output

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        if OPTIMIZE_FLAG == True:
            self.linear1 = OutGradOnlyLinear(in_features, hidden)
            self.linear2 = nn.ModuleList()
            for i in range(20):
                self.linear2.append(OutGradOnlyLinear(hidden, hidden))

            self.linear3 = nn.ModuleList()
            for i in range(20):
                self.linear3.append(OutGradOnlyLinear(hidden, hidden))

            self.linear4 = nn.ModuleList()
            for i in range(20):
                self.linear4.append(OutGradOnlyLinear(hidden, hidden))

            self.linear5 = nn.ModuleList()
            for i in range(20):
                self.linear5.append(OutGradOnlyLinear(hidden, hidden))
            self.linear6 = OutGradOnlyLinear(hidden, out_features)
            self.relu = nn.ReLU(inplace = True)

        else:

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

# 
if OPTIMIZE_FLAG == True:
    print_modules(model)
    run_for_hooking(model)


#model = GPipe(model, balance=[1,1,1,1], devices=['cpu', 'cpu', 'cpu', 'cpu'])
model = GPipe(model, balance=[2,2], devices=[4,2])


print(f'model len = {len(model)}')
print(model)

model.train()
optimizer = Adam(model.parameters(), lr=3e-5)

# GPU
in_device = model.devices[0]
out_device = model.devices[-1]

tick = time.time()

#sample_input = torch.rand(batch_size, in_features)
sample_output = torch.rand(batch_size, out_features)

for i in range(10):
    sample_input = torch.rand(batch_size, in_features)
    #sample_output = torch.rand(batch_size, out_features)

    # GPU
    # The input device is different from the output device
    sample_input = sample_input.to(in_device, non_blocking=True)
    sample_output = sample_output.to(out_device, non_blocking=True)

    optimizer.zero_grad()
    output = model(sample_input)
    loss = torch.nn.MSELoss()(output, sample_output)
    loss.backward()
    #
    if OPTIMIZE_FLAG == True:
        compute_weight_grad_all()
    optimizer.step()
    print(f'step {i}, Loss: {loss}')

tock = time.time()
elapsed_time = tock - tick
print('Time elapsed: %.3f sec' % (elapsed_time))

