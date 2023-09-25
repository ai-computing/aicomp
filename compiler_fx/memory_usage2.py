import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import init
import math

import psutil
import os
pid = os.getpid()
print(f">> Process ID: {pid}")

#use_reset_parameters = False
use_reset_parameters = True

print_flag = True

def print_memory_usage(str, print_flag):
    if print_flag == True:
        print(" =========", str, "=========")
        my_process = psutil.Process(pid)
        usage =  my_process.memory_info().rss / (1024 ** 3)   # GB unit
        print(f" Memory Usage: {usage:.3f} GB")


class Linear2(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int):
        super(Linear2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.bias = Parameter(torch.empty(out_features))

        if use_reset_parameters == True:
            self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

print_memory_usage("Before: m = Linear2(10000, 20000)", print_flag)

m = Linear2(10000, 20000)

print(f" ***** use reset_parameters() : {use_reset_parameters} *****")
print_memory_usage("After: m = Linear2(10000, 20000)", print_flag)
print(f"{m.weight}")
input = torch.randn(20000, 10000)
print_memory_usage("After: input = torch.randn(20000, 10000)", print_flag)
output = m(input)
print_memory_usage("After: output = m(input)", print_flag)
print(f"{output.size()}")

