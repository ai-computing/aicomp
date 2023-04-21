#
# Copyright (c) 2023-present, ETRI, All rights reserved.
#
# This is a PoC that trains a model using FX IR.
#     : To create an IR with backward logic and perform F/B on it, 
#         we have quoted or modified the PiPPy code (https://github.com/pytorch/PiPPy)."
#

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
import torch.nn as nn
from torch.optim import Adam
from torch import fx
from torch.fx.node import Node
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time



import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

torch.manual_seed(42)

batch_size = 64
in_features = 5120
out_features = 5120
hidden = 5120

#torch.autograd.set_detect_anomaly(True)

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

class TestModel2(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return x

t1 = TestModel()
#t1 = TestModel2()

t2 = copy.deepcopy(t1)

#
print(t1)
print("-----------------------")
print(t2)

# LossWrapper: cited form PiPPy
class LossWrapper(torch.nn.Module):
    def __init__(self, module, loss_fn):
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        raise NotImplementedError("LossWrapper: no forward implementation")

# SimpleLossWrapper: cited from PiPPy
class SimpleLossWrapper(LossWrapper):
    def forward(self, x, targets):
        out1 = self.module(x)
        return self.loss_fn(out1, targets)

loss_fn = torch.nn.MSELoss()
wrapper = SimpleLossWrapper(t1, loss_fn)

#gm1 = fx.symbolic_trace(t1)
gm1 = fx.symbolic_trace(wrapper)

for node in gm1.graph.nodes:
    print(f"node.op:{node.op}, node.target:{node.target}, node.name:{node.name}")

print("-----------------------")
print(gm1.code)
print("-----------------------")

# stage_backward function: cited from PiPPy
def stage_backward(
    stage_output,
    output_grads,
    input_values,
    outputs_with_grads_idxs: List[int],
):
    #
    #print(f"** stage_backward ** stage_output:{stage_output}, output_grads:{output_grads}, input_values:{input_values}, outputs_with_grads_idxs: {outputs_with_grads_idxs}")
    stage_output_with_grads = [
        stage_output[i] for i in outputs_with_grads_idxs
    ]
    output_grads_with_grads = [
        output_grads[i] for i in outputs_with_grads_idxs
    ]

    stage_output_tensors = []
    output_grad_tensors = []

    def extract_tensors_with_grads(output_val, grad_val):
        if isinstance(output_val, torch.Tensor):
            if not output_val.requires_grad and output_val.grad_fn is None:
                return
            stage_output_tensors.append(output_val)
            output_grad_tensors.append(grad_val)
        elif isinstance(output_val, (tuple, list)):
            if grad_val is None:
                return
            for ov, gv in zip(output_val, grad_val):
                extract_tensors_with_grads(ov, gv)
        elif isinstance(output_val, dict):
            if grad_val is None:
                return
            for k in output_val.keys():
                extract_tensors_with_grads(output_val[k], grad_val[k])
        else:
            print(f"... ignored in this case")
            pass

    extract_tensors_with_grads(stage_output_with_grads, output_grads_with_grads)

    torch.autograd.backward(stage_output_tensors, grad_tensors=output_grad_tensors)

    grad_inputs = []
    for val in input_values:
        if isinstance(val, torch.Tensor):
            grad_inputs.append(val.grad)
        else:
            grad_inputs.append(None)

    barrier_token = None
    return grad_inputs, barrier_token


# _get_loss_output: adapted from PiPPy
def _get_loss_output(graph: fx.Graph):
    output_nodes = [n for n in graph.nodes if n.op == "output"]
    assert len(output_nodes) == 1
    output_node = output_nodes[0]
    output_val = output_node.args[0]
    #
    loss_node = output_val

    return loss_node, output_node

# sync_barrier: cited from PiPPy
def sync_barrier(loss, barrier_tokens, last_grads):
    return loss, last_grads
    

loss_node, output_node = _get_loss_output(gm1.graph)

print(loss_node)
print(output_node)


# make_ir_for_backwards: adapted from PiPPy
def make_ir_for_backwards(graph: fx.Graph, loss_node: fx.Node, output_node: fx.Node):

    tuples: Dict[fx.Node, Tuple] = {}
    for node in reversed(graph.nodes):
        if node.op == 'call_function':
            indexed_value, node_idx = tuple(node.args)
            existing_list_size = (
                len(tuples[indexed_value]) if indexed_value in tuples else -1
            )
            new_list_size = max(node_idx + 1, existing_list_size)
            reconstructed_list = [None for _ in range(new_list_size)]
            if indexed_value in tuples:
                for i, val in enumerate(tuples[indexed_value]):
                    reconstructed_list[i] = val
            reconstructed_list[node_idx] = node
            tuples[indexed_value] = tuple(reconstructed_list)

    live_nodes = {loss_node: None}
    val_to_grad: Dict[fx.Node, Optional[fx.Node]] = {loss_node: None}

    def assign_or_accumulate_grad(forward_node, grad_value):
        #
        if forward_node in val_to_grad and forward_node.op != "placeholder":
            grad_value = g.call_function(
                _null_coalesce_accumulate,
                (val_to_grad[forward_node], grad_value),
            )
        val_to_grad[forward_node] = grad_value

    with graph.inserting_before(output_node):
        barrier_tokens = []
        last_grads = None
    
        for node in reversed(graph.nodes):
            if node not in live_nodes:
                continue

            def add_to_live_nodes(n):
                live_nodes.setdefault(n, None)

            fx.graph.map_arg(node.args, add_to_live_nodes)
            fx.graph.map_arg(node.kwargs, add_to_live_nodes)
            if node.op == 'call_module':
                output_grads: Union[Tuple[Optional[fx.Node], ...], Optional[fx.Node]]

                if node in tuples:
                    stage_output = tuples[node]
                    output_grads = tuple( val_to_grad.get(n, None) for n in tuples[node] )
                    outputs_with_grads_idxs = [
                        i for i, n in enumerate(tuples[node]) if n in live_nodes ]
                else:
                    stage_output = (node, )
                    output_grads = val_to_grad[node]
                    outputs_with_grads_idxs = [0]

                output_grads = ((output_grads,) if not isinstance(output_grads, tuple) else output_grads)

                grad_call = graph.call_function(
                    stage_backward,
                    kwargs={
                        "stage_output": stage_output,
                        "output_grads": output_grads,
                        "input_values": list(node.all_input_nodes),
                        "outputs_with_grads_idxs": outputs_with_grads_idxs,
                    },
                )

                grad_call_proxy = fx.Proxy(grad_call)
                grads, barrier_token = (
                    grad_call_proxy[0].node,
                    grad_call_proxy[1].node,
                )
                barrier_tokens.append(barrier_token)
                last_grads = grads

                input_nodes = list(node.all_input_nodes)
                grads_proxy = fx.Proxy(grads)
                for i, input_node in enumerate(input_nodes):
                    assign_or_accumulate_grad(input_node, grads_proxy[i].node)

        barrier_call = graph.call_function(sync_barrier, (output_node.args[0], barrier_tokens, last_grads))
        output_node.args = (barrier_call,)


print(f" =======> make_backwards\n\n")
make_ir_for_backwards(gm1.graph, loss_node, output_node)

# DEBUG
for node in gm1.graph.nodes:
    #print(f"node.op:{node.op}, node.target:{node.target}, node.name:{node.name}, node.all_input_nodes: {node.all_input_nodes}")
    print(f"node.op:{node.op}, node.target:{node.target}, node.name:{node.name}, node.args:{node.args}, node.all_input_nodes: {node.all_input_nodes}")
print("-----------------------")

class FXRun:

    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.loss = None
        self.fwd_cache: Dict[int, Tuple[Any, List[torch.Tensor]]] = {}

        # TODO
        self.stage_num = 0

        #self.env : Dict[Node, Any] = {}
        self.env : Dict[str, Any] = {}

    def run(self, *args):
        args_iter = iter(args)

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
                #print(f" [placeholder] result: {result}")

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

                if node.target == stage_backward:
                    #
                    def extract_tensor_args(b):
                        #a = self.env[b]
                        a = self.env[b.name]
                        #nonlocal flat_args
                        if isinstance(a, torch.Tensor):
                            #print(f"## a:{a} is torch.Tensor a.requires_grad:{a.requires_grad} !!!")
                            val = a.detach().requires_grad_(a.requires_grad)
                            #flat_args.append(val)
                            return val
                        else:
                            #print(f"## a:{a} is NOT torch.Tensor !!!!")
                            #flat_args.append(a)
                            return a

                    #args = fx.graph.map_arg(node.args, lambda n: self.env[n.name]) 
                    #args = fx.graph.map_arg(node.args, lambda n: self.env[n]) 
                    args = fx.graph.map_arg(node.args, extract_tensor_args) 


                    self.stage_num -= 1

                    #kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name]) 
                    #kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env[n]) 
                    kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args) 

                    kwargs = dict(kwargs)
                    ( kwargs["stage_output"], kwargs["input_values"],) = self.fwd_cache.pop(self.stage_num)
                    
                    # DEBUG
                    #print(f" ===> kwargs: {kwargs}")

                    #
                    #print(f" [call_function-stage_backward, stage_num[{self.stage_num}]]: node.target:{node.target}, node.name:{node.name}, len(args):{len(args)}, len(kwargs):{len(kwargs)}, args:{args}, kwargs:{kwargs}")

                    result = stage_backward(*args, **kwargs)
                    # TODO
                    #print(f" call_function: stage_backward, node.name:{node.name} --> result:{result}")
                else:
                    # TODO
                    args = fx.graph.map_arg(node.args, lambda n: self.env[n.name])
                    #args = fx.graph.map_arg(node.args, lambda n: self.env[n])
                    kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name])
                    #kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env[n])
                    #
                    #print(f" call_function: node.target:{node.target}, node.name:{node.name}, len(args):{len(args)}, len(kwargs):{len(kwargs)}")
                    result = node.target( \
                            *args, \
                            **kwargs)
                    #result = node.target( \
                    #        *fx.graph.map_arg(node.args, lambda n: self.env[n.name]), \
                    #        **fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name]))
                    #
                    #print(f" call_function: {node.target}, node.name:{node.name} --> result:{result}")

            elif node.op == 'call_method':
                self_obj, *args = fx.graph.map_arg(node.args, lambda n: self.env[n.name])
                #self_obj, *args = fx.graph.map_arg(node.args, lambda n: self.env[n])
                kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name])
                #kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env[n])
                result = getattr(self_obj, node.target)(*args, **kwargs)

            elif node.op == 'call_module':
                #print(f" ### call_module , node.target:{node.target}, node.name:{node.name}")

                #result = self.modules[node.target](\
                #        *fx.graph.map_arg(node.args, lambda n: self.env[n.name]),\
                #        **fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name]))


                flat_args = []
                def extract_tensor_args(b):
                    #a = self.env[b]
                    a = self.env[b.name]
                    nonlocal flat_args
                    if isinstance(a, torch.Tensor):
                        #print(f"## a:{a} is torch.Tensor a.requires_grad:{a.requires_grad} !!!")
                        val = a.detach().requires_grad_(a.requires_grad)
                        flat_args.append(val)
                        return val
                    else:
                        #print(f"## a:{a} is NOT torch.Tensor !!!!")
                        flat_args.append(a)
                        return a
                    #flat_args.append(a)
                    #print(f"## a isinstance --> {isinstance(a, torch.Tensor)}, flat_args:{flat_args}  !!!!")
                    return a

                args = fx.graph.map_arg(node.args, extract_tensor_args)
                kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)

                #
                #print(f" ### call_module , node.target:{node.target}, node.name:{node.name} --> fwd_cache[{self.stage_num}] ")

                #result = self.modules[node.target](*args, **kwargs)

                # TODO
                target_atoms = node.target.split('.')
                attr_itr = self.mod
                for i , atom in enumerate(target_atoms):
                    if not hasattr(attr_itr, atom):
                        raise RuntimeError(\
                                f"Node referenced nonexistant target{'.'.join(target_atoms[:i])}")
                    attr_itr = getattr(attr_itr, atom)
                submod = attr_itr
                result = submod(*args, **kwargs)

                # DEBUG
                if node.target == 'loss_fn':
                    #print(f" -- args:{args}, kwargs:{kwargs}, len(args):{len(args)}")
                    #print(f" -- args[0]:{Tensor.size(args[0])}, args[1]: {Tensor.size(args[1])}")
                    #print(f" -- node.all_input_nodes[0]: {node.all_input_nodes[0]}")
                    if not str(node.all_input_nodes[0]).startswith("target"):
                        self.output = self.env[str(node.all_input_nodes[0])]


                # TODO
                self.fwd_cache[self.stage_num] = \
                         ( result if isinstance(result, tuple) else (result,), \
                         flat_args, )

                self.stage_num += 1


                #result = self.modules[node.target](\
                #        *fx.graph.map_arg(node.args, lambda n: self.env[n.name]),\
                #        **fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name]))

                if node.target == 'loss_fn':
                    self.loss = result

            #elif node.op == 'output':
            #    result = node.args[0]
            #    print(f" ---------------------------> output:  {result}")

            #self.env[node] = result
            self.env[node.name] = result

        #return fx.graph.map_arg(self.env[node.name], lambda n: self.env[n.name])
        #return fx.graph.map_arg(self.env[node], lambda n: self.env[n])
        return self.output




t1.train()
t2.train()
optimizer1 = Adam(t1.parameters(), lr=3e-5)
optimizer2 = Adam(t2.parameters(), lr=3e-5)

#
print(f" begin  fx_run ... ")

fx_run = FXRun(gm1)

tick =  time.time()

#sample_input = torch.rand(batch_size, in_features)
sample_output = torch.rand(batch_size, out_features)

N = 20

for i in range(N):
    sample_input = torch.rand(batch_size, in_features)

    optimizer1.zero_grad()
    optimizer2.zero_grad()

    #loss1 = wrapper(sample_input, sample_output) # actual
    output1 = fx_run.run(sample_input, sample_output) # actual
    loss1 = fx_run.loss

    optimizer1.step()

    output2 = t2(sample_input) # expected
    loss2 = torch.nn.MSELoss()(output2, sample_output)
    loss2.backward()
    optimizer2.step()

    print(f'Step {i}, Loss1: {loss1}, Loss2: {loss2}')

    torch.testing.assert_close(output1, output2)
    #torch.allclose(output1, output2)

tock = time.time()
elapsed_time = tock - tick
print('Time elapsed: %.3f sec ' % (elapsed_time))
print(output1)
print("#######")
print(output2)

