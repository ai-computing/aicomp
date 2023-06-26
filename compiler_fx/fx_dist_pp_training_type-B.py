#
# Copyright (c) 2023-present, ETRI, All rights reserved.
#
#
#  This is a PoC that performs a GPipe-style pipeline-parallel training based on the FX IR partition.
#
#   In this PoC, FX compile generates the FX IR,
#        and partitions of FX IR are transferred to the distibuted hosts, 
#        and then pipeline parallel training is executed using N hosts.
#
#   Micro-batch is supported in this PoC.
#
#
#  Sample Usage:
#      <machine #0>
#            torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0
#                  --master_addr="X.X.X.X" --master_port=29500 fx_dist_pp_training_type-B.py
#      <machine #1>
#            torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1
#                  --master_addr="X.X.X.X" --master_port=29500 fx_dist_pp_training_type-B.py
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
#import torch.distributed.rpc as rpc

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torch.fx.graph_module import GraphModule
from torch.fx.passes.split_module import split_module


torch.manual_seed(42)

#
# Total process count
#
#num_rank=N
num_rank=4  
#num_rank=6  
#num_rank=8  
#num_rank=16

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

        #
        #self.linear5_0 = nn.ModuleList()
        #for i in range(2):
        #    self.linear5_0.append(nn.Linear(hidden, hidden))
        #self.linear5_1 = nn.ModuleList()
        #for i in range(2):
        #    self.linear5_1.append(nn.Linear(hidden, hidden))
        #self.linear5_2 = nn.ModuleList()
        #for i in range(2):
        #    self.linear5_2.append(nn.Linear(hidden, hidden))

        #self.linear5_3 = nn.ModuleList()
        #for i in range(2):
        #    self.linear5_3.append(nn.Linear(hidden, hidden))
        #self.linear5_4 = nn.ModuleList()
        #for i in range(2):
        #    self.linear5_4.append(nn.Linear(hidden, hidden))
        #self.linear5_5 = nn.ModuleList()
        #for i in range(2):
        #    self.linear5_5.append(nn.Linear(hidden, hidden))
        

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

        
        #for m in self.linear5_0:
        #    x = self.relu(m(x))
        #for m in self.linear5_1:
        #    x = self.relu(m(x))
        #for m in self.linear5_2:
        #    x = self.relu(m(x))
        #
        #for m in self.linear5_3:
        #    x = self.relu(m(x))
        #for m in self.linear5_4:
        #    x = self.relu(m(x))
        #for m in self.linear5_5:
        #    x = self.relu(m(x))
        

        x = self.linear6(x)
        x = self.relu(x)
        return x

# LossWrapper: cited from PiPPy
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


class Simple_split_test(object):
    def __init__(self):
        self.initialize_comm()

        self.model_ir = []

    def initialize_comm(self):

        if dist.is_initialized():
            print(f"Communication already initialized")
            return


        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.master_addr = os.getenv("MASTER_ADDR")
        self.master_port = os.getenv("MASTER_PORT")
        self.stage = 0


        #
        print(f" --- rank:{self.rank}, world_size:{self.world_size}, master:{self.master_addr}, port:{self.master_port}")

        self.backend = "gloo"
        init_method = "tcp://" + str(self.master_addr) + ":" + str(self.master_port)

        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size, init_method=init_method)

        #
        print(f" --- rank:{dist.get_rank()}, world_size:{dist.get_world_size()}")

        #options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=10, rpc_timeout=30)

        #rpc.init_rpc(f"worker{self.rank}", rank=self.rank, world_size=self.world_size, rpc_backend_options=options,)

        # DEBUG
        #print(f" --- after init_rpc -- rank:{dist.get_rank()}, world_size:{dist.get_world_size()}")

        # rpc.shutdown()


    def simple_split(self, gm, module, metadata_range):

        length = gm.graph.nodes.__len__()
        segment = length  // num_rank
        print(f"length:{length}, num_rank:{num_rank}, segment:{segment}")

        # simple assert
        assert length >= num_rank, f"Model length:{length} is smaller than # of workers:{num_rank}"

        self.last_flag = False

        def part_fn(node):
            last_idx, last_name = metadata_range[-1]

            if self.last_flag == True:
                idx = last_idx
                #print(f" part_fn:  node.name:{node.name}, --> {idx}")
                return idx

            idx = 0

            cur = node
            while cur.name != last_name:
                for i, m_name in metadata_range:
                    if cur.name == m_name:
                        idx = i
                        print(f" part_fn:  node.name:{node.name}, m_name:{m_name}, --> {idx}")
                        return idx

                cur = cur._next

            if cur.name == last_name:
                idx = last_idx
                self.last_flag = True

            print(f" part_fn:  node.name:{node.name}, --> {idx}")
            return idx


        k, cnt = 0, 0
        for n in gm.graph.nodes:
            if n.op == 'call_module':
                cnt = cnt + 1

            if cnt == segment:
                metadata_range.append((k, n.name))
                k = k + 1
                cnt = 0

            if k > num_rank - 1:
                break

        if len(metadata_range) <  num_rank:
            metadata_range.append((k, n.name))


        print(metadata_range)

        submodules = split_module(gm, module, part_fn, keep_original_order=True)

        return submodules


    def setup_pair_info(self):

        if self.rank == 0:
            self.rank_pair: Dict[int, List[int]] = {}
            rank_pair_obj = [self.rank_pair]

            for rank in range(self.world_size):
                if rank == 0:
                    continue
                self.rank_pair.setdefault(rank, [0, rank] )

            dist.broadcast_object_list(rank_pair_obj, src=0, device=self.device)

        else:
            self.rank_pair: Dict[int, List[int]] = {}

            rank_pair_obj = [None]
            dist.broadcast_object_list(rank_pair_obj, src=0, device=self.device)
            self.rank_pair = rank_pair_obj[0]

        print(f"## In rank[{self.rank}], setup_pair_info completed ==> rank_pair:{self.rank_pair}")


    def setup_ctrl_group(self):
        self.ctrl_group: Dict[int, Any] = {}

        for rank in range(self.world_size):
            if rank == 0:
                continue
            pair_ranks = self.rank_pair[rank]
            self.ctrl_group[rank] = dist.new_group(pair_ranks)

        print(f"##In rank[{self.rank}], setup_ctrl_group completed")


    def check_last_submods(self, submods):
        gmodule_cnt = 0
        mod_cnt = 0
        for submod in submods.modules():
            if isinstance(submod, fx.GraphModule):
                gmodule_cnt = gmodule_cnt + 1
                last_submod = submod
                continue

        assert gmodule_cnt > num_rank, f"GraphModule #:[{gmodule_cnt}] must have more than host #:[{num_rank}]"

        for node in last_submod.graph.nodes:
            print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.all_input_nodes:{node.all_input_nodes}")
            if node.op == 'call_module' and node.target != 'loss_fn':
                mod_cnt = mod_cnt + 1

        print(f">>> GraphModule cnt:{gmodule_cnt},  Last GraphModule's  mod_cnt ==> {mod_cnt}")

        assert mod_cnt > 0, f"Last partition has {mod_cnt} modules. It must have more than 0 modules"

        
    def metadata_transfer(self):

        self.metadata_range = []

        if self.rank == 0:
            t1 = TestModel()
            #gm = fx.symbolic_trace(t1)

            loss_fn = torch.nn.MSELoss()
            wrapper = SimpleLossWrapper(t1, loss_fn)

            gm = fx.symbolic_trace(wrapper)

            # DEBUG
            for n in gm.graph.nodes:
                print(f"n.op:{n.op}, n.name:{n.name}, n.target:{n.target}, n.args:{n.args}, n.all_input_nodes:{n.all_input_nodes}")
            print(f"------------------------------------------------------------")

        self.device = torch.device("cpu")

        self.setup_pair_info()
        self.setup_ctrl_group()


        if self.rank == 0:
            #submods = self.simple_split(gm, t1, self.metadata_range)
            submods = self.simple_split(gm, wrapper, self.metadata_range)

            self.check_last_submods(submods)

            skip = False
            to_rank = 0
            for submod in submods.modules():
                if skip == False and isinstance(submod, fx.GraphModule):
                    skip = True
                    continue
                if skip == True and isinstance(submod, fx.GraphModule):
                    print(f"submod:{submod._get_name()}")

                    if to_rank == 0:
                        print(f"### rank = 0 holding submod_0")

                        self.model_ir.append(submod)

                        #for node in submod.graph.nodes:
                        #    print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.all_input_nodes:{node.all_input_nodes}")

                    else:
                        print(f"### send IR partition to rank:{to_rank}")
                        object_list = [submod, to_rank]
                        dist.broadcast_object_list(object_list, src=0, group=self.ctrl_group[to_rank], device=self.device)

                    to_rank = to_rank + 1

                    #print(f" >> FROM:{self.rank} ==> TO:{to_rank} FX IR partition transferred")

        else:

            object_list = [None,None]
            dist.broadcast_object_list(object_list, src=0, group=self.ctrl_group[self.rank], device=self.device)

            submod = object_list[0]

            self.model_ir.append(submod)
            self.stage = object_list[1]

            if submod is None:
                print(f"In rank[{self.rank}], FX IR sync failed")
            else:
                print(f" ### rank:{self.rank}, stage:{self.stage} received <==  FX IR partition")

                #print(f" ############## rank[{self.rank}], stage:{self.stage} #################")
                #for node in submod.graph.nodes:
                #    print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.all_input_nodes:{node.all_input_nodes}")


# stage_backward function: cited from PiPPy
def stage_backward(
    stage_output,
    output_grads,
    input_values,
    outputs_with_grads_idxs: List[int],
):
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



class FXRun2:

    def __init__(self, split_info: Simple_split_test, device, mbsize):

        self.mod = split_info.model_ir[0]
        self.graph = self.mod.graph
        self.modules = dict(self.mod.named_modules())
        self.mbsize = mbsize  
        self.env2: List[Dict[str, Node]] = [{} for _ in range(mbsize)]
        self.metadata_range = split_info.metadata_range
        self.rank = split_info.rank
        self.world_size = split_info.world_size
        self.device = device
        self.stage = split_info.stage
        self.loss: List[Any] = [None for _ in range(mbsize)]
        self.fwd_cache2: List[Dict[str, Tuple[Any, List[torch.Tensor]]]] = [{} for _ in range(mbsize)]
        self.grads2: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        

    def get_destination(self, input_nodes, lst_):
    #def get_destination(self, input_nodes, set_):
        for i, m in enumerate(input_nodes):
            for n in self.graph.nodes:
                if n.name == m.name:
                    #if m.op == 'call_module' or m.op == 'call_method':
                    #if m.op == 'call_module' or m.op == 'call_method' or m.op == 'call_function':
                    #if m.op == 'call_module' or m.op == 'call_method' or m.op == 'call_function' or m.op == 'placeholder':
                    if m.op == 'call_module' or m.op == 'call_method' or  m.op == 'placeholder':
                        lst_.append(m)
                        #set_.add(m)
                        break

                    if m.op == 'call_function':
                        self.get_destination(m.all_input_nodes, lst_)
                        #self.get_destination(m.all_input_nodes, set_)



    def receive_tensor(self, from_rank):
        dimension = torch.tensor([0], dtype=torch.long)
        dist.recv(dimension, from_rank)

        shape = torch.tensor([0] * dimension, dtype=torch.long)
        dist.recv(shape, from_rank)
        shape = tuple(shape.tolist())

        obj = torch.zeros(size=shape)
        dist.recv(obj, from_rank)

        return obj

    def send_tensor(self, obj, to_rank):
        dimension = torch.tensor(len(obj.size()), dtype=torch.long) # ex. 2
        dist.send(dimension, to_rank)

        shape = torch.tensor(list(obj.size()), dtype=torch.long) # ex. [54, 5120]
        dist.send(shape, to_rank)

        dist.send(obj, to_rank)

    def receive_list(self, from_rank):
        length = torch.tensor([0], dtype=torch.long)
        dist.recv(length, from_rank)

        obj = []
        for _ in range(length.item()):
            n = self.receive_tensor(from_rank)
            obj.append(n)

        return obj

    def send_list(self, obj, to_rank):
        length = torch.tensor(len(obj), dtype=torch.long)
        dist.send(length, to_rank)

        for n in obj:
            self.send_tensor(n, to_rank)


    def print_range(self):
        print(f" # rank = {self.rank}, metadata_range:{self.metadata_range}")

        for node in self.mod.graph.nodes:
            print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.args:{node.args}, node.all_input_nodes:{node.all_input_nodes}")


    def fx_forward4(self, *args):
        #print(f" -----> rank{self.rank}: in fx_forward4, args[0]:{args[0]}")
        self.args_iter = iter(args)
        for n in self.mod.graph.nodes:
            if n.op == 'placeholder' and self.stage == 0:
                input = next(self.args_iter)

                #print(f">>>>> input:{input}, mbsize:{self.mbsize}")

                if isinstance(input, torch.Tensor):
                    mbatches = torch.chunk(input, self.mbsize)
                    for j in range(self.mbsize):
                        self.env2[j]["placeholder"] = mbatches[j]
                else:
                    print(f"### input:{input} not Tensor --> currently not supported!!")
                    sys.exit(1)
                break

        #print(f" * rank:{self.rank}, in run_micro_batch_forward() ..")
        for i in range(self.mbsize):
            result = self.fx_micro_forward(i)
            next(result)


    def get_last_module(self):
        if self.rank == self.world_size - 1:
            for n in reversed(self.mod.graph.nodes):
                if n.op == 'call_module' and n.target != 'loss_fn':
                    print(f"[Rank:{self.rank}] ==> got last module: {n.name}")
                    return n.name


    def make_output(self):
        output = None
        if self.rank ==  self.world_size - 1:
            target = self.get_last_module()
            outputs = tuple(mb[target] for mb in self.env2) 
            print(f" ---> RANK: {self.rank},  outputs = {outputs}, type(output):{type(outputs)}")
            output = torch.cat(outputs)

        return output


    def fx_micro_forward(self, mb_idx):

        for n in self.mod.graph.nodes:
            from_ = n
            break

        for n in reversed(self.mod.graph.nodes):
            to_ = n
            break

        if self.rank > 0:
            target_node_name = "placeholder"
            pre_split_rank = self.rank - 1
            #print(f"## rank:{self.rank}, receive activation from {pre_split_rank}, target_node_name:{target_node_name}")
            self.env2[mb_idx][target_node_name] = self.receive_tensor(pre_split_rank)


        cur = from_
        while cur != to_:
            self.fx_ir_run_node2(cur, mb_idx)
            cur = cur._next
        result = self.fx_ir_run_node2(cur, mb_idx)

        #print(f" rank:{self.rank}, cur.node name{cur.name}, target_node_name:{to_.name}")

        if self.rank < self.world_size - 1:
            target_node_name = "output"
            next_split_rank = self.rank + 1
            #print(f"### rank:{self.rank} send activation to {next_split_rank}, target_node_name:{target_node_name}")

            obj = self.env2[mb_idx][target_node_name]
            self.send_tensor(obj, next_split_rank)

        yield result



    #def restore_env(self, node: Node) -> Tuple[Tuple, Dict]:
    #    #print(f"## before restore_env, node:{node}, node.args:{node.args}, node.kwargs:{node.kwargs}")
    #
    #    args = fx.graph.map_arg(node.args, lambda n: self.env[n.name])
    #    assert isinstance(args, tuple)
    #
    #    kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name])
    #    assert isinstance(kwargs, dict)
    #
    #    #print(f">>> after restore_env, node:{node}, node.name:{node.name}, args:{args}, kwargs:{kwargs}")
    #
    #    return args, kwargs
        

    def fx_ir_run_node2(self, node, mb_idx):

        #args, kwargs = self.restore_env(node)

        result = Any

        if node.op == 'placeholder' and self.stage == 0:
            #result = next(self.args_iter)
            result = self.env2[mb_idx]["placeholder"]

        elif node.op == 'placeholder' and node.target != 'targets' and self.stage > 0:
            result = self.env2[mb_idx]["placeholder"]

        elif node.op == 'placeholder' and node.target == 'targets' and self.stage > 0:
            #print(f" ****** env2[{mb_idx}]['targets']: {self.env2[mb_idx]['targets']} ****")
            result = self.env2[mb_idx]["targets"]

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
                    *fx.graph.map_arg(node.args, lambda n: self.env2[mb_idx][n.name]), \
                    **fx.graph.map_arg(node.kwargs, lambda n: self.env2[mb_idx][n.name]))

        elif node.op == 'call_method':
            #self_obj, *args = fx.graph.map_arg(node.args, lambda n: self.env2[mb_idx][n.name])
            #kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env2[mb_idx][n.name])
            #result = getattr(self_obj, node.target)(*args, **kwargs)
            arg0_b = node.args[0]
            arg0_a = self.env2[mb_idx][arg0_b.name]
            if isinstance(arg0_a, torch.Tensor):
                self_obj = arg0_a.detach().requires_grad_(arg0_a.requires_grad)
            else:
                self_obj = arg0_a

            flat_args = [self_obj, ]

            def extract_tensor_args(b):
                a = self.env2[mb_idx][b.name]
                nonlocal flat_args
                if isinstance(a, torch.Tensor):
                    val = a.detach().requires_grad_(a.requires_grad)
                    flat_args.append(val)
                    return val
                else:
                    flat_args.append(a)
                    return a

                return a

            args = fx.graph.map_arg(node.args[1:], extract_tensor_args)
            wargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)

            result = getattr(self_obj, node.target)(*args, **kwargs)

            self.fwd_cache2[mb_idx][node.name] = \
                    ( result if isinstance(result, tuple) else (result,), \
                    flat_args, )
            #print(f" --> call_method:  result:[{type(result)}], flat_args:{type(flat_args)}")

        elif node.op == 'call_module':
            #result = self.modules[node.target](\
            #        *fx.graph.map_arg(node.args, lambda n: self.env2[mb_idx][n.name]),\
            #        **fx.graph.map_arg(node.kwargs, lambda n: self.env2[mb_idx][n.name]))
            flat_args = []
            def extract_tensor_args(b):
                a = self.env2[mb_idx][b.name]
                nonlocal flat_args
                if isinstance(a, torch.Tensor):
                    #val = a.detach().requires_grad_(a.requires_grad)
                    val = a.detach().requires_grad_(True) ####
                    flat_args.append(val)
                    return val
                else:
                    flat_args.append(a)
                    return a

                return a

            args = fx.graph.map_arg(node.args, extract_tensor_args)
            kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)

            target_atoms = node.target.split('.')
            attr_itr = self.mod
            for i , atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(\
                            f"Node referenced nonexistant target{'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            submod = attr_itr
            result = submod(*args, **kwargs)

            if node.target == 'loss_fn':
                #print(f" node.target == 'loss_fn' --> {self.env2[mb_idx][str(node.all_input_nodes[0])]}")
                if not str(node.all_input_nodes[0]).startswith("target"):
                    self.output = self.env2[mb_idx][str(node.all_input_nodes[0])]
                self.grads2[mb_idx][node.name] = (None,)

            self.fwd_cache2[mb_idx][node.name] = \
                    ( result if isinstance(result, tuple) else (result,), \
                    flat_args, )

            if node.target == 'loss_fn':
                self.loss[mb_idx] = result
            #print(f" --> call_module:  result:[{type(result)}], flat_args:{type(flat_args)}")

        elif node.op == 'output':
            result = fx.graph.map_arg(node.args[0], lambda n: self.env2[mb_idx][n.name])

        self.env2[mb_idx][node.name] = result

        #
        print(f" ## run [rank:{self.rank}, micro#:{mb_idx}] - node:{node.name}, node.op:{node.op}")

        return result


    def fx_backward4(self, *args):
        print(f" -----> rank{self.rank}: in fx_backward4, args[0]:{args[0]}")

        for i in range(self.mbsize):
            result = self.fx_micro_backward(i)
            next(result)


    def fx_micro_backward(self, mb_idx):
    
        if self.rank < self.world_size - 1:
            target_node_name = "output"
            pre_split_rank = self.rank + 1
            #print(f"## rank:{self.rank}, receive grads from {pre_split_rank}, target_node_name:{target_node_name}")
            self.grads2[mb_idx][target_node_name] = self.receive_list(pre_split_rank)
    
    
        for node in reversed(self.graph.nodes):
            if node.op == 'output' and self.rank < self.world_size - 1 :
                target_node_name = str(node.all_input_nodes[0])
                self.grads2[mb_idx][target_node_name] = self.grads2[mb_idx]["output"]
                continue
    
            if node.op == 'output':
                continue
    
    
            if node.op == 'call_module' or node.op == 'call_method':
            #if node.op == 'call_module' or node.op == 'call_method' or node.op == 'call_function':
    
                def extract_tensor_args(b):
                    a = self.env[b.name]
                    if isinstance(a, torch.Tensor):
                        val = a.detach().requires_grad_(a.requires_grad)
                        return val
                    else:
                        return a
    
                args = ()
                kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)
    
                kwargs = dict(kwargs)
                k1, k2 = self.fwd_cache2[mb_idx].pop(node.name)
                #print(f" node.name:{node.name}, fwd_cache2[{mb_idx}].pop({node.name}) ---> k1:{k1}, k2:{k2}") # DEBUG
                kwargs["stage_output"] = k1
                kwargs["input_values"] = k2
    
                kwargs["output_grads"] = self.grads2[mb_idx][node.name]
                #print(f" node.name:{node.name}, grads[{node.name}] ---> {self.grads2[mb_idx][node.name]}") # DEBUG
                kwargs["outputs_with_grads_idxs"] = [0]
    
                result = stage_backward(*args, **kwargs)
    
                next_ = []
                #next_ = set([])
                self.get_destination(node.all_input_nodes, next_)
                #print(f" node.name:{node.name}, get_destination --> next_:{next_}... --> result:{result} ") # DEBUG
    
                cnt = len(result[0])
    
                for m in next_:
                    if cnt > 1:
                        self.grads2[mb_idx][m.name] = tuple(result[0])
                    else:
                        self.grads2[mb_idx][m.name] = result[0]
    
                continue
    
    
            if node.op == 'placeholder' and node.target == 'targets':
                continue
    
            if node.op == 'placeholder' and node.target != 'targets' and self.rank > 0:
                #target_node_name = "placeholder"
                target_node_name = str(node.target)
                next_split_rank = self.rank - 1
                #print(f" ---- fx_backward: got {str(node.target)}'s grads --> to be sent to rank:{next_split_rank}") # DEBUG
                #print(f"### rank:{self.rank} send grads to {next_split_rank}, target_node_name:{target_node_name}")
    
                obj = self.grads2[mb_idx][target_node_name]
                self.send_list(obj, next_split_rank)

        yield 0

sim_split = Simple_split_test()
sim_split.metadata_transfer()


# TEST ONLY
micro_batch_size = num_rank // 2
fx_run2 = FXRun2(sim_split, sim_split.device, mbsize=micro_batch_size)
print(f">>> micro batch size = {fx_run2.mbsize}")

fx_run2.print_range()

if sim_split.rank == 0:
    tick = time.time()

## t1.train()
## optimizer1 = Adam(t1.parameters(), lr=3e-5)

fx_run2.mod.train()
optimizer1 = Adam(fx_run2.mod.parameters(), lr=3e-5)

if fx_run2.rank == 0:
    #
    # move sample_input to first host
    #
    sample_output = torch.rand(batch_size, out_features)
    sample_input = torch.rand(batch_size, in_features)
else:
    sample_input = None
    sample_output = None


#
# move sample_output to last host
#
if fx_run2.rank == 0:
    target_node_name = "targets"
    mbatches = torch.chunk(sample_output, fx_run2.mbsize)
    for j in range(fx_run2.mbsize):
        fx_run2.env2[j][target_node_name] = mbatches[j]

    for j in range(fx_run2.mbsize):
        obj = fx_run2.env2[j][target_node_name]
        fx_run2.send_tensor(obj, fx_run2.world_size - 1)
    #
    print(f"sent ====> {sample_output}")

if fx_run2.rank == fx_run2.world_size - 1:
    target_node_name = "targets"
    for j in range(fx_run2.mbsize):
        fx_run2.env2[j][target_node_name] = fx_run2.receive_tensor(0)

    outputs = tuple(mb["targets"] for mb in fx_run2.env2)
    sample_output = torch.cat(outputs)
    #
    print(f"received <==== env2[{fx_run2.mbsize-1}][{target_node_name}]: {fx_run2.env2[fx_run2.mbsize-1][target_node_name]}")



for i in range(5):
    optimizer1.zero_grad()
    output1 = fx_run2.fx_forward4(sample_input, sample_output)
    loss1 = fx_run2.loss

    if sim_split.rank == sim_split.world_size - 1:
        print(f'Step {i}, Loss1:{sum(loss1) / fx_run2.mbsize}')

    fx_run2.fx_backward4(loss1)
    
    optimizer1.step()
    


if sim_split.rank == 0:
    tock=time.time()
    elapsed_time = tock - tick

    print('Time elapsed: %.3f sec ' % (elapsed_time))

if sim_split.rank == sim_split.world_size - 1:
    print(f"RANK:{sim_split.rank} ###################### output #############")
    output1 = fx_run2.make_output()
    print(output1)
    print(f"###################################")

print(f"[rank:{sim_split.rank}, run completed ...")

#rpc.shutdown()

