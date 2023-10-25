#
# Copyright (c) 2023-present, ETRI, All rights reserved.
#
#
#  This is a PoC that performs a GPipe-style pipeline-parallel training based on the FX IR.
#
#   In this PoC, FX compile generates FX IR,
#       and each process is responsible for a subset of the entire FX IR,
#       and pipeline parallel training is executed across N processes.
#
#   Micro-batch is supported in this PoC, and applied to a synthetic model (GPU version)
#
#
#  Sample Usage:
#      <machine #0>
#            torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0
#                  --master_addr="X.X.X.X" --master_port=29500 fx_dist_pp_dp_training_type-A_gpu_validate.py --pp_size=2 --dp_size=2
#      <machine #1>
#            torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1
#                  --master_addr="X.X.X.X" --master_port=29500 fx_dist_pp_dp_training_type-A_gpu_validate.py --pp_size=2 --dp_size=2
#

import torch
from torch import Tensor, Size
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

import logging

import argparse
import psutil
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torch.fx.graph_module import GraphModule

from torch.nn.parallel import DistributedDataParallel as DDP

#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.ERROR)


torch.manual_seed(42)

use_wrapper = False
#use_wrapper = True

#use_gpu = False
use_gpu = True

print_mem = True
#print_mem = False

#
# Total process count
#
#num_rank=N
num_rank=int(os.environ["WORLD_SIZE"])
#pp_size = 2
pp_size = None
#dp_size = num_rank // pp_size
dp_size = None

#micro_batch_size = num_rank // 2 # TODO
micro_batch_size = 2 # TODO

batch_size = 64
#batch_size = batch_size // dp_size

# reduced size for print
in_features = 4
out_features = 4
hidden = 2

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
    def forward(self, x, labels):
        out1 = self.module(x)
        return self.loss_fn(out1, labels)


gm = None

NoneType=type(None)

def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params

pid = os.getpid()
def print_memory_usage(str, print_flag):
    if print_flag == True:
        #print(" =========", str, "=========")
        my_process = psutil.Process(pid)
        usage = my_process.memory_info().rss / (1024 ** 3) # GB unit
        print(f" === {str} === rank:[{int(os.environ['RANK'])}] >> Memory Usage: {usage:.3f} GB")

class Simple_split_test(object):
    def __init__(self):
        self.initialize_comm()
        self.model_ir = []
        self.range_metadata = []

    def initialize_comm(self):

        if dist.is_initialized():
            logging.info("Communication already initialized")
            return


        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.master_addr = os.getenv("MASTER_ADDR")
        self.master_port = os.getenv("MASTER_PORT")
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.stage = 0

        if use_gpu == True:
            gpu_cnt = torch.cuda.device_count()
            if self.local_rank == 0:
                print(f"Available GPUs per server: {gpu_cnt}")
            if self.local_rank + 1 > gpu_cnt:
                logging.error(f"This program cannot create more processes than the number of available GPUs:{gpu_cnt}")
                sys.exit(1)

        #
        logging.info(f" --- rank:{self.rank}, local_rank:{self.local_rank}, world_size:{self.world_size}, master:{self.master_addr}, port:{self.master_port}")

        if use_gpu == True:
            self.backend = "nccl"
        else:
            self.backend = "gloo"
        init_method = "tcp://" + str(self.master_addr) + ":" + str(self.master_port)

        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size, init_method=init_method)

        #
        logging.info(f" --- rank:{dist.get_rank()}, world_size:{dist.get_world_size()}")

        #options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=10, rpc_timeout=30)

        #rpc.init_rpc(f"worker{self.rank}", rank=self.rank, world_size=self.world_size, rpc_backend_options=options,)

        #logging.debug(f" --- after init_rpc -- rank:{dist.get_rank()}, world_size:{dist.get_world_size()}")

        #rpc.shutdown()


    def simple_split(self, g: fx.Graph):

        length = g.nodes.__len__()

        mod_cnt = 0
        for n in g.nodes:
            if n.op == 'call_module':
                mod_cnt = mod_cnt + 1


        # simple assert
        assert mod_cnt >= pp_size, f"Model length:{length} is smaller than # of workers:{pp_size}"

        target_cnt = mod_cnt // pp_size
        logging.info(f"simple_split >> length:{length}, pp_size:{pp_size}, mod_cnt:{mod_cnt}, target_cnt:{target_cnt}")

        k, m_cnt, cnt = 0, 0, 0
        for n in g.nodes:
            if n.op == 'call_module':
                m_cnt = m_cnt + 1

            if m_cnt == target_cnt and k < pp_size-1:
                self.range_metadata.append((k, n.name))
                k = k + 1
                m_cnt = 0

            cnt = cnt + 1

            if cnt == length:
                break

        logging.debug(f" >>> cnt: {cnt}, k:{k}, n:{n.name}, mod_cnt:{mod_cnt}, target_cnt:{target_cnt}")
        if len(self.range_metadata) < pp_size:
            self.range_metadata.append((k, n.name))

        logging.info(f" ------------------------------------------------------------")
        logging.info(self.range_metadata)
        print(f"   range_metadata: {self.range_metadata}")
        #logging.info(f"   range_metadata: {self.range_metadata}")
        logging.info(f" ------------------------------------------------------------")



    # transfer range metadata to all processes
    def metadata_transfer(self):

        global gm

        print_memory_usage("Before model instancing", print_mem)

        if use_wrapper == True:
            t1 = TestModel()
            print_memory_usage("After model instancing", print_mem)

            loss_fn = torch.nn.MSELoss()
            wrapper = SimpleLossWrapper(t1, loss_fn)
            gm = fx.symbolic_trace(wrapper)

        else:
            t1 = TestModel()
            print_memory_usage("After model instancing", print_mem)
            gm = fx.symbolic_trace(t1)

        print_memory_usage("After fx symbolic_trace", print_mem)

        logging.info(f"------------------ FX graph --------------------------------")
        for n in gm.graph.nodes:
            logging.info(f"n.op:{n.op}, n.name:{n.name}, n.target:{n.target}, n.args:{n.args}, n.all_input_nodes:{n.all_input_nodes}")
        logging.info(f"------------------------------------------------------------")

        if use_gpu == True:
            self.device = torch.device(f"cuda:{self.local_rank}")
            # TO DELETE
            print(f"Using GPU ... cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        self.model_ir.append(gm)
        #self.stage = self.rank # TODO
        self.stage = self.rank // dp_size

        if self.rank == 0:
            self.simple_split(gm.graph)
            dist.broadcast_object_list(self.range_metadata, src=0, device=self.device)

            logging.info(f" >> worker:{self.rank} ==> range metadata {self.range_metadata} transfer to all other workers")

        else:
            for i in range(pp_size):
                self.range_metadata.append(None)
                
            dist.broadcast_object_list(self.range_metadata, src=0, device=self.device)
        
            logging.info(f" worker: {self.rank} <==  range metadata:{self.range_metadata} transferred")
            logging.info(f" -----------------------------------------------------")





# stage_backward function: cited from PiPPy
def stage_backward(
    stage_output,
    output_grads,
    input_values,
    outputs_with_grads_idxs: List[int],
):
    #logging.debug(f"** stage_backward ** stage_output:{stage_output}, output_grads:{output_grads}, input_values:{input_values}, outputs_with_grads_idxs: {outputs_with_grads_idxs}")

    stage_output_with_grads = [
        stage_output[i] for i in outputs_with_grads_idxs
    ]
    output_grads_with_grads = [
        output_grads[i] for i in outputs_with_grads_idxs
    ]

    stage_output_tensors = []
    output_grad_tensors = []

    #no_tensor = False

    def extract_tensors_with_grads(output_val, grad_val):
        #nonlocal no_tensor
        if isinstance(output_val, torch.Tensor):
            if not output_val.requires_grad and output_val.grad_fn is None:
                logging.warning(f" ---------------- {output_val}: not requirs_grad and grad_fn None")
                return
            stage_output_tensors.append(output_val)
            output_grad_tensors.append(grad_val)
        elif isinstance(output_val, (tuple, list)):
            if grad_val is None:
                logging.warning(f" ---------------- {grad_val}: is None")
                return
            for ov, gv in zip(output_val, grad_val):
                extract_tensors_with_grads(ov, gv)
        elif isinstance(output_val, dict):
            if grad_val is None:
                logging.warning(f" ---------------- {grad_val}: is None")
                return
            for k in output_val.keys():
                extract_tensors_with_grads(output_val[k], grad_val[k])
        else:
            #no_tensor = True
            logging.critical(f"... ignored in this case")


    extract_tensors_with_grads(stage_output_with_grads, output_grads_with_grads)

    #if no_tensor == True:
    #    return ((None,), None)

    if len(stage_output_tensors) == 0:
        logging.debug(f">> stage_output_tensors == []")
        return ((None,), None)

    if stage_output_tensors[0] != None and output_grad_tensors[0] != None and stage_output_tensors[0].shape != output_grad_tensors[0].shape:
        logging.debug(f">> stage_backward ** stage_output_tensors-shape:{stage_output_tensors[0].shape}")
        logging.debug(f">> stage_backward ** output_grad_tensors-shape:{output_grad_tensors[0].shape}")
        stage_output_tensors[0] = stage_output_tensors[0].view(-1, stage_output_tensors[0].size(-1))
        logging.debug(f">> stage_backward ==> stage_output_tensors[0]--shape:{stage_output_tensors[0].shape}")
        logging.debug(f">> stage_backward ==> output_grad_tensors-shape:{output_grad_tensors[0].shape}")

    torch.autograd.backward(stage_output_tensors, grad_tensors=output_grad_tensors)

    grad_inputs = []
    for val in input_values:
        if isinstance(val, torch.Tensor):
            grad_inputs.append(val.grad)
        else:
            grad_inputs.append(None)

    barrier_token = None
    return grad_inputs, barrier_token


import gc

class FXRun2:

    def __init__(self, split_info: Simple_split_test, device, mbsize): # TODO

        self.mod = split_info.model_ir[0]
        self.ddp_mod = self.mod
        self.graph = self.mod.graph
        #self.modules = dict(self.mod.named_modules())
        self.mbsize = mbsize  
        self.env2: List[Dict[str, Node]] = [{} for _ in range(mbsize)]
        self.range_metadata = split_info.range_metadata
        self.rank = split_info.rank
        self.local_rank = split_info.local_rank
        self.world_size = split_info.world_size
        self.device = device
        self.stage = split_info.stage
        self.loss: List[Any] = [None for _ in range(mbsize)]
        self.fwd_cache2: List[Dict[str, Tuple[Any, List[torch.Tensor]]]] = [{} for _ in range(mbsize)]
        self.grads2: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        #self.window_size = 3 # TODO default = 0  # TO DELETE
        self.special_nodes: Dict[str, Tuple[int, int]] = {}  # { node_name : {rank#, needed-by-rank#),}

        # TODO
        self.ds_type2id = {
            Tensor: 100,
            tuple: 101,
            list: 102, 
            Size: 103, 
            int: 104, 
            NoneType: 105,
            type: 106, }


        self.ds_id2type = {v:k for k, v in self.ds_type2id.items()}
        
        self.tensor_type2id = {
            torch.float32: 0,
            torch.float64: 1,
            torch.complex64: 2,
            torch.complex128: 3,
            torch.float16: 4,
            torch.bfloat16: 5,
            torch.uint8: 6,
            torch.int8: 7,
            torch.int16: 8,
            torch.int32: 9,
            torch.int64: 10,
            torch.bool: 11, }

        self.tensor_id2type = {v:k for k,v in self.tensor_type2id.items()}

        if self.rank == 0:
            for stage in reversed(range(1, self.world_size // dp_size)):
                self.cross_reference_analyze(stage, self.graph)

            special_nodes_obj = [self.special_nodes]
            dist.broadcast_object_list(special_nodes_obj, src=0, device=self.device)

        else:
            special_nodes_obj = [None]
            dist.broadcast_object_list(special_nodes_obj, src=0, device=self.device)
            self.special_nodes = special_nodes_obj[0]

        logging.info(f" *********** rank:{self.rank} ==> cross-referenced nodes *****************")
        print(f" special_nodes: {self.special_nodes}")
        #logging.info(f" special_nodes: {self.special_nodes}")
        logging.info(f" *************************************************************************")

        print_memory_usage("Before clean_unused_modules", print_mem)

        # remove unused modules
        self.clean_unused_modules()

        gc.collect()
        print_memory_usage("After clean_unusd_modules", print_mem)

        if use_gpu == True:
            #fx_run2.prepare_module()
            self.mod.to(self.device) # move memory cleaned modules to GPU

        if dp_size > 1:
            self.activate_dp()


    def activate_dp(self):
        ddp_group = []

        for i in range(0, pp_size):
            start_rank = i * dp_size
            end_rank = (i+1) * dp_size
            dp_group = list(range(start_rank, end_rank))
            if self.rank in dp_group:
                print(self.rank, dp_group)
                ddp_group = dist.new_group(dp_group)
            else:
                dist.new_group(dp_group)

        self.ddp_mod = DDP(self.mod, process_group=ddp_group, find_unused_parameters=True)
        if isinstance(self.ddp_mod, torch.nn.parallel.DistributedDataParallel):
            self.mod = self.ddp_mod.module
            self.graph = self.mod.graph

    #
    def delete_param(self, mod, name):
        for param_name, param in mod.named_parameters():
            t = getattr(mod, param_name)
            setattr(mod, param_name, None)
            del t
            #print(f" ----- del param : {mod} --> {param_name}")

    #
    def has_child(self, mod):
        num_children = 0
        if mod is not None:
            num_children = len(list(mod.children()))
        return num_children

    #
    def partially_matched(self, name):
        for string in self.reference:
            if name in string:
                #print(f" >>>>> partially matched [{name}] in [{string}]")
                return True
        return False

    #
    #def delete_module(self, module_node):
    def delete_module(self, root, module_node):
        module_name = module_node.name
        module_target_name = str(module_node.target)
        if len(module_name) > len(module_target_name):
            self.reference.append(module_target_name)

        get_attr_flag = False
        if len(module_name) == len(module_target_name) and not self.partially_matched(module_target_name):
            target_atoms = module_target_name.split('.')
            #attr_itr = self.mod
            attr_itr = root
            if module_node.op == 'get_attr':
                if target_atoms[-1] == "weight" or target_atoms[-1] == "bias":
                    #print(f"> [{target_atoms}]")
                    target_atoms = target_atoms[:-1]
                    #print(f">  >  [{target_atoms}]")
                    get_attr_flag = True
                    if self.partially_matched(str('.'.join(target_atoms))):
                        return
                else:
                    return

            for i , atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(\
                            f"Node referenced nonexistant target {'.'.join(target_atoms[:i])} ... [{target_atoms}]")
                parent_ = attr_itr
                attr_itr = getattr(attr_itr, atom)

            #if self.has_child(attr_itr) == 0:
            if get_attr_flag == True or self.has_child(attr_itr) == 0:
                self.delete_param(attr_itr, atom)
                delattr(parent_, atom)
                del attr_itr
                #print(f">>> rank:{self.rank}, delete module:target:{module_target_name} ## atom:{atom},op:{module_node.op}")
                self.reference.append(module_target_name)
                #print(f"## reference: {self.reference} ")
            #else:
            #    print(f">>> delete_module:{module_name} requested, but {module_name} has child")

    #
    def clean_unused_modules(self):

        self.reference = []

        print(f"*** [rank:{self.rank}] clean_unused_modules *******")

        #for rank in reversed(range(self.world_size)):
        for stage in reversed(range(pp_size)):

             if stage != self.stage:
                 from_, to_ = self.get_range(stage, self.graph)

                 cur = to_
                 while cur != from_:
                     #if cur.op == 'call_module':
                     if cur.op == 'call_module' or cur.op == 'get_attr':
                         #self.delete_module(cur)
                         self.delete_module(self.mod, cur)
                     cur = cur._prev
                 #if cur.op == 'call_module':
                 if cur.op == 'call_module' or cur.op == 'get_attr':
                     #self.delete_module(cur)
                     self.delete_module(self.mod, cur)

        print(f"**********************************")
        self.reference = None

    def get_prev_rank(self, rank, stage):
        # TODO: for TP, need update for rank/stage (e.g., list)
        assert stage != 0, f"no prev rank {rank}, {stage}"
        prev_rank = (stage - 1) * dp_size + self.get_first_rank(rank)
        return prev_rank

    def get_next_rank(self, rank, stage):
        # TODO: for TP, need update for rank/stage (e.g., list)
        assert stage != pp_size-1, f"no next rank {rank}, {stage}"
        next_rank = (stage + 1) * dp_size + self.get_first_rank(rank)
        return next_rank

    def get_first_rank(self, rank):
        first_rank = rank % dp_size
        return first_rank

    def get_last_rank(self, rank):
        last_rank = (pp_size-1)*dp_size + rank % dp_size
        return last_rank

    # analyze IR graph and find the cross-layer referenced nodes
    def cross_reference_analyze(self, stage, g:fx.Graph):

        if stage == 0:
            return

        from_, to_ = self.get_range(stage, g)

        logging.debug(f" ***** stage:{stage} >>  from_:{from_.name}, to_:{to_.name}")

        cur = to_
        while (cur != from_) or (stage > 0 and cur == from_):

            # in process check - backward direction

            #for _, target_ in enumerate(cur.all_input_nodes):
            for i, target_ in enumerate(cur.all_input_nodes):
                # DEBUG
                if ((use_wrapper == False and cur.op == 'output') or (use_wrapper == True and cur.name == "loss_fn")) and i > 0:
                    break
                referenced_in = False
                referenced_out = False

                inner = cur._prev
                if inner != from_._prev:
                    while (inner != from_) or (stage > 0 and inner == from_):
                        if inner.name == target_.name:
                            #logging.debug(f" [cross_reference_analyze] ({target_.name}) referenced in current rank:{rank} !")
                            referenced_in = True
                            break

                        if inner == from_:
                            break

                        inner = inner._prev

                if referenced_in == True:
                    continue

                if referenced_in == False:

                    # output process check - forward direction

                    stage_ = 0
                    split_node_name = self.range_metadata[stage_][1]

                    for k in g.nodes:
                        first_node = k
                        break

                    outer = first_node
                    while outer != from_: 
                        # DEBUG
                        if outer.name == target_.name:
                            logging.info(f" [cross_reference_analyze] ({target_.name}) referenced in outer stage:{stage_} !!")

                            self.special_nodes[target_.name] = (stage_, stage)  # { node_name : {rank#, needed-by-rank#),}
                            referenced_out = True
                            break

                        if outer.name == split_node_name:
                            stage_ = stage_ + 1
                            split_node_name = self.range_metadata[stage_][1]

                        outer = outer._next


                if referenced_out == False:
                    logging.critical(f"[Error] cannot handle this case: {target_.name} !!!")
                    sys.exit(1)

            if cur == from_:
                break

            cur = cur._prev



    def get_range(self, stage, g:fx.Graph) -> (Node, Node):

        logging.info(f"stage:{stage} range_metadata: {self.range_metadata}")

        if stage == 0:
            from_node_name = "-1"
            for n in g.nodes:
                if n.op == 'placeholder':
                    from_node_name = n.name
                    logging.debug(f">>>> get_range: n.op == 'placeholder' --> from_node_name:{from_node_name}")
                break
        else:
            from_node_name = self.range_metadata[stage-1][1]

        to_node_name = self.range_metadata[stage][1]

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

        if stage == 0:
            return (from_node, to_node)
        else:
            return (from_node._next, to_node)


    def print_range(self):
        from_, to_ = self.get_range(self.stage, self.graph)

        logging.info(f"# print_range ==> stage = {self.stage}, from_:{from_.name}, to_:{to_.name}")

        cur = from_ # first node assigned to the process#{rank} in range_metadata
        while cur != to_:
            logging.info(f" ---- node:{cur.name}")
            cur = cur._next
        logging.info(f" ---- node:{cur.name}")  # last node assigned to the process#{rank} in range_metadata
        logging.info(f" -------------------------------")

    def prepare_module(self):
        if use_gpu == False:
            return
        from_, to_ = self.get_range(self.stage, self.graph)
        logging.info(f"# prepare_module ==> stage = {self.stage}, from_:{from_.name}, to_:{to_.name}")

        node = from_ # first node assigned to the process#{rank} in range_metadata
        while node != to_:
            logging.info(f" ---- node:{node.name}")
            if node.op == 'call_module':
                target_atoms = node.target.split('.')
                attr_itr = self.mod.module
                for i , atom in enumerate(target_atoms):
                    if not hasattr(attr_itr, atom):
                        raise RuntimeError(\
                            f"Node referenced nonexistant target{'.'.join(target_atoms[:i])}")
                    attr_itr = getattr(attr_itr, atom)
                submod = attr_itr

                submod.to(self.device)
                # TO DELETE
                print(f" module:{submod} --> moved to GPU({self.device})")

            node = node._next

        logging.info(f" ---- node:{node.name}")  # last node assigned to the process#{rank} in range_metadata
        if node.op == 'call_module':
            target_atoms = node.target.split('.')
            attr_itr = self.mod.module
            for i , atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(\
                        f"Node referenced nonexistant target{'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            submod = attr_itr

            submod.to(self.device)
            # TO DELETE
            print(f" module:{submod} --> moved to GPU({self.device})")


    # TODO
    def receive_data(self, from_rank):
        ds_type = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.recv(ds_type, from_rank)

        ds_type = self.ds_id2type[ds_type.item()]
        
        #if str(ds_type) == "<class 'Tensor'>":
        if ds_type is Tensor:
            return self.receive_tensor(from_rank)
        #elif str(ds_type) == "<class 'tuple'>":
        elif ds_type is tuple:
            return self.receive_tuple(from_rank)
        #elif str(ds_type) == "<class 'list'>":
        elif ds_type is list:
            return self.receive_list(from_rank)
        elif ds_type is Size:
            return self.receive_size(from_rank)
        elif ds_type is int:
            return self.receive_int(from_rank)
        elif ds_type is set:
            return self.receive_set(from_rank)
        elif ds_type is NoneType:
            return self.receive_none(from_rank)
        elif ds_type is type:
            return self.receive_type(from_rank)
        else:
            logging.critical(f"#### receive_data: not supported type!")
        # TODO


    # TODO
    def send_data(self, obj, to_rank):
        ds_type = self.ds_type2id[type(obj)]
        ds_type = torch.tensor(ds_type, dtype=torch.long, device=self.device)
        dist.send(ds_type, to_rank)

        if isinstance(obj, torch.Tensor):
            self.send_tensor(obj, to_rank)
        elif isinstance(obj, tuple):
            self.send_tuple(obj, to_rank)
        elif isinstance(obj, list):
            self.send_list(obj, to_rank)
        elif isinstance(obj, Size):
            self.send_size(obj, to_rank)
        elif isinstance(obj, int):
            self.send_int(obj, to_rank)
        elif isinstance(obj, set):
            self.send_set(obj, to_rank)
        elif obj is None:
            self.send_none(obj, to_rank)
        elif isinstance(obj, type):
            self.send_type(obj, to_rank)
        else:
            logging.critical(f"#### send_data: not supported type!")

        #TODO

    def receive_set(self, from_rank):
        return set(self.receive_list(from_rank))

    def send_set(self, obj, to_rank):
        self.send_list(list(obj), to_rank)

    def receive_int(self, from_rank):
        int_data = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.recv(int_data, from_rank)
        return int_data.item()

    def send_int(self, obj, to_rank):
        int_data = torch.tensor([obj], dtype=torch.long, device=self.device) # ex. 2
        dist.send(int_data, to_rank)


    def receive_size(self, from_rank):
        return Size(self.receive_list(from_rank))

    def send_size(self, obj, to_rank):
        self.send_list(list(obj), to_rank)

    def receive_tuple(self, from_rank):
        return tuple(self.receive_list(from_rank))

    def send_tuple(self, obj, to_rank):
        self.send_list(list(obj), to_rank)

    def send_none(self, obj, to_rank):
        logging.debug(f"send_none")

    def receive_none(self, from_rank):
        return None

    def send_type(self, obj, to_rank):
        type_data = torch.tensor([self.ds_type2id[type(obj)]], dtype=torch.long, device=self.device) # ex. 2
        dist.send(type_data, to_rank)

    def receive_type(self, from_rank):
        type_data = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.recv(type_data, from_rank)
        return self.ds_id2type[type_data.item()]

    def receive_tensor(self, from_rank):
        dimension = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.recv(dimension, from_rank)
        #logging.debug(f" >>>>> recv_tensor, dimension:{dimension} from rank:{from_rank}")

        shape = torch.tensor([0] * dimension.item(), dtype=torch.long, device=self.device)
        dist.recv(shape, from_rank)
        #logging.debug(f" >>>>> recv_tensor, shaple:{shape} from rank:{from_rank}")
        shape = tuple(shape.tolist())

        ttype = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.recv(ttype, from_rank)
        #logging.debug(f" >>>>> recv_tensor, ttype:{ttype} from rank:{from_rank}")

        ttype = self.tensor_id2type[ttype.item()]

        obj = torch.zeros(size=shape, dtype=ttype, device=self.device)
        dist.recv(obj, from_rank)
        #logging.debug(f" >>>>> recv_tensor, obj:{obj} from rank:{from_rank}")

        return obj

    def send_tensor(self, obj, to_rank):
        if isinstance(obj, torch.Tensor):
            obj_size = obj.size()
            dimension = torch.tensor(len(obj_size), dtype=torch.long, device=self.device) # ex. 2
            logging.debug(f" >>>>> send_tensor, obj.size():{obj_size}, len:{len(obj_size)}, dimension:{dimension}")
        dist.send(dimension, to_rank)

        if isinstance(obj, torch.Tensor):
            shape = torch.tensor(list(obj_size), dtype=torch.long, device=self.device) # ex. [54, 5120]
        dist.send(shape, to_rank)

        ttype = self.tensor_type2id[obj.dtype]
        ttype = torch.tensor(ttype, dtype=torch.long, device=self.device)
        dist.send(ttype, to_rank)
        #logging.debug(f" >>>>> send_tensor, ttype:{ttype}")

        if not obj.is_contiguous():
            obj = obj.contiguous()
            #logging.debug(f" >>> obj made to be contiguous")

        obj = obj.to(self.device)
        dist.send(obj, to_rank)
        #logging.debug(f" >>>>> send_tensor, obj:{obj}")

    def receive_list(self, from_rank):
        length = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.recv(length, from_rank)

        obj = []
        for _ in range(length.item()):
            n = self.receive_data(from_rank)
            obj.append(n)

        return obj

    def send_list(self, obj, to_rank):
        length = torch.tensor(len(obj), dtype=torch.long, device=self.device)
        dist.send(length, to_rank)

        for n in obj:
            self.send_data(n, to_rank)


    def fx_forward4(self, *args):
        #logging.debug(f" -----> rank{self.rank}: in fx_forward4, args[0]:{args[0]}")
        self.args_iter = iter(args)

        if self.stage == 0:
            #for n in self.mod.graph.nodes:
            for n in self.graph.nodes:
                if (use_wrapper == True and n.op == 'placeholder' and self.stage == 0 and n.name == 'x') or \
                        (use_wrapper == False and n.op == 'placeholder' and self.stage == 0 and n.name == 'x'):
                    input = next(self.args_iter)

                    #logging.debug(f">>>>> input:{input}, mbsize:{self.mbsize}")
                    if isinstance(input, torch.Tensor):
                        mbatches = torch.chunk(input, self.mbsize)
                        if self.mbsize == 1:
                            #self.env2[0]["placeholder"] = input
                            input = input.to(self.device)
                            self.env2[0]["placeholder"] = input
                        else:
                            for j in range(self.mbsize):
                                #self.env2[j]["placeholder"] = mbatches[j]
                                mbatch = mbatches[j].to(self.device)
                                self.env2[j]["placeholder"] = mbatch

                    else:
                        logging.critical(f"### input:{input} not Tensor --> currently not supported!!")
                        sys.exit(1)
                    break

        #logging.debug(f" * rank:{self.rank}, in run_micro_batch_forward() ..")

        result = None
        for i in range(self.mbsize):
            if dp_size > 1:
                with self.ddp_mod.no_sync():
                    result = self.fx_micro_forward(i)
            else:
                result = self.fx_micro_forward(i)
            next(result)


    def get_last_module(self):
        if self.rank == self.world_size - 1:
            from_, to_ = self.get_range(self.stage, self.graph)

            cur = to_
            while cur != from_:
                if cur.op == 'call_module' and cur.target != 'loss_fn':
                    logging.debug(f"[Rank:{self.rank}] ==> got last module: {cur.name}")
                    return cur.name

                cur = cur._prev


    def make_output(self):
        output = None
        if self.rank ==  self.world_size - 1:
            #target = "output"
            target = self.get_last_module()
            outputs = tuple(mb[target] for mb in self.env2) 
            print(f" ---> RANK: {self.rank},  outputs = {outputs}, type(output):{type(outputs)}")
            output = torch.cat(outputs)

        return output

    #def rewind(self, cur_node, window_size):
    #    i = window_size
    #    cur_ = cur_node
    #
    #    while i > 0:
    #        cur_ = cur_._prev
    #        i = i - 1
    #
    #    return cur_

    #def fastfwd(self, cur_node, window_size):
    #    i = window_size
    #    cur_ = cur_node
    #
    #    while i > 0:
    #        cur_ = cur_._next
    #        i = i - 1
    #
    #    return cur_


    def fx_micro_forward(self, mb_idx):

        from_, to_ = self.get_range(self.stage, self.graph)
        logging.info(f"## rank:{self.rank}, mb_idx:{mb_idx} world_size:{self.world_size}, from_:{from_.name}, to_:{to_.name}")

        if self.stage > 0:
            pre_split_rank = self.get_prev_rank(self.rank, self.stage)

            cur = from_._prev

            begin_ = cur
            #cur = self.rewind(cur, self.window_size)
            #while cur != begin_:         # rewind case
            #    node_name = cur.name
            #    pre_split_rank = self.rank - 1
            #    logging.info(f"## rank:{self.rank}, receive activation from {pre_split_rank}, node_name:{node_name}")
            #    self.env2[mb_idx][node_name] = self.receive_data(pre_split_rank)
            #    cur = cur._next

            for node_name, range_ in self.special_nodes.items():
                src_stage, needed_by_stage = range_
                if self.stage > src_stage and self.stage <= needed_by_stage:
                    logging.info(f"### stage:{self.stage}, receive cross_ref activation from {pre_split_rank}, node_name:{node_name}")
                    self.env2[mb_idx][node_name] = self.receive_data(pre_split_rank)
                    if isinstance(self.env2[mb_idx][node_name], torch.Tensor):
                        if not self.env2[mb_idx][node_name].requires_grad or self.env2[mb_idx][node_name].grad_fn is None:
                            self.env2[mb_idx][node_name].requires_grad_(True)
                            logging.info(f" ###### node name:{node_name} requires_grad(True) #####") 

            if cur == begin_:
                node_name = cur.name     # cur = from_._prev
                #pre_split_rank = self.rank - 1
                if node_name not in self.special_nodes:
                    logging.info(f"## stage:{self.stage}, receive activation from {pre_split_rank}, node_name:{node_name}")
                    self.env2[mb_idx][node_name] = self.receive_data(pre_split_rank)
                    if isinstance(self.env2[mb_idx][node_name], torch.Tensor):
                        if not self.env2[mb_idx][node_name].requires_grad or self.env2[mb_idx][node_name].grad_fn is None:
                            self.env2[mb_idx][node_name].requires_grad_(True)
                            logging.info(f" ###### ### node name{node_name} requires_grad(True) #####") 

        #if self.rank == 0:
        #    for n in self.graph.nodes:
        #        cur = n
        #        break
        #else:
        #    cur = from_
        cur = from_
        while cur != to_:
            self.fx_ir_run_node2(cur, mb_idx)
            cur = cur._next
        result = self.fx_ir_run_node2(cur, mb_idx)

        logging.info(f" stage:{self.stage}, cur.node name:{cur.name}, split_node_name:{to_.name}")

        if self.rank < self.get_last_rank(self.rank):
            next_split_rank = self.get_next_rank(self.rank, self.stage)
            begin_ = cur

            for node_name, range_ in self.special_nodes.items():
                src_stage, needed_by_stage = range_
                if self.stage >= src_stage and self.stage < needed_by_stage:
                    obj = self.env2[mb_idx][node_name]
                    self.send_data(obj, next_split_rank)
                    logging.info(f"### stage:{self.stage} send cross_ref activation to {next_split_rank}, node_name:{node_name}")

            #cur = self.rewind(cur, self.window_size)
            #while cur != begin_:           # rewind case
            #    node_name = cur.name
            #    next_split_rank = self.rank + 1
            #    logging.info(f"### rank:{self.rank} send activation to {next_split_rank}, node_name:{node_name}")
            #    obj = self.env2[mb_idx][node_name]
            #    self.send_data(obj, next_split_rank)
            #    cur = cur._next

            if cur == begin_:
                node_name = cur.name       # cur = to_
                #next_split_rank = self.rank + 1
                if node_name not in self.special_nodes:
                    logging.info(f"### rank:{self.rank} send activation to {next_split_rank}, node_name:{node_name}")
                    obj = self.env2[mb_idx][node_name]
                    self.send_data(obj, next_split_rank)

        yield result



    #def restore_env(self, node: Node) -> Tuple[Tuple, Dict]:
    #    #logging.info(f"## before restore_env, node:{node}, node.args:{node.args}, node.kwargs:{node.kwargs}")
    #
    #    args = fx.graph.map_arg(node.args, lambda n: self.env[n.name])
    #    assert isinstance(args, tuple)
    #
    #    kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name])
    #    assert isinstance(kwargs, dict)
    #
    #    #logging.info(f">>> after restore_env, node:{node}, node.name:{node.name}, args:{args}, kwargs:{kwargs}")
    #
    #    return args, kwargs
        

    def fx_ir_run_node2(self, node, mb_idx):

        #args, kwargs = self.restore_env(node)

        result = Any

        if (use_wrapper == True and node.op == 'placeholder' and self.stage == 0 and node.name == 'x') or \
                (use_wrapper == False and node.op == 'placeholder' and self.stage == 0 and node.name == 'x'):
            logging.debug(f" ------- [{node.op}] node.name:{node.name}, node.target:{node.target}")
            #result = next(self.args_iter)
            result = self.env2[mb_idx]["placeholder"]
        
        elif node.op == 'get_attr':
            logging.info(f" ------- [{node.op}] node.name:{node.name}, node.target:{node.target}")
            target_atoms = node.target.split('.')
            attr_itr = self.mod
            for i , atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(\
                            f"Node referenced nonexistant target{'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            result = attr_itr

        elif node.op == 'call_function':
            #logging.info(f" ------- [{node.op}] node.name:{node.name}, node.target:{node.target}")
            #result = node.target(\
            #        *fx.graph.map_arg(node.args, lambda n: self.env2[mb_idx][n.name]), \
            #        **fx.graph.map_arg(node.kwargs, lambda n: self.env2[mb_idx][n.name]))
            flat_args = []
            def extract_tensor_args(b):
                a = self.env2[mb_idx][b.name]
                nonlocal flat_args
                if isinstance(a, torch.Tensor):
                    val = a.detach().to(self.device)
                    val.requires_grad_(a.requires_grad)
                    flat_args.append(val)
                    # DEBUG
                    #logging.debug(f" >>>>>>>>>>>> call_function[node.name={node.name}] a is Tensor:{a}")
                    return val
                else:
                    flat_args.append(a)
                    #logging.debug(f" >>>>>>>>>>>>>> call_function[node.name={node.name}] a is not Tensor:{a}")
                    return a
                return a
            
            args = fx.graph.map_arg(node.args, extract_tensor_args)
            kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)
            
            #logging.debug(f" --> call_function[node.name:{node.name}:  args:{args}, kwargs:{kwargs}")
            result = node.target(*args, **kwargs)
            
            self.fwd_cache2[mb_idx][node.name] = \
                    ( result if isinstance(result, tuple) else (result,), \
                    flat_args, )
            #logging.debug(f" --> call_function:  result:[{type(result)}], flat_args:{type(flat_args)}")

        elif node.op == 'call_method':
            #logging.info(f" ------- [{node.op}] node.name:{node.name}, node.target:{node.target}")
            #self_obj, *args = fx.graph.map_arg(node.args, lambda n: self.env2[mb_idx][n.name])
            #kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env2[mb_idx][n.name])
            #result = getattr(self_obj, node.target)(*args, **kwargs)

            arg0_b = node.args[0]
            arg0_a = self.env2[mb_idx][arg0_b.name]
            if isinstance(arg0_a, torch.Tensor):
                self_obj = arg0_a.detach().to(self.device)
                self_obj.requires_grad_(arg0_a.requires_grad)
            else:
                self_obj = arg0_a
            
            flat_args = [self_obj, ]
            
            def extract_tensor_args(b):
                a = self.env2[mb_idx][b.name]
                nonlocal flat_args
                if isinstance(a, torch.Tensor):
                    val = a.detach().to(self.device)
                    val.requires_grad_(a.requires_grad)
                    flat_args.append(val)
                    return val
                else:
                    flat_args.append(a)
                    return a
            
                return a
            
            args = fx.graph.map_arg(node.args[1:], extract_tensor_args)
            kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)
            
            logging.debug(f" ----------- node.name:{node.name}, self_obj:{self_obj}, type(self_obj): {type(self_obj)} -----")
            result = getattr(self_obj, node.target)(*args, **kwargs)
            
            self.fwd_cache2[mb_idx][node.name] = \
                    ( result if isinstance(result, tuple) else (result,), \
                    flat_args, )
            #logging.debug(f" --> call_method:  result:[{type(result)}], flat_args:{type(flat_args)}")


        elif node.op == 'call_module':
            #logging.info(f" ------- [{node.op}] node.name:{node.name}, node.target:{node.target}")
            #result = self.modules[node.target](\
            #        *fx.graph.map_arg(node.args, lambda n: self.env2[mb_idx][n.name]),\
            #        **fx.graph.map_arg(node.kwargs, lambda n: self.env2[mb_idx][n.name]))

            flat_args = []
            def extract_tensor_args(b):
                a = self.env2[mb_idx][b.name]
                nonlocal flat_args

                if isinstance(a, torch.Tensor) and a.is_floating_point():
                    val = a.detach().to(self.device)
                    val.requires_grad_(a.requires_grad)
                    flat_args.append(val)
                    return val
                else:
                    flat_args.append(a)
                    return a

                return a

            target_atoms = node.target.split('.')
            attr_itr = self.mod
            for i , atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(\
                            f"Node referenced nonexistant target{'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            submod = attr_itr

            if use_wrapper == True and node.target == 'loss_fn':
                #key_ = node.args[0]['logits']
                #output1 = self.env2[mb_idx][str(key_)]
                #target1 = self.env2[mb_idx]["labels"]
                #
                #output1 = output1.view(-1, output1.size(-1))
                #target1 = target1.view(-1)
                #
                #kwargs = {}
                #
                #if isinstance(output1, torch.Tensor) and output1.is_floating_point():
                #    output1 = output1.detach().requires_grad_(output1.requires_grad)
                #    flat_args.append(output1)
                #    output1.grad = None 
                #
                #    #if not output1.is_leaf:
                #    #    output1.retain_grad()
                #
                #    logging.debug(f" >>> output1:{output1} ")
                #else:
                #    flat_args.append(output1)
                #    logging.debug(f" >>>> output1:{output1} ")
                #
                #if isinstance(target1, torch.Tensor) and target1.is_floating_point():
                #    target1 = target1.detach().requires_grad_(target1.requires_grad)
                #    flat_args.append(target1)
                #else:
                #    flat_args.append(target1)
                #
                #myargs = [None, None]
                #myargs[0] = output1
                #myargs[1] = target1
                #myargs = tuple(myargs)
                #
                #result = submod(*myargs, **kwargs)
                args = fx.graph.map_arg(node.args, extract_tensor_args)
                kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)
                result = submod(*args, **kwargs)


                #logging.debug(f" In forward --> node.target=='loss_fn' ==> result:{result}")
                logging.debug(f" ------- [{node.op}] node.name:{node.name}, node.target:{node.target}, *** shape:{result.shape}")

                if not str(node.all_input_nodes[0]).startswith("labels"):
                    self.output = self.env2[mb_idx][str(node.all_input_nodes[0])]
                self.grads2[mb_idx][node.name] = (None,)
                self.loss[mb_idx] = result

                self.fwd_cache2[mb_idx][node.name] = \
                    ( result if isinstance(result, tuple) else (result,), \
                    flat_args, )

            else:

                args = fx.graph.map_arg(node.args, extract_tensor_args)
                kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)
                result = submod(*args, **kwargs)
                logging.debug(f" ------- [{node.op}] node.name:{node.name}, node.target:{node.target}, *** shape:{result.shape}")

                self.fwd_cache2[mb_idx][node.name] = \
                    ( result if isinstance(result, tuple) else (result,), \
                    flat_args, )

            #logging.debug(f" --> call_module:  node.name:{node.name}, fwd_cache2[{mb_idx}][{node.name}] set !!")

        elif node.op == 'output':
            logging.info(f" ------- [{node.op}] node.name:{node.name}, node.target:{node.target}")
            if use_wrapper == True:
                result = fx.graph.map_arg(node.args[0], lambda n: self.env2[mb_idx][n.name])

            # TODO: Experimental
            elif use_wrapper == False:  #  loss_fn (output nodes's args[0],  mbsize-considered "labels")
                logging.info(f" ************ use_wrapper == False *****************")
                #logging.debug(f"node.args[0]['logits']: {node.args[0]['logits']}")
                #logging.debug(f">>>> p:{node.op}, name:{node.name}, target:{node.target}, args:{node.args}, all_input_nodes:{node.all_input_nodes}")
        
                #key_ = node.args[0]['logits']
                #output1 = self.env2[mb_idx][str(key_)]

                output1_ = self.env2[mb_idx][str(node.args[0])]
                target1_ = self.env2[mb_idx]["labels"]

                logging.debug(f" ########## output1: {output1_}, ######### target1:{target1_}")

                #output1 = output1_.view(-1, output1_.size(-1))
                #target1 = target1_.view(-1)

                kwargs = {}

                flat_args = []
                if isinstance(output1_, torch.Tensor) and output1_.is_floating_point():
                    output1 = output1_.detach().to(self.device)
                    output1.requires_grad_(output1_.requires_grad)
                    flat_args.append(output1)
                else:
                    output1 = output1_
                    flat_args.append(output1)
                
                if isinstance(target1_, torch.Tensor) and target1_.is_floating_point():
                    target1 = target1_.detach().to(self.device)
                    target1.requires_grad_(target1_.requires_grad)
                    flat_args.append(target1)
                else:
                    target1 = target1_
                    flat_args.append(target1)

                loss_fn = torch.nn.MSELoss()

                loss_fn = loss_fn.to(self.device)

                result = loss_fn(output1, target1)

                logging.debug(f" >>>> loss : {result}, result.shape:{result.shape}, flat_args:{flat_args} ")

                self.grads2[mb_idx][node.name] = (None,)
                self.loss[mb_idx] = result

                self.fwd_cache2[mb_idx][node.name] = \
                    ( result if isinstance(result, tuple) else (result,), \
                    flat_args, )

        self.env2[mb_idx][node.name] = result

        #logging.debug(f" ## run [rank:{self.rank}, micro#:{mb_idx}] - node:{node.name}, node.op:{node.op}")

        return result


    def fx_backward4(self, *args):
        #logging.debug(f" -----> rank{self.rank}: in fx_backward4, args[0]:{args[0]}")
    
        for i in range(self.mbsize):
            result = self.fx_micro_backward(i)
            next(result)

    def init_grad(self, mb_idx, from_node, to_node):
        cnt = 0

        node = to_node
        while node != from_node._prev:
            if (use_wrapper == True and node.target != 'loss_fn') or \
                    (use_wrapper == False and node.target != 'output'):
                self.grads2[mb_idx][node.name] = None
                cnt = cnt + 1

            node = node._prev

        logging.debug(f" --------- init_grad >> {cnt} grads2 initialized !!!")




    def fx_micro_backward(self, mb_idx):
        from_, to_ = self.get_range(self.stage, self.graph)

        self.init_grad(mb_idx, from_, to_)

        if self.rank < self.get_last_rank(self.rank):
            pre_split_rank = self.get_next_rank(self.rank, self.stage)

            #cur = to_._next
            cur = to_
            begin_ = cur

            for node_name, range_ in self.special_nodes.items():
                src_stage, needed_by_stage = range_
                if self.stage >= src_stage and self.stage < needed_by_stage:
                    self.grads2[mb_idx][node_name] = self.receive_data(pre_split_rank)
                    logging.debug(f"### stage:{self.stage}, receive cross_ref gradient from {pre_split_rank}, node_name:{node_name}")

            #cur = self.fastfwd(cur, self.window_size)
            #while cur != begin_:         # fastfwd case
            #    node_name = cur.name
            #    pre_split_rank = self.rank + 1
            #    logging.debug(f"## rank:{self.rank}, receive grads from {pre_split_rank}, node_name:{node_name}")
            #    self.grads2[mb_idx][node_name] = self.receive_data(pre_split_rank)
            #    cur = cur._prev

            if cur == begin_:
                node_name = cur.name     # cur = from_._prev
                #ipre_split_rank = self.rank + 1
                if node_name not in self.special_nodes:
                    logging.debug(f"## rank:{self.rank}, stage:{self.stage} receive grads from {pre_split_rank}, node_name:{node_name}")
                    self.grads2[mb_idx][node_name] = self.receive_data(pre_split_rank)

        node = to_
        while node != from_._prev:    
    
            if use_wrapper == True and node.op == 'output':
                node = node._prev
                continue

    
            if node.op == 'call_module' or node.op == 'call_method' or \
                    (node.op == 'call_function' and not node.name.startswith("getitem")) or \
                    (use_wrapper == False and node.op == 'output'):

                logging.debug(f" >>>>> fx_micro_backward[{mb_idx}] - node.name:{node.name}")

    
                def extract_tensor_args(b):
                    a = self.env2[mb_idx][b.name]
                    #if isinstance(a, torch.Tensor):
                    if isinstance(a, torch.Tensor) and a.is_floating_point():
                        val = a.detach().to(self.device)
                        val.requires_grad_(a.requires_grad)
                        return val
                    else:
                        return a
    
                args = ()
                kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)

                #logging.debug(f" ***** node:{node.name}, kwargs:{kwargs}")
    
                #kwargs = dict(kwargs)
                kwargs = dict() 
                k1, k2 = self.fwd_cache2[mb_idx].pop(node.name)

                kwargs["stage_output"] = k1
                kwargs["input_values"] = k2
                kwargs["output_grads"] = self.grads2[mb_idx][node.name]

                if self.grads2[mb_idx][node.name] == None:
                    logging.debug(f" >>> node:{node.name} - grads2 is None")
                    node = node._prev
                    continue

                #kwargs["outputs_with_grads_idxs"] = [0]
                if isinstance(k1, tuple):
                    num_nodes = len(k1)
                    if num_nodes > 1:
                        logging.debug(f" #### num_nodes: {num_nodes} ##")
                else:
                    num_nodes = 1
                kwargs["outputs_with_grads_idxs"] = [i for i in range(num_nodes)]

                if not isinstance(kwargs['stage_output'][0], torch.Tensor):
                    logging.debug(f" >>> node:{node.name} - not Tensor")
                    node = node._prev
                    continue

                elif isinstance(kwargs['stage_output'][0], torch.Tensor):
                    shape1 = kwargs['stage_output'][0].shape
                    #logging.debug(f" >>>>>> node:{node.name}, stage_output - shape: {kwargs['stage_output'][0].shape}, .... stage_output:{kwargs['stage_output']}")
                    logging.debug(f" >>>>>> node:{node.name}, stage_output - shape: {kwargs['stage_output'][0].shape}")
                    logging.debug(f" shape1: {shape1}")
                    if isinstance(self.grads2[mb_idx][node.name], (tuple,list)):
                        logging.debug(f" grads2 len: { len(self.grads2[mb_idx][node.name]) }")
                        for i in range(len(self.grads2[mb_idx][node.name])):

                            if self.grads2[mb_idx][node.name][i] != None:
                                logging.debug(f" --- self.grads2[{mb_idx}][{node.name}][{i}].shape: {self.grads2[mb_idx][node.name][i].shape}")
                            if self.grads2[mb_idx][node.name][i] != None and self.grads2[mb_idx][node.name][i].shape == shape1 and kwargs["output_grads"][0] != None and  kwargs["output_grads"][0].shape != shape1:
                                logging.debug(f" ***** self.grads2[{mb_idx}][{node.name}][{i}].shape : {self.grads2[mb_idx][node.name][i].shape} ==> set output_grads values <- shape:{self.grads2[mb_idx][node.name][i].shape}")
                                kwargs["output_grads"] = self.grads2[mb_idx][node.name][i] if isinstance(self.grads2[mb_idx][node.name][i], tuple) else (self.grads2[mb_idx][node.name][i],)
                                break

                if node.name.startswith("to_"):
                    logging.debug(f" ***** node:{node.name}, continue")
                    node = node._prev
                    continue

                if (use_wrapper == True and node.target != 'loss_fn' and self.grads2[mb_idx][node.name] == (None,)) or \
                        (use_wrapper == False and node.target != 'output' and self.grads2[mb_idx][node.name] == (None,)):
                    logging.debug(f" >>>>> self.grads2[{mb_idx}][{node.name}] == (None,)")
                    node = node._prev
                    continue

                # prepare_for_backward position.
                if isinstance(self.ddp_mod, torch.nn.parallel.distributed.DistributedDataParallel):
                    if mb_idx == self.mbsize - 1:
                        self.ddp_mod.reducer.prepare_for_backward(
                                list(
                                    torch.nn.parallel.distributed._find_tensors(
                                        kwargs['stage_output']
                                        )
                                    )
                                )
                        result = stage_backward(*args, **kwargs)
                    else:
                        with self.ddp_mod.no_sync():
                            result = stage_backward(*args, **kwargs)
                else:
                    result = stage_backward(*args, **kwargs)


                #if result[0] == (None,):
                #    node = node._prev
                #    continue
    
                next_ = []
                for i, m in enumerate(node.all_input_nodes):
                    next_.append(m)
                    if (use_wrapper == False and node.op == 'output') \
                            or (use_wrapper == True and node.target == 'loss_fn'):
                        break
    
                cnt = len(result[0])

                #logging.debug(f" >>>>>>>>>>>> node.name:{node.name}, get_desti ->{next_}, cnt={cnt}, stage_bacward --> result:{result}")
                logging.debug(f" >>>>>>>>>>>> node.name:{node.name}, get_desti ->{next_}, cnt={cnt}")


                for m in next_:
                    if cnt > 1:
                        self.grads2[mb_idx][m.name] = tuple(result[0])
                    else:
                        self.grads2[mb_idx][m.name] = result[0]


                node = node._prev
                continue
    
            if node.op == 'placeholder' and node.target == 'labels':
                node = node._prev
                continue

            node = node._prev

        if self.stage > 0:
            next_split_rank = self.get_prev_rank(self.rank, self.stage)

            cur = node

            begin_ = cur

            for node_name, range_ in self.special_nodes.items():
                src_stage, needed_by_stage = range_
                if self.stage > src_stage and self.stage <= needed_by_stage:
                    obj = self.grads2[mb_idx][node_name]
                    self.send_data(obj, next_split_rank)
                    logging.debug(f"### rank:{self.rank} send cross_ref gradient to {next_split_rank}, node_name:{node_name}")

            #cur = self.fastfwd(cur, self.window_size)
            #
            #while cur != begin_:           # fastfwd case
            #    node_name = cur.name
            #    next_split_rank = self.rank - 1
            #    logging.debug(f"### rank:{self.rank} send gradients to {next_split_rank}, node_name:{node_name}")
            #    obj = self.grads2[mb_idx][node_name]
            #    self.send_data(obj, next_split_rank)
            #    cur = cur._prev

            if cur == begin_:
                #cur = cur._prev 
                node_name = cur.name       #
                if node_name not in self.special_nodes:
                    logging.debug(f"### rank:{self.rank} send gradients to {next_split_rank}, node_name:{node_name}")
                    obj = self.grads2[mb_idx][node_name]
                    self.send_data(obj, next_split_rank)

        yield 0

parser = argparse.ArgumentParser()
parser.add_argument('--pp_size', default=1, type=int)
parser.add_argument('--dp_size', default=1, type=int)
args = parser.parse_args()

pp_size = args.pp_size
dp_size = args.dp_size

if dp_size > 1:
    batch_size = batch_size // dp_size

assert pp_size * dp_size == num_rank, f"There are unused GPUs"

if int(os.environ["RANK"]) == 0:
    print(f"total process count: {num_rank}")
    print(f"pipeline parallel size: {pp_size}")
    print(f"data parallel size: {dp_size}")
    print(f"batch size: {batch_size}")
    print(f"micro batch size: {micro_batch_size}")

sim_split = Simple_split_test()
sim_split.metadata_transfer()

fx_run2 = FXRun2(sim_split, sim_split.device, mbsize=micro_batch_size) 

fx_run2.print_range()

print(f">>> micro batch size = {fx_run2.mbsize}")


if fx_run2.rank == 0:
    print('Total parameters in model: {:,}'.format(get_total_params(fx_run2.mod)))
    tick = time.time()


fx_run2.mod.train()
optimizer1 = Adam(fx_run2.mod.parameters(), lr=3e-5)

torch.manual_seed(42 + fx_run2.rank)

if fx_run2.stage == 0:
    #
    # move sample_input to first host
    #
    sample_output = torch.rand(batch_size, out_features)
    sample_input = torch.rand(batch_size, in_features)
else:
    sample_input = None
    sample_output = None


print(sample_input)

#
# move sample_output to last host
#
if fx_run2.stage == 0 and pp_size > 1:
    target_node_name = "labels"
    mbatches = torch.chunk(sample_output, fx_run2.mbsize)
    for j in range(fx_run2.mbsize):
        fx_run2.env2[j][target_node_name] = mbatches[j]

    for j in range(fx_run2.mbsize):
        obj = fx_run2.env2[j][target_node_name]
        fx_run2.send_data(obj, fx_run2.get_last_rank(fx_run2.rank))
    #
    print(f"sent ====> {sample_output}")

if fx_run2.stage == pp_size - 1 and pp_size > 1:
    target_node_name = "labels"
    for j in range(fx_run2.mbsize):
        fx_run2.env2[j][target_node_name] = fx_run2.receive_data(fx_run2.get_first_rank(fx_run2.rank))

    outputs = tuple(mb["labels"] for mb in fx_run2.env2)
    sample_output = torch.cat(outputs)
    #
    print(f"received <==== env2[{fx_run2.mbsize-1}][{target_node_name}]: {fx_run2.env2[fx_run2.mbsize-1][target_node_name]}")

logFile = open('rank'+str(fx_run2.rank)+'_parameters.txt', 'w')

#for i in range(20):
for i in range(50):

    optimizer1.zero_grad()

    output1 = fx_run2.fx_forward4(sample_input, sample_output)
    loss1 = fx_run2.loss

    if fx_run2.rank == fx_run2.get_last_rank(fx_run2.rank):
        print(f'Step {i}, {fx_run2.rank} Loss1:{sum(loss1) / fx_run2.mbsize}')

    fx_run2.fx_backward4(loss1)

    optimizer1.step()

    print('step', i, file = logFile)
    for param in fx_run2.mod.parameters():
        print(param.data.tolist(), file = logFile)
    print('\n', file = logFile)

logFile.close()

if fx_run2.rank == 0:
    tock=time.time()
    elapsed_time = tock - tick

    print('Time elapsed: %.3f sec ' % (elapsed_time))

if fx_run2.rank == fx_run2.get_last_rank(fx_run2.rank):
    print(f"RANK:{sim_split.rank} ###################### output #############")
    output1 = fx_run2.make_output()
    print(output1)
    print(f"###################################")

print(f"[rank:{fx_run2.rank}, run completed ...")

#rpc.shutdown()

