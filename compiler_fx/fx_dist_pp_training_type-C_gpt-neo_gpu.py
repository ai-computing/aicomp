#
# Copyright (c) 2023-present, ETRI, All rights reserved.
#
#
#  This is a PoC that performs a GPipe-style pipeline-parallel training based on the FX IR partition.
#
#   In this PoC, FX compile generates FX IR,
#       and partitions of FX IR are transferred to the distributed processes,
#       and then pipeline parallel training is executed across processes.
#
#   Micro-batch is supported in this PoC, and applied to the GPT-Neo model (GPU version)
#
#
#  Sample Usage:
#      <machine #0>
#            torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0
#                  --master_addr="X.X.X.X" --master_port=29500 fx_dist_pp_training_type-C_gpt-neo_gpu.py
#      <machine #1>
#            torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1
#                  --master_addr="X.X.X.X" --master_port=29500 fx_dist_pp_training_type-C_gpt-neo_gpu.py
#

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoConfig
from transformers import PreTrainedModel   
from transformers.models.gpt_neo import GPTNeoPreTrainedModel
import transformers.utils.fx as hf_fx
import inspect

from torch import Tensor, Size
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch.optim import Adam
from torch import fx
from torch.fx.node import Node
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time

import torch.distributed as dist
import datetime
#import torch.distributed.rpc as rpc

import logging

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torch.fx.graph_module import GraphModule
from torch.fx.passes.split_module import split_module


from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer


#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.ERROR)

torch.manual_seed(42)

use_gpu = True


#
# Total process count
#
num_rank=int(os.environ["WORLD_SIZE"])

#batch_size = 64
batch_size = 32
#batch_size = 128

micro_batch_size = num_rank // 2 # TODO

if int(os.environ["RANK"]) == 0:
    print(f"total process count: {num_rank}")
    print(f"batch size: {batch_size}")
    print(f"micro batch size: {micro_batch_size}")

gm = None

NoneType=type(None)


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params



class Simple_split_test(object):
    def __init__(self):
        self.initialize_comm()
        self.model_ir = []
        self.part_idx = 0

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
        logging.info(f" --- rank:{self.rank}, world_size:{self.world_size}, master:{self.master_addr}, port:{self.master_port}")

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


    def simple_split(self, gm, module):
        length = gm.graph.nodes.__len__()

        modcnt = 0
        for n in gm.graph.nodes:
            if n.op == 'call_module':
                modcnt = modcnt + 1
        
        segment = modcnt // num_rank
        print(f"length:{length}, modcnt:{modcnt}, num_rank:{num_rank}, segment:{segment}")


        ## simple assert
        assert length >= num_rank, f"Model length:{length} is smaller than # of workers:{num_rank}"
        
        self.last_flag = False

        def part_fn(node):
            last_idx, last_name = self.metadata_range[-1]
            
            if self.last_flag == True:
                idx = last_idx
                #print(f" part_fn:  node.name:{node.name}, --> {idx}")
                return idx
            
            idx = 0
            
            cur = node
            while cur.name != last_name:
                for i, m_name in self.metadata_range:
                    if cur.name == m_name:
                        idx = i
                        #print(f" part_fn:  node.name:{node.name}, m_name:{m_name}, --> {idx}")
                        return idx
            
                cur = cur._next
            
            if cur.name == last_name:
                idx = last_idx
                self.last_flag = True
            
            #print(f" part_fn:  node.name:{node.name}, --> {idx}")
            return idx
            

        k, cnt = 0, 0
        for n in gm.graph.nodes:
            if n.op == 'call_module':
                cnt = cnt + 1
        
            if cnt == segment:
                self.metadata_range.append((k, n.name))
                k = k + 1
                cnt = 0
        
            if k > num_rank - 1:
                break
        
        if len(self.metadata_range) <  num_rank:
            self.metadata_range.append((k, n.name))

        if self.rank == 0:
            print(f" ------------------------------------------------------------")
            print(f"  rank:{self.rank},  first metadata_range: {self.metadata_range}")
            print(f" ------------------------------------------------------------")

        submodules = split_module(gm, module, part_fn, keep_original_order=True)
        #if self.rank == 0:
        #    print(f" ------------------------------------------------------------")
        #    print(f" rank:{self.rank},  submodules: {submodules}")
        #    print(f" ------------------------------------------------------------")


        def move_parameters(split_graph_module, user_target, parameter_value, use_index, _buffer):

            assert isinstance(parameter_value, torch.Tensor), f"Not torch.Tensor but {type(parameter_value)} received."

            target = split_graph_module.get_submodule(user_target)
            new_parameter_name = f"moved_{node.target.replace('.', '_')}"

            assert not hasattr(target, new_parameter_name), f"{user_target} has parameter[{new_parameter_name}]"

            if _buffer:
                target.register_buffer(new_parameter_name, parameter_value)
            else:
                setattr(target, new_parameter_name, parameter_value)

            placeholder_cnt = 0
            for snode in target.graph.nodes:
                if snode.op == "placeholder":
                    if placeholder_cnt == use_index:
                        with target.graph.inserting_before(snode):
                            get_attr = target.graph.get_attr(new_parameter_name)
                            snode.replace_all_uses_with(get_attr)
                            target.graph.erase_node(snode)
                    placeholder_cnt += 1

            target.graph.lint()
            target.recompile()

            return get_attr

                    
        def remove_reference(node, user, delete_node=True):
            assert len(user.kwargs) == 0
            use_indices = [i for i, arg in enumerate(user.args) if arg == node]
            assert len(use_indices) == 1
            args_copy = list(user.args)
            args_copy.pop(use_indices[0])
            user.args = tuple(args_copy)
            if delete_node:
                node.graph.erase_node(node)

            return use_indices[0]


        remove_candidates = list()
        for node in submodules.graph.nodes:
            if node.op == "get_attr" and len(node.users) == 1:
                user = list(node.users)[0]
                assert user.op == "call_module"
                use_index = remove_reference(node, user)

                atoms = node.target.split(".")
                module_itr = submodules
                for atom in atoms[:-1]:
                    module_itr = getattr(module_itr, atom)
                parameter_value = getattr(module_itr, atoms[-1])
                _buffer = atoms[-1] in module_itr._buffers

                move_parameters(submodules, user.target, parameter_value, use_index, _buffer)

                remove_candidates.append((module_itr, atoms))

        for module_itr, atoms in remove_candidates:
            delattr(module_itr, atoms[-1])
        submodules.graph.lint()
        submodules.recompile()

        self.metadata_range = []

        cnt = 0
        for n in submodules.graph.nodes:
            if n.op == 'call_module':
                self.metadata_range.append((cnt, n.name))
                cnt = cnt + 1

        print(f" ------------------------------------------------------------")
        print(f"  rank:{self.rank},  second metadata_range: {self.metadata_range}")
        print(f" ------------------------------------------------------------")

        assert len(self.metadata_range) == num_rank

        return submodules

        
    def check_last_submods(self, submods):
        gmodule_cnt = 0
        mod_cnt = 0
        for submod in submods.modules():
            if isinstance(submod, fx.GraphModule):
                gmodule_cnt = gmodule_cnt + 1
                last_submod = submod
                continue

        #print(f">> check_last_submods: gmodule_cnt:{gmodule_cnt}, num_rank:{num_rank}")

        assert gmodule_cnt > num_rank, f"GraphModule #:[{gmodule_cnt}] must have more than the number of processes #:[{num_rank}]"

        for node in last_submod.graph.nodes:
            #print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.all_input_nodes:{node.all_input_nodes}")
            if node.op == 'call_module' and node.target != 'loss_fn':
                mod_cnt = mod_cnt + 1

        #print(f">>> GraphModule cnt:{gmodule_cnt},  Last GraphModule's  mod_cnt ==> {mod_cnt}")

        assert mod_cnt > 0, f"Last partition has {mod_cnt} modules. It must have more than 0 modules"



    # transfer range metadata to all processes
    def metadata_transfer(self):

        global gm

        self.metadata_range = []

        input_names = model.dummy_inputs.keys()
        sig = inspect.signature(model.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }

        tracer = hf_fx.HFTracer()

        traced_graph = tracer.trace(model, concrete_args=concrete_args)
        gm = torch.fx.GraphModule(model, traced_graph)

        if use_gpu == True:
            self.device = torch.device(f"cuda:{self.local_rank}")
            print(f">>> Using GPU ... cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
            print(f">>> Using CPU ...")



        submods = self.simple_split(gm, model)
        self.check_last_submods(submods)

        #if self.rank == 0:
        #    print(f"------------------ FX graph --------------------------------")
        #    for n in submods.graph.nodes:
        #        print(f"n.op:{n.op}, n.name:{n.name}, n.target:{n.target}, n.args:{n.args}, n.all_input_nodes:{n.all_input_nodes}")
        #    print(f"------------------------------------------------------------")

        self.model_ir.append(submods)
        self.stage = self.rank # TODO

        #if self.rank == 0:
        #    print(f">> ------------------ FX graph --------------------------------")
        #    for n in self.model_ir[0].graph.nodes:
        #    #for n in submods.graph.nodes:
        #        print(f"n.op:{n.op}, n.name:{n.name}, n.target:{n.target}, n.args:{n.args}, n.all_input_nodes:{n.all_input_nodes}")
        #    print(f">> ------------------------------------------------------------")



def core_backward(forward_output, forward_output_gradient, forward_input, valid_index: List[int],):

    forward_output_with_grads = [forward_output[i] for i in valid_index]
    forward_output_gradient_with_grads = [forward_output_gradient[i] for i in valid_index]

    forward_output_list = []
    forward_output_gradient_list = []


    def extract_tensor_for_gradients(output_val, grad_val):
        if isinstance(output_val, torch.Tensor):
            if not output_val.requires_grad and output_val.grad_fn is None:
                #logging.warning(f" ---------------- {output_val}: not requirs_grad and grad_fn None")
                print(f" ---------------- {output_val}: not requirs_grad and grad_fn None")
                return
            forward_output_list.append(output_val)
            forward_output_gradient_list.append(grad_val)
        elif isinstance(output_val, (tuple, list)):
            if grad_val is None:
                #logging.warning(f" ---------------- {grad_val}: is None")
                print(f" ---------------- {grad_val}: is None")
                return
            for ov, gv in zip(output_val, grad_val):
                extract_tensor_for_gradients(ov, gv)
        elif isinstance(output_val, dict):
            if grad_val is None:
                #logging.warning(f" ---------------- {grad_val}: is None")
                print(f" ---------------- {grad_val}: is None")
                return
            for k in output_val.keys():
                extract_tensor_for_gradients(output_val[k], grad_val[k])
        else:
            logging.critical(f"... ignored in this case")


    extract_tensor_for_gradients(forward_output_with_grads, forward_output_gradient_with_grads)


    if isinstance(forward_output_gradient_list[0], list):
        forward_output_gradient_list[0] = forward_output_gradient_list[0][0]

    if forward_output_list[0] != None and forward_output_gradient_list[0] != None and forward_output_list[0].shape != forward_output_gradient_list[0].shape:
        forward_output_list[0] = forward_output_list[0].view(-1, forward_output_list[0].size(-1))

    torch.autograd.backward(forward_output_list, grad_tensors=forward_output_gradient_list)
    #inputs_with_grad = []
    #for val in forward_input:
    #    if isinstance(val, torch.Tensor) and val.requires_grad:
    #        inputs_with_grad.append(val)
    #forward_input_gradient = torch.autograd.grad(forward_output_list, inputs_with_grad, forward_output_gradient_list,)


    forward_input_gradient = []
    for v in forward_input:
        if isinstance(v, torch.Tensor):
            forward_input_gradient.append(v.grad)
        else:
            forward_input_gradient.append(None)

    return forward_input_gradient, None



class FXRun3:

    def __init__(self, split_info: Simple_split_test, device, mbsize): 

        self.mod = split_info.model_ir[0]
        self.graph = self.mod.graph
        self.rank = split_info.rank
        self.name = None
        self.node = None
        self.stage = split_info.stage
        self.submod = None
        self.mbsize = mbsize  
        self.env2: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.env2_recv_mark: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.env2_send_mark: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.env2_grad_recv_mark: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.env2_grad_send_mark: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.metadata_range = split_info.metadata_range
        self.world_size = split_info.world_size
        self.device = device
        self.loss: List[Any] = [None for _ in range(mbsize)]
        self.flat_args2: List[Dict[str, List[torch.Tensor]]] = [{} for _ in range(mbsize)]
        self.grads2: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.special_nodes: Dict[str, Tuple[int, int]] = {}  # { node_name : {rank#, needed-by-rank#),}

        self.getitem_dic : Dict[str, Any] = {}

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

        self.get_submod()

        self.build_getitem_dic()

        if self.rank == 0:
            self.print_range()
            self.print_getitem_dic()


        if self.rank == 0:
            for rank in reversed(range(1, self.world_size)):
                self.cross_reference_analyze(rank, self.graph)
         
            special_nodes_obj = [self.special_nodes]
            dist.broadcast_object_list(special_nodes_obj, src=0, device=self.device)
         
        else:
            special_nodes_obj = [None]
            dist.broadcast_object_list(special_nodes_obj, src=0, device=self.device)
            self.special_nodes = special_nodes_obj[0]
         
        print(f" *********** rank:{self.rank} ==> cross-referenced nodes *****************")
        print(f"   special_nodes: {self.special_nodes}")
        print(f" *************************************************************************")




    # analyze IR graph and find the cross-layer referenced nodes
    def cross_reference_analyze(self, rank, g:fx.Graph):
    
        if rank == 0:
            return
    
        from_, to_ = self.get_range(rank, g)
    
        #logging.debug(f" ***** rank:{rank} >>  from_:{from_.name}, to_:{to_.name}")
        print(f" ***** rank:{rank} >>  from_:{from_.name}, to_:{to_.name}")
    
        cur = to_
        while (cur != from_) or (rank > 0 and cur == from_):
    
            # in process check - backward direction
    
            #for _, target_ in enumerate(cur.all_input_nodes):
            for i, target_ in enumerate(cur.all_input_nodes):
                if cur.name == "loss_fn" and i > 0:
                    break
                referenced_in = False
                referenced_out = False
    
                inner = cur._prev
                if inner != from_._prev:
                    while (inner != from_) or (rank > 0 and inner == from_):
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
    
                    rank_ = 0
                    split_node_name = self.metadata_range[rank_][1]
    
                    for k in g.nodes:
                        first_node = k
                        break
    
                    outer = first_node
                    while outer != from_: 
                        # DEBUG
                        if outer.name == target_.name:
                            logging.info(f" [cross_reference_analyze] ({target_.name}) referenced in outer rank:{rank_} !!")
    
                            if target_.name not in self.special_nodes:
                                self.special_nodes[target_.name] = (rank_, rank)  # { node_name : {rank#, needed-by-rank#),}
                            referenced_out = True
                            break
    
                        if outer.name == split_node_name:
                            rank_ = rank_ + 1
                            split_node_name = self.metadata_range[rank_][1]
    
                        outer = outer._next
    
    
                if referenced_out == False:
                    logging.critical(f"[Error] cannot handle this case: {target_.name} !!!")
                    sys.exit(1)
    
            if cur == from_:
                break
    
            cur = cur._prev



    def get_submod(self):
        for n, m in self.mod.named_children():
             if n == f"submod_{self.stage}" and isinstance(m, GraphModule):
                 self.name = n
                 self.submod = m
                 break

        if self.name is None:
            print(f"ERROR: Not found name(submod_{self.stage})")
            sys.exit(0)

        #print(f" ## Rank:{self.rank}, name:{self.name}")

        for n in self.graph.nodes:
            if n.name == self.name:
               self.node = n
               break

        if self.node is None:
            print(f"ERROR: Not found node({self.name})")
            sys.exit(0)

        #print(f" ## Rank:{self.rank}, name:{self.node.name}")

        self.submod.to(self.device)



    def get_prev_nodename(self, rank):
        if rank > 0:
            return self.metadata_range[rank-1][1]
        elif rank == 0:
            return None



    def get_range(self, rank, g:fx.Graph) -> (Node, Node):
     
        if rank == 0:
            from_node_name = "-1"
            for n in g.nodes:
                if n.op == 'placeholder':
                    from_node_name = n.name
                    logging.debug(f">>>> get_range: n.op == 'placeholder' --> from_node_name:{from_node_name}")
                break
        else:
            from_node_name = self.metadata_range[rank-1][1]
     
        to_node_name = self.metadata_range[rank][1]
    
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
     
        if rank == 0:
            return (from_node, to_node)
        else:
            return (from_node._next, to_node)



    def print_range(self):
        print(f" # rank = {self.rank}, metadata_range:{self.metadata_range}")

        for node in self.mod.graph.nodes:
            print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.args:{node.args}, node.all_input_nodes:{node.all_input_nodes}")


    def print_getitem_dic(self):
        print(f" ========= getitem_dic =========")
        for k, v in self.getitem_dic.items():
            print(f" --- key:{k}, values:{v[0],v[1]}")
        print(f" ===============================")


    def build_getitem_dic(self):
        for node in self.mod.graph.nodes:
            if node.op == 'call_function' and node.name.startswith("getitem"):
                self.getitem_dic[node.name] = (node.args[0].name, node.args[1])



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

        if self.rank == 0:
            for n in self.mod.graph.nodes:
                if (n.op == 'placeholder' and self.stage == 0 and n.name == 'x') or \
                        (n.op == 'placeholder' and self.stage == 0 and n.name == 'input_ids'):
                    input = next(self.args_iter)

                    if isinstance(input, torch.Tensor):
                        mbatches = torch.chunk(input, self.mbsize)
                        if self.mbsize == 1:
                            input = input.to(self.device)
                            self.env2[0]["placeholder"] = input
                        else:
                            for j in range(self.mbsize):
                                mbatch = mbatches[j].to(self.device)
                                self.env2[j]["placeholder"] = mbatch
                    else:
                        logging.critical(f"### input:{input} not Tensor --> currently not supported!!")
                        sys.exit(1)
                    break

        #logging.debug(f" * rank:{self.rank}, in run_micro_batch_forward() ..")
        for i in range(self.mbsize):
            result = self.fx_micro_forward(i)
            next(result)


    def get_last_module(self):
        assert self.rank == self.world_size - 1
        return self.node.name


    #def make_output(self):
    #    output = None
    #    if self.rank ==  self.world_size - 1:
    #        target = self.get_last_module()
    #
    #        outputs = tuple(mb[target] for mb in self.env2) 
    #        #print(f" ---> RANK: {self.rank},  outputs = {outputs}, type(output):{type(outputs)}")
    #        #output = torch.cat(outputs)
    #        output = outputs
    #
    #    return output

    def get_output_node(self):
        for n in reversed(self.graph.nodes):
            if n.op == 'output':
                return n

    def get_next_node_name(self, rank):
        # TODO: last stage processing
        assert rank < self.world_size - 1

        next_node_name = self.metadata_range[rank+1][1]

        return next_node_name


    def run_loss(self):
        # TODO: last stage processing
        assert self.rank == self.world_size - 1

        node = self.get_output_node()
        key_ = node.args[0]['logits']

        for mb_idx in range(self.mbsize):
            if str(key_) in self.getitem_dic:
                a_submod = self.getitem_dic[str(key_)][0]
                a_idx = self.getitem_dic[str(key_)][1]
                output1_ = self.env2[mb_idx][a_submod][a_idx]
            else:
                output1_ = self.env2[mb_idx][str(key_)]

            target1_ = self.env2[mb_idx]["labels"]

            output1_ = output1_.view(-1, output1_.size(-1))
            target1_ = target1_.view(-1)


            flat_args = []
            if isinstance(output1_, torch.Tensor) and output1_.is_floating_point():
                output1 = output1_.detach().to(self.device)
                output1.requires_grad_(output1_.requires_grad)
                #output1.requires_grad_(True)
                flat_args.append(output1)
                output1.grad = None
            else:
                output1 = output1_
                flat_args.append(output1)

            if isinstance(target1_, torch.Tensor) and target1_.is_floating_point():
                target1 = target1_.detach().to(self.device)
                target1.requires_grad_(True)
                #flat_args.append(target1)
                flat_args.append(target1)
            else:
                target1 = target1_
                flat_args.append(target1)

            criterion = nn.CrossEntropyLoss()

            criterion = criterion.to(self.device)

            result = criterion(output1, target1)

            #print(f" >>>> loss: {result}, result.shape:{result.shape}")

            self.grads2[mb_idx][node.name] = (None,)

            self.loss[mb_idx] = result

            #self.fwd_cache2[mb_idx][node.name] = \
            #        ( result if isinstance(result, tuple) else (result,), \
            #        flat_args, )
            self.env2[mb_idx][node.name] = result
            self.flat_args2[mb_idx][node.name] = flat_args



    def init_env2_mark(self, mb_idx):
        for i in range(len(self.metadata_range)):
            self.env2_recv_mark[mb_idx][self.metadata_range[i][1]] = None
            self.env2_send_mark[mb_idx][self.metadata_range[i][1]] = None



    def fx_micro_forward(self, mb_idx):

        self.init_env2_mark(mb_idx)

        from_, to_ = self.get_range(self.rank, self.graph)

        if self.rank == 0:
            target_node_name = "placeholder"
            #self.env2[mb_idx]["x"] = self.env2[mb_idx][target_node_name]
            self.env2[mb_idx]["input_ids"] = self.env2[mb_idx][target_node_name]
        #if self.rank > 0:
        #    # TODO
        #    target_node_name = self.get_prev_nodename(self.rank)
        #    pre_split_rank = self.rank - 1
        #    self.env2[mb_idx][target_node_name] = self.receive_data(pre_split_rank)
        #    # DEBUG
        #    print(f" #### rank:{self.rank} <== received from [{pre_split_rank}] ####")

        if self.rank > 0:
            pre_split_rank = self.rank - 1
        
            cur = from_._prev
        
            begin_ = cur
            for node_name, range_ in self.special_nodes.items():
                src_rank, needed_by_rank = range_
                if self.rank > src_rank and self.rank <= needed_by_rank:
                    #print(f"MBF[{mb_idx}]: ### rank:{self.rank}, receive cross_ref activation from {pre_split_rank}, node_name:{node_name}")
                    if node_name in self.getitem_dic:
                        submod_name = self.getitem_dic[node_name][0]
                        if self.env2_recv_mark[mb_idx][submod_name] is None:
                            self.env2[mb_idx][submod_name] = self.receive_data(pre_split_rank)
                            self.env2_recv_mark[mb_idx][submod_name] = 1

                        if isinstance(self.env2[mb_idx][submod_name], torch.Tensor):
                            if not self.env2[mb_idx][submod_name].requires_grad or self.env2[mb_idx][submod_name].grad_fn is None:
                                self.env2[mb_idx][submod_name].requires_grad_(True)
                                logging.info(f" ###### node name:{submod_name} requires_grad(True) #####") 
                    else:
                        if self.env2_recv_mark[mb_idx][node_name] is None:
                            self.env2[mb_idx][node_name] = self.receive_data(pre_split_rank)
                            self.env2_recv_mark[mb_idx][node_name] = 1
                        if isinstance(self.env2[mb_idx][node_name], torch.Tensor):
                            if not self.env2[mb_idx][node_name].requires_grad or self.env2[mb_idx][node_name].grad_fn is None:
                                self.env2[mb_idx][node_name].requires_grad_(True)
                                logging.info(f" ###### node name:{node_name} requires_grad(True) #####") 
        

        #forward one chunk !!
        flat_args = []
        def extract_tensor_args(b):
            # TODO
            if b.name in self.getitem_dic:
                a_submod = self.getitem_dic[b.name][0]
                a_idx = self.getitem_dic[b.name][1]
                a = self.env2[mb_idx][a_submod][a_idx]
            else:
                a = self.env2[mb_idx][b.name]
            #a = self.env2[mb_idx][b.name]

            nonlocal flat_args
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                val = a.detach().to(self.device)
                #val.requires_grad_(a.requires_grad)
                val.requires_grad_(True)
                flat_args.append(val)
                return val
            else:
                flat_args.append(a)
                return a
            return a


        args = fx.graph.map_arg(self.node.args, extract_tensor_args)
        kwargs = fx.graph.map_arg(self.node.kwargs, extract_tensor_args)

        result = self.submod(*args, **kwargs)

        #self.fwd_cache2[mb_idx][self.name] = \
        #        ( result if isinstance(result, tuple) else (result,), \
        #        flat_args, )
        self.flat_args2[mb_idx][self.name] = flat_args

        #print(f" >>>> rank:{self.rank}, run fx_micro_forward( mb_idx:{mb_idx}, name:{self.name})")

        self.env2[mb_idx][self.name] = result

        if self.rank < self.world_size - 1:
            next_split_rank = self.rank + 1
            #begin_ = cur
        
            for node_name, range_ in self.special_nodes.items():
                src_rank, needed_by_rank = range_
                if self.rank >= src_rank and self.rank < needed_by_rank:
                    if node_name in self.getitem_dic:
                        submod_name = self.getitem_dic[node_name][0]
                        if self.env2_send_mark[mb_idx][submod_name] is None:
                            obj = self.env2[mb_idx][submod_name]
                            self.send_data(obj, next_split_rank)
                            self.env2_send_mark[mb_idx][submod_name] = 1
                    else:
                        if self.env2_send_mark[mb_idx][node_name] is None:
                            obj = self.env2[mb_idx][node_name]
                            self.send_data(obj, next_split_rank)
                            self.env2_send_mark[mb_idx][node_name] = 1

                    #print(f"MBF[{mb_idx}]: ### rank:{self.rank} send cross_ref activation to {next_split_rank}, node_name:{node_name}")

        yield result


    def free_mem(self):
        #self.env2 = [{} for _ in range(self.mbsize)]
        #self.flat_args2 = [{} for _ in range(self.mbsize)]
        #self.grads2 = [{} for _ in range(self.mbsize)]

        torch.cuda.empty_cache()


    def fx_backward4(self, *args):
    
        for i in range(self.mbsize):
            result = self.fx_micro_backward(i)
            next(result)

        self.free_mem()


    def init_env2_grad_mark(self, mb_idx):
        for i in range(len(self.metadata_range)):
            self.env2_grad_recv_mark[mb_idx][self.metadata_range[i][1]] = None
            self.env2_grad_send_mark[mb_idx][self.metadata_range[i][1]] = None

            self.grads2[mb_idx][self.metadata_range[i][1]] = None


    def get_num_nodes(self, name):
        cnt = 0
        for k, v in self.getitem_dic.items():
            if k == name:
                cnt = cnt +  1

        if cnt == 0:
            return 1

        return cnt



    def run_core_backward(self, mb_idx, node, grads):

        args = ()
        kwargs = dict()
        #k1, k2 = self.fwd_cache2[mb_idx].pop(node.name)
        k1 = self.env2[mb_idx].pop(node.name)
        #if not isinstance(k1, tuple):
        #    k1 = (k1)
        k1 = ((k1,) if not isinstance(k1, tuple) else k1)
        k2 = self.flat_args2[mb_idx].pop(node.name)

        kwargs["forward_output"] = k1
        kwargs["forward_input"] = k2
        kwargs["forward_output_gradient"] = grads 

        num_nodes = self.get_num_nodes(node.name) 
        kwargs["valid_index"] = [i for i in range(num_nodes)]

        result = core_backward(*args, **kwargs)

        return result



    def fx_micro_backward(self, mb_idx):

        self.init_env2_grad_mark(mb_idx)

        grads = None

        if self.rank < self.world_size - 1:
            pre_split_rank = self.rank + 1

            node_name = self.get_next_node_name(self.rank)
            if self.env2_grad_recv_mark[mb_idx][node_name] is None:
                self.grads2[mb_idx][node_name] = self.receive_data(pre_split_rank)
                grads = self.grads2[mb_idx][node_name]
                self.env2_grad_recv_mark[mb_idx][node_name] = 1


        # TODO: last stage ?
        if self.rank == self.world_size - 1:
            node = self.get_output_node()
            grads = self.grads2[mb_idx][node.name]
            result = self.run_core_backward(mb_idx, node, grads)
            result = ((result,) if not isinstance(result, tuple) else result)
            self.grads2[mb_idx][node.name] = result

            grads = self.grads2[mb_idx][node.name]

        node = self.node
        result = self.run_core_backward(mb_idx, node, grads)

        if self.rank == self.world_size - 1:
            node = self.get_output_node()
            self.grads2[mb_idx][node.name] = None
            node = self.node

        result = ((result,) if not isinstance(result, tuple) else result)

        self.grads2[mb_idx][node.name] = result

        if self.rank > 0:
            next_split_rank = self.rank - 1

            node_name = self.name
            if self.env2_grad_send_mark[mb_idx][node_name] is None:
                obj = self.grads2[mb_idx][node_name]
                self.send_data(obj, next_split_rank)
                self.env2_grad_send_mark[mb_idx][node_name] = 1


        yield 0

device = torch.device("cpu")

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


config = GPTNeoConfig(use_cache=False)

model = GPTNeoForCausalLM(config)
model = model.from_pretrained("EleutherAI/gpt-neo-2.7B")

print(f"model loaded ...")

if int(os.environ["RANK"]) == 0:
    print ('Total parameters in model: {:,}'.format(get_total_params(model)))


sim_split = Simple_split_test()
sim_split.metadata_transfer()


fx_run3 = FXRun3(sim_split, sim_split.device, mbsize=micro_batch_size) 

print(f">>> micro batch size = {fx_run3.mbsize}")

fx_run3.submod.train()
lr = 5.0
optimizer1 = torch.optim.SGD(fx_run3.submod.parameters(), lr=lr)
#lr = 3e-5
#optimizer1 = torch.optim.Adam(fx_run3.submod.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, 1.0, gamma=0.95)


datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(record) for record in datasets if len(str(record)) < 500]
dataloader = DataLoader(datasets, batch_size=batch_size, num_workers=4)
data_size=len(dataloader.dataset)   # total count of data
print(f"data_size={data_size}")
nbatches = len(dataloader)      # total count of data / batch size
print(f"nbatches={nbatches}")    

#epochs = 2 # The number of epochs
epochs = 1 # The number of epochs

def train():


    fx_run3.submod.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        data = None
        labels = None

        if fx_run3.rank == 0:

            # move data to first host
            # move labels to last host

            tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
            data, labels = tokens.input_ids, tokens.input_ids

            target_node_name = "labels"
            mbatches = torch.chunk(labels, fx_run3.mbsize)
            if fx_run3.mbsize == 1:
                fx_run3.env2[0][target_node_name] = labels
            else:
                for j in range(fx_run3.mbsize):
                    fx_run3.env2[j][target_node_name] = mbatches[j]

            for j in range(fx_run3.mbsize):
                obj = fx_run3.env2[j][target_node_name]
                fx_run3.send_data(obj, fx_run3.world_size - 1)
                #logging.debug(f">>>> [rank:0] sent [j:{j}] ==> {labels}")
            

        if fx_run3.rank == fx_run3.world_size - 1:
            #logging.debug(f" << RANK:{fx_run3.rank},  WORLD_SIZE:{fx_run3.world_size}, mbsize:{fx_run3.mbsize}")
            target_node_name = "labels"
            for j in range(fx_run3.mbsize):
                fx_run3.env2[j][target_node_name] = fx_run3.receive_data(0)
                #logging.debug(f">>>> received <==== env2[{j}][{target_node_name}]: {fx_run3.env2[j][target_node_name]}")
            if fx_run3.mbsize == 1:
                labels = fx_run3.env2[0][target_node_name]
            else:
                outputs = tuple(mb["labels"] for mb in fx_run3.env2)
                labels = torch.cat(outputs)


        optimizer1.zero_grad()

        fx_run3.fx_forward4(data, labels)

        if fx_run3.rank == fx_run3.world_size - 1:
            fx_run3.run_loss()
            loss1 = fx_run3.loss
        else:
            loss1 = None


        fx_run3.fx_backward4(loss1)
        
        torch.nn.utils.clip_grad_norm_(fx_run3.submod.parameters(), 0.5)
        optimizer1.step()
        
        if sim_split.rank == sim_split.world_size - 1:
            loss =  sum(loss1) / fx_run3.mbsize
            total_loss += loss
            log_interval = 10

            #fx_run3.loss = [None for _ in range(fx_run3.mbsize)]

            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, i, nbatches, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
         
                total_loss = 0
                start_time = time.time()


if sim_split.rank == 0:
    tick = time.time()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    scheduler.step()

if sim_split.rank == 0:
    tock=time.time()
    elapsed_time = tock - tick

    print('Time elapsed: %.3f sec ' % (elapsed_time))

if sim_split.rank == sim_split.world_size - 1:
    #print(f"RANK:{sim_split.rank} ###################### output #############")
    #output1 = fx_run3.make_output()
    ##print(output1)
    print(f"###################################")

print(f"[rank:{sim_split.rank}, run completed ...")

#rpc.shutdown()

