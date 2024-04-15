import torch
import torch.distributed as dist
import logging
import argparse
import sys

import torch.nn as nn

from opt_prime.comm import Comm
from opt_prime.IR import IR
from opt_prime.schedule import ScheduleGPipe

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch.fx.graph_module import GraphModule


#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.ERROR)

SCHEDULE = {
    "gpipe": ScheduleGPipe,
    # TODO
    }

class Run_Info:

    def __init__(self, ir, device, mbsize, rank, world_size):
        self.mod = ir.model_ir[0] # TODO: beautify
        self.graph = self.mod.graph
        self.name = None
        self.node = None
        self.stage = rank # TODO
        self.submod = None
        self.env: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.env_recv_mark: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.env_send_mark: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.env_grad_recv_mark: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.env_grad_send_mark: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.device = device
        self.loss: List[Any] = [None for _ in range(mbsize)]
        self.flat_args: List[Dict[str, List[torch.Tensor]]] = [{} for _ in range(mbsize)]
        self.grads: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.getitem_dic : Dict[str, Any] = {}

        self.mbsize = mbsize
        #self.special_nodes: Dict[str, Tuple[int, int]] = ir.special_nodes
        #self.special_nodes = ir.special_nodes
        self.rank = rank
        self.world_size = world_size
        self.metadata_range = ir.metadata_range


    def setup_special_nodes(self, ir):
        self.special_nodes = ir.special_nodes

    def setup_submod(self):

        #for n, m in self.ir.model_ir[0].named_children():
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


        self.submod.to(self.device)

        # TODO:
        print(f" ## Rank:{self.rank}, name:{self.node.name}, move {self.name} to {self.device}")


    def build_getitem_dic(self):
        for node in self.mod.graph.nodes:
            if node.op == 'call_function' and node.name.startswith("getitem"):
                self.getitem_dic[node.name] = (node.args[0].name, node.args[1])


    def print_graph(self, ir):
        print(f" # rank = {self.rank}, metadata_range:{ir.metadata_range}")
        for node in self.mod.graph.nodes:
            print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.args:{node.args}, node.all_input_nodes:{node.all_input_nodes}")

    def print_getitem_dic(self):
        print(f" ========= getitem_dic =========")
        for k, v in self.getitem_dic.items():
            print(f" --- key:{k}, values:{v[0],v[1]}")
        print(f" ===============================")



class Optimus_p:

    def __init__(self, module:nn.Module, mbsize, use_gpu=False, dp_size=1):

        #self.model_ir = []
        #self.pp_size=pp_size
        self.dp_size=dp_size # TODO
        self.mbsize = mbsize

        
        #self.special_nodes: Dict[str, Tuple[int, int]] = {}  # { node_name : {rank#, needed-by-rank#),}

        self.use_gpu = use_gpu

        #use_gpu = True

        self.comm = Comm(use_gpu=use_gpu)

        self.rank = self.comm.rank
        self.world_size = self.comm.world_size
        self.local_rank = self.comm.local_rank

        if use_gpu == True:
            self.device = torch.device(f"cuda:{self.local_rank}")
            print(f">>> Using GPU ... cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
            print(f">>> Using CPU ...")


        self.ir = IR(module)
        #self.model_ir.append(IR(module))
        self.stage = self.rank # TODO

        self.ir.retrieve_IR(module)
        self.ir.split_IR(module, "simple", num_rank=self.world_size) # TODO: num_rank <-- Now, total process count 

        # TODO:
        self.run_info = Run_Info(self.ir, self.device, mbsize, self.rank, self.world_size) # TODO: rank, world_size
        self.run_info.setup_submod() 
        self.run_info.build_getitem_dic()

        if self.rank == 0:
            self.run_info.print_graph(self.ir)
            self.run_info.print_getitem_dic()

        if self.rank == 0:
            for rank in reversed(range(1, self.world_size)):
                self.ir.cross_reference_analyze(rank, self.run_info.graph)

            special_nodes_obj = [self.ir.special_nodes]
            dist.broadcast_object_list(special_nodes_obj, src=0, device=self.device)
        else:
            special_nodes_obj = [None]
            dist.broadcast_object_list(special_nodes_obj, src=0, device=self.device)
            self.ir.special_nodes = special_nodes_obj[0]

        print(f" *********** rank:{self.rank} ==> cross-referenced nodes *****************")
        print(f"   special_nodes: {self.ir.special_nodes}")
        print(f" *************************************************************************")

        self.run_info.setup_special_nodes(self.ir)


    def prepare_labels(self, labels):

        # TODO
        if self.rank != 0:
            print(f"[rank:{self.rank}] This function must be used in first rank!")
            sys.exit(1)

        target_node_name = "labels"
        mbatches = torch.chunk(labels, self.mbsize)
        if self.mbsize == 1:
            self.run_info.env[0][target_node_name] = labels
        else:
            for j in range(self.mbsize):
                self.run_info.env[j][target_node_name] = mbatches[j]

        for j in range(self.mbsize):
            obj = self.run_info.env[j][target_node_name]
            self.comm.send_data(obj, self.world_size - 1, self.device)


    def ready_labels(self):

        # TODO
        if self.rank != self.world_size - 1:
            print(f"[rank:{self.rank}] This function must be used in last rank!")
            sys.exit(1)

        target_node_name = "labels"
        for j in range(self.mbsize):
            self.run_info.env[j][target_node_name] = self.comm.receive_data(0, self.device)
        if self.mbsize == 1:
            labels = self.run_info.env[0][target_node_name]
        else:
            outputs = tuple(mb["labels"] for mb in self.run_info.env)
            labels = torch.cat(outputs)

        return labels

    def run(self, data, labels, mode="gpipe"):
        schedule = SCHEDULE[mode](self.run_info, self.ir, self.comm)

        schedule.run(data, labels)
        

    def parameters(self):
        return self.run_info.submod.parameters()

    def train(self):
        return self.run_info.submod.train()

    def get_loss(self):
        return self.run_info.loss
