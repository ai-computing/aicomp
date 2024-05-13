import torch
import torch.distributed as dist
import logging
import argparse
import sys

import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel

from opt_prime.comm import Comm
from opt_prime.IR import IR
from opt_prime.schedule import ScheduleGPipe 
from opt_prime.schedule import Schedule1F1B 

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch.fx.graph_module import GraphModule


#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.ERROR)

SCHEDULE = {
    "gpipe": ScheduleGPipe,
    "1f1b": Schedule1F1B, 
    }

class Topology:

    def __init__(self, rank, local_rank, world_size, pp_size, dp_size):
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.set_stage() 


    def set_stage(self):
        #if self.dp_size == 1: # PP only
        #    self.stage = self.rank
        self.stage = self.rank // self.dp_size # PP + DP

    def get_stage(self):
        return self.stage

    def get_num_stage(self):
        return self.pp_size

    def is_first_stage(self):
        #if self.dp_size == 1: # PP only
        #    return self.rank == 0:
        return self.stage == 0

    def is_last_stage(self):
        #if self.dp_size == 1: # PP only
        #    return self.rank == self.world_size - 1
        return self.stage == self.pp_size - 1


    def get_first_stage(self):
        #if self.dp_size == 1: # PP only
        #    return 0
        return 0
            
    def get_last_stage(self):
        #if self.dp_size == 1: # PP only
        #    return self.world_size - 1
        return self.pp_size - 1

    def get_next_stage(self):
        #if self.dp_size == 1: # PP only
        #    assert self.stage < self.pp_size - 1
        #    return self.stage + 1
        assert self.stage < self.get_last_stage()
        return self.stage + 1

    def get_prev_stage(self):
        #if self.dp_size == 1: # PP only
        #    assert self.stage > 0
        #    return self.stage - 1
        assert self.stage > self.get_first_stage()
        return self.stage - 1


    def get_first_rank(self):
        return self.rank % self.dp_size

    def get_last_rank(self):
        return (self.pp_size - 1)*self.dp_size + self.rank % self.dp_size

    def get_prev_rank(self):
        assert self.stage > self.get_first_stage()
        return (self.stage - 1) * self.dp_size + self.get_first_rank()

    def get_next_rank(self):
        assert self.stage < self.get_last_stage()
        return (self.stage + 1)*self.dp_size + self.get_first_rank()

    def print_stage_info(self):
        print(f"rank:{self.rank}, get_stage: {self.get_stage()}")
        print(f"rank:{self.rank}, get_first_stage: {self.get_first_stage()}")
        print(f"rank:{self.rank}, get_last_stage: {self.get_last_stage()}")
        print(f"rank:{self.rank}, get_first_rank: {self.get_first_rank()}")
        print(f"rank:{self.rank}, get_last_rank: {self.get_last_rank()}")




class Run_Info:

    def __init__(self, ir, device, mbsize, rank, world_size, stage, preserve_output, display_mem):
        self.mod = ir.model_ir[0] # TODO
        self.graph = self.mod.graph
        self.name = None
        self.node = None
        self.submod = None
        self.env: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.env_recv_mark: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.env_send_mark: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.env_grad_recv_mark: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.env_grad_send_mark: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.device = device
        self.loss: List[Any] = [None for _ in range(mbsize)]
        self.output: List[Any] = [None for _ in range(mbsize)]
        self.flat_args: List[Dict[str, List[torch.Tensor]]] = [{} for _ in range(mbsize)]
        self.grads: List[Dict[str, Any]] = [{} for _ in range(mbsize)]
        self.getitem_dic : Dict[str, Any] = {}

        self.mbsize = mbsize
        #self.special_nodes: Dict[str, Tuple[int, int]] = ir.special_nodes
        #self.special_nodes = ir.special_nodes
        self.rank = rank
        self.world_size = world_size
        self.metadata_range = ir.metadata_range

        self.stage = stage
        self.preserve_output = preserve_output
        self.display_mem = display_mem



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

        # TODO
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

    def __init__(self, module:nn.Module, mbsize, use_gpu=False, dp_size=1, preserve_output=False, activation_ckpt=False, display_mem=False):

        #self.model_ir = []
        self.mbsize = mbsize

        #self.special_nodes: Dict[str, Tuple[int, int]] = {}  # { node_name : {stage#, needed-by-stage#),}

        self.use_gpu = use_gpu

        self.comm = Comm(use_gpu=use_gpu)

        self.activation_ckpt = activation_ckpt

        self.rank = self.comm.rank
        self.world_size = self.comm.world_size
        self.local_rank = self.comm.local_rank

        if dp_size < 1 or self.world_size % dp_size != 0:
            print(f"Data Parallel Size(dp_size option) is not valid")
            sys.exit(1)

        pp_size = self.world_size // dp_size

        if self.rank == 0:
            print(f"> Pipeline Parallel Size: {pp_size}")  
            if dp_size > 1:
                print(f"> Data Parallel Size: {dp_size}")


        self.tpl = Topology(self.rank, self.local_rank, self.world_size, pp_size, dp_size)

        if use_gpu == True:
            self.device = torch.device(f"cuda:{self.local_rank}")
            print(f">>> Using GPU ... cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
            print(f">>> Using CPU ...")


        self.ir = IR(module)
        #self.model_ir.append(IR(module))

        self.ir.retrieve_IR(module)
        self.ir.split_IR(module, "simple", num_stage=self.tpl.get_num_stage())

        self.run_info = Run_Info(self.ir, self.device, mbsize, self.rank, self.world_size, self.tpl.stage, preserve_output, display_mem) 
        self.run_info.setup_submod() 
        self.run_info.build_getitem_dic()

        if dp_size > 1:
            self.prepare_dp_group()

        if self.rank == 0:
            self.run_info.print_graph(self.ir)
            self.run_info.print_getitem_dic()

        #self.tpl.print_stage_info() # TODO: DEBUG

        if self.rank == 0:
            for stage in reversed(range(1, self.tpl.get_num_stage())):
                self.ir.cross_reference_analyze(stage, self.run_info.graph)

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

        if self.tpl.is_first_stage():

            target_node_name = "labels"
            mbatches = torch.chunk(labels, self.mbsize)
            if self.mbsize == 1:
                self.run_info.env[0][target_node_name] = labels
            else:
                for j in range(self.mbsize):
                    self.run_info.env[j][target_node_name] = mbatches[j]

            for j in range(self.mbsize):
                obj = self.run_info.env[j][target_node_name]
                self.comm.send_data(obj, self.tpl.get_last_rank(), self.device)


    def ready_labels(self):

        if self.tpl.is_last_stage():
            target_node_name = "labels"
            for j in range(self.mbsize):
                self.run_info.env[j][target_node_name] = self.comm.receive_data(self.tpl.get_first_rank(), self.device)
            if self.mbsize == 1:
                labels = self.run_info.env[0][target_node_name]
            else:
                outputs = tuple(mb["labels"] for mb in self.run_info.env)
                labels = torch.cat(outputs)
            return labels
        return None


    def move_labels2last_stage(self, labels):
        self.prepare_labels(labels)

        return self.ready_labels()


    def run(self, data, labels, mode="gpipe"):
        #schedule = SCHEDULE[mode](self.run_info, self.ir, self.comm, self.tpl)
        #
        #schedule.run(data, labels)
        self.schedule = SCHEDULE[mode](self.run_info, self.ir, self.comm, self.tpl, self.activation_ckpt)

        self.schedule.run(data, labels)
        

    def parameters(self):
        return self.run_info.submod.parameters()

    def train(self):
        return self.run_info.submod.train()

    def get_loss(self):
        return self.schedule.run_info.loss

    def is_first_stage(self):
        return self.tpl.is_first_stage()

    def is_last_stage(self):
        return self.tpl.is_last_stage()


    def prepare_dp_group(self):

        for i in range(0, self.tpl.pp_size):
            start_rank = i * self.tpl.dp_size
            end_rank = (i+1) * self.tpl.dp_size
            dp_group = list(range(start_rank, end_rank))
            if self.rank in dp_group:
                ddp_group = dist.new_group(dp_group)
                self.run_info.submod = DistributedDataParallel(self.run_info.submod, process_group=ddp_group, find_unused_parameters=False)
                print(f"Preparing DP group: {dp_group}")
            else:
                dist.new_group(dp_group)


    def get_output(self):
        if self.run_info.preserve_output == True and self.tpl.is_last_stage() == True:
            return self.run_info.output
        else:
            return None
