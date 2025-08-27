import torch
import torch.distributed as dist
import logging
import argparse
import sys

import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from opt_prime.comm import Comm
from opt_prime.IR import IR, IR_Anal
from opt_prime.schedule import ScheduleGPipe 
from opt_prime.schedule import Schedule1F1B 

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch.fx.graph_module import GraphModule

import psutil
import os
import glob


#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.ERROR)

SCHEDULE = {
    "gpipe": ScheduleGPipe,
    "1f1b": Schedule1F1B, 
    }



class Topology:

    def __init__(self, rank, local_rank, world_size, pp_size, dp_size, tp_size):
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.tp_size = tp_size

        self.device_mesh = init_device_mesh("cuda", mesh_shape=(pp_size, dp_size, tp_size), mesh_dim_names=("pp", "dp", "tp"))
        self.tp_group = self.device_mesh["tp"].get_group()
        self.dp_group = self.device_mesh["dp"].get_group()
        self.pp_group = self.device_mesh["pp"].get_group()
        self.tp_mesh = self.device_mesh["tp"]
        self.dp_mesh = self.device_mesh["dp"]
        self.pp_mesh = self.device_mesh["pp"]

        #if rank == 0:
        #    print(f"> pp group:{self.pp_mesh}, tp group:{self.tp_mesh}, dp group:{self.dp_mesh}")
        print(f"> [rank:{rank}] pp group:{self.pp_mesh}, tp group:{self.tp_mesh}, dp group:{self.dp_mesh}")


        #
        self.stage2rank = {}
        self.set_stage2rank()

        self.set_stage()
        self.num_stage = self.get_num_stage()
        self.setup_rank_topology()


    def set_stage2rank(self):
        #for i in range(self.pp_size):
        #    self.stage2rank[i] = [i*self.dp_size + j for j in range(self.dp_size)]
        pp_dim = 0
        total_ranks = self.device_mesh.mesh
        for pp_stage in range(self.pp_size):
            stage_ranks = total_ranks[pp_stage, :, :].flatten().tolist()
            self.stage2rank[pp_stage] = stage_ranks

        if self.rank == 0:
            print(f" ----------------------------------")
            print(f"stage2rank = { self.stage2rank }")
            print(f" ----------------------------------")


    def get_rank2stage(self, rank):
        for stage, ranks in self.stage2rank.items():
            if rank in ranks:
                return stage
        return None

    def set_stage(self):
        ## PP only
        ##if self.dp_size == 1: 
        ##    self.stage = self.rank

        # PP + DP: using calculation
        #self.stage = self.rank // self.dp_size

        # PP + DP: using data structure
        self.stage = self.get_rank2stage(self.rank)

    def get_stage(self):
        return self.stage

    def get_num_stage(self):
        # PP, PP + DP
        #return self.pp_size
        return len(self.stage2rank)

    def is_first_stage(self):
        # PP only
        #if self.dp_size == 1:
        #    return self.rank == 0:

        return self.stage == 0

    def is_last_stage(self):
        ## PP only
        ##if self.dp_size == 1:
        ##    return self.rank == self.world_size - 1

        # PP + DP
        #return self.stage == self.pp_size - 1
        return self.stage == self.num_stage - 1


    def get_first_stage(self):
        # PP only
        #if self.dp_size == 1:
        #    return 0

        return 0
            
    def get_last_stage(self):
        ## PP only
        ##if self.dp_size == 1: 
        ##    return self.world_size - 1

        # PP + DP
        #return self.pp_size - 1
        return self.num_stage - 1

    def get_next_stage(self):
        # PP only
        #if self.dp_size == 1: 
        #    assert self.stage < self.pp_size - 1
        #    return self.stage + 1

        # PP + DP
        assert self.stage < self.get_last_stage()
        return self.stage + 1

    def get_prev_stage(self):
        # PP only
        #if self.dp_size == 1:
        #    assert self.stage > 0
        #    return self.stage - 1

        # PP + DP
        assert self.stage > self.get_first_stage()
        return self.stage - 1


    def setup_rank_topology(self):
        stage = self.get_rank2stage(self.rank)
        tlist = self.stage2rank[stage]
        index = tlist.index(self.rank)

        self.first_rank = self.stage2rank[0][index]
        self.last_rank = self.stage2rank[self.get_last_stage()][index]
        if self.stage < self.get_last_stage():
            self.next_rank = self.stage2rank[self.get_next_stage()][index]
        if self.stage > self.get_first_stage():
            self.prev_rank = self.stage2rank[self.get_prev_stage()][index]


    def get_first_rank(self):
        # PP + DP: using calculation
        #return self.rank % self.dp_size

        # PP + DP: using data structure
        return self.first_rank

    def get_last_rank(self):
        # PP + DP: using calculation
        #return (self.pp_size - 1)*self.dp_size + self.rank % self.dp_size

        # PP + DP: using data structure
        return self.last_rank


    def get_prev_rank(self):
        assert self.stage > self.get_first_stage()

        # PP + DP: using calculation
        #return (self.stage - 1) * self.dp_size + self.get_first_rank()

        # PP + DP: using data structure
        return self.prev_rank


    def get_next_rank(self):
        assert self.stage < self.get_last_stage()
        # PP + DP: using calculation
        #return (self.stage + 1)*self.dp_size + self.get_first_rank()

        # PP + DP: using data structure
        return self.next_rank


    def print_stage_info(self):
        print(f"rank:{self.rank}, get_stage: {self.get_stage()}")
        print(f"rank:{self.rank}, get_first_stage: {self.get_first_stage()}")
        print(f"rank:{self.rank}, get_last_stage: {self.get_last_stage()}")
        print(f"rank:{self.rank}, get_first_rank: {self.get_first_rank()}")
        print(f"rank:{self.rank}, get_last_rank: {self.get_last_rank()}")




class Run_Info:

    #def __init__(self, ir, device, mbsize):
    def __init__(self, device, mbsize, num_classes):
        #self.mod = ir.model_ir[0] # TODO
        #self.graph = self.mod.graph
        self.name = None
        self.node = None
        self.submod = None

        self.output_node = None
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

        #self.special_nodes: Dict[str, Tuple[int, int]] = ir.special_nodes
        self.special_nodes: Dict[str, Tuple[int, int]] = {}  # { node_name : {stage#, needed-by-stage#),}
        self.metadata_range = []

        self.state_dict_cpu = {}

        self.num_classes = num_classes


    #def setup_special_nodes(self, ir):
    #    self.special_nodes = ir.special_nodes


    # TODO: delele ??
    #def setup_submod(self, stage, rank):
    #
    #    name, submod, node = None, None, None
    #
    #    for n, m in self.ir.model_ir[0].named_children():
    #    #for n, m in self.mod.named_children():
    #         if n == f"submod_{stage}" and isinstance(m, GraphModule):
    #             #self.name = n
    #             #self.submod = m
    #             name = n
    #             submod = m
    #             break
    #
    #    #if self.name is None:
    #    if name is None:
    #        print(f"ERROR: Not found name(submod_{stage})")
    #        sys.exit(0)
    #
    #    #print(f" ## Rank:{rank}, name:{self.name}")
    #
    #    for n in self.graph.nodes:
    #        #if n.name == self.name:
    #        if n.name == name:
    #           #self.node = n
    #           node = n
    #           break
    #
    #    #if self.node is None:
    #    if node is None:
    #        print(f"ERROR: Not found node({self.name})")
    #        sys.exit(0)
    #
    #
    #    #self.submod.to(self.device)
    #    #submod.to(self.device)
    #
    #    # TODO
    #    print(f" ## Rank:{rank}, name:{self.node.name}, move {self.name} to {self.device}")
    #
    #    return name, submod, node


    # TODO: move to IR, then broadcast the result
    #def build_getitem_dic(self):
    #    for node in self.mod.graph.nodes:
    #        if node.op == 'call_function' and node.name.startswith("getitem"):
    #            self.getitem_dic[node.name] = (node.args[0].name, node.args[1])


    #def print_graph(self, ir, rank):
    #    print(f" # rank = {rank}, metadata_range:{ir.metadata_range}")
    #    for node in self.mod.graph.nodes:
    #        print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.args:{node.args}, node.all_input_nodes:{node.all_input_nodes}")

    def print_getitem_dic(self):
        print(f" ========= getitem_dic =========")
        for k, v in self.getitem_dic.items():
            print(f" --- key:{k}, values:{v[0],v[1]}")
        print(f" ===============================")


    def clean_run_info(self, mbsize):
        self.env = [{} for _ in range(mbsize)]
        self.flat_args = [{} for _ in range(mbsize)]
        self.grads = [{} for _ in range(mbsize)]


pid = os.getpid()
def print_cpu_memory_usage(str, print_flag = False):
    if print_flag == True:
        my_process = psutil.Process(pid)
        usage = my_process.memory_info().rss / (1024 ** 3) # GB unit
        print(f" === {str} === rank:[{int(os.environ['RANK'])}] >> Memory Usage: {usage:.3f} GB")


class Optimus_p:

    def __init__(self, module:nn.Module, mbsize, use_gpu=False, pp_size=1, dp_size=1, tp_size=1, preserve_output=False, activation_ckpt=False, force_free_mem=False, display_mem=False, swap_opt_in_fwdbwd=False, swap_model_in_optstep=False, ir_analyze: IR_Anal = IR_Anal.PARALLEL, use_padding=True, pre_barrier=None, checkpoint=False, ckpt_dir_postfix: Optional[str]= None, prt_shape=False, swap_use_disk=False):

        #self.model_ir = []
        self.mbsize = mbsize

        #self.special_nodes: Dict[str, Tuple[int, int]] = {}  # { node_name : {stage#, needed-by-stage#),}

        self.use_gpu = use_gpu

        self.comm = Comm(use_gpu=use_gpu, ir_analyze=ir_analyze)

        self.activation_ckpt = activation_ckpt


        rank = self.comm.rank
        world_size = self.comm.world_size
        local_rank = self.comm.local_rank

        if world_size == 1:
            print(f"> WORLD SIZE is 1. mbsize reset to 1")
            self.mbsize = mbsize = 1

        assert tp_size >= 1 and world_size % tp_size == 0, f"world size({world_size}) must be divisible by tp size({tp_size})"

        assert dp_size >= 1 and world_size % dp_size == 0, f"world size({world_size}) must be divisible by dp size({dp_size})"


        if pp_size == 1:
            pp_size = world_size // tp_size // dp_size

        assert pp_size >= 1 and world_size == pp_size * dp_size * tp_size, f"world size({world_size}) == pp_size({pp_size}) * dp_size({dp_size}) * tp_size({tp_size})"

        if tp_size > 1:
            assert "llama" in  module.__class__.__name__.lower(), f"Tensor parallel (size={tp_size}) is only supported for Llama models."

        ##pp_size = world_size // dp_size
        #pp_size = world_size // tp_size // dp_size

        if rank == 0:
            print(f"> World Size: {world_size}")

            print(f"> Pipeline Parallel Size: {pp_size}")  

            if dp_size > 1:
                print(f"> Data Parallel Size: {dp_size}")

            if tp_size > 1:
                print(f"> Tensor Parallel Size: {tp_size}")

            print(f">> ir_analyze: {ir_analyze}")


        self.tpl = Topology(rank, local_rank, world_size, pp_size, dp_size, tp_size)

        self.checkpoint = checkpoint
        self.ckpt_dir = f"checkpoint_{ckpt_dir_postfix}_{self.tpl.stage}" or f"checkpoint_{self.tpl.stage}"
        if self.checkpoint == True:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self.prt_shape = prt_shape

        if use_gpu == True:
            torch.cuda.set_device(local_rank) # TODO
            self.device = torch.device(f"cuda:{local_rank}")
            print(f">>> Using GPU ... cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
            print(f">>> Using CPU ...")


        # num_classes auto config
        self.ignore_index = -100

        self.run_info = Run_Info(device=self.device, mbsize=mbsize, num_classes=self.ignore_index)
        #self.model2type = { "hf" : 50, "sy" : 51,}
        self.model2type = { "hf" : 50, "sy" : 51, "vt" : 52,}
        self.model_type = None

        self.clean_module_memory = True

        # TODO
        split_method = "llama-tp-split" if module.__class__.__name__.startswith("Llama") and tp_size > 1 else "simple"
        print(f">> model class name: {module.__class__.__name__}")
        print(f">> split method: {split_method}")

        if ir_analyze == IR_Anal.SEQUENTIAL:
            print(f"SEQUENTIAL mode >> [rank:{rank}, local_world_size:{self.comm.local_world_size}]")

            for i in range(self.comm.local_world_size):
                if local_rank == i:
                    self.ir = IR(module, self)

                    self.model_type = self.ir.retrieve_IR(module)
                    #self.ir.split_IR(module, "simple", num_stage=self.tpl.get_num_stage())
                    self.ir.split_IR(module, split_method, num_stage=self.tpl.get_num_stage())

                    self.ir.setup_submod(self.tpl.stage, rank) # setup name, submod, node
                    self.ir.build_getitem_dic()

                    self.run_info.submod.to(self.run_info.device)
                    print(f" ### Rank:{rank}, name:{self.run_info.node.name}, move {self.run_info.name} to {self.run_info.device}")

                    self.run_info.output_node = self.ir.get_output_node()

                    if rank == 0:
                        self.ir.print_graph(rank)
                        self.run_info.print_getitem_dic()

                    #    for stage in reversed(range(1, self.tpl.get_num_stage())):
                    #        self.ir.cross_reference_analyze(stage, self.ir.model_ir[0].graph)

                    for stage in reversed(range(1, self.tpl.get_num_stage())):
                         self.ir.cross_reference_analyze(stage, self.ir.model_ir[0].graph)
                    self.run_info.special_nodes = self.ir.special_nodes
                    self.run_info.metadata_range = self.ir.metadata_range
                    self.run_info.getitem_dic = self.run_info.getitem_dic

                    if self.prt_shape == True:
                        self.ir.print_module_shape_before_parallel()

                    if self.clean_module_memory == True:
                        print_cpu_memory_usage(f"[Rank:{rank}] Before: clean_module_memory")
                        self.ir.clean_module_memory()
                        print(f" ### Rank:{rank}, clean_module_memory ...")
                        print_cpu_memory_usage(f"[Rank:{rank}] After: clean_module_memory")
                    print(f"[rank:{rank}, local_rank:{local_rank}] SEQUENTIAL MODE PROCESSING ...")

                dist.barrier()


        if (ir_analyze == IR_Anal.SINGLE and rank == 0) or ir_analyze == IR_Anal.PARALLEL:
            # IR effective at #0 process when IR_Anal.SINGLE

            self.ir = IR(module, self)
            #self.model_ir.append(IR(module))

            self.model_type = self.ir.retrieve_IR(module)
            #self.ir.split_IR(module, "simple", num_stage=self.tpl.get_num_stage())
            self.ir.split_IR(module, split_method, num_stage=self.tpl.get_num_stage())

            self.ir.setup_submod(self.tpl.stage, rank) # setup name, submod, node
            self.ir.build_getitem_dic()

            if ir_analyze == IR_Anal.SINGLE and rank == 0:

                to_name, to_submod, to_node = None, None, None

                for stage in range(self.tpl.num_stage):
                    #for n, m in self.ir.model_ir[0].named_children():
                    #    if n == f"submod_{stage}" and isinstance(m, GraphModule):
                    #        to_name = n
                    #        to_submod = m
                    #        break
                    to_name = f"submod_{stage}"
                    to_submod = self.ir.model_ir[0].get_submodule(to_name)

                    for nd in self.ir.model_ir[0].graph.nodes:
                        if nd.name == to_name:
                            to_node = nd
                            break

                    object_list = [to_name, to_submod, to_node]

                    for to_rank in self.tpl.stage2rank[stage]:

                        if to_rank == 0:
                            continue
                        else:
                            print(f"[Rank:0] >> Send IR partition to rank:{to_rank} ...")
                            dist.broadcast_object_list(object_list, src=0, group=self.comm.ctrl_group[to_rank], device=self.run_info.device)
                    to_name, to_submod, to_node = None, None, None
                    object_list = []

            elif ir_analyze == IR_Anal.PARALLEL:
                self.run_info.submod.to(self.run_info.device)
                print(f" ### Rank:{rank}, name:{self.run_info.node.name}, move {self.run_info.name} to {self.run_info.device}")

                self.run_info.output_node = self.ir.get_output_node()

                if rank == 0:
                    self.ir.print_graph(rank)
                    self.run_info.print_getitem_dic()

                for stage in reversed(range(1, self.tpl.get_num_stage())):
                    self.ir.cross_reference_analyze(stage, self.ir.model_ir[0].graph)
                self.run_info.special_nodes = self.ir.special_nodes
                self.run_info.metadata_range = self.ir.metadata_range
                self.run_info.getitem_dic = self.run_info.getitem_dic

                if self.prt_shape == True:
                    self.ir.print_module_shape_before_parallel()

                if self.clean_module_memory == True:
                    print_cpu_memory_usage(f"[Rank:{rank}] Before: clean_module_memory")
                    self.ir.clean_module_memory()
                    print(f" ### Rank:{rank}, clean_module_memory ...")
                    print_cpu_memory_usage(f"[Rank:{rank}] After: clean_module_memory")
                print(f"[rank:{rank}, local_rank:{local_rank}] PARALLEL MODE PROCESSING ...")

                # TODO
                if pre_barrier is not None:
                    dist.barrier(group=pre_barrier)
                    pre_barrier_remaining = self.comm.local_world_size - self.comm.local_rank - 1
                    for j in range(pre_barrier_remaining):
                        dist.barrier(group=pre_barrier)


        elif ir_analyze == IR_Anal.SINGLE and rank != 0:
            object_list = [None, None, None]
            dist.broadcast_object_list(object_list, src=0, group=self.comm.ctrl_group[rank], device=self.run_info.device)
            self.run_info.name = object_list[0]
            print(f"<< [Rank:{rank}, Stage:{self.tpl.stage}] <== Received {self.run_info.name} ...")
            self.run_info.submod = object_list[1]
            self.run_info.node = object_list[2]

        if ir_analyze == IR_Anal.SINGLE:
            self.run_info.submod.to(self.run_info.device)
            print(f" ### Rank:{rank}, name:{self.run_info.node.name}, move {self.run_info.name} to {self.run_info.device}")

            if rank == 0:
                #self.run_info.output_node = self.ir.get_output_node()
                object_list2 = [self.ir.get_output_node()]
                for to_rank in self.tpl.stage2rank[self.tpl.get_last_stage()]:
                    if to_rank == 0:
                        continue
                    dist.broadcast_object_list(object_list2, src=0, group=self.comm.ctrl_group[to_rank], device=self.run_info.device)
                    print(f" [Rank:0] >>>> Send output node[:{self.run_info.output_node}] to rank:{to_rank} ...")
                self.run_info.output_node = object_list2[0]
                #print(f" >>>> Send output node to rank:{to_rank} ...")
            elif rank in self.tpl.stage2rank[self.tpl.get_last_stage()]:
                if rank != 0:
                    object_list2 = [None]
                    dist.broadcast_object_list(object_list2, src=0, group=self.comm.ctrl_group[rank], device=self.run_info.device)
                    self.run_info.output_node = object_list2[0]
                    print(f"[Rank:{rank}, Stage:{self.tpl.stage}] <<<< Received output node[{self.run_info.output_node}] ...")

        if ir_analyze == IR_Anal.SINGLE:
            if rank == 0:
                self.ir.print_graph(rank)
                self.run_info.print_getitem_dic()

                for stage in reversed(range(1, self.tpl.get_num_stage())):
                    self.ir.cross_reference_analyze(stage, self.ir.model_ir[0].graph)

                if self.prt_shape == True:
                    self.ir.print_module_shape_before_parallel()

                if self.clean_module_memory == True:
                    print_cpu_memory_usage(f"[Rank:{rank}] Before: clean_module_memory")
                    self.ir.clean_module_memory()
                    print(f" ### Rank:{rank}, clean_module_memory ...")
                    print_cpu_memory_usage(f"[Rank:{rank}] After: clean_module_memory")
                print(f"[rank:{rank}, local_rank:{local_rank}] SINGLE MODE PROCESSING ...")

        self.preserve_output = preserve_output

        self.force_free_mem = force_free_mem
        self.free_threshold = 4294967296 # 4GB # For forcefully garbage collection/cache cleaning
        self.free_threshold2 = 5368709120 # 5GB # For optimizer offloading
        #self.free_threshold3 = 22548578304 # 21GB # For model offloading
        self.free_threshold3 = 26843545600 # 25GB # For model offloading

        self.display_mem = display_mem

        if tp_size > 1:
            self.prepare_tp_group()

        if dp_size > 1:
            self.prepare_dp_group()

        #if rank == 0:
        #    self.ir.print_graph(rank)
        #    self.run_info.print_getitem_dic()

        if ir_analyze == IR_Anal.SINGLE:
            if rank == 0:
                #for stage in reversed(range(1, self.tpl.get_num_stage())):
                #    self.ir.cross_reference_analyze(stage, self.ir.model_ir[0].graph)

                special_nodes_obj = [self.ir.special_nodes, self.ir.metadata_range, self.run_info.getitem_dic, self.model_type]
                print(f" ### Rank:{rank}, local_rank:{local_rank} - before broadcast_object_list.. ")
                dist.broadcast_object_list(special_nodes_obj, src=0, device=self.device)
                print(f" ### Rank:{rank}, local_rank:{local_rank} - after broadcast_object_list.. ")
                self.run_info.special_nodes = special_nodes_obj[0]
                self.run_info.metadata_range = special_nodes_obj[1]
                self.run_info.getitem_dic = special_nodes_obj[2]
            else:
                special_nodes_obj = [None, None, None, None]
                print(f" ### Rank:{rank}, local_rank:{local_rank} - before broadcast_object_list.. ")
                dist.broadcast_object_list(special_nodes_obj, src=0, device=self.device)
                print(f" ### Rank:{rank}, local_rank:{local_rank} - after broadcast_object_list.. ")
                self.run_info.special_nodes = special_nodes_obj[0]
                self.run_info.metadata_range = special_nodes_obj[1]
                self.run_info.getitem_dic = special_nodes_obj[2]
                self.model_type = special_nodes_obj[3]


        print(f" *********** rank:{rank} cross-referenced nodes *****************")
        print(f"   special_nodes: {self.run_info.special_nodes}")
        print(f" *************************************************************************")

        #if self.clean_module_memory == True:
        #    if (ir_analyze == IR_Anal.PARALLEL) or (ir_analyze == IR_Anal.SINGLE and rank == 0):
        #        print_cpu_memory_usage(f"[Rank:{rank}] Before: clean_module_memory")
        #        self.ir.clean_module_memory()
        #        print(f" ### Rank:{rank}, clean_module_memory ...")
        #        print_cpu_memory_usage(f"[Rank:{rank}] After: clean_module_memory")

        self.optimizer = None  # TODO
        self.swap_opt_in_fwdbwd = swap_opt_in_fwdbwd 
        self.swap_model_in_optstep = swap_model_in_optstep 
        self.use_padding = use_padding  # padding option

        if self.prt_shape == True:
            self.ir.print_module_shape_after_parallel()

        self.swap_use_disk = swap_use_disk


    def prepare_labels(self, labels):
        if self.tpl.is_first_stage():
            target_node_name = "labels"

            # labels padding
            if self.use_padding and labels.size(0) % self.mbsize != 0:
                padding_size = self.mbsize - (labels.size(0) % self.mbsize)
                # Use class value as padding
                padding_value = self.ignore_index  
                padding = torch.full((padding_size, *labels.size()[1:]), padding_value, device=labels.device, dtype=labels.dtype)
                labels = torch.cat([labels, padding], dim=0)

            mbatches = torch.chunk(labels, self.mbsize)
            assert len(mbatches) == self.mbsize, f"len(mbatches):[{len(mbatches)}] is not equal to mbsize:[{self.mbsize}]"
            if self.mbsize == 1:
                self.run_info.env[0][target_node_name] = labels
            else:
                for j in range(self.mbsize):
                    self.run_info.env[j][target_node_name] = mbatches[j]

            if self.comm.world_size > 1:
                for j in range(self.mbsize):
                    obj = self.run_info.env[j][target_node_name]
                    self.comm.send_data(obj, self.tpl.get_last_rank(), self.device)
            else:
                self.run_info.env[0][target_node_name] = self.run_info.env[0][target_node_name].to(self.device)


    def ready_labels(self):

        if self.tpl.is_last_stage():
            target_node_name = "labels"

            if self.comm.world_size > 1:
                for j in range(self.mbsize):
                    self.run_info.env[j][target_node_name] = self.comm.receive_data(self.tpl.get_first_rank(), self.device)
            if self.mbsize == 1:
                labels = self.run_info.env[0][target_node_name]
            else:
                outputs = tuple(mb["labels"] for mb in self.run_info.env)
                labels = torch.cat(outputs)
            return labels
        return None

    
    def prepare_dataloader(self, datasets, batch_size):
        if self.tpl.dp_size > 1:
            dp_rank = (self.get_rank() // self.tpl.tp_size) % self.tpl.dp_size
            return DataLoader(datasets, batch_size=batch_size, num_workers=4, sampler=DistributedSampler(datasets, shuffle=True, num_replicas=self.tpl.dp_size, rank=dp_rank))
        else:
            return DataLoader(datasets, batch_size=batch_size, num_workers=4)


    def move_labels2last_stage(self, labels):
        self.prepare_labels(labels)

        return self.ready_labels()


    def run(self, data, labels, mode="gpipe"):
        #schedule = SCHEDULE[mode](self.run_info, self.ir, self.comm, self.tpl)
        #
        #schedule.run(data, labels)
        #self.schedule = SCHEDULE[mode](self.run_info, self.ir, self.comm, self.tpl, self.activation_ckpt)
        self.schedule = SCHEDULE[mode](self)

        self.schedule.run(data, labels)
        

    def parameters(self):
        return self.run_info.submod.parameters()

    def train(self):
        return self.run_info.submod.train()

    def get_loss(self):
        return self.run_info.loss

    #def get_loss2(self):
    #    loss = sum(self.run_info.loss) / self.mbsize
    #    return loss.item()


    def is_first_stage(self):
        return self.tpl.is_first_stage()

    def is_last_stage(self):
        return self.tpl.is_last_stage()


    def prepare_dp_group(self):
    #
    #    for i in range(0, self.tpl.pp_size):
    #        start_rank = i * self.tpl.dp_size
    #        end_rank = (i+1) * self.tpl.dp_size
    #        dp_group = list(range(start_rank, end_rank))
    #        if self.tpl.rank in dp_group:
    #            ddp_group = dist.new_group(dp_group)
    #            #self.run_info.submod = DistributedDataParallel(self.run_info.submod, process_group=ddp_group, find_unused_parameters=False)
    #            self.run_info.submod = DistributedDataParallel(self.run_info.submod, process_group=ddp_group, find_unused_parameters=True)
    #            print(f"Preparing DP group: {dp_group}")
    #        else:
    #            dist.new_group(dp_group)
    #
        self.run_info.submod = DistributedDataParallel(self.run_info.submod, find_unused_parameters=True, device_mesh=self.tpl.dp_mesh)

    def check_tp_post(self, arg0):
        if isinstance(arg0, torch.fx.Node):
            arg0 = arg0.name
        parts = arg0.split('_')

        if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
            layer_id = int(parts[2])
            if parts[3] == "self" and parts[4] == "attn" and parts[5] == "q" and parts[6] == "proj": # self_attn_q_proj
                return 0, layer_id
            elif parts[3] == "self" and parts[4] == "attn" and parts[5] == "k" and parts[6] == "proj": # self_attn_k_proj
                return 1, layer_id
            elif parts[3] == "self" and parts[4] == "attn" and parts[5] == "v" and parts[6] == "proj": # self_attn_v_proj
                return 2, layer_id

        return -1, -1

    
    # TODO
    def prepare_tp_group(self):
        #
        #print(f"############## TP feature is coming soon. #############################")
        #sys.exit(0)
        # TODO

        tp_plan = {}
        for name, module in self.run_info.submod.named_modules():
            parts = name.split('_')
            if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                layer_id = int(parts[2])
                if parts[3] == "self" and parts[4] == "attn" and parts[5] == "q" and parts[6] == "proj": # self_attn_q_proj
                    tp_plan[f"model_layers_{layer_id}_self_attn_q_proj"] = ColwiseParallel()
                elif parts[3] == "self" and parts[4] == "attn" and parts[5] == "k" and parts[6] == "proj": # self_attn_k_proj
                    tp_plan[f"model_layers_{layer_id}_self_attn_k_proj"] = ColwiseParallel()
                elif parts[3] == "self" and parts[4] == "attn" and parts[5] == "v" and parts[6] == "proj": # self_attn_v_proj
                    tp_plan[f"model_layers_{layer_id}_self_attn_v_proj"] = ColwiseParallel()
                elif parts[3] == "self" and parts[4] == "attn" and parts[5] == "o" and parts[6] == "proj": # self_attn_o_proj
                    tp_plan[f"model_layers_{layer_id}_self_attn_o_proj"] = RowwiseParallel()
                elif parts[3] == "mlp" and parts[4] == "gate" and parts[5] == "proj": # mlp_gate_proj
                    tp_plan[f"model_layers_{layer_id}_mlp_gate_proj"] = ColwiseParallel()
                elif parts[3] == "mlp" and parts[4] == "down" and parts[5] == "proj": # mlp_down_proj
                    tp_plan[f"model_layers_{layer_id}_mlp_down_proj"] = RowwiseParallel()
                elif parts[3] == "mlp" and parts[4] == "up" and parts[5] == "proj": # mlp_up_proj
                    tp_plan[f"model_layers_{layer_id}_mlp_up_proj"] = ColwiseParallel()

        print(f" >> rank:{self.get_rank()} -----------------------------------------------")
        print(f"rank: {self.get_rank()}, #### last layer id:{layer_id}")
        print(f" >> rank:{self.get_rank()} -----------------------------------------------")


        print(f">>>> self.tpl.tp_mesh.size(): {self.tpl.tp_mesh.size()}")

        for node in self.run_info.submod.graph.nodes:
            if node.op == 'call_method' and node.name.startswith("view"):
                result, layer_id = self.check_tp_post(node.args[0])

                if result == 0:
                    print(f">model_layers_{layer_id}_self_attn_q_proj ==> node.args[3]:{node.args[3]}")
                    new_args = list(node.args)
                    new_args[3] = new_args[3] // self.tpl.tp_mesh.size()
                    node.args = tuple(new_args)
                    print(f"> >model_layers_{layer_id}_self_attn_q_proj ==> node.args[3]:{node.args[3]}")

                elif result == 1:
                    print(f">>model_layers_{layer_id}_self_attn_k_proj ===> node.args[3]:{node.args[3]}")
                    new_args = list(node.args)
                    new_args[3] = new_args[3] // self.tpl.tp_mesh.size()
                    node.args = tuple(new_args)
                    print(f">> >> model_layers_{layer_id}_self_attn_k_proj ===> node.args[3]:{node.args[3]}")

                elif result == 2:
                    print(f">>>model_layers_{layer_id}_self_attn_v_proj ===> node.args[3]:{node.args[3]}")
                    new_args = list(node.args)
                    new_args[3] = new_args[3] // self.tpl.tp_mesh.size()
                    node.args = tuple(new_args)
                    print(f">>> >>>model_layers_{layer_id}_self_attn_v_proj ===> node.args[3]:{node.args[3]}")

        self.run_info.submod.recompile()
        self.run_info.submod = parallelize_module(module=self.run_info.submod, device_mesh=self.tpl.tp_mesh, parallelize_plan=tp_plan)


    def get_output(self):
        if self.preserve_output == True and self.tpl.is_last_stage() == True:
            return self.run_info.output
        else:
            return None

    def get_rank(self):
        return self.tpl.rank

    def get_local_rank(self):
        return self.tpl.local_rank

    def get_world_size(self):
        return self.tpl.world_size

    def save_ckpt(self, step, epoch):
        if self.checkpoint != True:
            return

        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_step_{step}_epoch_{epoch}.pt")

        torch.save({
            'epoch': epoch,
            'stage': self.tpl.stage,
            'stage_state_dict': self.run_info.submod.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        ckpt_files = sorted(
                glob.glob(os.path.join(self.ckpt_dir, "ckpt_step_*.pt")),
                key=os.path.getmtime,
                reverse=True
                )
        for old_ckpt in ckpt_files[2:]:
            os.remove(old_ckpt)
            print(f"Old checkpoint deleted: {old_ckpt}")

    def load_ckpt(self, step, epoch):
        if self.checkpoint != True:
            return

        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_step_{step}_epoch_{epoch}.pt")

        ckpt = torch.load(ckpt_path)

        self.run_info.submod.load_state_dict(ckpt['stage_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        print(f"Checkpoint loaded: {ckpt_path}")
