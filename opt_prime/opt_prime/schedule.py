#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#

import torch
import torch.nn as nn

from torch import Tensor, Size
from torch import fx
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn.parallel import DistributedDataParallel

import gc



sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.ERROR)

optimizer_offloaded = False
model_offloaded = False

class Schedule:

    def __init__(self, optimus): 
        self.optimus = optimus

        if self.optimus.force_free_mem == True:
            self.total_mem = torch.cuda.get_device_properties(self.optimus.tpl.local_rank).total_memory 
            self.allocated_mem = torch.cuda.memory_allocated(self.optimus.tpl.local_rank) 
            self.cached_mem = torch.cuda.memory_reserved(self.optimus.tpl.local_rank) 

        global optimizer_offloaded

        global model_offloaded

        ##
        self.placeholder_name = self.get_placeholder_name()

    ##
    def get_placeholder_name(self):
        if self.optimus.model_type == self.optimus.model2type["hf"]:
            return "input_ids"
        elif self.optimus.model_type == self.optimus.model2type["vt"]:
            return "pixel_values"
        elif self.optimus.model_type == self.optimus.model2type["sy"]:
            return "x"


    def init_env_mark(self, mb_idx):
        #self.optimus.run_info.env_recv_mark[mb_idx]["input_ids"] = None # TODO: Seq Cls.
        #self.optimus.run_info.env_send_mark[mb_idx]["input_ids"] = None # TODO: Seq Cls.
        self.optimus.run_info.env_recv_mark[mb_idx][self.placeholder_name] = None # TODO: Seq Cls.
        self.optimus.run_info.env_send_mark[mb_idx][self.placeholder_name] = None # TODO: Seq Cls.
        for i in range(len(self.optimus.run_info.metadata_range)):
            self.optimus.run_info.env_recv_mark[mb_idx][self.optimus.run_info.metadata_range[i][1]] = None
            self.optimus.run_info.env_send_mark[mb_idx][self.optimus.run_info.metadata_range[i][1]] = None


    def init_env_grad_mark(self, mb_idx):
        for i in range(len(self.optimus.run_info.metadata_range)):
            self.optimus.run_info.env_grad_recv_mark[mb_idx][self.optimus.run_info.metadata_range[i][1]] = None
            self.optimus.run_info.env_grad_send_mark[mb_idx][self.optimus.run_info.metadata_range[i][1]] = None

            self.optimus.run_info.grads[mb_idx][self.optimus.run_info.metadata_range[i][1]] = None


    def get_input(self, *args):
        self.args_iter = iter(args)

        if self.optimus.tpl.is_first_stage():
            input = next(self.args_iter)
            if isinstance(input, torch.Tensor):
                # Check if input size is smaller than mbsize before chunking
                if self.optimus.use_padding and input.size(0) % self.optimus.mbsize != 0:
                    padding_size = self.optimus.mbsize - (input.size(0) % self.optimus.mbsize)
                    padding_batch = torch.zeros_like(input[0:1]) 
                    padding = torch.cat([padding_batch] * padding_size)
                    input = torch.cat([input, padding])

                # Now chunk the padded input
                mbatches = torch.chunk(input, self.optimus.mbsize)
                # Now proceed as usual
                if self.optimus.mbsize == 1:
                    input = input.to(self.optimus.run_info.device)
                    self.optimus.run_info.env[0]["placeholder"] = input
                else:
                    for j in range(self.optimus.mbsize):
                        mbatch = mbatches[j].to(self.optimus.run_info.device)
                        self.optimus.run_info.env[j]["placeholder"] = mbatch
            else:
                logging.critical(f"### input:{input} not Tensor --> currently not supported!!")
                sys.exit(1)


    #def get_output_node(self):
    #    for n in reversed(self.optimus.run_info.graph.nodes):
    #        if n.op == 'output':
    #            return n


    def get_next_node_name(self):
        assert self.optimus.tpl.get_stage() < self.optimus.tpl.get_last_stage()

        next_node_name = self.optimus.run_info.metadata_range[self.optimus.tpl.get_next_stage()][1]
        return next_node_name

    def offload_optimizer(self):
        if self.optimus.swap_opt_in_fwdbwd == False:
            print(f"offload_optimizer() should be used when swap_opt_in_fwdbwd == True")
            return

        if self.optimus.optimizer == None:
            print(f"optimus.optimizer not set when swap_opt_in_fwdbwd == True")
            return

        state_dict = self.optimus.optimizer.state_dict()
        for state in state_dict['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

        if self.optimus.swap_use_disk == True:
            disk_path_opt = f"temp_optistat_{self.optimus.tpl.rank}"
            torch.save({'optimizer_state_dict': self.optimus.optimizer.state_dict()}, disk_path_opt)
            #print(f"[W] optimizer state dict --> DISK({disk_path_opt})")


    def load_optimizer(self):
        if self.optimus.swap_opt_in_fwdbwd == False:
            print(f"load_optimizer() should be used when swap_opt_in_fwdbwd == True")
            return

        if self.optimus.optimizer == None:
            print(f"optimus.optimizer not set when swap_opt_in_fwdbwd == True")
            return

        if self.optimus.swap_use_disk == True:
            disk_path_opt = f"temp_optistat_{self.optimus.tpl.rank}"
            opt_state = torch.load(disk_path_opt)
            self.optimus.optimizer.load_state_dict(opt_state['optimizer_state_dict'])
            #print(f"[R] optimizer state dict <-- DISK({disk_path_opt})")

        state_dict = self.optimus.optimizer.state_dict()
        for state in state_dict['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.optimus.run_info.device)

    def offload_model(self):
        if self.optimus.swap_model_in_optstep == False:
            print(f"offload_model() should be used when swap_model_in_optstep == True")
            return

        self.optimus.run_info.submod.to('cpu')

        if self.optimus.display_mem == True:
            print(f" >>> >>> [rank:{self.optimus.tpl.rank}], offload model ...")

        if self.optimus.swap_use_disk == True:
            disk_path_model = f"temp_model_{self.optimus.tpl.rank}"
            torch.save({'model_state_dict': self.optimus.run_info.submod.state_dict()}, disk_path_model)
            #print(f"[W] model state dict --> DISK({disk_path_model})")



    def load_model(self):
        if self.optimus.swap_model_in_optstep == False:
            print(f"load_model() should be used when swap_model_in_optstep == True")
            return

        if self.optimus.swap_use_disk == True:
            disk_path_model = f"temp_model_{self.optimus.tpl.rank}"
            model_state = torch.load(disk_path_model)
            self.optimus.run_info.submod.load_state_dict(model_state['model_state_dict'])
            #print(f"[R] model state dict <-- DISK({disk_path_model})")

        self.optimus.run_info.submod.to(self.optimus.run_info.device)

        if self.optimus.display_mem == True:
            print(f" >>> >>> [rank:{self.optimus.tpl.rank}], load model ...")

    def check_swap_model_in_optstep(self):
        global model_offloaded
        global optimizer_offloaded

        if self.optimus.swap_model_in_optstep == False:
            print(f"load_model() should be used when swap_model_in_optstep == True")
            return

        self.allocated_mem = torch.cuda.memory_allocated(self.optimus.tpl.local_rank) 
        self.cached_mem = torch.cuda.memory_reserved(self.optimus.tpl.local_rank)
        remain_mem = self.total_mem - self.cached_mem
        #if self.optimus.display_mem == True:
        #    print(f"###[rank:{self.optimus.tpl.rank}], remain[:{remain_mem}], total[:{self.total_mem}], cached[:{self.cached_mem}] ... in check_swap_model_in_optstep ...") # TO DELETE
        if remain_mem < self.optimus.free_threshold3:
            if model_offloaded == False:
                self.offload_model()
                model_offloaded = True
                #if self.optimus.display_mem == True:
                #    print(f" >>> >>> [rank:{self.optimus.tpl.rank}], offload model [remain:{remain_mem}]...")

                if optimizer_offloaded == False:
                    self.offload_optimizer()
                    optimizer_offloaded = True
                    if self.optimus.display_mem == True:
                        print(f" >>> [rank:{self.optimus.tpl.rank}], offload optimizer ...")


    def run_loss(self, mb_idx):
        assert self.optimus.tpl.is_last_stage() == True

        assert mb_idx < self.optimus.mbsize

        #node = self.get_output_node()
        node = self.optimus.run_info.output_node
        #if self.optimus.model_type == self.optimus.model2type["hf"]:
        if self.optimus.model_type == self.optimus.model2type["hf"] or self.optimus.model_type == self.optimus.model2type["vt"]:
            key_ = node.args[0]['logits']
        elif self.optimus.model_type == self.optimus.model2type["sy"]:
            key_ = node.args[0]


        if str(key_) in self.optimus.run_info.getitem_dic:
            a_submod = self.optimus.run_info.getitem_dic[str(key_)][0]
            a_idx = self.optimus.run_info.getitem_dic[str(key_)][1]
            output1_ = self.optimus.run_info.env[mb_idx][a_submod][a_idx]
        else:
            output1_ = self.optimus.run_info.env[mb_idx][str(key_)]

        target1_ = self.optimus.run_info.env[mb_idx]["labels"]

        #if self.optimus.model_type == self.optimus.model2type["hf"]:
        if self.optimus.model_type == self.optimus.model2type["hf"] or self.optimus.model_type == self.optimus.model2type["vt"]:
            output1_ = output1_.view(-1, output1_.size(-1))
            target1_ = target1_.view(-1)


        flat_args = []
        if isinstance(output1_, torch.Tensor) and output1_.is_floating_point():
            output1 = output1_.detach().to(self.optimus.run_info.device)
            output1.requires_grad_(output1_.requires_grad)
            #output1.requires_grad_(True)
            flat_args.append(output1)
            output1.grad = None
        else:
            output1 = output1_
            flat_args.append(output1)

        if isinstance(target1_, torch.Tensor) and target1_.is_floating_point():
            target1 = target1_.detach().to(self.optimus.run_info.device)
            target1.requires_grad_(True)
            #flat_args.append(target1)
        else:
            target1 = target1_
            #flat_args.append(target1)

        self.optimus.run_info.env[mb_idx]["labels"] = None 
        #if self.optimus.model_type == self.optimus.model2type["hf"]:
        if self.optimus.model_type == self.optimus.model2type["hf"] or self.optimus.model_type == self.optimus.model2type["vt"]:
            criterion = nn.CrossEntropyLoss(ignore_index=self.optimus.ignore_index)
        elif self.optimus.model_type == self.optimus.model2type["sy"]:
            criterion = nn.MSELoss()

        criterion = criterion.to(self.optimus.run_info.device)

        if output1.size(0) != target1.size(0):
            # Check for padding in target1 by looking for ignore_index
            if self.optimus.ignore_index in target1:
                # Find first padding index
                pad_start = (target1 == self.optimus.ignore_index).nonzero()[0].item()
                output1 = output1[:pad_start]
                target1 = target1[:pad_start]
            else:
                # If no padding found, take minimum size as fallback
                min_size = min(output1.size(0), target1.size(0))
                output1 = output1[:min_size]
                target1 = target1[:min_size]
        result = criterion(output1, target1)

        #print(f" >>>> loss: {result}, result.shape:{result.shape}")


        self.optimus.run_info.grads[mb_idx][node.name] = (None,)

        #self.optimus.run_info.loss[mb_idx] = result 
        if isinstance(result, torch.Tensor):
            self.optimus.run_info.loss[mb_idx] = result.item()
        else:
            self.optimus.run_info.loss[mb_idx] = result 

        self.optimus.run_info.env[mb_idx][node.name] = result
        self.optimus.run_info.flat_args[mb_idx][node.name] = flat_args




    def pre_fx_micro_forward_core(self, mb_idx):
        #from_, to_ = self.optimus.ir.get_range(self.optimus.tpl.get_stage(), self.optimus.run_info.graph)

        if self.optimus.tpl.is_first_stage():
            target_node_name = "placeholder"
            #if self.optimus.model_type == self.optimus.model2type["hf"]:
            #    self.optimus.run_info.env[mb_idx]["input_ids"] = self.optimus.run_info.env[mb_idx][target_node_name]
            #elif self.optimus.model_type == self.optimus.model2type["sy"]:
            #    self.optimus.run_info.env[mb_idx]["x"] = self.optimus.run_info.env[mb_idx][target_node_name]
            #else:
            #    print(f"Not supported model type!")
            #    sys.exit(1)
            self.optimus.run_info.env[mb_idx][self.placeholder_name] = self.optimus.run_info.env[mb_idx][target_node_name]

        if self.optimus.tpl.get_stage() > self.optimus.tpl.get_first_stage():
            pre_split_rank = self.optimus.tpl.get_prev_rank()
        
            for node_name, range_ in self.optimus.run_info.special_nodes.items():
                src_stage, needed_by_stage = range_
                if self.optimus.tpl.stage > src_stage and self.optimus.tpl.stage <= needed_by_stage:
                    if node_name in self.optimus.run_info.getitem_dic:
                        submod_name = self.optimus.run_info.getitem_dic[node_name][0]
                        if self.optimus.run_info.env_recv_mark[mb_idx][submod_name] is None:
                            self.optimus.run_info.env[mb_idx][submod_name] = self.optimus.comm.receive_data(pre_split_rank, self.optimus.run_info.device)
                            self.optimus.run_info.env_recv_mark[mb_idx][submod_name] = 1

                        if isinstance(self.optimus.run_info.env[mb_idx][submod_name], torch.Tensor):
                            if not self.optimus.run_info.env[mb_idx][submod_name].requires_grad or self.optimus.run_info.env[mb_idx][submod_name].grad_fn is None:
                                self.optimus.run_info.env[mb_idx][submod_name].requires_grad_(True)
                                logging.info(f" ###### node name:{submod_name} requires_grad(True) #####") 
                    else:
                        if self.optimus.run_info.env_recv_mark[mb_idx][node_name] is None:
                            self.optimus.run_info.env[mb_idx][node_name] = self.optimus.comm.receive_data(pre_split_rank, self.optimus.run_info.device)
                            self.optimus.run_info.env_recv_mark[mb_idx][node_name] = 1
                        # TODO: Seq Cls.
                        #if isinstance(self.optimus.run_info.env[mb_idx][node_name], torch.Tensor):
                        #if node_name != "input_ids" and isinstance(self.optimus.run_info.env[mb_idx][node_name], torch.Tensor):
                        if node_name != self.placeholder_name and isinstance(self.optimus.run_info.env[mb_idx][node_name], torch.Tensor):
                            if not self.optimus.run_info.env[mb_idx][node_name].requires_grad or self.optimus.run_info.env[mb_idx][node_name].grad_fn is None:
                                self.optimus.run_info.env[mb_idx][node_name].requires_grad_(True)
                                logging.info(f" ###### node name:{node_name} requires_grad(True) #####") 




    def fx_micro_forward_core(self, mb_idx):

        #self.init_env_mark(mb_idx)

        #forward one chunk !!
        flat_args = []
        def extract_tensor_args(b):
            # TODO
            if b.name in self.optimus.run_info.getitem_dic:
                a_submod = self.optimus.run_info.getitem_dic[b.name][0]
                a_idx = self.optimus.run_info.getitem_dic[b.name][1]
                a = self.optimus.run_info.env[mb_idx][a_submod][a_idx]
            else:
                a = self.optimus.run_info.env[mb_idx][b.name]
            #a = self.optimus.run_info.env[mb_idx][b.name]

            nonlocal flat_args
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                val = a.detach().to(self.optimus.run_info.device)
                #val.requires_grad_(a.requires_grad)
                val.requires_grad_(True)
                flat_args.append(val)
                return val
            else:
                flat_args.append(a)
                return a
            return a


        #print(f" [rank:{self.optimus.tpl.rank}] fx_micro_forward({mb_idx}), node.args:{self.optimus.run_info.node.args} .....")

        args = fx.graph.map_arg(self.optimus.run_info.node.args, extract_tensor_args)
        kwargs = fx.graph.map_arg(self.optimus.run_info.node.kwargs, extract_tensor_args)

        if isinstance(self.optimus.run_info.submod, DistributedDataParallel):
            with self.optimus.run_info.submod.no_sync():
                #logging.info(f" [FWD] DDP no_sync ... rank:{self.optimus.tpl.rank}, mb_idx:{mb_idx}")
                #result = self.optimus.run_info.submod(*args, **kwargs)
                result = self.optimus.run_info.submod(*args, **kwargs)
            #result = self.optimus.run_info.submod(*args, **kwargs)
        else:
            result = self.optimus.run_info.submod(*args, **kwargs)

        self.optimus.run_info.flat_args[mb_idx][self.optimus.run_info.name] = flat_args
        self.optimus.run_info.env[mb_idx][self.optimus.run_info.name] = result


    # For Act ckpt
    def produce_forward_output(self, mb_idx, node_name):

        flat_args = []
        def extract_tensor_args(b):
            # TODO
            if b.name in self.optimus.run_info.getitem_dic:
                a_submod = self.optimus.run_info.getitem_dic[b.name][0]
                a_idx = self.optimus.run_info.getitem_dic[b.name][1]
                a = self.optimus.run_info.env[mb_idx][a_submod][a_idx]
            else:
                a = self.optimus.run_info.env[mb_idx][b.name]
            #a = self.optimus.run_info.env[mb_idx][b.name]

            nonlocal flat_args
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                val = a.detach().to(self.optimus.run_info.device)
                #val.requires_grad_(a.requires_grad)
                val.requires_grad_(True) # TODO
                flat_args.append(val)
                return val
            else:
                flat_args.append(a)
                return a
            return a


        #print(f" [rank:{self.optimus.tpl.rank}] fx_micro_forward({mb_idx}), node.args:{self.optimus.run_info.node.args} .....")

        args = fx.graph.map_arg(self.optimus.run_info.node.args, extract_tensor_args)
        kwargs = fx.graph.map_arg(self.optimus.run_info.node.kwargs, extract_tensor_args)

        result = self.optimus.run_info.submod(*args, **kwargs)
        #with torch.no_grad():
        #    result = self.optimus.run_info.submod(*args, **kwargs)

        #print(f" [ACT CKPT] rank:{self.optimus.tpl.rank}, mb_idx:{mb_idx}, node name:{node_name}")

        self.optimus.run_info.flat_args[mb_idx][self.optimus.run_info.name] = flat_args
        #self.optimus.run_info.env[mb_idx][self.optimus.run_info.name] = result
        return result



    def post_fx_micro_forward_core(self, mb_idx):

        if self.optimus.tpl.stage < self.optimus.tpl.get_last_stage():
            next_split_rank = self.optimus.tpl.get_next_rank()
        
            for node_name, range_ in self.optimus.run_info.special_nodes.items():
                src_stage, needed_by_stage = range_
                if self.optimus.tpl.stage >= src_stage and self.optimus.tpl.stage < needed_by_stage:
                    if node_name in self.optimus.run_info.getitem_dic:
                        submod_name = self.optimus.run_info.getitem_dic[node_name][0]
                        if self.optimus.run_info.env_send_mark[mb_idx][submod_name] is None:
                            obj = self.optimus.run_info.env[mb_idx][submod_name]
                            self.optimus.comm.send_data(obj, next_split_rank, self.optimus.run_info.device)
                            self.optimus.run_info.env_send_mark[mb_idx][submod_name] = 1
                            if self.optimus.activation_ckpt == True and needed_by_stage - src_stage == 1: # For Act ckpt
                                self.optimus.run_info.env[mb_idx][submod_name] = None  
                                self.optimus.run_info.flat_args[mb_idx][submod_name] = None 
                                #print(f".. [Act ckpt] rank:{self.optimus.tpl.rank}, mb_idx:{mb_idx}, node name:{submod_name} <- None") 
                    else:
                        if self.optimus.run_info.env_send_mark[mb_idx][node_name] is None:
                            obj = self.optimus.run_info.env[mb_idx][node_name]
                            self.optimus.comm.send_data(obj, next_split_rank, self.optimus.run_info.device)
                            self.optimus.run_info.env_send_mark[mb_idx][node_name] = 1
                            if self.optimus.activation_ckpt == True and needed_by_stage - src_stage == 1: # For Act ckpt
                                self.optimus.run_info.env[mb_idx][node_name] = None 
                                self.optimus.run_info.flat_args[mb_idx][node_name] = None
                                #print(f"... [Act ckpt] rank:{self.optimus.tpl.rank}, mb_idx:{mb_idx}, node name:{node_name} <- None") 

        yield 0


    def get_num_nodes(self, name):
        cnt = 0
        for k, v in self.optimus.run_info.getitem_dic.items():
            if k == name:
                cnt = cnt +  1

        if cnt == 0:
            return 1

        return cnt


    def core_backward(self, forward_output, forward_output_gradient, forward_input, valid_index: List[int],):

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


    def run_core_backward(self, mb_idx, node, grads):

        if self.optimus.force_free_mem == True:
            self.cond_free_mem_()

        args = ()
        kwargs = dict()
        #if self.optimus.activation_ckpt == True and node.name != "output":
        if self.optimus.activation_ckpt == True and node.name != "output" and not self.optimus.tpl.is_last_stage():
            src, needed_by_stage = self.optimus.run_info.special_nodes[node.name]
            if needed_by_stage - src > 1:
                k1 = self.optimus.run_info.env[mb_idx].pop(node.name)
            else:
                k1 = self.produce_forward_output(mb_idx, node.name)
        else:
            k1 = self.optimus.run_info.env[mb_idx].pop(node.name)
            
        k1 = ((k1,) if not isinstance(k1, tuple) else k1)
        k2 = self.optimus.run_info.flat_args[mb_idx].pop(node.name)

        if self.optimus.preserve_output == True:
            self.optimus.run_info.output[mb_idx] = k1

        kwargs["forward_output"] = k1
        kwargs["forward_input"] = k2
        kwargs["forward_output_gradient"] = grads 

        num_nodes = self.get_num_nodes(node.name) 
        kwargs["valid_index"] = [i for i in range(num_nodes)]

        if isinstance(self.optimus.run_info.submod, DistributedDataParallel):
            if mb_idx == self.optimus.mbsize - 1:
                #logging.info(f" DDP ... [node.name:{node.name}], [mb_idx:{mb_idx}], prepare_for_backward ...") 
                self.optimus.run_info.submod.reducer.prepare_for_backward(list(torch.nn.parallel.distributed._find_tensors(kwargs['forward_output'])))
                result = self.core_backward(*args, **kwargs)
            
            else:
                with self.optimus.run_info.submod.no_sync():
                    result = self.core_backward(*args, **kwargs)
        else:
            result = self.core_backward(*args, **kwargs)


        if self.optimus.force_free_mem == True:
            self.cond_free_mem_()

        return result


    def pre_fx_micro_backward_core(self, mb_idx):
        grads = None
        if self.optimus.tpl.stage < self.optimus.tpl.get_last_stage():
            pre_split_rank = self.optimus.tpl.get_next_rank()

            node_name = self.get_next_node_name()
            if self.optimus.run_info.env_grad_recv_mark[mb_idx][node_name] is None:
                self.optimus.run_info.grads[mb_idx][node_name] = self.optimus.comm.receive_data(pre_split_rank, self.optimus.run_info.device)
                grads = self.optimus.run_info.grads[mb_idx][node_name]
                self.optimus.run_info.env_grad_recv_mark[mb_idx][node_name] = 1

        return grads


    def fx_micro_backward_core(self, mb_idx, grads):

        #self.init_env_grad_mark(mb_idx)

        #if self.optimus.tpl.rank == self.optimus.tpl.world_size - 1:
        if self.optimus.tpl.is_last_stage():
            #node = self.get_output_node()
            node = self.optimus.run_info.output_node
            grads = self.optimus.run_info.grads[mb_idx][node.name]
            result = self.run_core_backward(mb_idx, node, grads)
            result = ((result,) if not isinstance(result, tuple) else result)
            #self.optimus.run_info.grads[mb_idx][node.name] = result 
            #grads = self.optimus.run_info.grads[mb_idx][node.name] 
            grads = result

        node = self.optimus.run_info.node
        result = self.run_core_backward(mb_idx, node, grads)

        result = ((result,) if not isinstance(result, tuple) else result)

        self.optimus.run_info.grads[mb_idx][node.name] = result



    def post_fx_micro_backward_core(self, mb_idx):

        if self.optimus.tpl.get_stage() > 0:
            next_split_rank = self.optimus.tpl.get_prev_rank()

            node_name = self.optimus.run_info.name
            if self.optimus.run_info.env_grad_send_mark[mb_idx][node_name] is None:
                obj = self.optimus.run_info.grads[mb_idx][node_name]
                self.optimus.comm.send_data(obj, next_split_rank, self.optimus.run_info.device)
                self.optimus.run_info.env_grad_send_mark[mb_idx][node_name] = 1
                if self.optimus.activation_ckpt == True:
                    self.optimus.run_info.grads[mb_idx][node_name] = None

        yield 0


    def force_free_mem(self):
        gc.collect()
        torch.cuda.empty_cache()
        #print(f"###[rank:{self.optimus.tpl.rank}], .. forcefully free memory")

            

    def cond_free_mem_(self):
        global optimizer_offloaded

        self.allocated_mem = torch.cuda.memory_allocated(self.optimus.tpl.local_rank) 
        self.cached_mem = torch.cuda.memory_reserved(self.optimus.tpl.local_rank)
        remain_mem = self.total_mem - self.cached_mem
        if remain_mem < self.optimus.free_threshold:
            if self.optimus.display_mem == True:
                print(f"###[rank:{self.optimus.tpl.rank}], remain[:{remain_mem}], total[:{self.total_mem}], cached[:{self.cached_mem}] ... forcefully free memory")

            gc.collect()
            torch.cuda.empty_cache()

        if remain_mem < self.optimus.free_threshold2:
            #if self.optimizer_offloaded == False:
            if optimizer_offloaded == False:
                if self.optimus.swap_opt_in_fwdbwd == True:
                    self.offload_optimizer()
                    if self.optimus.display_mem == True:
                        print(f" >>> [rank:{self.optimus.tpl.rank}], offload optimizer [remain:{remain_mem}]...")
                    #self.optimizer_offloaded = True
                    optimizer_offloaded = True


class ScheduleGPipe(Schedule):

    def __init__(self, optimus): 
        super().__init__(optimus)

    
    # run GPipe schedule
    def run(self, data, labels):

        global model_offloaded
        global optimizer_offloaded

        if self.optimus.tpl.is_first_stage():
            #self.get_input(data, labels)
            self.get_input(data)

        for i in range(self.optimus.mbsize):
            self.init_env_mark(i)
            self.init_env_grad_mark(i)

        if self.optimus.force_free_mem == True:
            self.cond_free_mem_()
            if self.optimus.swap_model_in_optstep == True and model_offloaded == True:
                self.load_model()
                #model_offloaded = False

                if optimizer_offloaded == True and model_offloaded == False:
                    self.load_optimizer()
                    optimizer_offloaded = False
                    if self.optimus.display_mem == True:
                        print(f" >>> [rank:{self.optimus.tpl.rank}], load optimizer ...")

                model_offloaded = False

        for i in range(self.optimus.mbsize):
            self.pre_fx_micro_forward_core(i)
            self.fx_micro_forward_core(i)
            result = self.post_fx_micro_forward_core(i)
            next(result)

        for i in range(self.optimus.mbsize):
            if self.optimus.tpl.is_last_stage():
                self.run_loss(i)
            grads = self.pre_fx_micro_backward_core(i)
            self.fx_micro_backward_core(i, grads)
            result = self.post_fx_micro_backward_core(i)
            next(result)

        if self.optimus.force_free_mem == True:
            self.optimus.run_info.clean_run_info(self.optimus.mbsize)
            if self.optimus.swap_model_in_optstep == True:
                self.check_swap_model_in_optstep()
            self.force_free_mem()
            if optimizer_offloaded == True and model_offloaded == False:
                if self.optimus.swap_opt_in_fwdbwd == True:
                    self.load_optimizer()
                    if self.optimus.display_mem == True:
                        print(f" >>>>>> [rank:{self.optimus.tpl.rank}], load optimizer ...")
                optimizer_offloaded = False




class Schedule1F1B(Schedule):

    def __init__(self, optimus): 
        super().__init__(optimus)

    
    # run 1F1B schedule
    def run(self, data, labels):
        global model_offloaded
        global optimizer_offloaded

        #num_warmup_microbatches = self.optimus.tpl.world_size - self.optimus.tpl.stage - 1
        num_warmup_microbatches = self.optimus.tpl.get_last_stage() - self.optimus.tpl.stage
        num_warmup_microbatches = min(num_warmup_microbatches, self.optimus.mbsize)
        remaining = self.optimus.mbsize - num_warmup_microbatches

        if self.optimus.tpl.is_first_stage():
            #self.get_input(data, labels)
            self.get_input(data)

        for i in range(self.optimus.mbsize):
            self.init_env_mark(i)
            self.init_env_grad_mark(i)

        if self.optimus.force_free_mem == True:
            self.cond_free_mem_()
            #print(f" >>>>>> [rank:{self.optimus.tpl.rank}], after first cond_free_mem  ..., model_offloaded:{model_offloaded}")
            if self.optimus.swap_model_in_optstep == True and model_offloaded == True:
                #print(f" >>>>>> [rank:{self.optimus.tpl.rank}], before calling load_model() ...") # TO DELETE
                self.load_model()
                #print(f" >>>>>> [rank:{self.optimus.tpl.rank}], after calling load_model() ...") # TO DELETE
                #model_offloaded = False

                if optimizer_offloaded == True and model_offloaded == False:
                    self.load_optimizer()
                    optimizer_offloaded = False
                    if self.optimus.display_mem == True:
                        print(f" >>> [rank:{self.optimus.tpl.rank}], load optimizer ...")

                model_offloaded = False

        for i in range(num_warmup_microbatches):
            self.pre_fx_micro_forward_core(i)
            self.fx_micro_forward_core(i)
            result = self.post_fx_micro_forward_core(i)
            next(result)

        reorder_mbi = -1
        for i in range(remaining): # steady
            forward_i = i + num_warmup_microbatches
            backward_i = i

            self.pre_fx_micro_forward_core(forward_i)
            if reorder_mbi >= 0:
                result = self.post_fx_micro_backward_core(reorder_mbi)
                next(result)
                reorder_mbi = -1

            self.fx_micro_forward_core(forward_i)
            result = self.post_fx_micro_forward_core(forward_i)
            next(result)

            if self.optimus.tpl.is_last_stage():
                self.run_loss(backward_i)


            grads = self.pre_fx_micro_backward_core(backward_i)
            self.fx_micro_backward_core(backward_i, grads)

            reorder_mbi = backward_i

        if num_warmup_microbatches == 0 and reorder_mbi >= 0: 
            result = self.post_fx_micro_backward_core(reorder_mbi)
            next(result)
            reorder_mbi = -1

        for i in range(num_warmup_microbatches):
            backward_i = i + remaining

            if self.optimus.tpl.is_last_stage():
                self.run_loss(backward_i)

            if reorder_mbi >= 0:
                result = self.post_fx_micro_backward_core(reorder_mbi)
                next(result)
                reorder_mbi = -1

            grads = self.pre_fx_micro_backward_core(backward_i)
            self.fx_micro_backward_core(backward_i, grads)

            result = self.post_fx_micro_backward_core(backward_i)
            next(result)


        if self.optimus.force_free_mem == True:
            self.optimus.run_info.clean_run_info(self.optimus.mbsize)
            if self.optimus.swap_model_in_optstep == True:
                self.check_swap_model_in_optstep()
            self.force_free_mem()
            if optimizer_offloaded == True and model_offloaded == False:
                if self.optimus.swap_opt_in_fwdbwd == True:
                    self.load_optimizer()
                    if self.optimus.display_mem == True:
                        print(f" >>>>>> [rank:{self.optimus.tpl.rank}], load optimizer ...")
                optimizer_offloaded = False

