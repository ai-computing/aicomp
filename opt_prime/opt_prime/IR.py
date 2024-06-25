#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#

import sys
import math
import torch
import torch.nn as nn

from transformers import GPT2Config
from transformers import OPTConfig
from transformers import GPT2LMHeadModel
from transformers import GPTNeoForCausalLM
from transformers import BertLMHeadModel
from transformers import GPTJForCausalLM
from transformers import BartForCausalLM
from transformers import MBartForCausalLM
from transformers import OPTForCausalLM

import transformers.utils.fx as hf_fx
import inspect

from torch import Tensor, Size
from torch import fx
from torch.fx.node import Node
from torch.fx.graph_module import GraphModule
from torch.fx.passes.split_module import split_module
import copy
import time

import torch.distributed as dist
import datetime
import logging
import os
import gc
from enum import Enum


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

logging.basicConfig(level=logging.ERROR)

class IR_Anal(Enum):
    SINGLE = 1      # Experimental
    PARALLEL = 2
    SEQUENTIAL = 3


huggingface_model_class = [
        GPT2LMHeadModel, 
        GPTNeoForCausalLM, 
        BertLMHeadModel, 
        GPTJForCausalLM, 
        BartForCausalLM,
        MBartForCausalLM, 
        OPTForCausalLM,
        # TODO
        ]


class IR(object):
    def __init__(self, model: nn.Module, optimus):

        self.gm = None
        self.model_ir = []
        self.metadata_range = []

        self.optimus = optimus

        self.special_nodes: Dict[str, Tuple[int, int]] = {}  # { node_name : {stage#, needed-by-stage#),}


    def retrieve_IR(self, model: nn.Module):

        if model.__class__ in huggingface_model_class:
            input_names = model.dummy_inputs.keys()
            sig = inspect.signature(model.forward)
            concrete_args = {
                p.name: p.default
                for p in sig.parameters.values()
                if p.name not in input_names
            }

            tracer = hf_fx.HFTracer()

            traced_graph = tracer.trace(model, concrete_args=concrete_args)
            self.gm = torch.fx.GraphModule(model, traced_graph)
            return self.optimus.model2type["hf"]

        elif isinstance(model, nn.Module):
            self.gm = fx.symbolic_trace(model)
            return self.optimus.model2type["sy"]

        else:
            print(f"Not supported model!")
            sys.exit(1)


    def split_IR(self, model: nn.Module, method, num_stage):

        if method not in [ "simple", ]:
            print(f"Not supported split method!")
            sys.exit(1)

        if method == "simple":
            submods = self.simple_split(model, num_stage)

        # TODO: add new split method
        #elif method == ...
        #
            
        self.check_last_submods(submods, num_stage)

        self.model_ir.append(submods)

        if int(os.environ["RANK"]) == 0:
            print(f">> ------------------ FX graph --------------------------------")
            for n in self.model_ir[0].graph.nodes:
                print(f"n.op:{n.op}, n.name:{n.name}, n.target:{n.target}, n.args:{n.args}, n.all_input_nodes:{n.all_input_nodes}")
            print(f">> ------------------------------------------------------------")


    def simple_split(self, module, num_stage):
        length = self.gm.graph.nodes.__len__()

        modcnt = 0
        for n in self.gm.graph.nodes:
            if n.op == 'call_module':
                modcnt = modcnt + 1
        
        segment = modcnt // num_stage
        print(f"length:{length}, modcnt:{modcnt}, num_stage:{num_stage}, segment:{segment}")


        ## simple assert
        assert length >= num_stage, f"Model length:{length} is smaller than # of workers:{num_stage}"
        
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
        for n in self.gm.graph.nodes:
            if n.op == 'call_module':
                cnt = cnt + 1
        
            if cnt == segment:
                self.metadata_range.append((k, n.name))
                k = k + 1
                cnt = 0
        
            if k > num_stage - 1:
                break
        
        if len(self.metadata_range) <  num_stage:
            self.metadata_range.append((k, n.name))

        if int(os.environ["RANK"]) == 0:
            print(f" ------------------------------------------------------------")
            print(f"  rank:{self.optimus.tpl.rank},  first metadata_range: {self.metadata_range}")
            print(f" ------------------------------------------------------------")

        submodules = split_module(self.gm, module, part_fn, keep_original_order=True)


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
        print(f"  rank:{self.optimus.tpl.rank},  second metadata_range: {self.metadata_range}")
        print(f" ------------------------------------------------------------")

        assert len(self.metadata_range) == num_stage

        return submodules

        
    def check_last_submods(self, submods, num_stage):
        gmodule_cnt = 0
        mod_cnt = 0
        for submod in submods.modules():
            if isinstance(submod, fx.GraphModule):
                gmodule_cnt = gmodule_cnt + 1
                last_submod = submod
                continue


        assert gmodule_cnt > num_stage, f"GraphModule #:[{gmodule_cnt}] must have more than the number of stages #:[{num_stage}]"

        for node in last_submod.graph.nodes:
            #print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.all_input_nodes:{node.all_input_nodes}")
            if node.op == 'call_module' and node.target != 'loss_fn':
                mod_cnt = mod_cnt + 1

        #print(f">>> GraphModule cnt:{gmodule_cnt},  Last GraphModule's  mod_cnt ==> {mod_cnt}")

        assert mod_cnt > 0, f"Last partition has {mod_cnt} modules. It must have more than 0 modules"



    # analyze IR graph and find the cross-layer referenced nodes
    def cross_reference_analyze(self, stage, g:fx.Graph):
    
        if stage == 0:
            return
    
        from_, to_ = self.get_range(stage, g)
    
        #logging.debug(f" ***** stage:{stage} >>  from_:{from_.name}, to_:{to_.name}")
        print(f" ***** stage:{stage} >>  from_:{from_.name}, to_:{to_.name}")
    
        cur = to_
        while (cur != from_) or (stage > 0 and cur == from_):
    
            # in process check - backward direction
    
            #for _, target_ in enumerate(cur.all_input_nodes):
            for i, target_ in enumerate(cur.all_input_nodes):
                if cur.name == "loss_fn" and i > 0:
                    break
                referenced_in = False
                referenced_out = False
    
                inner = cur._prev
                if inner != from_._prev:
                    while (inner != from_) or (stage > 0 and inner == from_):
                        if inner.name == target_.name:
                            #logging.debug(f" [cross_reference_analyze] ({target_.name}) referenced in current stage:{stage} !")
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
                    split_node_name = self.metadata_range[stage_][1]
    
                    for k in g.nodes:
                        first_node = k
                        break
    
                    outer = first_node
                    while outer != from_: 
                        # DEBUG
                        if outer.name == target_.name:
                            logging.info(f" [cross_reference_analyze] ({target_.name}) referenced in outer stage:{stage_} !!")
    
                            if target_.name not in self.special_nodes:
                                self.special_nodes[target_.name] = (stage_, stage)  # { node_name : {stage#, needed-by-stage#),}
                            referenced_out = True
                            break
    
                        if outer.name == split_node_name:
                            stage_ = stage_ + 1
                            split_node_name = self.metadata_range[stage_][1]
    
                        outer = outer._next
    
    
                if referenced_out == False:
                    logging.critical(f"[Error] cannot handle this case: {target_.name} !!!")
                    sys.exit(1)
    
            if cur == from_:
                break
    
            cur = cur._prev



    def get_range(self, stage, g:fx.Graph) -> (Node, Node):
     
        if stage == 0:
            from_node_name = "-1"
            for n in g.nodes:
                if n.op == 'placeholder':
                    from_node_name = n.name
                    logging.debug(f">>>> get_range: n.op == 'placeholder' --> from_node_name:{from_node_name}")
                break
        else:
            from_node_name = self.metadata_range[stage-1][1]
     
        to_node_name = self.metadata_range[stage][1]
    
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


    def setup_submod(self, stage, rank):

        for n, m in self.model_ir[0].named_children():
            if n == f"submod_{stage}" and isinstance(m, GraphModule):
                self.optimus.run_info.name = n
                self.optimus.run_info.submod = m
                break
        
        if self.optimus.run_info.name is None:
            print(f"ERROR: Not found name(submod_{stage})")
            sys.exit(0)

        #print(f" ## Rank:{rank}, name:{self.name}")

        for n in self.model_ir[0].graph.nodes:
            if n.name == self.optimus.run_info.name:
                self.optimus.run_info.node = n
                break

        if self.optimus.run_info.node is None:
            print(f"ERROR: Not found node({self.name})")
            sys.exit(0)


        #self.submod.to(self.run_info.device)
        #print(f" ## Rank:{rank}, name:{self.optimus.run_info.node.name}, move {self.optimus.run_info.name} to {self.optimus.run_info.device}")


    def build_getitem_dic(self):
        for node in self.model_ir[0].graph.nodes:
            if node.op == 'call_function' and node.name.startswith("getitem"):
                self.optimus.run_info.getitem_dic[node.name] = (node.args[0].name, node.args[1])

    #def print_graph(self, ir, rank):
    def print_graph(self, rank):
        print(f" # rank = {rank}, metadata_range:{self.metadata_range}")
        for node in self.model_ir[0].graph.nodes:
            print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.args:{node.args}, node.all_input_nodes:{node.all_input_nodes}")


    def get_output_node(self):
        for n in reversed(self.model_ir[0].graph.nodes):
            if n.op == 'output':
                return n


    def delete_param(self, mod, name):
        for param_name, param in mod.named_parameters():
            t = getattr(mod, param_name)
            setattr(mod, param_name, None)
            del t

    def has_child(self, mod):
        num_children = 0
        if mod is not None:
            num_children = len(list(mod.children()))
        return num_children

    def partially_matched(self, name):
        for string in self.reference:
            if name in string:
                return True
        return False

    def delete_module(self, root, module_node):

        module_name = module_node.name
        module_target_name = str(module_node.target)
        if len(module_name) > len(module_target_name):
            self.reference.append(module_target_name)

        get_attr_flag = False
        if len(module_name) == len(module_target_name) and not self.partially_matched(module_target_name):
            target_atoms = module_target_name.split('.')
            attr_itr = root
            if module_node.op == 'get_attr':
                if target_atoms[-1] == "weight" or target_atoms[-1] == "bias":
                    target_atoms = target_atoms[:-1]
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
            #if get_attr_flag == True or self.has_child(attr_itr) == 0:
            if True:
                self.delete_param(attr_itr, atom)
                delattr(parent_, atom)
                del attr_itr
                #print(f">>> rank:{self.optimus.tpl.rank}, delete module:target:{module_target_name} ## atom:{atom},op:{module_node.op}")
                self.reference.append(module_target_name)
            else:
                print(f">>> delete_module:{module_name} requested, but {module_name} has child")



    def delete_intermediate_module(self, module_name):
        del_submod = self.model_ir[0].get_submodule(module_name)

        for n in del_submod.graph.nodes:
            if n.op == 'call_module':
                self.delete_module(del_submod, n)


    def clean_module_memory(self):
        self.reference = []

        for stage in reversed(range(self.optimus.tpl.num_stage)):
            if stage != self.optimus.tpl.get_stage():
                del_name = f"submod_{stage}"
                self.delete_intermediate_module(del_name)

        #print(f"[rank:{self.optimus.tpl.rank}] reference:{self.reference}")
        self.reference = None

        self.model_ir = []
        self.gm = None

        gc.collect()
        torch.cuda.empty_cache()

