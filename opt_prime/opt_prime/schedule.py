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


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.ERROR)





class ScheduleGPipe:

    def __init__(self, rinfo, ir, comm): 
        self.run_info = rinfo
        self.ir = ir
        self.comm = comm

    
    # TODO
    def run(self, data, labels):
        self.fx_forward(data, labels)

        if self.run_info.rank == self.run_info.world_size - 1:
            self.run_loss()
            loss = self.run_info.loss
        else:
            loss = None

        self.fx_backward(loss)



    def fx_forward(self, *args):
        #logging.debug(f" -----> rank{self.run_info.rank}: in fx_forward, args[0]:{args[0]}")
        self.args_iter = iter(args)

        if self.run_info.rank == 0: # TODO
            for n in self.run_info.mod.graph.nodes:
                if (n.op == 'placeholder' and self.run_info.stage == 0 and n.name == 'x') or \
                        (n.op == 'placeholder' and self.run_info.stage == 0 and n.name == 'input_ids'):
                    input = next(self.args_iter)

                    if isinstance(input, torch.Tensor):
                        mbatches = torch.chunk(input, self.run_info.mbsize)
                        if self.run_info.mbsize == 1:
                            input = input.to(self.run_info.device)
                            self.run_info.env[0]["placeholder"] = input
                        else:
                            for j in range(self.run_info.mbsize):
                                mbatch = mbatches[j].to(self.run_info.device)
                                self.run_info.env[j]["placeholder"] = mbatch
                    else:
                        logging.critical(f"### input:{input} not Tensor --> currently not supported!!")
                        sys.exit(1)
                    break

        #logging.debug(f" * rank:{self.run_info.rank}, in run_micro_batch_forward() ..")
        for i in range(self.run_info.mbsize):
            result = self.fx_micro_forward(i)
            next(result)



    def get_last_module(self):
        assert self.run_info.rank == self.run_info.world_size - 1
        return self.run_info.node.name
    
    
    #def make_output(self):
    #    output = None
    #    if self.run_info.rank == self.run_info.world_size - 1:
    #        target = self.get_last_module()
    #
    #        outputs = tuple(mb[target] for mb in self.run_info.env) 
    #        #print(f" ---> RANK: {self.run_info.rank},  outputs = {outputs}, type(output):{type(outputs)}")
    #        #output = torch.cat(outputs)
    #        output = outputs
    #
    #    return output

    def get_output_node(self):
        for n in reversed(self.run_info.graph.nodes):
            if n.op == 'output':
                return n

    def get_next_node_name(self, rank):
        # TODO: last stage processing
        assert rank < self.run_info.world_size - 1

        next_node_name = self.run_info.metadata_range[rank+1][1]
        return next_node_name


    def run_loss(self):
        # TODO: last stage processing
        assert self.run_info.rank == self.run_info.world_size - 1

        node = self.get_output_node()
        key_ = node.args[0]['logits']

        for mb_idx in range(self.run_info.mbsize):
            if str(key_) in self.run_info.getitem_dic:
                a_submod = self.run_info.getitem_dic[str(key_)][0]
                a_idx = self.run_info.getitem_dic[str(key_)][1]
                output1_ = self.run_info.env[mb_idx][a_submod][a_idx]
            else:
                output1_ = self.run_info.env[mb_idx][str(key_)]

            target1_ = self.run_info.env[mb_idx]["labels"]

            output1_ = output1_.view(-1, output1_.size(-1))
            target1_ = target1_.view(-1)


            flat_args = []
            if isinstance(output1_, torch.Tensor) and output1_.is_floating_point():
                output1 = output1_.detach().to(self.run_info.device)
                output1.requires_grad_(output1_.requires_grad)
                #output1.requires_grad_(True)
                flat_args.append(output1)
                output1.grad = None
            else:
                output1 = output1_
                flat_args.append(output1)

            if isinstance(target1_, torch.Tensor) and target1_.is_floating_point():
                target1 = target1_.detach().to(self.run_info.device)
                target1.requires_grad_(True)
                #flat_args.append(target1)
                flat_args.append(target1)
            else:
                target1 = target1_
                flat_args.append(target1)

            criterion = nn.CrossEntropyLoss()

            criterion = criterion.to(self.run_info.device)

            result = criterion(output1, target1)

            #print(f" >>>> loss: {result}, result.shape:{result.shape}")

            self.run_info.grads[mb_idx][node.name] = (None,)
            self.run_info.loss[mb_idx] = result
            #self.fwd_cache2[mb_idx][node.name] = \
            #        ( result if isinstance(result, tuple) else (result,), \
            #        flat_args, )
            self.run_info.env[mb_idx][node.name] = result
            self.run_info.flat_args[mb_idx][node.name] = flat_args



    def init_env_mark(self, mb_idx):
        for i in range(len(self.run_info.metadata_range)):
            self.run_info.env_recv_mark[mb_idx][self.run_info.metadata_range[i][1]] = None
            self.run_info.env_send_mark[mb_idx][self.run_info.metadata_range[i][1]] = None



    def fx_micro_forward(self, mb_idx):

        self.init_env_mark(mb_idx)

        from_, to_ = self.ir.get_range(self.run_info.rank, self.run_info.graph)

        #print(f" [rank:{self.run_info.rank}] fx_micro_forward({mb_idx}), from_:{from_}, to_:{to_} .....")

        if self.run_info.rank == 0:
            target_node_name = "placeholder"
            #self.run_info.env[mb_idx]["x"] = self.run_info.env[mb_idx][target_node_name]
            self.run_info.env[mb_idx]["input_ids"] = self.run_info.env[mb_idx][target_node_name]
        #if self.run_info.rank > 0:
        #    # TODO
        #    target_node_name = self.get_prev_nodename(self.run_info.rank)
        #    pre_split_rank = self.run_info.rank - 1
        #    self.run_info.env[mb_idx][target_node_name] = self.comm.receive_data(pre_split_rank, self.run_info.device)
        #    # DEBUG
        #    print(f" #### rank:{self.run_info.rank} <== received from [{pre_split_rank}] ####")

        if self.run_info.rank > 0:
            pre_split_rank = self.run_info.rank - 1
        
            cur = from_._prev

        
            begin_ = cur
            for node_name, range_ in self.run_info.special_nodes.items():
                src_rank, needed_by_rank = range_
                if self.run_info.rank > src_rank and self.run_info.rank <= needed_by_rank:
                    #print(f"MBF[{mb_idx}]: ### rank:{self.run_info.rank}, receive cross_ref activation from {pre_split_rank}, node_name:{node_name}")
                    if node_name in self.run_info.getitem_dic:
                        submod_name = self.run_info.getitem_dic[node_name][0]
                        if self.run_info.env_recv_mark[mb_idx][submod_name] is None:
                            self.run_info.env[mb_idx][submod_name] = self.comm.receive_data(pre_split_rank, self.run_info.device)
                            self.run_info.env_recv_mark[mb_idx][submod_name] = 1

                        if isinstance(self.run_info.env[mb_idx][submod_name], torch.Tensor):
                            if not self.run_info.env[mb_idx][submod_name].requires_grad or self.run_info.env[mb_idx][submod_name].grad_fn is None:
                                self.run_info.env[mb_idx][submod_name].requires_grad_(True)
                                logging.info(f" ###### node name:{submod_name} requires_grad(True) #####") 
                    else:
                        if self.run_info.env_recv_mark[mb_idx][node_name] is None:
                            self.run_info.env[mb_idx][node_name] = self.comm.receive_data(pre_split_rank, self.run_info.device)
                            self.run_info.env_recv_mark[mb_idx][node_name] = 1
                        if isinstance(self.run_info.env[mb_idx][node_name], torch.Tensor):
                            if not self.run_info.env[mb_idx][node_name].requires_grad or self.run_info.env[mb_idx][node_name].grad_fn is None:
                                self.run_info.env[mb_idx][node_name].requires_grad_(True)
                                logging.info(f" ###### node name:{node_name} requires_grad(True) #####") 
        

        #forward one chunk !!
        flat_args = []
        def extract_tensor_args(b):
            # TODO
            if b.name in self.run_info.getitem_dic:
                a_submod = self.run_info.getitem_dic[b.name][0]
                a_idx = self.run_info.getitem_dic[b.name][1]
                a = self.run_info.env[mb_idx][a_submod][a_idx]
            else:
                a = self.run_info.env[mb_idx][b.name]
            #a = self.run_info.env[mb_idx][b.name]

            nonlocal flat_args
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                val = a.detach().to(self.run_info.device)
                #val.requires_grad_(a.requires_grad)
                val.requires_grad_(True)
                flat_args.append(val)
                return val
            else:
                flat_args.append(a)
                return a
            return a


        #print(f" [rank:{self.run_info.rank}] fx_micro_forward({mb_idx}), node.args:{self.run_info.node.args} .....")

        args = fx.graph.map_arg(self.run_info.node.args, extract_tensor_args)
        kwargs = fx.graph.map_arg(self.run_info.node.kwargs, extract_tensor_args)

        result = self.run_info.submod(*args, **kwargs)

        #self.fwd_cache2[mb_idx][self.run_info.name] = \
        #        ( result if isinstance(result, tuple) else (result,), \
        #        flat_args, )
        self.run_info.flat_args[mb_idx][self.run_info.name] = flat_args


        #print(f" >>>> rank:{self.run_info.rank}, run fx_micro_forward( mb_idx:{mb_idx}, name:{self.run_info.name})")

        self.run_info.env[mb_idx][self.run_info.name] = result


        if self.run_info.rank < self.run_info.world_size - 1:
            next_split_rank = self.run_info.rank + 1
            #begin_ = cur
        
            for node_name, range_ in self.run_info.special_nodes.items():
                src_rank, needed_by_rank = range_
                if self.run_info.rank >= src_rank and self.run_info.rank < needed_by_rank:
                    if node_name in self.run_info.getitem_dic:
                        submod_name = self.run_info.getitem_dic[node_name][0]
                        if self.run_info.env_send_mark[mb_idx][submod_name] is None:
                            obj = self.run_info.env[mb_idx][submod_name]
                            self.comm.send_data(obj, next_split_rank, self.run_info.device)
                            self.run_info.env_send_mark[mb_idx][submod_name] = 1
                    else:
                        if self.run_info.env_send_mark[mb_idx][node_name] is None:
                            obj = self.run_info.env[mb_idx][node_name]
                            self.comm.send_data(obj, next_split_rank, self.run_info.device)
                            self.run_info.env_send_mark[mb_idx][node_name] = 1

                    #print(f"MBF[{mb_idx}]: ### rank:{self.run_info.rank} send cross_ref activation to {next_split_rank}, node_name:{node_name}")

        yield result


    def free_mem(self):
        #self.run_info.env = [{} for _ in range(self.run_info.mbsize)]
        #self.run_info.flat_args = [{} for _ in range(self.run_info.mbsize)]
        #self.run_info.grads = [{} for _ in range(self.run_info.mbsize)]
        torch.cuda.empty_cache()

    def fx_backward(self, *args):
        for i in range(self.run_info.mbsize):
            result = self.fx_micro_backward(i)
            next(result)

        self.free_mem()


    def init_env_grad_mark(self, mb_idx):
        for i in range(len(self.run_info.metadata_range)):
            self.run_info.env_grad_recv_mark[mb_idx][self.run_info.metadata_range[i][1]] = None
            self.run_info.env_grad_send_mark[mb_idx][self.run_info.metadata_range[i][1]] = None

            self.run_info.grads[mb_idx][self.run_info.metadata_range[i][1]] = None


    def get_num_nodes(self, name):
        cnt = 0
        for k, v in self.run_info.getitem_dic.items():
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

        args = ()
        kwargs = dict()
        #k1, k2 = self.fwd_cache2[mb_idx].pop(node.name)
        k1 = self.run_info.env[mb_idx].pop(node.name)
        k1 = ((k1,) if not isinstance(k1, tuple) else k1)
        k2 = self.run_info.flat_args[mb_idx].pop(node.name)

        kwargs["forward_output"] = k1
        kwargs["forward_input"] = k2
        kwargs["forward_output_gradient"] = grads 

        num_nodes = self.get_num_nodes(node.name) 
        kwargs["valid_index"] = [i for i in range(num_nodes)]

        result = self.core_backward(*args, **kwargs)

        return result



    def fx_micro_backward(self, mb_idx):

        self.init_env_grad_mark(mb_idx)

        grads = None

        if self.run_info.rank < self.run_info.world_size - 1:
            pre_split_rank = self.run_info.rank + 1

            node_name = self.get_next_node_name(self.run_info.rank)
            if self.run_info.env_grad_recv_mark[mb_idx][node_name] is None:
                self.run_info.grads[mb_idx][node_name] = self.comm.receive_data(pre_split_rank, self.run_info.device)
                grads = self.run_info.grads[mb_idx][node_name]
                self.run_info.env_grad_recv_mark[mb_idx][node_name] = 1


        # TODO: last stage ?
        if self.run_info.rank == self.run_info.world_size - 1:
            node = self.get_output_node()
            grads = self.run_info.grads[mb_idx][node.name]
            result = self.run_core_backward(mb_idx, node, grads)
            result = ((result,) if not isinstance(result, tuple) else result)
            self.run_info.grads[mb_idx][node.name] = result

            grads = self.run_info.grads[mb_idx][node.name]

        node = self.run_info.node
        result = self.run_core_backward(mb_idx, node, grads)

        result = ((result,) if not isinstance(result, tuple) else result)

        self.run_info.grads[mb_idx][node.name] = result

        if self.run_info.rank > 0:
            next_split_rank = self.run_info.rank - 1

            node_name = self.run_info.name
            if self.run_info.env_grad_send_mark[mb_idx][node_name] is None:
                obj = self.run_info.grads[mb_idx][node_name]
                self.comm.send_data(obj, next_split_rank, self.run_info.device)
                self.run_info.env_grad_send_mark[mb_idx][node_name] = 1

        yield 0
