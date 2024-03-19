#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#
#
#  This is a test program for running PiPPy with pipeline-parallel training.
#
#
#  Sample Usage:
#      <machine #0>
#            torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0
#                  --master_addr="X.X.X.X" --master_port=29501 pippy_pp_training_gpt2-xl.py
#      <machine #1>
#            torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1
#                  --master_addr="X.X.X.X" --master_port=29501 pippy_pp_training_gpt2-xl.py
#



import torch
import torch.distributed.rpc as rpc

import torch.nn as nn
import time

import torch.distributed as dist
import datetime
import torch.distributed.rpc as rpc


import psutil
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


from pippy.IR import Pipe, pipe_split, annotate_split_points, PipeSplitWrapper
from pippy import split_into_equal_size
from pippy.PipelineDriver import PipelineDriverFillDrain
from pippy.microbatch import sum_reducer, TensorChunkSpec, Replicate
#from pippy.IR import LossWrapper

from pippy.hf import PiPPyHFTracer

from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2 import GPT2PreTrainedModel
import inspect

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

torch.manual_seed(42)

use_gpu = True
#use_gpu = False

world_size=int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

device = None


if use_gpu == True:
    #device = torch.device(f"cuda:{rank}")
    #print(f"Using GPU ... cuda:{rank}")
    device = torch.device(f"cuda:{local_rank}")
    print(f"Using GPU ... cuda:{local_rank}")

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=30)
    n_devs = torch.cuda.device_count()
    dev_id = rank % n_devs
    #dev_id = local_rank % n_devs
    for i in range(world_size):
        options.set_device_map(f"worker{i}", {dev_id: i % n_devs})
        #options.set_device_map(f"worker{i}", {dev_id: rank})
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)
else:
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

batch_size = 32

#micro_batch_size = world_size // 2 # TODO
micro_batch_size = 2 # TODO
CHUNKS = micro_batch_size

#if int(os.environ["RANK"]) == 0:
if rank == 0:
    print(f"total process count: {world_size}")
    print(f"batch size: {batch_size}")
    print(f"micro batch size: {micro_batch_size}")

'''
class LossWrapper(GPT2PreTrainedModel):
    def __init__(self, config, module, loss_fn):
        super().__init__(config)
        self.module = module
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        raise NotImplementedError("LossWrapper: no forward implementation")

class SimpleLossWrapper(LossWrapper, torch.nn.Module):
    def forward(self, x, labels):
        out1 = self.module(x)
        return self.loss_fn(out1, labels)
'''

total_parameters = 0
def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


if rank == 0:
    #config = GPT2Config(use_cache=False)
    #config = GPT2Config(n_layer=24, use_cache=False)
    #config = GPT2Config(n_layer=36, use_cache=False)
    config = GPT2Config(n_layer=48, use_cache=False)

    #model = GPT2LMHeadModel.from_pretrained("gpt2-xl", config=config, ignore_mismatched_sizes=True
    model = GPT2LMHeadModel(config)
    model = model.from_pretrained("gpt2-xl")


    #loss_fn = torch.nn.CrossEntropyLoss().to(device)
    #wrapper = SimpleLossWrapper(config, model, loss_fn)

    #total_parameters = get_total_params(wrapper)
    total_parameters = get_total_params(model)
    print('Total parameters in model: {:,}'.format(total_parameters))

    split_policy = split_into_equal_size(world_size)

    # need specific input for concrete_args
    gpt2_input_dict = {
            'input_ids': torch.empty(batch_size, 100, dtype=torch.long, device=device).random_(50257),
            'labels': torch.empty(batch_size, 100, dtype=torch.long, device=device).random_(50257),
            'position_ids': torch.arange(0, 100, dtype=torch.long, device=device)
    }

    input_names = gpt2_input_dict.keys()
    #input_names = model.dummy_inputs.keys()
    sig = inspect.signature(model.forward)
    concrete_args = {
        p.name: p.default
        for p in sig.parameters.values()
        if p.name not in input_names
    }

    #wrapper.train()
    output_loss_value_spec = {'loss': True, 'logits': False,
                              'past_key_values': [[False for _ in range(2)] for _ in range(48)]}
                              #'past_key_values': [[False for _ in range(2)] for _ in range(36)]}
                              #'past_key_values': [[False for _ in range(2)] for _ in range(24)]}
                              #'past_key_values': [[False for _ in range(2)] for _ in range(12)]}
    #pipe = Pipe.from_tracing(wrapper, tracer=PiPPyHFTracer(), concrete_args=concrete_args, output_loss_value_spec=True, split_policy=split_policy)
    pipe = Pipe.from_tracing(model, tracer=PiPPyHFTracer(), concrete_args=concrete_args, output_loss_value_spec=output_loss_value_spec, split_policy=split_policy)
    print(pipe)

    if use_gpu == True:
        pipe.to(device)
        print(f">> Moving to: {device}")

    #args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    #kwargs_chunk_spec = {}
    kwargs_chunk_spec = {'input_ids': TensorChunkSpec(0), 'labels': TensorChunkSpec(0), 'position_ids': Replicate}
    #output_chunk_spec = TensorChunkSpec(0)
    from pippy.microbatch import LossReducer
    #output_chunk_spec = LossReducer(0.0, lambda a, b: a+b)
    output_chunk_spec = {'loss': sum_reducer, 'logits': TensorChunkSpec(0),
                         'past_key_values': [[TensorChunkSpec(0) for _ in range(2)] for _ in range(config.n_layer)]}

    #driver = PipelineDriverFillDrain(
    #        pipe, CHUNKS, args_chunk_spec=args_chunk_spec, kwargs_chunk_spec=kwargs_chunk_spec,
    #        output_chunk_spec=output_chunk_spec, world_size=world_size)
    driver = PipelineDriverFillDrain(
            pipe, CHUNKS, kwargs_chunk_spec=kwargs_chunk_spec,
            output_chunk_spec=output_chunk_spec, world_size=world_size)

    optimizer1 = driver.instantiate_optimizer(torch.optim.Adam)
    lr_scheduler = driver.instantiate_lr_scheduler(torch.optim.lr_scheduler.LinearLR, total_iters=100)


if rank == 0:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(record) for record in datasets if len(str(record)) < 500]
    dataloader = DataLoader(datasets, batch_size=batch_size, num_workers=1)


    #print('Total parameters in model: {:,}'.format(get_total_params(wrapper)))

    tick = time.time()

    for i, batch in enumerate(dataloader):
        data = None
        labels = None

        tokens = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        data, labels = tokens.input_ids, tokens.input_ids

        
        gpt2_input_dict = {
        #'input_ids': torch.chunk(data, device=device),
        'input_ids': data.to(device),
        #'labels': torch.chunk(labels, device=device),
        'labels': labels.to(device),
        'position_ids': torch.arange(0, data.size(1), device=device)}

        optimizer1.zero_grad()

        #pipe_loss = driver(data, labels)
        pipe_loss = driver(**gpt2_input_dict)
        #print(f'Step {i}, Loss: {pipe_loss}')
        print(f"Step {i}, Loss: {pipe_loss['loss']}")

        optimizer1.step()
        lr_scheduler.step()


    tock=time.time()
    elapsed_time = tock - tick

    print('Time elapsed: %.3f sec ' % (elapsed_time))
    print('Total parameters in model: {:,}'.format(total_parameters))



rpc.shutdown()
print(f"[rank:{rank}, run completed ...")

