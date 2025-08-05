#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
#   This program fine-tunes the LLaMa model using the Transformers library installed locally."
#
#    *** This program was tested with torch 2.5.0 and transformers 4.46.2.
#     The version of transformers used must be consistent across all machines used for testing ***


import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.distributed as dist


#
# This program use the locally installed transformers for profiling purpose
#
sys.path.insert(0, "/workspace/transformers-4.46.2/src")
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from packaging import version

from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.nn.parallel import DistributedDataParallel



batch_size = 32

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
master_addr = os.getenv("MASTER_ADDR")
master_port = os.getenv("MASTER_PORT")


if len(sys.argv) > 1:
    os.environ['LLAMA_ACCESS_TOKEN'] = sys.argv[1]

# Llama access token
access_token = os.getenv('LLAMA_ACCESS_TOKEN')
if access_token is None:
    raise ValueError("LLAMA_ACCESS_TOKEN environment variable is not set."
                     "       [Usage:] torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> --master_addr=<IP_of_rank_0> --master_port=29500 llama_2d.py <llama_access_token>")
elif rank == 0:
    print(f"LLAMA_ACCESS_TOKEN is processed.")


# torch version
required_version = "2.3.1"
current_version = torch.__version__
if version.parse(current_version) >= version.parse(required_version):
    print(f"[rank:{int(os.environ['RANK'])}] torch version 2.3.1 or higher --> OK")
else:
    print(f"[rank:{int(os.environ['RANK'])}] current torch version is {current_version}.")
    raise ValueError('This program needs torch version 2.3.1 or higher.')

# transformers version
required_tf_version = "4.46.2"
current_tf_version = transformers.__version__
if version.parse(current_tf_version) >= version.parse(required_tf_version):
    print(f"[rank:{int(os.environ['RANK'])}] transformers version 4.46.2 or higher --> OK")
else:
    print(f"[rank:{int(os.environ['RANK'])}] current transformers version is {current_tf_version}.")
    raise ValueError('This program needs transformers version 4.46.2 or higher.')


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token)
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token, use_cache=False)



class Optimus_t:
    def __init__(self, model, dp_size, tp_size):
        self.dp_size = dp_size
        self.tp_size = tp_size

        self.setup(model)


    def prepare_tp(self, model):

        self.parallelized_model = parallelize_module(model.to(device="cuda"), self.tp_mesh,
                {
                    "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(),),
                    "lm_head": ColwiseParallel(output_layouts=Replicate(),),
                }
        )

        for layer_id, transformer_block in enumerate(self.parallelized_model.model.layers):
            layer_tp_plan = {
                    f"model.layers.{layer_id}.self_attn.q_proj": ColwiseParallel(),
                    f"model.layers.{layer_id}.self_attn.k_proj": ColwiseParallel(),
                    f"model.layers.{layer_id}.self_attn.v_proj": ColwiseParallel(),
                    f"model.layers.{layer_id}.self_attn.o_proj": RowwiseParallel(),
                    f"model.layers.{layer_id}.mlp.gate_proj": ColwiseParallel(),
                    f"model.layers.{layer_id}.mlp.down_proj": RowwiseParallel(),
                    f"model.layers.{layer_id}.mlp.up_proj": ColwiseParallel(),
            }


            attn_layer = transformer_block.self_attn
            attn_layer.num_heads = attn_layer.num_heads // self.tp_mesh.size()
            attn_layer.num_key_value_heads = attn_layer.num_key_value_heads // self.tp_mesh.size()

            parallelize_module(module=transformer_block, device_mesh=self.tp_mesh, parallelize_plan=layer_tp_plan)


    def setup(self, model):
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        self.device = torch.device(f"cuda:{local_rank}")

        self.device_mesh = init_device_mesh("cuda", mesh_shape=(self.dp_size, self.tp_size), mesh_dim_names=("dp", "tp"))
        dp_group = self.device_mesh["dp"].get_group()
        tp_group = self.device_mesh["tp"].get_group()
        self.dp_mesh = self.device_mesh["dp"]
        self.tp_mesh = self.device_mesh["tp"]

        #
        print(f"[{rank}] >>>  tp group:{self.tp_mesh}, dp_group:{self.dp_mesh}")

        #self.prepare_tp(model)
        ##self.parallelized_model = FSDP(self.parallelized_model, device_mesh=self.dp_mesh, sharding_strategy=ShardingStrategy.NO_SHARD)
        #self.parallelized_model = DistributedDataParallel(self.parallelized_model, find_unused_parameters=True, device_mesh=self.dp_mesh)
        if self.tp_size == 1 and self.dp_size == 1:
            self.parallelized_model = model
            self.parallelized_model.to(self.device)

        else:
            self.prepare_tp(model)
            self.parallelized_model = DistributedDataParallel(self.parallelized_model, find_unused_parameters=True, device_mesh=self.dp_mesh)
        self.criterion = nn.CrossEntropyLoss().cuda(rank)
        self.optimizer = torch.optim.Adam(self.parallelized_model.parameters(), lr=3e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)


    def run(self, data, label):
        self.optimizer.zero_grad()

        outputs = self.parallelized_model(data)

        outputs_ = outputs.logits
        outputs_ = outputs_.view(-1, outputs_.size(-1))
        label_ = label.view(-1)

        loss = self.criterion(outputs_, label_)

        loss.backward()


        return loss


datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(record) for record in datasets if len(str(record)) < 500]
dataloader = DataLoader(datasets, batch_size=batch_size, num_workers=4)
data_size=len(dataloader.dataset)
print(f"data_size={data_size}")
nbatches = len(dataloader)
print(f"nbatches={nbatches}")

optimus_t = Optimus_t(model, dp_size=1, tp_size=world_size)
#optimus_t = Optimus_t(model, dp_size=1, tp_size=2)
#optimus_t = Optimus_t(model, dp_size=1, tp_size=4)

if rank == 0:
    print(f">> tp_size:{optimus_t.tp_size}, dp_size:{optimus_t.dp_size}")

epochs = 1 # The number of epochs


def train():

    optimus_t.parallelized_model.train() # turn on the train mode

    total_loss = 0
    start_time = time.time()

    for i, batch in enumerate(dataloader):

        data, labels = None, None

        # prepare input and label
        tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
        data, labels = tokens.input_ids.to(optimus_t.device), tokens.input_ids.to(optimus_t.device)

        loss = optimus_t.run(data, labels)


        torch.nn.utils.clip_grad_norm_(optimus_t.parallelized_model.parameters(), 0.5)
        optimus_t.optimizer.step()

        if local_rank == 0:
            total_loss += loss
            log_interval = 10
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('[rank:{:3d}| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        rank, epoch, i, nbatches, optimus_t.scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))

                total_loss = 0
                start_time = time.time()

if rank == 0:
    tick = time.time()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    optimus_t.scheduler.step()

if rank == 0:
    tock = time.time()
    elapsed_time = tock - tick

    print('Time elapsed: %.3f sec ' % (elapsed_time))

print(f"[rank:{rank}, run completed ...")




