#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#

import deepspeed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


if len(sys.argv) > 1:
    os.environ['LLAMA_ACCESS_TOKEN'] = sys.argv[3]

access_token = os.getenv('LLAMA_ACCESS_TOKEN')
if access_token is None:
    raise ValueError("LLAMA_ACCESS_TOKEN environment variable is not set."
            "   [Usage] deepspeed --hostfile=<hostfile> --no_ssh --node_rank=<current_node_rank> --master_addr=<IP_of_rank_0> --master_port=29500  deep_llama-8b.py --deepspeed <llama_access_token>")

print(f">> llama access token: {access_token}")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", token=access_token, use_cache=False)


deepspeed.init_distributed("nccl")

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params

if rank == 0:
    print ('Total parameters in model: {:,}'.format(get_total_params(model)))


train_batch_size = 32 # TODO
#train_batch_size = 60 # TODO
#train_batch_size = 36 # TODO
batch_size_per_process = int(train_batch_size // world_size)

config = {
        "train_batch_size": train_batch_size,
        #"train_micro_batch_size_per_gpu": 15,
        "zero_optimization": {
            "stage": 2,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 3e-05,
                "betas": [ 0.9, 0.999 ],
                "eps": 1e-8
            }
        },
}

if rank == 0:
    print(f"total process count: {os.environ['WORLD_SIZE']}")
    print(f"global batch size: {train_batch_size}")
    print(f"per process batch size: {batch_size_per_process}")

model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config_params=config, model_parameters=model.parameters())


datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(record) for record in datasets if len(str(record)) < 500]
dataloader = DataLoader(datasets, batch_size=batch_size_per_process, shuffle=False, sampler=DistributedSampler(datasets, shuffle=False))
data_size=len(dataloader.dataset)
print(f"data_size={data_size}")
nbatches = len(dataloader)
print(f"nbatches={nbatches}")


epochs = 1 # The number of epochs

def train():

    model_engine.train() # turn on the train mode

    total_loss = 0

    for i, batch in enumerate(dataloader):

        data, labels = None, None

        # prepare input and label
        tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
        data, labels = tokens.input_ids, tokens.input_ids

        optimizer.zero_grad()

        outputs = model_engine(data.to(model.device), labels=labels.to(model.device))

        loss = outputs.loss

        model_engine.backward(loss)

        model_engine.step()

        if rank == 0:
            total_loss += loss
            log_interval = 10
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                print('{:5d}/{:5d} batches |  loss {:5.2f}'.format(i, nbatches, cur_loss))
                total_loss = 0

if rank == 0:
    tick = time.time()

for epoch in range(1, epochs + 1):
    train()

if rank == 0:
    tock = time.time()
    elapsed_time = tock - tick

    print('Time elapsed: %.3f sec ' % (elapsed_time))

print(f"##### [rank:{rank}, run completed ...")
