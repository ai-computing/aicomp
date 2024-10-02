#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#
#
#   Usage: torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="X.X.X.X" --master_port=29500 <program name>.py
#

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.utils.data.distributed import DistributedSampler
import os

torch.manual_seed(42)

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
master_addr = os.getenv("MASTER_ADDR")
master_port = os.getenv("MASTER_PORT")

batch_size=64


def prepare_data():
    dataset = load_dataset("imdb")

    dataset = dataset["train"]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")


    range_ = 6000

    group = rank // optimus_t.tp_size
    small_train_dataset = tokenized_datasets.select(range(range_+batch_size*group))
    train_dataloader = DataLoader(small_train_dataset, shuffle=False, batch_size=batch_size)
    train_dataloader = iter(train_dataloader)
    for _ in range(group):
        batch = next(train_dataloader)

    return train_dataloader



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(512, 24)
        self.fc2 = nn.Linear(24, 2)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.weight.data.uniform_(-initrange, initrange)

        self.fc1.bias.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return x



class Optimus_t:
    def __init__(self, dp_size, tp_size):
        self.dp_size = dp_size
        self.tp_size = tp_size

        self.setup()

        device_type = "cuda"
        self.model = MLP().to(device=device_type)

        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    

        self.device_mesh = init_device_mesh("cuda", mesh_shape=(self.dp_size, self.tp_size), mesh_dim_names=("dp", "tp"))
        tp_group = self.device_mesh["tp"].get_group()
        dp_group = self.device_mesh["dp"].get_group()
        dp_mesh = self.device_mesh["dp"]
        tp_mesh = self.device_mesh["tp"]

        #
        print(f"[{rank}] >>>  tp group:{tp_mesh}, dp_group:{dp_mesh}")

        parallel_dict = { 'fc1': ColwiseParallel(), 'fc2': RowwiseParallel() }

        parallelized_model = parallelize_module(self.model, tp_mesh, parallel_dict)

        self.parallelized_model = FSDP(parallelized_model, device_mesh=dp_mesh, sharding_strategy=ShardingStrategy.NO_SHARD)

        self.criterion = nn.CrossEntropyLoss().cuda(rank)
        self.optimizer = torch.optim.Adam(self.parallelized_model.parameters(), lr=3e-5)



    def setup(self):
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        self.device = torch.device(f"cuda:{local_rank}")


    def run(self, data, label):
        self.optimizer.zero_grad()

        outputs = self.parallelized_model(data)
        loss = self.criterion(outputs, label)
        loss.backward()

        self.optimizer.step()

        return loss


optimus_t = Optimus_t(dp_size=4, tp_size=2)

if rank == 0:
    print(f">> tp_size:{optimus_t.tp_size}, dp_size:{optimus_t.dp_size}")

train_dataloader = prepare_data()

num_epochs = 1

i = 0


for epoch in range(num_epochs):
    for batch in train_dataloader:
        inputs = batch['input_ids'].to(torch.float32).to(optimus_t.device)
        targets = batch['label'].to(optimus_t.device)

        if i == 0:
            print(f"[{i}] >> rank:{rank}, inputs ==> {inputs}")

        loss = optimus_t.run(inputs, targets)

        if rank == 0:
            print(f"Loss: {loss.item()}")

        i += 1


print(f"[rank:[{rank}], fc1.weight:{optimus_t.parallelized_model.fc1.weight}, fc2.weight:{optimus_t.parallelized_model.fc2.weight}")
print(f" -------------------------------------------------------------")
