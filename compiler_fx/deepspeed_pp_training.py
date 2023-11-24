#
# Copyright (c) 2023-present, ETRI, All rights reserved.
#
#
#  This is a test program for running DeepSpeed with pipeline-parallel training.
#
#
#  Sample Usage for Pipeline-Parallel execution:
#
#      # deepspeed deepspeed_pp_training.py --deepspeed_config=ds_config.json -p 8 --steps=100
#
#



import os
import argparse

import torch
import torch.distributed as dist
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from itertools import cycle
import time

import deepspeed
from deepspeed.pipe import PipelineModule

from deepspeed.utils import logger,logging

# For details, comment the following line
logger.setLevel(logging.log_levels["error"])

batch_size = 64

in_features = 5120
out_features = 5120
hidden = 5120


TestModel = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace = False),

        nn.Linear(hidden, hidden),
        nn.ReLU(inplace = False),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace = False),

        nn.Linear(hidden, hidden),
        nn.ReLU(inplace = False),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace = False),

        nn.Linear(hidden, hidden),
        nn.ReLU(inplace = False),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace = False),

        nn.Linear(hidden, hidden),
        nn.ReLU(inplace = False),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace = False),

        nn.Linear(hidden, out_features),
        nn.ReLU(inplace = False)
        )


class MyDataset(Dataset):
    def __init__(self):
        self.outputs = torch.rand(batch_size, out_features)
        self.inputs = torch.rand(batch_size, in_features)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        #if idx >= batch_size:
        #    idx = idx % batch_size
        #inputs = torch.FloatTensor(self.inputs[idx])
        #outputs = torch.FloatTensor(self.outputs[idx])
        inputs = torch.FloatTensor(self.inputs)
        outputs = torch.FloatTensor(self.outputs)
        return inputs, outputs



def get_args():
    parser = argparse.ArgumentParser(description='TESTMODEL')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args



#def train_pipe(args, part='parameters'):
def train_pipe(args, part='uniform'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)


    net = PipelineModule(layers=TestModel,
                         loss_fn=torch.nn.MSELoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    trainset = MyDataset()
    train_loader = DataLoader(trainset, batch_size=batch_size)
    #train_iter = iter(train_loader)
    train_iter = iter(cycle(train_loader))

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    if int(os.environ['RANK']) == 0:
        tick = time.time()

    for step in range(args.steps):
        loss = engine.train_batch(data_iter=train_iter)

    if int(os.environ['RANK']) == 0:
        tock = time.time()
        elapsed_time = tock - tick
        print('Time elapsed: %.3f sec ' % (elapsed_time))


if __name__ == '__main__':
    args = get_args()

    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    train_pipe(args)
