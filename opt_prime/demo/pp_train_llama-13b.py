#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#
# Usage: torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> 
#                 --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama.py <llama_access_token>
#
# *** This program was tested with torch 2.3.1 and transformers 4.42.4.
#     The version of transformers used must be consistent across all machines used for testing ***
#


import torch
import torch.nn as nn


import torch.distributed as dist
import datetime
import logging
import os
import sys
import math
import time
from packaging import version

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

### add tqdm & tensorboard
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p
from opt_prime.IR import IR_Anal

from torch.amp import autocast

logging.basicConfig(level=logging.ERROR)


#
# This program needs 'access token' for Llama. First, obtain your access token for Llama !!!
#
if len(sys.argv) > 1:
    os.environ['LLAMA_ACCESS_TOKEN'] = sys.argv[1]

access_token = os.getenv('LLAMA_ACCESS_TOKEN')
if access_token is None:
    raise ValueError("LLAMA_ACCESS_TOKEN environment variable is not set."
                    "       [Usage:] torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama.py <llama_access_token>")


#
# This program needs torch version 2.3.1 or higher !!!
#
required_version = "2.3.1"
current_version = torch.__version__

if version.parse(current_version) >= version.parse(required_version):
    print(f"[rank:{int(os.environ['RANK'])}] torch version 2.3.1 or higher --> OK")
else:
    raise ValueError('This program needs torch version 2.3.1 or higher. But current torch version is {current_version}.')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token=access_token, use_cache=False)

def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


if int(os.environ["RANK"]) == 0:
    print ('Total parameters in model: {:,}'.format(get_total_params(model)))


batch_size = 384
micro_batch_size = 64

if int(os.environ["RANK"]) == 0:
    print(f"total process count: {os.environ['WORLD_SIZE']}")
    print(f"batch size: {batch_size}")
    print(f"micro batch size: {micro_batch_size}")

#optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True)
#optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=False, ir_analyze=IR_Anal.SEQUENTIAL)
optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, activation_ckpt=False, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=False, swap_model_in_optstep=False, ir_analyze=IR_Anal.SEQUENTIAL)
print(f" rank={optimus_p.get_rank()} ...")

optimus_p.train()

#optimus_p.optimizer = torch.optim.SGD(optimus_p.parameters(), lr=5.0)
optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimus_p.optimizer, 1.0, gamma=0.95)

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(record) for record in datasets if len(str(record)) < 500]
dataloader = DataLoader(datasets, batch_size=batch_size, num_workers=4)
data_size=len(dataloader.dataset)
print(f"data_size={data_size}")
nbatches = len(dataloader)
print(f"nbatches={nbatches}")


# TensorBoard SummaryWriter
# runs/date-time folder
if optimus_p.is_last_stage():
    log_dir = os.path.join("/workspace/runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)


epochs = 1 # The number of epochs

def train():

    optimus_p.train() # turn on the train mode

    total_loss = 0
    start_time = time.time()

    # add tqdm
    if optimus_p.is_last_stage():    
        pbar = tqdm(enumerate(dataloader), total=nbatches, desc=f"Epoch {epoch}")
    else:
        pbar = enumerate(dataloader)

    # for i, batch in enumerate(dataloader):
    for i, batch in pbar:
        data, labels = None, None

        # prepare input and label
        if optimus_p.is_first_stage():
            tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
            data, labels = tokens.input_ids, tokens.input_ids

        labels = optimus_p.move_labels2last_stage(labels)

        optimus_p.optimizer.zero_grad()

        #optimus_p.run(data, labels)
        #optimus_p.run(data, labels, mode="gpipe")
        with autocast(device_type='cuda', dtype=torch.bfloat16):    
            optimus_p.run(data, labels, mode="1f1b")

        if optimus_p.is_last_stage():
            loss = optimus_p.get_loss() 
        else:
            loss = None

        torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)
        optimus_p.optimizer.step()

        # if optimus_p.is_last_stage():
        #     loss = sum(loss) / optimus_p.mbsize
        #     total_loss += loss
        #     log_interval = 10
        #     if i % log_interval == 0 and i > 0:
        #         cur_loss = total_loss / log_interval
        #         elapsed = time.time() - start_time
        #         print('| epoch {:3d} | {:5d}/{:5d} batches | '
        #             'lr {:02.2f} | ms/batch {:5.2f} | '
        #             'loss {:5.2f} | ppl {:8.2f}'.format(
        #                 epoch, i, nbatches, scheduler.get_lr()[0],
        #                 elapsed * 1000 / log_interval,
        #                 cur_loss, math.exp(cur_loss)))

        #         total_loss = 0
        #         start_time = time.time()

        # 로그 및 TensorBoard 기록 (마지막 스테이지에서만)
        if optimus_p.is_last_stage():
            # 여러 micro-batch의 loss 합이 리스트로 반환되므로 sum 후 평균
            loss_value = sum(loss) / optimus_p.mbsize
            total_loss += loss_value

            log_interval = 1
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                ms_per_batch = elapsed * 1000 / log_interval
                ppl = math.exp(cur_loss)

                # if optimus_p.get_rank() == 0:
                # tqdm 진행 표시줄 업데이트
                if isinstance(pbar, tqdm):
                    pbar.set_postfix(loss=f"{cur_loss:.2f}",
                                     ppl=f"{ppl:.2f}",
                                     ms_batch=f"{ms_per_batch:.2f}")

                # TensorBoard 기록
                if writer is not None:
                    # 전체 스텝: (epoch-1)*nbatches + 현재 배치 인덱스
                    global_step = (epoch - 1) * nbatches + i
                    writer.add_scalar("Train/Loss", cur_loss, global_step)
                    writer.add_scalar("Train/Perplexity", ppl, global_step)
                    writer.add_scalar("Train/ms_per_batch", ms_per_batch, global_step)

                # log_interval마다 누적된 loss 초기화
                total_loss = 0.0
                start_time = time.time()

if optimus_p.get_rank() == 0:
    tick = time.time()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    scheduler.step()

if optimus_p.get_rank() == 0:
    tock = time.time()
    elapsed_time = tock - tick

    print('Time elapsed: %.3f sec ' % (elapsed_time))

print(f"[rank:{optimus_p.get_rank()}, run completed ...")
