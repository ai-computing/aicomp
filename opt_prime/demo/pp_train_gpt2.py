#
# Copyright (c) 2024-present, ETRI, All rights reserved.
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

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p

logging.basicConfig(level=logging.ERROR)

# 1) Tokenizer & Model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

config = GPT2Config(use_cache=False)
model = GPT2LMHeadModel(config)
model = model.from_pretrained("gpt2")

def get_total_params(module: nn.Module):
    return sum(p.numel() for p in module.parameters())

# 2) 환경 변수, Model 정보 출력
if int(os.environ["RANK"]) == 0:
    print('Total parameters in model: {:,}'.format(get_total_params(model)))

batch_size = 32
micro_batch_size = int(os.environ["WORLD_SIZE"]) // 2  # TODO

if int(os.environ["RANK"]) == 0:
    print(f"total process count: {os.environ['WORLD_SIZE']}")
    print(f"batch size: {batch_size}")
    print(f"micro batch size: {micro_batch_size}")

# 3) Optimus_p 초기화
optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True)
print(f" rank={optimus_p.get_rank()} ...")

# 4) Optimizer & Scheduler
optimus_p.train()
optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# 5) Dataset & Dataloader
datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(record) for record in datasets if len(str(record)) < 500]
dataloader = DataLoader(datasets, batch_size=batch_size, num_workers=4)
data_size = len(dataloader.dataset)
nbatches = len(dataloader)
print(f"data_size={data_size}")
print(f"nbatches={nbatches}")

# 6) TensorBoard SummaryWriter
# runs/date-time
if optimus_p.is_last_stage():
    log_dir = os.path.join("/workspace/runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

# 7) 학습 함수 (epoch 인자로 받기)
epochs = 1

def train_one_epoch(epoch):
    optimus_p.train()  # turn on the train mode

    total_loss = 0.0
    start_time = time.time()

    # tqdm 진행 표시줄은 "마지막 스테이지 에서만 띄웁니다.
    # if optimus_p.is_last_stage() and optimus_p.get_rank() == 0:
    if optimus_p.is_last_stage():    
        pbar = tqdm(enumerate(dataloader), total=nbatches, desc=f"Epoch {epoch}")
    else:
        pbar = enumerate(dataloader)

    for i, batch in pbar:
        data, labels = None, None

        # 첫 스테이지면 토크나이징
        if optimus_p.is_first_stage():
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            data, labels = tokens.input_ids, tokens.input_ids

        # label은 마지막 스테이지로 이동
        labels = optimus_p.move_labels2last_stage(labels)

        # Forward
        optimizer.zero_grad()
        optimus_p.run(data, labels, mode="1f1b")

        # Loss 계산은 마지막 스테이지
        if optimus_p.is_last_stage():
            loss = optimus_p.get_loss()
        else:
            loss = None

        # Backward
        torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)
        optimizer.step()

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

# 8) 학습 루프
if optimus_p.get_rank() == 0:
    tick = time.time()

for epoch in range(1, epochs + 1):
    train_one_epoch(epoch)
    scheduler.step()

if optimus_p.get_rank() == 0:
    tock = time.time()
    print(f"Time elapsed: {tock - tick:.3f} sec")
    if writer is not None:
        writer.close()

print(f"[rank:{optimus_p.get_rank()}] run completed ...")
