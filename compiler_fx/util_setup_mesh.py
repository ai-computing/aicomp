import os
import sys
import time
import math

import torch
import torch.distributed as dist

from torch.distributed.device_mesh import init_device_mesh


rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
master_addr = os.getenv("MASTER_ADDR")
master_port = os.getenv("MASTER_PORT")

#
# world size 8
#
pp_size = 2
tp_size = 2
dp_size = 2

#
# world size 16
#
#pp_size = 8
#tp_size = 2
#dp_size = 1

#
# world size 12
#
#pp_size = 3
#dp_size = 2
#tp_size = 2


assert world_size == pp_size * dp_size * tp_size, f"pp_size({pp_size}) * dp_size({dp_size}) * tp_size({tp_size}) must be equal to world_size({world_size})"
assert world_size % tp_size == 0, f"world size({world_size}) must be divisible by tp size({tp_size})"
assert world_size % dp_size == 0, f"world size({world_size}) must be divisible by dp size({dp_size})"


dist.init_process_group("nccl", rank=rank, world_size=world_size)
torch.cuda.set_device(local_rank)

device = torch.device(f"cuda:{local_rank}")

device_mesh = init_device_mesh("cuda", mesh_shape=(pp_size, dp_size, tp_size), mesh_dim_names=("pp", "dp", "tp"))
tp_group = device_mesh["tp"].get_group()
dp_group = device_mesh["dp"].get_group()
pp_group = device_mesh["pp"].get_group()
tp_mesh = device_mesh["tp"]
dp_mesh = device_mesh["dp"]
pp_mesh = device_mesh["pp"]


print(f"[{rank}] >>>  pp group:{pp_mesh}, dp_group:{dp_mesh}, tp_group:{tp_mesh}")

time.sleep(2)

print(f"[rank:{rank}, run completed ...")
