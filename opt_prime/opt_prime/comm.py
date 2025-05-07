import torch
import torch.distributed as dist
import logging
import os
from torch import Tensor, Size

from opt_prime.IR import IR_Anal

from torch.distributed._functional_collectives import AsyncCollectiveTensor



logging.basicConfig(level=logging.ERROR)

NoneType=type(None)


class Comm:

    def __init__(self, use_gpu=False, ir_analyze: IR_Anal = IR_Anal.PARALLEL):

        self.ds_type2id = {
            Tensor: 100,
            tuple: 101,
            list: 102, 
            Size: 103, 
            int: 104, 
            NoneType: 105,
            type: 106,
            AsyncCollectiveTensor: 107, }

        self.ds_id2type = {v:k for k, v in self.ds_type2id.items()}
        
        self.tensor_type2id = {
            torch.float32: 0,
            torch.float64: 1,
            torch.complex64: 2,
            torch.complex128: 3,
            torch.float16: 4,
            torch.bfloat16: 5,
            torch.uint8: 6,
            torch.int8: 7,
            torch.int16: 8,
            torch.int32: 9,
            torch.int64: 10,
            torch.bool: 11, }

        self.tensor_id2type = {v:k for k,v in self.tensor_type2id.items()}

        self.init_comm(use_gpu)

        if ir_analyze == IR_Anal.SINGLE:
            self.setup_ctrl_group()


    def init_comm(self, use_gpu):
        torch.manual_seed(42)

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.master_addr = os.getenv("MASTER_ADDR")
        self.master_port = os.getenv("MASTER_PORT")
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

        if use_gpu == True:
            gpu_cnt = torch.cuda.device_count()
            if self.local_rank == 0:
                print(f"Available GPUs per server: {gpu_cnt}")
            if self.local_rank + 1 > gpu_cnt:
                logging.error(f"This program cannot create more processes than the number of available GPUs:{gpu_cnt}")
                sys.exit(1)

            self.backend = "nccl"
            print(f"GPU mode is used.")
        else:
            self.backend = "gloo"
            print(f"CPU mode is used.")

        if dist.is_initialized():
            print(f"Communication already initialized")
            return

        init_method = "tcp://" + str(self.master_addr) + ":" + str(self.master_port)
        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size, init_method=init_method)

        logging.info(f" --- rank:{dist.get_rank()}, world_size:{dist.get_world_size()}")



    def receive_data(self, from_rank, device):
        ds_type = torch.tensor([0], dtype=torch.long, device=device)
        dist.recv(ds_type, from_rank)

        ds_type = self.ds_id2type[ds_type.item()]
        
        #if str(ds_type) == "<class 'Tensor'>":
        if ds_type is Tensor:
            return self.receive_tensor(from_rank, device)
        #elif str(ds_type) == "<class 'tuple'>":
        elif ds_type is tuple:
            return self.receive_tuple(from_rank, device)
        #elif str(ds_type) == "<class 'list'>":
        elif ds_type is list:
            return self.receive_list(from_rank, device)
        elif ds_type is Size:
            return self.receive_size(from_rank, device)
        elif ds_type is int:
            return self.receive_int(from_rank, device)
        elif ds_type is set:
            return self.receive_set(from_rank, device)
        elif ds_type is NoneType:
            return self.receive_none(from_rank)
        elif ds_type is type:
            return self.receive_type(from_rank, device)
        elif ds_type is AsyncCollectiveTensor:
            return self.receive_tensor(from_rank, device)
        else:
            logging.critical(f"#### receive_data: not supported type!")
        # TODO


    def send_data(self, obj, to_rank, device):
        ds_type = self.ds_type2id[type(obj)]
        ds_type = torch.tensor(ds_type, dtype=torch.long, device=device)
        dist.send(ds_type, to_rank)

        if isinstance(obj, torch.Tensor):
            self.send_tensor(obj, to_rank, device)
        elif isinstance(obj, tuple):
            self.send_tuple(obj, to_rank, device)
        elif isinstance(obj, list):
            self.send_list(obj, to_rank, device)
        elif isinstance(obj, Size):
            self.send_size(obj, to_rank, device)
        elif isinstance(obj, int):
            self.send_int(obj, to_rank, device)
        elif isinstance(obj, set):
            self.send_set(obj, to_rank, device)
        elif obj is None:
            self.send_none(obj, to_rank)
        elif isinstance(obj, type):
            self.send_type(obj, to_rank, device)
        elif isinstance(obj, AsyncCollectiveTensor):
            obj = obj.wait()
            self.send_tensor(obj, to_rank, device)
        else:
            logging.critical(f"#### send_data: not supported type!")


    def receive_set(self, from_rank, device):
        return set(self.receive_list(from_rank, device))

    def send_set(self, obj, to_rank, device):
        self.send_list(list(obj), to_rank, device)

    def receive_int(self, from_rank, device):
        int_data = torch.tensor([0], dtype=torch.long, device=device)
        dist.recv(int_data, from_rank)
        return int_data.item()

    def send_int(self, obj, to_rank, device):
        int_data = torch.tensor([obj], dtype=torch.long, device=device) # ex. 2
        dist.send(int_data, to_rank)


    def receive_size(self, from_rank, device):
        return Size(self.receive_list(from_rank, device))

    def send_size(self, obj, to_rank, device):
        self.send_list(list(obj), to_rank, device)

    def receive_tuple(self, from_rank, device):
        return tuple(self.receive_list(from_rank, device))

    def send_tuple(self, obj, to_rank, device):
        self.send_list(list(obj), to_rank, device)

    def send_none(self, obj, to_rank):
        logging.debug(f"send_none")

    def receive_none(self, from_rank):
        return None

    def send_type(self, obj, to_rank, device):
        type_data = torch.tensor([self.ds_type2id[type(obj)]], dtype=torch.long, device=device) # ex. 2
        dist.send(type_data, to_rank)

    def receive_type(self, from_rank, device):
        type_data = torch.tensor([0], dtype=torch.long, device=device)
        dist.recv(type_data, from_rank)
        return self.ds_id2type[type_data.item()]

    def receive_tensor(self, from_rank, device):
        dimension = torch.tensor([0], dtype=torch.long, device=device)
        dist.recv(dimension, from_rank)
        #logging.debug(f" >>>>> recv_tensor, dimension:{dimension} from rank:{from_rank}")

        shape = torch.tensor([0] * dimension.item(), dtype=torch.long, device=device)
        dist.recv(shape, from_rank)
        #logging.debug(f" >>>>> recv_tensor, shaple:{shape} from rank:{from_rank}")
        shape = tuple(shape.tolist())

        ttype = torch.tensor([0], dtype=torch.long, device=device)
        dist.recv(ttype, from_rank)
        #logging.debug(f" >>>>> recv_tensor, ttype:{ttype} from rank:{from_rank}")

        ttype = self.tensor_id2type[ttype.item()]

        obj = torch.zeros(size=shape, dtype=ttype, device=device)
        dist.recv(obj, from_rank)
        #logging.debug(f" >>>>> recv_tensor, obj:{obj} from rank:{from_rank}")

        return obj

    def send_tensor(self, obj, to_rank, device):
        if isinstance(obj, torch.Tensor):
            obj_size = obj.size()
            dimension = torch.tensor(len(obj_size), dtype=torch.long, device=device) # ex. 2
            logging.debug(f" >>>>> send_tensor, obj.size():{obj_size}, len:{len(obj_size)}, dimension:{dimension}")
        dist.send(dimension, to_rank)

        if isinstance(obj, torch.Tensor):
            shape = torch.tensor(list(obj_size), dtype=torch.long, device=device) # ex. [54, 5120]
        dist.send(shape, to_rank)

        ttype = self.tensor_type2id[obj.dtype]
        ttype = torch.tensor(ttype, dtype=torch.long, device=device)
        dist.send(ttype, to_rank)
        #logging.debug(f" >>>>> send_tensor, ttype:{ttype}")

        if not obj.is_contiguous():
            obj = obj.contiguous()
            #logging.debug(f" >>> obj made to be contiguous")

        obj = obj.to(device)
        dist.send(obj, to_rank)
        #logging.debug(f" >>>>> send_tensor, obj:{obj}")

    def receive_list(self, from_rank, device):
        length = torch.tensor([0], dtype=torch.long, device=device)
        dist.recv(length, from_rank)

        obj = []
        for _ in range(length.item()):
            n = self.receive_data(from_rank, device)
            obj.append(n)

        return obj

    def send_list(self, obj, to_rank, device):
        length = torch.tensor(len(obj), dtype=torch.long, device=device)
        dist.send(length, to_rank)

        for n in obj:
            self.send_data(n, to_rank, device)


    def setup_ctrl_group(self):
        print(f"[rank:{self.rank}] ir_analyze=IR_Anal.SINGLE")
        self.ctrl_group: Dict[int, Any] = {}
        rank_pair: Dict[int, List[int]] = {}

        for rank in range(self.world_size):
            if rank == 0:
                continue
            rank_pair.setdefault(rank, [0, rank])


        for rank in range(self.world_size):
            if rank == 0:
                continue
            pair_ranks = rank_pair[rank]
            self.ctrl_group[rank] = dist.new_group(pair_ranks)

        print(f"[rank:{self.rank}], setup_ctrl_group completed")
