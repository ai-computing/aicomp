"""Opt-in bounded direct receive into owned pinned native-layout KV storage."""
import numpy as np
import torch
import zmq
from memory_receiver import CanonicalAssembler
from protocol import MAX_POSITIONS


class PinnedPacket:
    def __init__(self, positions):
        self.host = torch.empty((72,positions,8,128),dtype=torch.bfloat16,pin_memory=True)
        self.nbytes = self.host.numel()*self.host.element_size()

    def torch_layers(self):
        return [(self.host[2*i].transpose(0,1)[None],self.host[2*i+1].transpose(0,1)[None]) for i in range(36)]


class PinnedAssembler(CanonicalAssembler):
    def __init__(self, request_id, positions):
        if not 1 <= positions <= MAX_POSITIONS: raise ValueError('unsupported position count')
        self.request_id,self.positions = request_id,positions
        self.data = PinnedPacket(positions)
        self.array = self.data.host.view(torch.uint16).numpy()
        self.seen = np.zeros(72,dtype=bool)
        self.expected_positions = np.arange(positions,dtype=np.uint32)
        self.final = False
        self.failed = False

    def receive(self, header, socket):
        if self.failed: raise RuntimeError('failed assembler cannot be reused')
        try:
            slots,final = self.validate_header(header)
            first,last = int(slots[0]),int(slots[-1])+1
            if not np.array_equal(slots,np.arange(first,last)):
                raise ValueError('direct receive requires consecutive slots')
            target = memoryview(self.array[first:last]).cast('B')
            received = socket.recv_into(target)
            if received != target.nbytes or socket.getsockopt(zmq.RCVMORE):
                raise ValueError('invalid payload size or extra multipart frame')
            self.seen[slots] = True
            self.final = final
            return final
        except BaseException:
            self.failed = True
            raise
