"""Receive-owner H2D copies into isolated storage, never an active decode cache."""
import torch


class StreamingUpload:
    def __init__(self, packet, device, stream=None):
        host=packet.host
        if (host.device.type!='cpu' or host.dtype!=torch.bfloat16 or host.ndim!=4
                or host.shape[0]!=72 or host.shape[2:]!=(8,128)
                or not 1<=host.shape[1]<2048 or not host.is_contiguous()
                or not host.is_pinned('cuda')):
            raise ValueError('expected owned pinned native KV')
        self.packet=packet
        self.stream=stream if stream is not None else torch.cuda.Stream(device=device)
        with torch.cuda.stream(self.stream):
            self.device_data=torch.empty(host.shape,dtype=host.dtype,device=device)
        self.event=torch.cuda.Event()
        self.seen=set()
        self.closed=False

    @torch.inference_mode()
    def stage(self, first, last):
        if self.closed: raise RuntimeError('upload closed')
        if (type(first) is not int or type(last) is not int or not 0<=first<last<=72
                or self.seen.intersection(range(first,last))):
            raise ValueError('invalid or repeated slots')
        with torch.cuda.stream(self.stream):
            self.device_data[first:last].copy_(self.packet.host[first:last],non_blocking=True)
            self.event.record(self.stream)
        self.seen.update(range(first,last))

    def finish(self):
        if self.closed: raise RuntimeError('upload closed')
        if len(self.seen)!=72: raise ValueError('incomplete upload')
        consumer=torch.cuda.current_stream(self.device_data.device)
        consumer.wait_event(self.event)
        self.device_data.record_stream(consumer)
        return self.device_data

    def close(self):
        if not self.closed:
            if self.seen: self.event.synchronize()
            self.closed=True
            self.packet=None
