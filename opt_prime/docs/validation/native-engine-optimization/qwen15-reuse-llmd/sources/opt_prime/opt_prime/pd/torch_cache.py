"""Optimus KV storage and explicit Torch tensor-copy transport."""
from dataclasses import dataclass
import torch
from opt_prime.kv_cache import KVCacheManager


@dataclass
class TorchCache:
    manager: KVCacheManager


def fence(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class TorchCacheTransport:
    """Explicit tensor-copy transport; never selected as a P2P fallback."""
    def __init__(self, *, source, destination, spec):
        self.source, self.destination, self.spec = source, destination, spec

    def supports(self, source, destination, spec):
        return (source, destination, spec) == (self.source, self.destination, self.spec)

    @torch.inference_mode()
    def transfer(self, request_id, source, destination):
        if source.spec != destination.spec or not self.supports(source.endpoint, destination.endpoint, source.spec):
            raise ValueError('unconfigured tensor-copy route')
        src, dst = source.payload.manager, destination.payload.manager
        position = src.get_seq_len()
        if not 0 < position <= dst.max_seq_len:
            raise ValueError('invalid source sequence length')
        copied = 0
        for i in range(src.layer_start, src.layer_end):
            keys, values = src.get(i)
            dst.update(i, keys.to(dst.device), values.to(dst.device), position=0)
            copied += keys.numel() * keys.element_size() + values.numel() * values.element_size()
        dst.set_seq_len(position)
        fence(src.device)
        fence(dst.device)
        return dict(transport='torch-copy', bytes=copied, cache_position=position, completed=True)

    def finish(self, request_id):
        pass
