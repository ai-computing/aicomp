"""Optimus KV storage and explicit Torch tensor-copy transport."""
from dataclasses import dataclass
import torch
from opt_prime.kv_cache import KVCacheManager


@dataclass
class TorchCache:
    manager: KVCacheManager


class LayeredKVCache:
    """Single-request K/V storage with its own shape per layer (models whose layers keep differently shaped K/V,
    e.g. Gemma 4). Same interface as the parts of KVCacheManager the PD engine and transport use."""

    def __init__(self, layer_dims, max_seq_len, dtype, device):
        self.layer_start, self.layer_end = 0, len(layer_dims)
        self.max_seq_len, self.dtype, self.device = max_seq_len, dtype, torch.device(device)
        self.k = [torch.zeros(1, h, max_seq_len, d, dtype=dtype, device=self.device) for h, d in layer_dims]
        self.v = [torch.zeros(1, h, max_seq_len, d, dtype=dtype, device=self.device) for h, d in layer_dims]
        self._len = 0

    def allocate(self, batch_size=1):
        if batch_size != 1:
            raise ValueError('one request per layered cache')

    def update(self, layer, keys, values, position=0):
        n = keys.shape[2]
        if position + n > self.max_seq_len:
            raise ValueError('cache capacity exceeded')
        self.k[layer][:, :, position:position + n] = keys
        self.v[layer][:, :, position:position + n] = values

    def get(self, layer):
        return self.k[layer][:, :, :self._len], self.v[layer][:, :, :self._len]

    def set_seq_len(self, n):
        self._len = n

    def get_seq_len(self):
        return self._len


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
