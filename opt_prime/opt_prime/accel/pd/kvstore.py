"""Shared KV stores behind the paged prefix cache (#128): LMCache-compatible chunks in Mooncake Store or any byte store.

The in-process host tier (`paged.HostKVTier`) keeps evicted pages in this process. A `RemoteKVTier` instead writes
the KV of every full prompt chunk to a shared store and reads chunks back on a later prefix miss, so other engines -
other Optimus Prime replicas, or vLLM + LMCache configured the same way - can reuse it.

Compatibility follows LMCache's multiprocess (MP) mode, whose keys are reproducible outside LMCache:
  - chunks of `chunk_tokens` tokens (LMCache default 256), full chunks only;
  - chunk hash: blake3 chained over the chunk's tokens packed as big-endian uint32, seeded with
    blake3(int64 0 big-endian || uint32 0) (LMCache `v1/multiprocess/token_hasher.py`);
  - key string: "{model}@{kv_rank:08x}@{group:x}@{hash hex}[@salt]", kv_rank = world_size<<24 | rank<<16 |
    local_world_size<<8 | local_rank (LMCache `distributed/api.py` ObjectKey, L2 adapter key format);
  - value: the chunk in LMCache's KV_2LTD layout [2 (K, V), layers, tokens, kv_heads * head_dim], KV dtype.
`OptimusKeys` (our chained 64-bit page hashes) is the alternative for stores used only by Optimus Prime engines.

Stores: `MooncakeKVStore` (mooncake.store.MooncakeDistributedStore: put/get/is_exist, TCP or RDMA) and
`MemoryKVStore` (tests). Writes go through a background thread; reads are synchronous on admission.
"""
from concurrent.futures import ThreadPoolExecutor
import struct
import threading

import torch


# ---- keys ----
class LMCacheMPKeys:
    """LMCache multiprocess-mode chunk keys (blake3 chain); `model` must be the name the vLLM side uses"""

    def __init__(self, model, chunk_tokens=256, world_size=1, rank=0, local_world_size=1, local_rank=0, group=0,
                 salt=None):
        import blake3                                            # optional dependency (pip install blake3)
        self._blake3 = blake3.blake3
        self.model, self.chunk_tokens, self.group, self.salt = model, chunk_tokens, group, salt
        self.kv_rank = world_size << 24 | rank << 16 | local_world_size << 8 | local_rank
        self.none_hash = self._blake3(int(0).to_bytes(8, "big", signed=True) + struct.pack(">I", 0)).digest()

    def chunk_hashes(self, tokens):
        out, prefix = [], self.none_hash
        for i in range(len(tokens) // self.chunk_tokens):
            chunk = tokens[i * self.chunk_tokens:(i + 1) * self.chunk_tokens]
            prefix = self._blake3(prefix + struct.pack(f">{len(chunk)}I", *chunk)).digest()
            out.append(prefix)
        return out

    def keys(self, tokens):
        tail = f"@{self.salt}" if self.salt else ""
        return [f"{self.model}@{self.kv_rank:08x}@{self.group:x}@{h.hex()}{tail}" for h in self.chunk_hashes(tokens)]


class OptimusKeys:
    """our own chained 64-bit block hashes over chunks (stores shared by Optimus Prime engines only)"""

    def __init__(self, model, chunk_tokens=256):
        self.model, self.chunk_tokens = model, chunk_tokens

    def keys(self, tokens):
        from opt_prime.accel.pd.paged import block_hashes
        return [f"{self.model}@opt@{h:016x}" for h in block_hashes(tokens, self.chunk_tokens)]


# ---- stores (bytes by key) ----
class MemoryKVStore:
    def __init__(self):
        self.data, self._lock = {}, threading.Lock()

    def put(self, key, value: bytes):
        with self._lock:
            self.data[key] = bytes(value)
        return 0

    def get(self, key):
        with self._lock:
            return self.data.get(key)

    def exists(self, key):
        with self._lock:
            return key in self.data


class MooncakeKVStore:
    """Mooncake Store client. `config`: MooncakeDistributedStore.setup arguments, e.g.
    {"local_hostname": "...", "metadata_server": "http://<master>:8080/metadata", "master_server_addr": "<master>:50051",
     "protocol": "tcp" | "rdma", "rdma_devices": "", "global_segment_size": ..., "local_buffer_size": ...}"""

    def __init__(self, config, store=None):
        if store is None:
            from mooncake.store import MooncakeDistributedStore
            store = MooncakeDistributedStore()
            rc = store.setup({k: str(v) for k, v in config.items()})
            if rc:
                raise RuntimeError(f"Mooncake store setup failed ({rc})")
        self.store = store

    def put(self, key, value: bytes):
        return self.store.put(key, value)

    def get(self, key):
        if not self.store.is_exist(key):
            return None
        value = self.store.get(key)
        return value if value else None

    def exists(self, key):
        return bool(self.store.is_exist(key))

    def close(self):
        self.store.close()


# ---- the tier ----
def to_kv_2ltd(keys, values):
    """our span [layers, kv_heads, tokens, head_dim] (K and V) -> LMCache KV_2LTD [2, layers, tokens, kv_heads*head_dim]"""
    L, H, T, D = keys.shape
    return torch.stack([keys.permute(0, 2, 1, 3).reshape(L, T, H * D),
                        values.permute(0, 2, 1, 3).reshape(L, T, H * D)])


def from_kv_2ltd(chunk, kv_heads, head_dim):
    """KV_2LTD [2, L, T, H*D] -> keys, values [L, H, T, D]"""
    _, L, T, _ = chunk.shape
    kv = chunk.reshape(2, L, T, kv_heads, head_dim).permute(0, 1, 3, 2, 4)
    return kv[0], kv[1]


class RemoteKVTier:
    """chunks of `keys.chunk_tokens` tokens in a shared `store`, written on prompt completion, read on prefix misses"""

    def __init__(self, store, keys, *, layers, kv_heads, head_dim, dtype, medium="cpu", writers=2):
        self.store, self.keys = store, keys
        self.chunk_tokens = keys.chunk_tokens
        self.shape = (2, layers, self.chunk_tokens, kv_heads * head_dim)
        self.kv_heads, self.head_dim, self.dtype, self.medium = kv_heads, head_dim, dtype, medium
        self._pool = ThreadPoolExecutor(max_workers=writers, thread_name_prefix="kv-remote-put")
        self.stats = dict(put=0, put_skipped=0, get=0, hit_tokens=0, bytes_put=0, bytes_get=0)

    def chunk_keys(self, tokens):
        return self.keys.keys(tokens)

    def put_async(self, key, keys, values):
        """keys/values [layers, kv_heads, chunk, head_dim] on any device; copied to host now, stored in background"""
        chunk = to_kv_2ltd(keys, values).to("cpu", self.dtype).contiguous()
        self._pool.submit(self._put, key, chunk)

    def _put(self, key, chunk):
        if self.store.exists(key):
            self.stats["put_skipped"] += 1
            return
        data = chunk.view(torch.uint8).numpy().tobytes()
        self.store.put(key, data)
        self.stats["put"] += 1
        self.stats["bytes_put"] += len(data)

    def get(self, key):
        """-> keys, values [layers, kv_heads, chunk, head_dim] (host) or None"""
        data = self.store.get(key)
        if data is None:
            return None
        flat = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(self.dtype)
        if flat.numel() != torch.Size(self.shape).numel():
            return None                                         # another model / layout under this key: ignore
        self.stats["get"] += 1
        self.stats["bytes_get"] += len(data)
        return from_kv_2ltd(flat.view(self.shape), self.kv_heads, self.head_dim)

    def flush(self):
        """wait for queued writes (tests, shutdown)"""
        self._pool.shutdown(wait=True)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kv-remote-put")
