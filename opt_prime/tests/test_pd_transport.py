"""TorchCacheTransport on hetero-kv keeps the tensor-copy contract: prefix [0, seq_len) of every layer and batch
entry, receipt bytes, published seq_len, and a reusable lane after rejections and recovery."""
import pytest
import torch

from opt_prime.accel.pd import KVLease, KVSpec
from opt_prime.accel.pd.torch_engine import LayeredKVCache, TorchCache, TorchCacheTransport
from opt_prime.kv_cache import KVCacheManager

SPEC = KVSpec("tiny", "seed17", "torch.float32", "torch-bhsd-full-attention-v1:test", (1,))


def kv_manager(layers, batch, heads, dim, dtype):
    def make(capacity):
        m = KVCacheManager(layers, batch, capacity, heads, dim, dtype=dtype, device="cpu")
        m.allocate(batch)
        return m
    return make


def fill(manager, n):
    gen = torch.Generator().manual_seed(n)
    for i in range(manager.layer_start, manager.layer_end):
        for t in manager.storage(i):
            t[:, :, :n] = torch.randn(t[:, :, :n].shape, generator=gen).to(t.dtype)
    manager.set_seq_len(n)


def leases(src, dst, endpoint="destination"):
    return KVLease(SPEC, src, TorchCache(src), "source"), KVLease(SPEC, dst, TorchCache(dst), endpoint)


@pytest.mark.parametrize("make", [
    kv_manager(3, 1, 2, 8, torch.float32),
    kv_manager(2, 2, 2, 4, torch.bfloat16),                                    # batch > 1
    lambda capacity: LayeredKVCache([(2, 8), (1, 16), (2, 8)], capacity, torch.float32, "cpu"),
])
def test_copies_prefix_and_publishes_seq_len(make):
    src, dst = make(16), make(16)
    fill(src, 7)
    transport = TorchCacheTransport(source="source", destination="destination", spec=SPEC)
    expected = sum(t.numel() * t.element_size() for i in range(src.layer_end) for t in src.get(i))
    receipt = transport.transfer("r", *leases(src, dst))
    assert receipt == dict(transport="torch-copy", bytes=expected, cache_position=7, completed=True)
    assert dst.get_seq_len() == 7
    for i in range(src.layer_end):
        for a, b in zip(src.get(i), dst.get(i)):
            torch.testing.assert_close(b, a, rtol=0, atol=0)
    transport.finish("r")
    fill(src, 3)
    assert transport.transfer("r2", *leases(src, dst))["cache_position"] == 3
    transport.finish("r2")
    assert transport.recover() is True


def test_rejections_and_recovery_leave_lane_usable():
    make = kv_manager(1, 1, 1, 4, torch.float32)
    src, dst = make(8), make(8)
    transport = TorchCacheTransport(source="source", destination="destination", spec=SPEC)
    with pytest.raises(ValueError, match="invalid source sequence length"):
        transport.transfer("r", *leases(src, dst))
    with pytest.raises(ValueError, match="unconfigured tensor-copy route"):
        transport.transfer("r", *leases(src, dst, "elsewhere"))
    fill(src, 8)
    transport.transfer("r", *leases(src, dst))
    assert transport.recover() is True              # decoder failed: PDRuntime.recover() settles the lane
    transport.transfer("r3", *leases(src, dst))
    transport.finish("r3")
