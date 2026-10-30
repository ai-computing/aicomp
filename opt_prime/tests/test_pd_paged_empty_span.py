"""An empty KV span (nothing left to import, e.g. a fully cached prompt) must index as a no-op (#47)."""
import torch
from opt_prime.accel.pd.paged import PagedKV


def pool():
    cache = PagedKV(2, 1, 1, 2, torch.float32, 'cpu', pages=4, page_size=4, max_context=16)
    assert cache.alloc(0, 8) == 0
    return cache


def test_empty_span_index_is_long_and_empty():
    idx = pool().span_index(0, 5, 5)
    assert idx.dtype == torch.long and idx.numel() == 0


def test_empty_span_write_is_noop():
    cache = pool()
    empty = torch.ones(2, 1, 0, 2)
    cache.write_span(0, 5, empty, empty)
    assert all(not x.any() for x in cache.gk + cache.gv)
