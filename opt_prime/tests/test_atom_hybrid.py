"""Hybrid SDK contract: local cache belongs to lanes, global cache to blocks."""

import pytest
import torch
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.platforms.atom.batching import AtomBatcher
from test_atom_batching import FakeBlocks, _greedy, _llama, _prompts


class HybridFake(FakeBlocks):
    def __init__(self):
        super().__init__(_llama(), batch=2, blocks=4)
        self.rbln_config.cache_impl = "hybrid"
        self.rbln_config.use_position_ids = True
        self.dec_attn_mask = torch.ones(2, 128)
        self.decoders = {2: self}
        self.local_calls = []

    def prefill_decoder(self, *, local_block_tables=None, **kw):
        assert local_block_tables is not None, "missing local cache ownership"
        assert local_block_tables.tolist() == [kw["batch_idx"]]
        self.local_calls.append((kw["batch_idx"], kw["block_tables"].tolist()))
        return super().prefill_decoder(**kw)

    def __call__(self, *, local_block_tables=None, position_ids=None, **kw):
        assert local_block_tables.tolist() == [[0], [1]]
        assert torch.equal(position_ids, kw["cache_position"])
        for lane in range(2):
            if int(kw["block_tables"][lane, 0]) == 3:
                continue
            pos = int(kw["cache_position"][lane, 0])
            self.dec_attn_mask[lane, pos] = 1
            expected = torch.arange(128) <= pos
            assert torch.equal(self.dec_attn_mask[lane].bool(), expected), (
                "stale or missing prompt mask"
            )
        return self._decode(**kw)


def test_hybrid_refill_reinitializes_only_request_lane():
    model = HybridFake()
    b = AtomBatcher(model)
    prompts = _prompts(4, 20, 32)
    for i, p in enumerate(prompts):
        f = b.submit(PDRequest(str(i), p, 3))
        b.run_until_idle()
        assert f.result() == _greedy(model.m, p, 3)
    assert any(lane != blocks[0] for lane, blocks in model.local_calls)


@pytest.mark.parametrize(
    "options",
    [{"prefix_cache": True}, {"host_kv_gib": 1}, {"prefill_worker": object()}],
)
def test_hybrid_unrestored_cache_modes_fail_before_sdk(options):
    model = HybridFake()
    with pytest.raises(ValueError, match="sliding"):
        AtomBatcher(model, **options)
    assert not model.local_calls


def test_hybrid_staggered_cancel_and_refill_preserves_other_lane():
    from opt_prime.accel.pd.runtime import RequestCancelled

    model = HybridFake()
    b = AtomBatcher(model)
    p, q, r = _prompts(3, 20, 35)
    first = b.submit(PDRequest("a", p, 12))
    b.step()
    second = b.submit(PDRequest("b", q, 8))
    b.step()
    assert b.cancel("a")
    third = b.submit(PDRequest("c", r, 4))
    b.run_until_idle()
    with pytest.raises(RequestCancelled):
        first.result()
    assert second.result() == _greedy(model.m, q, 8)
    assert third.result() == _greedy(model.m, r, 4)
    assert sorted(b.free) == [0, 1]
