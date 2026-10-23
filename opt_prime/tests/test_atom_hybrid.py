"""Hybrid SDK contract: local cache belongs to lanes, global cache to blocks."""

import pytest
import torch
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.platforms.atom.batching import AtomBatcher
from test_atom_batching import FakeBlocks, _greedy, _llama, _prompts, fake_rebel  # noqa: F401


class HybridFake(FakeBlocks):
    def __init__(self):
        super().__init__(_llama(), batch=2, blocks=4)
        self.rbln_config.cache_impl = "hybrid"
        self.rbln_config.use_position_ids = True
        for meta in self.rbln_config.kvcache_metas:
            meta["layer_type"] = "full_attention"
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
    [{"prefix_cache": True}, {"host_kv_gib": 1}],
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


def test_hybrid_remote_prefill_resets_masks_and_reuses_lanes(fake_rebel):  # noqa: F811
    import copy

    from opt_prime.accel.pd.pd_batching import PrefillWorker, TorchPrefillEngine

    model = HybridFake()
    fake_rebel["model"] = model
    worker = PrefillWorker(
        TorchPrefillEngine(
            copy.deepcopy(model.m), max_context=128, chunk=16, page_size=16
        )
    )
    try:
        batcher = AtomBatcher(model, prefill_worker=worker, local_prefill=False)
        prompts = _prompts(5, 20, 35)
        futures = [
            batcher.submit(PDRequest(str(i), p, 5)) for i, p in enumerate(prompts)
        ]
        batcher.run_until_idle()
        assert [f.result() for f in futures] == [
            _greedy(model.m, p, 5) for p in prompts
        ]
        assert batcher.stats_pd["remote"] == 5 and batcher.stats_pd["local"] == 0
        assert not model.local_calls
        assert sorted(batcher.free) == [0, 1]
    finally:
        worker.close()


def test_hybrid_import_uses_lane_tail_and_global_block_with_physical_dtype(fake_rebel):  # noqa: F811
    from types import SimpleNamespace

    import numpy as np
    from opt_prime.accel.platforms.atom.batching import AtomKVBlocks, bf16_to_dlfloat

    metas = [
        {"name": f"kv{i}", "shape": shape, "dtype": "float32", "layer_type": kind}
        for i, (shape, kind) in enumerate(
            [
                ([2, 1, 4, 2], "sliding_attention"),
                ([2, 1, 4, 2], "sliding_attention"),
                ([4, 1, 8, 2], "full_attention"),
                ([4, 1, 8, 2], "full_attention"),
            ]
        )
    ]
    writes = []
    rt = SimpleNamespace(
        get_kv_cache_size_for_layer=lambda name: (
            int(np.prod(metas[int(name[2:])]["shape"])) * 2
        ),
        set_kv_cache_for_layer=lambda name, block, start, n, buf: writes.append(
            (name, block, start, n, buf.copy())
        ),
    )
    rc = SimpleNamespace(
        cache_impl="hybrid",
        prefill_chunk_size=2,
        kvcache_num_blocks=4,
        kvcache_metas=metas,
    )
    model = SimpleNamespace(rbln_config=rc, model=[rt, rt])
    blocks = AtomKVBlocks(model, 2, prefix_cache=False)
    blocks.admit(1, tuple(range(6)))
    assert blocks.lane_block[1] == 0
    keys = torch.arange(24).reshape(2, 1, 6, 2).float()
    values = keys + 30
    blocks.import_span(1, 0, keys, values)
    assert len(writes) == 4
    for i, (name, block, start, n, buf) in enumerate(writes):
        assert buf.dtype == np.uint16
        assert (block, start, n) == ((1, 0, 4) if i < 2 else (0, 0, 6))
        source = (keys if i % 2 == 0 else values)[i // 2, :, -n:]
        expected = bf16_to_dlfloat(
            source.to(torch.bfloat16)
            .contiguous()
            .view(torch.int16)
            .numpy()
            .view(np.uint16)
        )
        np.testing.assert_array_equal(buf[block, :, :n], expected)
    with pytest.raises(ValueError, match="full"):
        blocks.import_span(1, 1, keys, values)
    assert len(writes) == 4


def test_partial_hybrid_result_fails_request_without_poisoning_lane(fake_rebel):  # noqa: F811
    import copy

    from opt_prime.accel.pd.pd_batching import PrefillWorker, TorchPrefillEngine

    model = HybridFake()
    fake_rebel["model"] = model
    engine = TorchPrefillEngine(
        copy.deepcopy(model.m), max_context=128, chunk=16, page_size=16
    )
    original = engine.prefill
    calls = 0

    def prefill(request, send_from=0):
        nonlocal calls
        out = original(request, send_from)
        calls += 1
        if calls == 1:
            out.start = 1
            out.keys, out.values = out.keys[:, :, 1:], out.values[:, :, 1:]
        return out

    engine.prefill = prefill
    worker = PrefillWorker(engine)
    try:
        batcher = AtomBatcher(model, prefill_worker=worker, local_prefill=False)
        prompt = _prompts(1, 20, 25)[0]
        bad = batcher.submit(PDRequest("partial", prompt, 3))
        batcher.run_until_idle()
        with pytest.raises(ValueError, match="full"):
            bad.result()
        good = batcher.submit(PDRequest("next", prompt, 3))
        batcher.run_until_idle()
        assert good.result() == _greedy(model.m, prompt, 3)
        assert not batcher.poisoned
    finally:
        worker.close()


def test_hybrid_cancel_pending_then_reuse_ignores_late_result(fake_rebel):  # noqa: F811
    import copy
    from concurrent.futures import Future

    from opt_prime.accel.pd.pd_batching import TorchPrefillEngine
    from opt_prime.accel.pd.runtime import RequestCancelled

    class Worker:
        def __init__(self):
            self.pending = []

        def submit(self, request, send_from=0, slot=None):
            future = Future()
            self.pending.append((request, future))
            return future

    model = HybridFake()
    fake_rebel["model"] = model
    worker = Worker()
    engine = TorchPrefillEngine(
        copy.deepcopy(model.m), max_context=128, chunk=16, page_size=16
    )
    batcher = AtomBatcher(model, prefill_worker=worker, local_prefill=False)
    p, q = _prompts(2, 20, 35)
    gone = batcher.submit(PDRequest("gone", p, 5))
    batcher.step()
    assert batcher.cancel("gone")
    batcher.step()
    with pytest.raises(RequestCancelled):
        gone.result()
    good = batcher.submit(PDRequest("reuse", q, 5))
    batcher.step()
    for request, future in worker.pending:
        if not future.done():
            future.set_result(engine.prefill(request))
    batcher.run_until_idle()
    assert good.result() == _greedy(model.m, q, 5)
    assert not batcher.by_id and sorted(batcher.free) == [0, 1]
