"""#44 stage 3: AtomBatcher over an optimum-rbln style model (prefill per lane, one B-lane decode graph).

The fake model keeps one HF cache per lane and checks the positions the batcher passes, including the SDK's
padded_cache_lengths offset. Hardware runs: docs/validation/batching/ATOM.md."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from transformers.cache_utils import DynamicCache

from opt_prime.accel.pd import PDRequest, SamplingParams
from opt_prime.accel.pd.runtime import RequestCancelled
from opt_prime.accel.platforms.atom.batching import AtomBatcher
from opt_prime.accel.pd.paged import block_hashes
from opt_prime.accel.serve.kv_events import ListKVEventPublisher

from test_pd_batching import _greedy, _llama, _prompts


class FakeRbln:
    def __init__(self, model, batch, max_seq_len=128, pad=0, fail_after=None):
        self.m, self.pad, self.fail_after, self.calls = model, pad, fail_after, 0
        self.config = model.config
        self.rbln_config = SimpleNamespace(batch_size=batch, max_seq_len=max_seq_len)
        self.cache, self.length = [None] * batch, [0] * batch
        self.decoders = {batch: self._decode}

    def _tick(self):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("SDK failure")

    def prefill_decoder(self, input_ids, cache_position, batch_idx):
        self._tick()
        assert cache_position.dtype == torch.int32 and cache_position[0, 0] == 0
        cache = DynamicCache()
        out = self.m(input_ids, past_key_values=cache, use_cache=True)
        self.cache[batch_idx], self.length[batch_idx] = out.past_key_values, input_ids.shape[1]
        return SimpleNamespace(logits=out.logits, padded_cache_lengths=torch.tensor(self.pad))

    def _decode(self, input_ids, cache_position):
        self._tick()
        B = len(self.cache)
        assert input_ids.shape == (B, 1) and cache_position.shape == (B, 1)
        logits = torch.zeros(B, 1, self.config.vocab_size)
        for lane in range(B):
            pos = int(cache_position[lane, 0])
            if pos == 0:                                       # idle lane: dummy token on its own page
                continue
            assert pos == self.length[lane] + self.pad
            out = self.m(input_ids[lane:lane + 1], past_key_values=self.cache[lane], use_cache=True,
                         position_ids=torch.tensor([[self.length[lane]]]))
            self.length[lane] += 1
            logits[lane] = out.logits[0]
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("pad", [0, 8])
def test_staggered_arrivals_refill_lanes_and_match(pad):
    model = _llama()
    b = AtomBatcher(FakeRbln(model, 3, pad=pad))
    prompts = _prompts(7)
    lengths = [5, 17, 1, 9, 24, 3, 12]
    futures = []
    for i, (p, n) in enumerate(zip(prompts, lengths)):
        futures.append(b.submit(PDRequest(f"r{i}", p, n)))
        for _ in range(i % 3):
            b.step()
    b.run_until_idle()
    for p, n, f in zip(prompts, lengths, futures):
        assert f.result() == _greedy(model, p, n)
    assert b.stats["decode_rows"] > b.stats["decode_steps"]          # lanes really decoded together
    assert b.stats["prefill_chunks"] == 7 and sorted(b.free) == [0, 1, 2]


def test_one_prefill_per_iteration():
    b = AtomBatcher(FakeRbln(_llama(), 4))
    for i, p in enumerate(_prompts(4, seed=1)):
        b.submit(PDRequest(f"r{i}", p, 6))
    b.step()                                                  # prefill r0, then r0 joins this step's decode
    assert len(b.decoding) == 1 and len(b.waiting) == 3 and b.stats["decode_rows"] == 1
    b.step()
    assert len(b.decoding) == 2 and len(b.waiting) == 2 and b.stats["decode_rows"] == 3


def test_stop_cancel_and_sampling():
    model = _llama()
    b = AtomBatcher(FakeRbln(model, 2))
    p, q = _prompts(2, seed=4)
    first = _greedy(model, p, 3)
    stop = b.submit(PDRequest("stop", p, 20, SamplingParams(stop_token_ids=(first[2],))))
    gone = b.submit(PDRequest("gone", q, 20))
    b.step(); b.step()
    assert b.cancel("gone")
    sampled = b.submit(PDRequest("s", q, 8, SamplingParams(temperature=0.8, seed=3)))
    b.run_until_idle()
    assert stop.result() == first
    with pytest.raises(RequestCancelled):
        gone.result()
    assert len(sampled.result()) == 8


def test_sdk_failure_quarantines():
    b = AtomBatcher(FakeRbln(_llama(), 2, fail_after=4))                # 2 warm-up prefills, then 2 calls
    b.submit(PDRequest("a", _prompts(1)[0], 10))
    with pytest.raises(RuntimeError, match="SDK failure"):
        b.run_until_idle()
    with pytest.raises(RuntimeError, match="quarantined"):
        b.step()


def test_rejections():
    model = _llama()
    fake = FakeRbln(model, 2, max_seq_len=32)
    fake.decoders = {1: fake._decode}
    with pytest.raises(ValueError, match="batch 2"):
        AtomBatcher(fake)
    b = AtomBatcher(FakeRbln(model, 2, max_seq_len=32))
    with pytest.raises(ValueError):
        b.submit(PDRequest("long", tuple(range(1, 30)), 8))


def test_served_by_batched_service():
    from opt_prime.accel.serve.batched import BatchedService
    model = _llama()
    b = AtomBatcher(FakeRbln(model, 2))
    svc = BatchedService(b, model="m", vocab_size=97, max_output=8, render=str)
    b.submit(PDRequest("a", _prompts(1)[0], 4)); b.step()
    assert svc.gauges() == (1, 0, 0.5)
    b.run_until_idle()
    assert svc.gauges() == (0, 0, 0.0)


# ---- #123: own block tables, device prefix cache, host tier (fake SDK with per-block KV) ----
class FakeBlocks:
    """optimum-rbln style model with kvcache_num_blocks blocks of max_seq_len tokens, external block tables,
    copy/get/set KV APIs (the device KV is a [blocks, heads, ctx, head_dim] tensor per K/V per layer)"""

    def __init__(self, model, batch, blocks, max_seq_len=128, chunk=16):
        self.m, self.config = model, model.config
        c = model.config
        layers, heads, hd = c.num_hidden_layers, c.num_key_value_heads, c.head_dim
        shape = [blocks, heads, max_seq_len, hd]
        self.kv = [torch.zeros(shape) for _ in range(2 * layers)]
        self.rbln_config = SimpleNamespace(batch_size=batch, max_seq_len=max_seq_len, kvcache_num_blocks=blocks,
                                           prefill_chunk_size=chunk,
                                           kvcache_metas=[dict(name=f"past_key_values_{i}", shape=shape, dtype="float32")
                                                          for i in range(2 * layers)])
        rt = SimpleNamespace(get_kv_cache_size_for_layer=lambda name: self.kv[0].numel() * 4,
                             get_kv_cache_for_layer=self._get, set_kv_cache_for_layer=self._set)
        self.model = [rt, rt]
        self.decoders = {batch: self._decode}
        self.calls = []

    def _cache(self, block, n):
        from transformers.cache_utils import DynamicCache
        return DynamicCache(ddp_cache_data=[(self.kv[2 * l][block, :, :n][None], self.kv[2 * l + 1][block, :, :n][None])
                                            for l in range(len(self.kv) // 2)])

    def _store(self, block, cache, start, n):
        for l, entry in enumerate(cache.layers):
            self.kv[2 * l][block, :, start:start + n] = entry.keys[0, :, start:start + n]
            self.kv[2 * l + 1][block, :, start:start + n] = entry.values[0, :, start:start + n]

    def prefill_decoder(self, input_ids, cache_position, batch_idx, block_tables):
        assert block_tables.shape == (1,)
        block, start, n = int(block_tables[0]), int(cache_position[0, 0]), input_ids.shape[1]
        assert start % self.rbln_config.prefill_chunk_size == 0
        self.calls.append(("prefill", block, start, n))
        with torch.inference_mode():
            out = self.m(input_ids, past_key_values=self._cache(block, start), use_cache=True,
                         position_ids=torch.arange(start, start + n)[None])
        self._store(block, out.past_key_values, start, n)
        return SimpleNamespace(logits=out.logits, padded_cache_lengths=0)

    def _decode(self, input_ids, cache_position, block_tables):
        B = input_ids.shape[0]
        assert block_tables.shape == (B, 1)
        logits = torch.zeros(B, 1, self.config.vocab_size)
        for lane in range(B):
            block, pos = int(block_tables[lane, 0]), int(cache_position[lane, 0])
            if self.rbln_config.kvcache_num_blocks > 1 and block == self.rbln_config.kvcache_num_blocks - 1:
                continue                                       # scratch block: idle lane
            with torch.inference_mode():
                out = self.m(input_ids[lane:lane + 1], past_key_values=self._cache(block, pos), use_cache=True,
                             position_ids=torch.tensor([[pos]]))
            self._store(block, out.past_key_values, pos, 1)
            logits[lane] = out.logits[0]
        return SimpleNamespace(logits=logits)

    def _idx(self, name):
        return int(name.rsplit("_", 1)[1])

    def _get(self, name, block, offset, n, buf):
        buf.reshape(self.kv[0].shape)[block, :, offset:offset + n] = self.kv[self._idx(name)][block, :, offset:offset + n].numpy()

    def _set(self, name, block, offset, n, buf):
        self.kv[self._idx(name)][block, :, offset:offset + n] = torch.from_numpy(buf.reshape(self.kv[0].shape)[block, :, offset:offset + n].copy())

    def copy(self, rt, src, dst, n):
        for t in self.kv:
            t[dst, :, :n] = t[src, :, :n]


@pytest.fixture
def fake_rebel(monkeypatch):
    import sys, types
    import numpy as np
    holder = {}
    kv = types.ModuleType("rebel.kv_cache")
    kv.copy_kv_cache = lambda rt, src, dst, n: holder["model"].copy(rt, src, dst, n)
    kv.aligned_tensor = lambda size, dtype=np.float16, alignment=0, tensor_type="np": np.zeros(size, dtype=dtype)
    monkeypatch.setitem(sys.modules, "rebel", types.ModuleType("rebel"))
    monkeypatch.setitem(sys.modules, "rebel.kv_cache", kv)
    return holder


def _blocks_model(holder, model, batch=2, blocks=5, **kw):
    fake = FakeBlocks(model, batch, blocks, **kw)
    holder["model"] = fake
    return fake


def test_own_blocks_match_and_refill(fake_rebel):
    model = _llama()
    b = AtomBatcher(_blocks_model(fake_rebel, model, batch=3, blocks=4))
    prompts, lengths = _prompts(6), [5, 9, 3, 12, 7, 4]
    fs = []
    for i, (p, n) in enumerate(zip(prompts, lengths)):
        fs.append(b.submit(PDRequest(f"r{i}", p, n)))
        b.step()
    b.run_until_idle()
    for p, n, f in zip(prompts, lengths, fs):
        assert f.result() == _greedy(model, p, n)
    assert sorted(b.free) == [0, 1, 2] and b.blocks.lane_block == [None] * 3


def test_device_prefix_copy_and_kv_events(fake_rebel):
    model = _llama()
    events = ListKVEventPublisher(block_size=16)
    fake = _blocks_model(fake_rebel, model, batch=2, blocks=5)
    b = AtomBatcher(fake, prefix_cache=True, kv_events=events)
    system = _prompts(1, 50, 51, seed=3)[0]
    a, c = system + (5, 6), system + (9,)
    for rid, p in (("a", a), ("c", c)):
        f = b.submit(PDRequest(rid, p, 5)); b.run_until_idle()
        assert f.result() == _greedy(model, p, 5)
    assert fake.calls[-1][2] == 48                           # c prefilled from 48: three 16-token units copied
    assert b.blocks.stats["device_copies"] == 1 and (b.hit_tokens, b.query_tokens) == (48, 52 + 51)
    stored = [e for batch in events.batches for e in batch[1] if e["type"] == "BlockStored"]
    assert stored[0]["block_hashes"] == block_hashes(a, 16)


def test_host_tier_reload_after_eviction(fake_rebel):
    model = _llama()
    events = ListKVEventPublisher(block_size=16)
    fake = _blocks_model(fake_rebel, model, batch=1, blocks=3)          # 1 lane + 1 cache block + scratch
    b = AtomBatcher(fake, prefix_cache=True, host_kv_gib=1.0, kv_events=events)
    p, q = _prompts(2, 40, 41, seed=7)
    for rid, x in (("p", p), ("q", q), ("q2", q), ("p2", p)):     # q evicts p's entry to host; p2 reloads it
        f = b.submit(PDRequest(rid, x, 6)); b.run_until_idle()
        assert f.result() == _greedy(model, x, 6)
    st = b.blocks.stats
    assert st["host_offloads"] >= 1 and st["host_reloads"] == 1 and b.blocks.host_hit_tokens == 32
    flat = [e for batch in events.batches for e in batch[1]]
    assert any(e["type"] == "BlockStored" and e["medium"] == "CPU" for e in flat)


def test_prefix_cache_needs_extra_blocks(fake_rebel):
    with pytest.raises(ValueError, match="kv_blocks"):
        AtomBatcher(FakeRbln(_llama(), 2), prefix_cache=True)


def test_prefix_cache_needs_an_evictable_block(fake_rebel):
    """#172: with kv_blocks == lanes + 1 the only cache entry is also the copy source, so a prefix hit with every
    other lane busy found no block and poisoned the batcher; the configuration is rejected at construction"""
    model = _llama()
    with pytest.raises(ValueError, match="kv_blocks"):
        AtomBatcher(_blocks_model(fake_rebel, model, batch=1, blocks=2), prefix_cache=True)
    b = AtomBatcher(_blocks_model(fake_rebel, model, batch=1, blocks=2))      # no prefix cache: one block is enough
    p = _prompts(1, 40, 41, seed=7)[0]
    for rid in ("p", "p2"):
        f = b.submit(PDRequest(rid, p, 4)); b.run_until_idle()
        assert f.result() == _greedy(model, p, 4)


def test_pd_mode_imports_prefilled_kv(fake_rebel):
    """ATOM decodes only: prompts are prefilled by a torch worker and their KV imported into the lanes' blocks"""
    import copy
    from opt_prime.accel.pd.pd_batching import PrefillWorker, TorchPrefillEngine
    model = _llama()
    fake = _blocks_model(fake_rebel, model, batch=2, blocks=5)
    worker = PrefillWorker(TorchPrefillEngine(copy.deepcopy(model), max_context=128, chunk=16, page_size=16))
    b = AtomBatcher(fake, prefix_cache=True, prefill_worker=worker, local_prefill=False)
    system = _prompts(1, 40, 41, seed=13)[0]
    prompts = [system + (i, i + 3) for i in range(1, 5)]
    fs = [b.submit(PDRequest(f"p{i}", p, 7)) for i, p in enumerate(prompts)]
    b.run_until_idle()
    for p, f in zip(prompts, fs):
        assert f.result() == _greedy(model, p, 7)
    assert not [c for c in fake.calls if c[0] == "prefill"]          # no prefill ran on the decode device
    assert b.blocks.stats["device_copies"] >= 1                       # later requests reused the cached prefix
    worker.close()


def test_pd_mode_routes_prompts_between_remote_and_local(fake_rebel):
    """with local prefill allowed, a burst is split between the prefill worker and ATOM (in-flight limit, then the
    measured cost of each side); tokens stay exact"""
    import copy
    from opt_prime.accel.pd.pd_batching import PrefillWorker, TorchPrefillEngine
    model = _llama()
    fake = _blocks_model(fake_rebel, model, batch=4, blocks=6)
    worker = PrefillWorker(TorchPrefillEngine(copy.deepcopy(model), max_context=128, chunk=16, page_size=16))
    b = AtomBatcher(fake, prefill_worker=worker, remote_inflight=1)
    prompts = _prompts(6, seed=17)
    fs = [b.submit(PDRequest(f"b{i}", p, 6)) for i, p in enumerate(prompts)]
    b.run_until_idle()
    for p, f in zip(prompts, fs):
        assert f.result() == _greedy(model, p, 6)
    assert b.stats_pd["remote"] >= 1 and b.stats_pd["local"] >= 1
    assert [c for c in fake.calls if c[0] == "prefill"]       # some prompts did run on the decode device
    worker.close()


def test_pd_remote_only_with_host_tier_events_and_metrics(fake_rebel):
    """PD on ATOM with every serving feature on: all prompts prefilled remotely, decode-side prefix reuse through
    the host tier, KV events, and /metrics showing the routing and the host traffic"""
    import copy
    from opt_prime.accel.pd.pd_batching import PrefillWorker, TorchPrefillEngine
    from opt_prime.accel.serve.batched import BatchedService
    model = _llama()
    events = ListKVEventPublisher(block_size=16)
    fake = _blocks_model(fake_rebel, model, batch=1, blocks=3)          # 1 lane + 1 cache block + scratch
    worker = PrefillWorker(TorchPrefillEngine(copy.deepcopy(model), max_context=128, chunk=16, page_size=16))
    b = AtomBatcher(fake, prefix_cache=True, host_kv_gib=1.0, kv_events=events, prefill_worker=worker,
                    local_prefill=False)
    svc = BatchedService(b, model="m", vocab_size=97, max_output=8, render=str)
    p, q = _prompts(2, 40, 41, seed=7)
    for rid, x in (("p", p), ("q", q), ("q2", q), ("p2", p)):     # q evicts p's entry to host; p2 reloads it
        f = b.submit(PDRequest(rid, x, 6)); b.run_until_idle()
        assert f.result() == _greedy(model, x, 6)
    assert not [c for c in fake.calls if c[0] == "prefill"]
    st = b.blocks.stats
    assert st["host_reloads"] == 1 and b.blocks.host_hit_tokens == 32
    text = svc.metrics.render(*svc.gauges())
    assert 'airm_pd_prefills_total{model_name="m",side="remote"} 4' in text
    assert 'airm_pd_prefills_total{model_name="m",side="local"} 0' in text
    assert 'airm_kv_reload_blocks_total{model_name="m"} 1' in text
    assert 'airm_kv_host_hit_tokens_total{model_name="m"} 32' in text
    flat = [e for batch in events.batches for e in batch[1]]
    assert any(e["type"] == "BlockStored" and e["medium"] == "CPU" for e in flat)
    worker.close()


@pytest.mark.parametrize("partial", [False, True])
def test_single_block_pd_reuses_full_prefix_without_cache(fake_rebel, partial):
    import copy
    from opt_prime.accel.pd.pd_batching import PrefillWorker, TorchPrefillEngine
    model = _llama()
    fake = _blocks_model(fake_rebel, model, batch=1, blocks=1)
    engine = TorchPrefillEngine(copy.deepcopy(model), max_context=128, chunk=16, page_size=16)
    original = engine.prefill
    calls = 0
    def prefill(request, send_from=0):
        nonlocal calls
        out = original(request, send_from)
        calls += 1
        if partial and calls == 1:
            out.start = 1
            out.keys, out.values = out.keys[:, :, 1:], out.values[:, :, 1:]
        return out
    engine.prefill = prefill
    worker = PrefillWorker(engine)
    try:
        b = AtomBatcher(fake, prefix_cache=False, prefill_worker=worker, local_prefill=False)
        prompts = _prompts(4, 20, 70, seed=236)
        fs = [b.submit(PDRequest(str(i), p, 5)) for i,p in enumerate(prompts)]
        b.run_until_idle()
        for i,(p,f) in enumerate(zip(prompts,fs)):
            if partial and i == 0:
                with pytest.raises(ValueError, match="full prompt"):
                    f.result()
            else:
                assert f.result() == _greedy(model,p,5)
        assert b.stats_pd == {"remote":4,"local":0}
        assert b.blocks.free == [0] and b.blocks.lane_block == [None]
        assert not fake.calls and not b.by_id and not b.poisoned
    finally:
        worker.close()


@pytest.mark.parametrize("batch,blocks,prefix,host", [(1,1,True,0),(1,1,False,1),(2,2,False,0)])
def test_single_block_pd_keeps_cache_and_multilane_guards(fake_rebel,batch,blocks,prefix,host):
    model = _llama()
    fake = _blocks_model(fake_rebel,model,batch=batch,blocks=blocks)
    with pytest.raises(ValueError):
        AtomBatcher(fake,prefix_cache=prefix,host_kv_gib=host,prefill_worker=object(),local_prefill=False)


def test_single_block_pd_cancel_late_result_and_reuse(fake_rebel):
    import copy
    from concurrent.futures import Future
    from opt_prime.accel.pd.pd_batching import TorchPrefillEngine

    class Worker:
        def __init__(self):
            self.pending = []

        def submit(self, request, send_from=0, slot=None):
            assert send_from == slot == 0
            f = Future()
            self.pending.append((request, f))
            return f

    model = _llama()
    fake = _blocks_model(fake_rebel, model, batch=1, blocks=1)
    worker = Worker()
    engine = TorchPrefillEngine(copy.deepcopy(model), max_context=128, chunk=16, page_size=16)
    b = AtomBatcher(fake, prefix_cache=False, prefill_worker=worker, local_prefill=False)
    p,q = _prompts(2,20,35)
    gone = b.submit(PDRequest('gone',p,5))
    b.step()
    assert b.cancel('gone')
    b.step()
    with pytest.raises(RequestCancelled):
        gone.result()
    good = b.submit(PDRequest('next',q,5))
    b.step()
    for request,f in worker.pending:
        if not f.done():
            f.set_result(engine.prefill(request))
    b.run_until_idle()
    assert good.result() == _greedy(model,q,5)
    assert b.blocks.free == [0] and b.blocks.lane_block == [None]
    assert not b.by_id and not b.poisoned
