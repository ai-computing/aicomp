"""#46: prefix caching on the paged batcher and KV events in vLLM's wire format."""
import pytest

torch = pytest.importorskip("torch")

from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.paged import block_hashes
from opt_prime.accel.serve.kv_events import ListKVEventPublisher, encode

from test_pd_batching import _greedy, _llama, _prompts


def _run(b, rid, prompt, n):
    f = b.submit(PDRequest(rid, prompt, n))
    b.run_until_idle()
    return f.result()


def test_shared_prefix_skips_prefill_and_matches():
    model = _llama()
    events = ListKVEventPublisher(block_size=16)
    b = ContinuousBatcher(model, kv="paged", slots=2, max_context=128, prefill_chunk=16, page_size=16,
                          prefix_cache=True, kv_events=events)
    system = _prompts(1, 50, 51, seed=3)[0]
    a, c = system + (5, 6, 7), system + (9, 10)
    assert _run(b, "a", a, 6) == _greedy(model, a, 6)
    chunks = b.stats["prefill_chunks"]
    assert _run(b, "c", c, 6) == _greedy(model, c, 6)
    assert b.stats["prefill_chunks"] - chunks == 1            # 48 cached tokens, only the tail is prefilled
    assert (b.kv.hit_tokens, b.kv.query_tokens) == (48, len(a) + len(c))
    stored = [e for batch in events.batches for e in batch[1] if e["type"] == "BlockStored"]
    assert stored[0]["block_hashes"] == block_hashes(a, 16) and stored[0]["parent_block_hash"] is None
    assert stored[0]["token_ids"] == list(a[:48]) and len(stored) == 1        # c's blocks were already cached


def test_full_prompt_hit_still_computes_the_last_token():
    model = _llama()
    b = ContinuousBatcher(model, kv="paged", slots=1, max_context=64, prefill_chunk=64, page_size=16, prefix_cache=True)
    p = _prompts(1, 32, 33, seed=5)[0]
    assert _run(b, "x", p, 4) == _run(b, "y", p, 4) == _greedy(model, p, 4)
    assert b.kv.hit_tokens == 16                              # one block kept back: the last prompt token is computed


def test_eviction_reuses_cached_pages_and_reports_removal():
    model = _llama()
    events = ListKVEventPublisher(block_size=16)
    b = ContinuousBatcher(model, kv="paged", slots=1, max_context=64, prefill_chunk=64, page_size=16, pages=4,
                          prefix_cache=True, kv_events=events)
    p, q = _prompts(2, 33, 34, seed=7)
    assert _run(b, "p", p, 8) == _greedy(model, p, 8)        # 3 pages, 2 full prompt blocks cached
    assert len(b.kv.evictable) == 2 and len(b.kv.free_pages) == 2
    assert _run(b, "q", q, 8) == _greedy(model, q, 8)        # needs 3 pages: evicts p's oldest block
    removed = [e for batch in events.batches for e in batch[1] if e["type"] == "BlockRemoved"]
    assert removed == [dict(type="BlockRemoved", block_hashes=[block_hashes(p, 16)[0]], medium="GPU")]
    assert sum(b.kv.refs) == 0 and len(b.kv.free_pages) + len(b.kv.evictable) == 4


def test_shared_pages_are_reference_counted_while_both_run():
    model = _llama()
    b = ContinuousBatcher(model, kv="paged", slots=3, max_context=128, prefill_chunk=16, page_size=16, prefix_cache=True)
    system = _prompts(1, 40, 41, seed=9)[0]
    _run(b, "warm", system, 1)
    prompts = [system + (i, i + 1) for i in range(1, 4)]
    futures = [b.submit(PDRequest(f"r{i}", p, 10)) for i, p in enumerate(prompts)]
    b.step()
    shared = b.kv.cached[block_hashes(system, 16)[0]]
    assert b.kv.refs[shared] == 3
    b.run_until_idle()
    for p, f in zip(prompts, futures):
        assert f.result() == _greedy(model, p, 10)
    assert sum(b.kv.refs) == 0


def test_wire_format_matches_vllm():
    msgpack = pytest.importorskip("msgpack")
    ts, events = encode([("stored", [1, 2], None, [5] * 32), ("removed", [1])], block_size=16)
    assert events == [dict(type="BlockStored", block_hashes=[1, 2], parent_block_hash=None, token_ids=[5] * 32,
                           block_size=16, lora_id=None, medium="GPU", lora_name=None),
                      dict(type="BlockRemoved", block_hashes=[1], medium="GPU")]
    assert msgpack.unpackb(msgpack.packb([ts, events], use_bin_type=True)) == [ts, events]


def test_zmq_publisher_multipart():
    zmq = pytest.importorskip("zmq")
    msgpack = pytest.importorskip("msgpack")
    from opt_prime.accel.serve.kv_events import ZmqKVEventPublisher
    import time
    pub = ZmqKVEventPublisher("tcp://127.0.0.1:*", topic="kv@192.0.2.5:8000@m", block_size=16)   # binds, as vLLM
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.connect(pub.socket.getsockopt(zmq.LAST_ENDPOINT).decode())
    sub.setsockopt(zmq.SUBSCRIBE, b"kv@")
    try:
        time.sleep(0.2)                                       # PUB/SUB join
        pub.publish([("removed", [7])])
        assert sub.poll(2000)
        topic, seq, payload = sub.recv_multipart()
        assert topic == b"kv@192.0.2.5:8000@m" and int.from_bytes(seq, "big") == 0
        assert msgpack.unpackb(payload)[1] == [dict(type="BlockRemoved", block_hashes=[7], medium="GPU")]
    finally:
        pub.close(); sub.close(linger=0)


def test_metrics_expose_cache_config_and_hits():
    from opt_prime.accel.serve.batched import BatchedService
    model = _llama()
    b = ContinuousBatcher(model, kv="paged", slots=1, max_context=64, prefill_chunk=64, page_size=16, prefix_cache=True)
    p = _prompts(1, 40, 41, seed=11)[0]
    _run(b, "a", p, 2); _run(b, "b", p, 2)
    svc = BatchedService(b, model="m", vocab_size=97, max_output=8, render=str)
    text = svc.metrics.render(*svc.gauges())
    assert 'vllm:cache_config_info{model_name="m",block_size="16",num_gpu_blocks="4",enable_prefix_caching="True"} 1' in text
    assert 'vllm:prefix_cache_hits_total{model_name="m"} 32' in text
    assert 'vllm:prefix_cache_queries_total{model_name="m"} 80' in text
    assert 'vllm:kv_cache_usage_perc{model_name="m"} 0.0' in text      # cached, unreferenced pages are reclaimable


# ---- #52: host-DRAM tier ----
def _tiered(model, pages, host_kv_gib, **kw):
    return ContinuousBatcher(model, kv="paged", slots=1, max_context=64, prefill_chunk=64, page_size=16, pages=pages,
                             prefix_cache=True, host_kv_gib=host_kv_gib, **kw)


BLOCK_GIB = 2 * 2 * 2 * 16 * 8 * 4 / 2 ** 30          # one block of the test llama (layers 2, kv heads 2, fp32)


def test_evicted_blocks_reload_from_host_and_match():
    model = _llama()
    events = ListKVEventPublisher(block_size=16)
    b = _tiered(model, 4, 8 * BLOCK_GIB, kv_events=events)
    p, q = _prompts(2, 33, 34, seed=7)
    _run(b, "p", p, 8)
    _run(b, "q", q, 8)                                     # evicts p's blocks from the device into the host tier
    host = b.kv.host
    assert host.blocks == 8 and host.stats["offloaded"] >= 1
    hits = b.kv.hit_tokens
    assert _run(b, "p2", p, 8) == _greedy(model, p, 8)     # p's prefix comes back from the host tier
    assert b.kv.hit_tokens - hits == 32 and host.stats["reloaded"] >= 1 and host.stats["hit_tokens"] >= 16
    flat = [e for batch in events.batches for e in batch[1]]
    cpu = [e for e in flat if e["medium"] == "CPU"]
    assert cpu and all(e["type"] == "BlockStored" and e["token_ids"] == [] for e in cpu)
    first = block_hashes(p, 16)[0]
    order = [(e["type"], e["medium"]) for e in flat if first in e["block_hashes"]]
    assert order[:3] == [("BlockStored", "GPU"), ("BlockStored", "CPU"), ("BlockRemoved", "GPU")]
    assert ("BlockStored", "GPU") in order[3:]             # reloaded: on the device again


def test_host_only_vs_device_only_hit_rate():
    """three prompts cycling through a pool that holds one: without the host tier nothing is ever reused"""
    model = _llama()
    prompts = _prompts(3, 33, 34, seed=21)
    rates = {}
    for gib in (0, 16 * BLOCK_GIB):
        b = _tiered(model, 4, gib)
        for rnd in range(3):
            for i, p in enumerate(prompts):
                assert _run(b, f"{gib}-{rnd}-{i}", p, 8) == _greedy(model, p, 8)
        rates[gib] = b.kv.hit_tokens / b.kv.query_tokens
    assert rates[0] == 0 and rates[16 * BLOCK_GIB] > 0.5


def test_host_lru_eviction_and_reload_rule():
    model = _llama()
    events = ListKVEventPublisher(block_size=16)
    b = _tiered(model, 4, 2 * BLOCK_GIB, kv_events=events)           # two host blocks
    for i, p in enumerate(_prompts(4, 33, 34, seed=31)):
        _run(b, f"r{i}", p, 8)
    host = b.kv.host
    assert len(host.index) == 2 and host.stats["evicted"] >= 1
    assert any(e["type"] == "BlockRemoved" and e["medium"] == "CPU" for batch in events.batches for e in batch[1])
    slow = _tiered(model, 4, 2 * BLOCK_GIB, reload_bandwidth=1.0, prefill_rate=1e6)   # recompute beats 1 B/s
    fast = _tiered(model, 4, 2 * BLOCK_GIB, reload_bandwidth=1e12, prefill_rate=1.0)
    assert not slow.kv.host.reload and fast.kv.host.reload


def test_host_tier_needs_prefix_cache():
    with pytest.raises(ValueError):
        ContinuousBatcher(_llama(), kv="paged", page_size=16, max_context=64, host_kv_gib=1)


def test_host_tier_metrics():
    from opt_prime.accel.serve.batched import BatchedService
    model = _llama()
    b = _tiered(model, 4, 8 * BLOCK_GIB)
    p, q = _prompts(2, 33, 34, seed=7)
    for rid, x in (("p", p), ("q", q), ("p2", p)):
        _run(b, rid, x, 8)
    svc = BatchedService(b, model="m", vocab_size=97, max_output=8, render=str)
    text = svc.metrics.render(*svc.gauges())
    assert 'airm_kv_host_capacity_blocks{model_name="m"} 8' in text
    assert f'airm_kv_reload_bytes_total{{model_name="m"}} {b.kv.host.stats["reloaded"] * b.kv.host.block_bytes}' in text
    assert b.kv.host.block_bytes == 2 * 2 * 2 * 16 * 8 * 4
