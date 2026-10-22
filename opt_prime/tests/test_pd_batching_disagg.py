"""#120: batched PD - prefill by a worker (thread / process), decode lanes batched; tokens equal single runs."""
import copy
import functools

import pytest

torch = pytest.importorskip("torch")

from opt_prime.accel.pd import PDRequest, SamplingParams
from opt_prime.accel.pd.pd_batching import (PDBatcher, PrefillWorker, ProcessPrefillWorker, SubprocessPrefillWorker,
                                           TorchPrefillEngine)
from opt_prime.accel.pd.runtime import RequestCancelled
from opt_prime.accel.serve.kv_events import ListKVEventPublisher

from test_pd_batching import _greedy, _llama, _prompts


def _pd(model, *, slots=3, prefix_cache=False, prefill_prefix=True, **kw):
    engine = TorchPrefillEngine(copy.deepcopy(model), max_context=128, chunk=16, page_size=16,
                                prefix_cache=prefill_prefix)
    return PDBatcher(model, PrefillWorker(engine), slots=slots, max_context=128, page_size=16,
                     prefix_cache=prefix_cache, **kw)


def test_staggered_arrivals_match_single_runs():
    model = _llama()
    b = _pd(model)
    prompts, lengths = _prompts(7), [5, 17, 1, 9, 24, 3, 12]
    futures = []
    for i, (p, n) in enumerate(zip(prompts, lengths)):
        futures.append(b.submit(PDRequest(f"r{i}", p, n)))
        for _ in range(i % 3):
            b.step()
    b.run_until_idle()
    for p, n, f in zip(prompts, lengths, futures):
        assert f.result() == _greedy(model, p, n)
    assert b.stats["prefills"] == 7 and b.stats["decode_rows"] > b.stats["decode_steps"]
    assert sorted(b.free) == [0, 1, 2] and b.stats["prefill_chunks"] == 0      # no prefill in the decode loop


def test_prefix_reuse_on_both_sides():
    model = _llama()
    events = ListKVEventPublisher(block_size=16)
    b = _pd(model, slots=1, prefix_cache=True, kv_events=events)
    system = _prompts(1, 50, 51, seed=3)[0]
    a, c = system + (5, 6, 7), system + (9, 10)
    f = b.submit(PDRequest("a", a, 6)); b.run_until_idle()
    assert f.result() == _greedy(model, a, 6)
    f = b.submit(PDRequest("c", c, 6)); b.run_until_idle()
    assert f.result() == _greedy(model, c, 6)
    s = b.stats
    assert s["decode_cached_tokens"] == 48                    # c's first 3 blocks were not transferred
    assert s["transferred_tokens"] == len(a) + len(c) - 48
    assert s["prefill_cached_tokens"] == 48                   # and not recomputed on the prefill side
    assert any(e["type"] == "BlockStored" for batch in events.batches for e in batch[1])


def test_cancel_while_prefilling_and_stop_tokens():
    model = _llama()
    b = _pd(model, slots=2)
    p, q = _prompts(2, seed=4)
    first = _greedy(model, p, 3)
    stop = b.submit(PDRequest("stop", p, 20, SamplingParams(stop_token_ids=(first[2],))))
    gone = b.submit(PDRequest("gone", q, 20))
    b.step()                                                  # both admitted, prefills submitted
    assert b.cancel("gone")
    b.run_until_idle()
    assert stop.result() == first
    with pytest.raises(RequestCancelled):
        gone.result()
    assert sorted(b.free) == [0, 1]


def test_prefill_failure_fails_only_that_request():
    model = _llama()

    class Flaky(TorchPrefillEngine):
        def prefill(self, request, send_from=0):
            if request.request_id == "bad":
                raise RuntimeError("prefill device lost")
            return super().prefill(request, send_from)

    b = PDBatcher(model, PrefillWorker(Flaky(copy.deepcopy(model), max_context=128, chunk=16, page_size=16)),
                  slots=2, max_context=128, page_size=16)
    p, q = _prompts(2, seed=6)
    bad, ok = b.submit(PDRequest("bad", p, 4)), b.submit(PDRequest("ok", q, 4))
    b.run_until_idle()
    with pytest.raises(RuntimeError, match="device lost"):
        bad.result()
    assert ok.result() == _greedy(model, q, 4)


def _engine_factory(state):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(17)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=97, max_position_embeddings=256, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=2, head_dim=8, num_attention_heads=4, num_key_value_heads=2))
    m.load_state_dict(state)
    return TorchPrefillEngine(m.eval(), max_context=128, chunk=16, page_size=16)


def test_process_worker():
    model = _llama()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    worker = ProcessPrefillWorker(functools.partial(_engine_factory, state), out_device="cpu")
    try:
        b = PDBatcher(model, worker, slots=2, max_context=128, page_size=16)
        prompts = _prompts(3, seed=8)
        fs = [b.submit(PDRequest(f"p{i}", p, 6)) for i, p in enumerate(prompts)]
        b.run_until_idle()
        for p, f in zip(prompts, fs):
            assert f.result() == _greedy(model, p, 6)
    finally:
        worker.close()


def test_served_by_batched_service():
    from opt_prime.accel.serve.batched import BatchedService
    b = _pd(_llama(), slots=2)
    svc = BatchedService(b, model="m", vocab_size=97, max_output=8, render=str)
    b.submit(PDRequest("a", _prompts(1)[0], 4)); b.step()
    running, waiting, used = svc.gauges()
    assert running == 1 and waiting == 0 and used > 0
    b.run_until_idle()
    assert svc.gauges()[0] == 0


def test_subprocess_worker_via_prefill_server(tmp_path):
    """the prefill server under this interpreter (stands for another venv), KV through the shared-memory slab"""
    import sys
    model = _llama()
    model.save_pretrained(tmp_path)
    c = model.config
    worker = SubprocessPrefillWorker(sys.executable, ["--engine", "torch", "--model", str(tmp_path), "--dtype",
                                                      "float32", "--chunk", "16"],
                                     slots=2, layers=c.num_hidden_layers, kv_heads=c.num_key_value_heads,
                                     head_dim=c.head_dim, max_context=128, startup_timeout=300)
    try:
        b = PDBatcher(model, worker, slots=2, max_context=128, page_size=16)
        prompts = _prompts(4, seed=21)
        fs = [b.submit(PDRequest(f"s{i}", p, 6)) for i, p in enumerate(prompts)]
        b.run_until_idle()
        for p, f in zip(prompts, fs):
            # KV crosses the process boundary in bfloat16: compare with a bf16-rounded prefix only loosely
            assert f.result()[0] == _greedy(model, p, 1)[0]
        assert b.stats["prefills"] == 4
    finally:
        worker.close()


def test_pd_decode_side_host_tier():
    """the decode pool's host-DRAM tier works under PD: evicted prefix blocks come back without being re-sent"""
    model = _llama()
    engine = TorchPrefillEngine(copy.deepcopy(model), max_context=64, chunk=16, page_size=16, prefix_cache=False)
    block_gib = 2 * 2 * 2 * 16 * 8 * 4 / 2 ** 30
    b = PDBatcher(model, PrefillWorker(engine), slots=1, max_context=64, page_size=16, pages=4, prefix_cache=True,
                  host_kv_gib=8 * block_gib)
    p, q = _prompts(2, 33, 34, seed=7)
    for rid, x in (("p", p), ("q", q), ("p2", p)):
        f = b.submit(PDRequest(rid, x, 8)); b.run_until_idle()
        assert f.result() == _greedy(model, x, 8)
    assert b.kv.host.stats["reloaded"] >= 1 and b.stats["decode_cached_tokens"] == 32


def test_pd_serves_prefix_cache_host_tier_and_events_with_metrics():
    """one PD batcher with every serving feature on: decode-side prefix cache backed by the host tier, KV events,
    and /metrics showing the prefills, handed-over tokens and host reloads"""
    from opt_prime.accel.serve.batched import BatchedService
    model = _llama()
    engine = TorchPrefillEngine(copy.deepcopy(model), max_context=64, chunk=16, page_size=16, prefix_cache=False)
    block_gib = 2 * 2 * 2 * 16 * 8 * 4 / 2 ** 30
    events = ListKVEventPublisher(block_size=16)
    b = PDBatcher(model, PrefillWorker(engine), slots=1, max_context=64, page_size=16, pages=4, prefix_cache=True,
                  host_kv_gib=8 * block_gib, kv_events=events)
    svc = BatchedService(b, model="m", vocab_size=97, max_output=8, render=str)
    p, q = _prompts(2, 33, 34, seed=7)
    for rid, x in (("p", p), ("q", q), ("p2", p)):
        f = b.submit(PDRequest(rid, x, 8)); b.run_until_idle()
        assert f.result() == _greedy(model, x, 8)
    text = svc.metrics.render(*svc.gauges())
    assert 'airm_pd_prefills_total{model_name="m",side="remote"} 3' in text
    assert f'airm_pd_transferred_tokens_total{{model_name="m"}} {b.stats["transferred_tokens"]}' in text
    assert 'vllm:prefix_cache_hits_total{model_name="m"} 32' in text
    assert f'airm_kv_host_hit_tokens_total{{model_name="m"}} {b.kv.host.stats["hit_tokens"]}' in text
    assert b.kv.host.stats["reloaded"] >= 1
    assert any(e["type"] == "BlockStored" for batch in events.batches for e in batch[1])


def test_prefill_pool_spreads_requests_and_matches():
    from opt_prime.accel.pd.pd_batching import PrefillWorkerPool
    model = _llama()
    workers = [PrefillWorker(TorchPrefillEngine(copy.deepcopy(model), max_context=128, chunk=16, page_size=16))
               for _ in range(2)]
    seen = []
    for i, w in enumerate(workers):
        orig = w.engine.prefill
        w.engine.prefill = lambda r, s=0, i=i, orig=orig: (seen.append(i), orig(r, s))[1]
    pool = PrefillWorkerPool(workers)
    b = PDBatcher(model, pool, slots=4, max_context=128, page_size=16)
    prompts = _prompts(6, seed=31)
    fs = [b.submit(PDRequest(f"q{i}", p, 6)) for i, p in enumerate(prompts)]
    b.run_until_idle()
    for p, f in zip(prompts, fs):
        assert f.result() == _greedy(model, p, 6)
    assert set(seen) == {0, 1}                         # both workers prefilled
    pool.close()
