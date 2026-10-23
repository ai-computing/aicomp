"""#122: FuriosaLLMBatcher over an AsyncLLMEngine-shaped fake (cumulative streaming outputs, abort, cached tokens).
Hardware runs: docs/validation/batching/RNGD.md."""
import asyncio
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from opt_prime.accel.pd import PDRequest, SamplingParams
from opt_prime.accel.pd.paged import block_hashes
from opt_prime.accel.pd.runtime import RequestCancelled
from opt_prime.accel.platforms.rngd.batching import FuriosaLLMBatcher
from opt_prime.accel.serve.kv_events import ListKVEventPublisher

from test_pd_batching import _greedy, _llama, _prompts


class FakeEngine:
    """greedy decoding with the tiny llama, one token per await; prefix hits counted in 16-token blocks"""

    def __init__(self, model):
        self.model, self.aborted, self.seen = model, set(), set()

    async def generate(self, prompt, params, rid):
        ids = list(prompt["prompt_token_ids"])
        cached = 0
        for h in block_hashes(ids, 16)[:(len(ids) - 1) // 16]:
            if h not in self.seen:
                break
            cached += 16
        self.seen.update(block_hashes(ids, 16))
        out = []
        for _ in range(params.max_tokens):
            await asyncio.sleep(0)
            if rid in self.aborted:
                return
            with torch.inference_mode():
                t = int(self.model(torch.tensor([ids + out])).logits[0, -1].argmax())
            out.append(t)
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(out))], num_cached_tokens=cached)
            if params.stop_token_ids and t in params.stop_token_ids:
                return

    async def abort(self, rid):
        self.aborted.add(rid)


@pytest.fixture
def furiosa_sampling(monkeypatch):
    import sys
    mod = SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setitem(sys.modules, "furiosa_llm", mod)


@pytest.mark.parametrize("mode", ["rounds", "native"])
def test_concurrent_requests_match_and_stream(furiosa_sampling, mode):
    model = _llama()
    b = FuriosaLLMBatcher(FakeEngine(model), max_context=128, max_concurrency=4, mode=mode, quantum_busy=2,
                          quantum_idle=3)
    prompts, lengths = _prompts(5), [5, 9, 3, 12, 7]
    streamed = {i: [] for i in range(5)}
    fs = [b.submit(PDRequest(f"r{i}", p, n), on_token=streamed[i].append) for i, (p, n) in enumerate(zip(prompts, lengths))]
    b.run_until_idle()
    for i, (p, n, f) in enumerate(zip(prompts, lengths, fs)):
        assert f.result() == _greedy(model, p, n) == streamed[i]
    assert b.idle and b.stats["requests"] == 5
    b.close()


@pytest.mark.parametrize("mode", ["rounds", "native"])
def test_stop_and_cancel(furiosa_sampling, mode):
    model = _llama()
    b = FuriosaLLMBatcher(FakeEngine(model), max_context=128, max_concurrency=2, mode=mode, quantum_busy=2)
    p, q = _prompts(2, seed=4)
    first = _greedy(model, p, 3)
    stop = b.submit(PDRequest("stop", p, 20, SamplingParams(stop_token_ids=(first[2],))))
    gone = b.submit(PDRequest("gone", q, 50))
    assert b.cancel("gone")
    b.run_until_idle()
    assert stop.result() == first
    with pytest.raises(RequestCancelled):
        gone.result()
    b.close()


def test_prefix_hits_and_mirrored_kv_events(furiosa_sampling):
    model = _llama()
    events = ListKVEventPublisher(block_size=16)
    b = FuriosaLLMBatcher(FakeEngine(model), max_context=128, max_concurrency=2, kv_events=events, block_size=16,
                          mirror_blocks=3)
    system = _prompts(1, 50, 51, seed=3)[0]
    for rid, p in (("a", system + (5,)), ("c", system + (9,)), ("d", _prompts(1, 40, 41, seed=9)[0])):
        f = b.submit(PDRequest(rid, p, 2)); b.run_until_idle(); f.result()
    b.step()
    assert (b.hit_tokens, b.query_tokens) == (48, 51 + 51 + 40)
    flat = [e for batch in events.batches for e in batch[1]]
    assert flat[0]["type"] == "BlockStored" and flat[0]["block_hashes"] == block_hashes(system + (5,), 16)
    assert any(e["type"] == "BlockRemoved" for e in flat)               # the mirror holds 3 blocks
    b.close()


def test_served_by_batched_service(furiosa_sampling):
    from opt_prime.accel.serve.batched import BatchedService
    b = FuriosaLLMBatcher(FakeEngine(_llama()), max_context=128, max_concurrency=4)
    svc = BatchedService(b, model="m", vocab_size=97, max_output=8, render=str)
    assert svc.gauges() == (0, 0, 0.0)
    f = b.submit(PDRequest("a", _prompts(1)[0], 4)); b.run_until_idle(); f.result()
    text = svc.metrics.render(*svc.gauges())
    assert "vllm:prefix_cache_queries_total" in text
    b.close()


def test_rounds_admit_waiting_requests_between_quanta(furiosa_sampling):
    """5 requests, 2 lanes: later requests start after a quantum, not after the running ones finish"""
    model = _llama()
    b = FuriosaLLMBatcher(FakeEngine(model), max_context=128, max_concurrency=2, quantum_busy=2, quantum_idle=50)
    prompts = _prompts(5, seed=12)
    first = {}
    fs = [b.submit(PDRequest(f"r{i}", p, 12), on_token=lambda t, i=i: first.setdefault(i, b.stats["iterations"]))
          for i, p in enumerate(prompts)]
    b.run_until_idle()
    for p, f in zip(prompts, fs):
        assert f.result() == _greedy(model, p, 12)
    assert first[2] < 12 and b.stats["decode_rows"] > b.stats["iterations"]   # admitted early; quanta batched
    b.close()
