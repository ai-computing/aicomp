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
