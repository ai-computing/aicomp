"""#44: continuous batching with chunked prefill - every request's tokens equal its own single-request run."""
import copy

import pytest

torch = pytest.importorskip("torch")
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from opt_prime.accel.pd import PDRequest, SamplingParams
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.runtime import RequestCancelled


def _llama():
    torch.manual_seed(17)
    return LlamaForCausalLM(LlamaConfig(vocab_size=97, max_position_embeddings=256, hidden_size=32,
                                        intermediate_size=64, num_hidden_layers=2, head_dim=8,
                                        num_attention_heads=4, num_key_value_heads=2)).eval()


def _greedy(model, prompt, n, stop=()):
    full, out = list(prompt), []
    with torch.inference_mode():
        for _ in range(n):
            t = int(model(torch.tensor([full])).logits[0, -1].argmax())
            out.append(t); full.append(t)
            if t in stop:
                break
    return out


def _prompts(k, lo=3, hi=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [tuple(torch.randint(1, 97, (int(torch.randint(lo, hi, (1,), generator=g)),), generator=g).tolist())
            for _ in range(k)]


KV = pytest.mark.parametrize("kv", ["slot", "paged"])


def _batcher(model, kv, **kw):
    if kv == "paged":
        kw.setdefault("page_size", 16)
    return ContinuousBatcher(model, kv=kv, **kw)


@KV
@pytest.mark.parametrize("chunk", [1, 7, 256])
def test_staggered_arrivals_match_single_request_runs(chunk, kv):
    model = _llama()
    b = _batcher(model, kv, slots=3, max_context=128, prefill_chunk=chunk)
    prompts = _prompts(7)
    lengths = [5, 17, 1, 9, 24, 3, 12]
    futures = []
    for i, (p, n) in enumerate(zip(prompts, lengths)):
        futures.append(b.submit(PDRequest(f"r{i}", p, n)))
        for _ in range(i % 3):            # arrivals between iterations: the batch changes every few steps
            b.step()
    b.run_until_idle()
    for p, n, f in zip(prompts, lengths, futures):
        assert f.result() == _greedy(model, p, n)
    assert b.stats["decode_rows"] > b.stats["decode_steps"]          # steps really batched several requests
    assert sorted(b.free) == [0, 1, 2]


def test_chunked_prefill_kv_equals_full_prefill():
    model = _llama()
    p = _prompts(1, 40, 41)[0]
    b = ContinuousBatcher(model, slots=1, max_context=64, prefill_chunk=6)
    f = b.submit(PDRequest("c", p, 1))
    b.run_until_idle()
    with torch.inference_mode():
        ref = model(torch.tensor([p]), use_cache=True)
    for layer, entry in enumerate(ref.past_key_values.layers):
        torch.testing.assert_close(b.kv.k[layer, 0, :, :len(p)], entry.keys[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(b.kv.v[layer, 0, :, :len(p)], entry.values[0], atol=1e-5, rtol=1e-5)
    assert f.result() == [int(ref.logits[0, -1].argmax())]
    assert b.stats["prefill_chunks"] == 7


@KV
def test_prefill_chunks_interleave_with_decode(kv):
    model = _llama()
    b = _batcher(model, kv, slots=2, max_context=128, prefill_chunk=4)
    short = b.submit(PDRequest("s", (1, 2, 3), 20))
    b.step()                                                 # short request prefilled, now decoding
    long_ = b.submit(PDRequest("l", tuple(range(1, 41)), 3))
    seen = []
    while not b.idle:
        before = len(b.decoding)
        b.step()
        seen.append((bool(b.prefilling), before))
    assert any(p and d for p, d in seen)                     # decode kept running while the long prompt prefilled
    assert short.result() == _greedy(model, (1, 2, 3), 20)
    assert long_.result() == _greedy(model, tuple(range(1, 41)), 3)


@KV
def test_stop_tokens_cancellation_and_slot_reuse(kv):
    model = _llama()
    p = (5, 6, 7, 8)
    ref = _greedy(model, p, 12)
    b = _batcher(model, kv, slots=1, max_context=64, prefill_chunk=8)
    stop = b.submit(PDRequest("stop", p, 12, SamplingParams(stop_token_ids=(ref[4],))))
    gone = b.submit(PDRequest("gone", p, 12))                # waits for the only slot
    b.cancel("gone")
    after = b.submit(PDRequest("after", p, 12))
    b.run_until_idle()
    assert stop.result() == ref[:ref.index(ref[4]) + 1]
    with pytest.raises(RequestCancelled):
        gone.result()
    assert after.result() == ref and b.free == [0]


@KV
def test_moe_family_and_rejections(kv):
    torch.manual_seed(3)
    moe = Qwen3MoeForCausalLM(Qwen3MoeConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
                                             moe_intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
                                             num_key_value_heads=2, head_dim=8, num_experts=4,
                                             num_experts_per_tok=2, max_position_embeddings=128)).eval()
    b = _batcher(moe, kv, slots=2, max_context=64, prefill_chunk=5)
    prompts = _prompts(3, 3, 20, seed=5)
    fs = [b.submit(PDRequest(f"m{i}", p, 6)) for i, p in enumerate(prompts)]
    b.run_until_idle()
    for p, f in zip(prompts, fs):
        assert f.result() == _greedy(moe, p, 6)
    with pytest.raises(ValueError):
        b.submit(PDRequest("big", tuple(range(1, 60)), 10))     # context exceeded



def test_paged_admission_follows_pages_not_slots():
    """with paged KV a request takes only the pages it needs; admission waits for pages, not for max_context"""
    model = _llama()
    b = ContinuousBatcher(model, kv="paged", slots=8, max_context=96, prefill_chunk=64, page_size=16, pages=6)
    prompts = _prompts(5, 10, 30, seed=2)
    fs = [b.submit(PDRequest(f"p{i}", p, 8)) for i, p in enumerate(prompts)]     # each needs 2-3 pages
    b.step()
    assert len(b.prefilling) + len(b.decoding) in (2, 3)                          # 6 pages hold 2-3 requests
    b.run_until_idle()
    for p, f in zip(prompts, fs):
        assert f.result() == _greedy(model, p, 8)
    assert len(b.kv.free_pages) == 6 and all(not o for o in b.kv.owned)


def test_paged_pool_smaller_than_one_request_is_rejected():
    """#169: a request admitted by max_context must fit the pool, or it blocks the FIFO queue forever"""
    with pytest.raises(ValueError):
        ContinuousBatcher(_llama(), kv="paged", slots=2, max_context=64, page_size=16, pages=2)
