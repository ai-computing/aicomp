"""#42: sampling parameters, seeds, stop tokens / strings and finish_reason on the PD path."""
import copy

import pytest

torch = pytest.importorskip("torch")
from transformers import LlamaConfig, LlamaForCausalLM

from opt_prime.accel.pd import PDRequest, PDRuntime, SamplingParams
from opt_prime.accel.pd.torch_engine import TorchCausalLMEngine, TorchCacheTransport
from opt_prime.accel.serve.llmd import PDStageService, StageError
from test_pd_llmd import service, body


def _runtime(prefix=False):
    torch.manual_seed(17)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=41, max_position_embeddings=128, hidden_size=32,
                                         intermediate_size=64, num_hidden_layers=2, head_dim=8,
                                         num_attention_heads=4, num_key_value_heads=2)).eval()
    src = TorchCausalLMEngine(copy.deepcopy(model), model_id="llama", revision="r", max_context=128,
                              endpoint="source", prefix_only=prefix)
    dst = TorchCausalLMEngine(copy.deepcopy(model), model_id="llama", revision="r", max_context=128,
                              endpoint="destination")
    return model, PDRuntime(src, dst, TorchCacheTransport(source="source", destination="destination", spec=src.spec))


PROMPT = tuple(range(1, 12))


def _hf_greedy(model, n):
    full = list(PROMPT)
    with torch.inference_mode():
        for _ in range(n):
            full.append(int(model(torch.tensor([full])).logits[0, -1].argmax()))
    return full[len(PROMPT):]


@pytest.mark.parametrize("prefix", [False, True])
def test_greedy_path_unchanged(prefix):
    model, rt = _runtime(prefix)
    assert rt.generate(PDRequest("g", PROMPT, 12)) == _hf_greedy(model, 12)
    assert rt.generate(PDRequest("g1", PROMPT, 12, SamplingParams(temperature=0.7, top_k=1))) == _hf_greedy(model, 12)


@pytest.mark.parametrize("prefix", [False, True])
def test_seeded_sampling_is_deterministic_across_engines(prefix):
    _, rt = _runtime(prefix)
    s = SamplingParams(temperature=1.3, top_p=0.9, top_k=20, seed=1234)
    a = rt.generate(PDRequest("a", PROMPT, 16, s))
    _, other = _runtime(prefix)                                   # fresh engines, as on another host
    assert other.generate(PDRequest("b", PROMPT, 16, s)) == a
    seeds = {tuple(rt.generate(PDRequest(f"s{i}", PROMPT, 16, SamplingParams(temperature=1.3, seed=i))))
             for i in range(4)}
    assert len(seeds) > 1                                         # the seed matters


def test_stop_token_ends_generation_early():
    model, rt = _runtime()
    greedy = _hf_greedy(model, 12)
    stop = greedy[4]
    first = greedy.index(stop)
    out = rt.generate(PDRequest("st", PROMPT, 12, SamplingParams(stop_token_ids=(stop,))))
    assert out == greedy[:first + 1]


def _letters(tokens):
    return "".join(chr(97 + t % 26) for t in tokens)


def test_service_finish_reason_eos_stop_string_and_greedy_only_engines():
    # fixture engines: prefill emits 7 ('h'), decode emits 8 ('i') repeatedly; they support greedy only
    base = service()
    engine = PDStageService(base.prefill, base.decode, base.transport, model="model", vocab_size=100, context=64,
                            max_output=32, render=_letters, ttl=10, eos_token_ids=(8,))
    engine.execute("prefill", "e", body(1))
    r = engine.execute("decode", "e", body(6))["choices"][0]
    assert r["token_ids"] == [7, 8] and r["finish_reason"] == "stop" and r["text"] == "hi"

    engine.execute("prefill", "n", dict(body(1), ignore_eos=True))
    r = engine.execute("decode", "n", dict(body(6), ignore_eos=True))["choices"][0]
    assert r["token_ids"] == [7] + [8] * 5 and r["finish_reason"] == "length"

    engine.execute("prefill", "s", dict(body(1), ignore_eos=True, stop=["iii"]))
    r = engine.execute("decode", "s", dict(body(6), ignore_eos=True, stop=["iii"]))["choices"][0]
    assert r["token_ids"] == [7, 8, 8, 8] and r["text"] == "h" and r["finish_reason"] == "stop"

    with pytest.raises(StageError) as err:
        engine.execute("prefill", "t", dict(body(1), temperature=0.8))
    assert err.value.status == 400
    with pytest.raises(StageError):
        engine.execute("prefill", "u", dict(body(1), top_p=0))
