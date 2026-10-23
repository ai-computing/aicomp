"""#127: hybrid families in the paged batcher - Gemma 3 (sliding + full) and Gemma 4 (sliding + K=V global layers with
another head geometry, MoE): tokens equal HF greedy beyond the sliding window, through batching, prefix cache and PD."""
import copy

import pytest

torch = pytest.importorskip("torch")
from transformers import AutoConfig, AutoModelForCausalLM

from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.pd_batching import PDBatcher, PrefillWorker, TorchPrefillEngine
from opt_prime.accel.pd.torch_models import describe_model

W = 16


def _family(module):
    pytest.importorskip(f"transformers.models.{module}")


def _gemma3():
    _family("gemma3")
    c = AutoConfig.for_model("gemma3_text", vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=6,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=W,
                             layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
                             max_position_embeddings=512, query_pre_attn_scalar=16)
    torch.manual_seed(5)
    return AutoModelForCausalLM.from_config(c).eval()


def _gemma4_globals():
    """the global layers' geometry: per-layer config (transformers >= 5.15) or global_* keys (earlier)"""
    import inspect
    from transformers import Gemma4TextConfig
    if "global_head_dim" in inspect.signature(Gemma4TextConfig.__init__).parameters:
        return dict(global_head_dim=32, num_global_key_value_heads=1)
    return dict(per_layer_config={"02": {"head_dim": 32, "num_key_value_heads": 1},
                                  "05": {"head_dim": 32, "num_key_value_heads": 1}})


def _gemma4():
    _family("gemma4")
    c = AutoConfig.for_model(
        "gemma4_text", vocab_size=128, vocab_size_per_layer_input=128, hidden_size=64, intermediate_size=96,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=W,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2, max_position_embeddings=512,
        attention_k_eq_v=True, enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=32,
        hidden_size_per_layer_input=0, num_kv_shared_layers=0, **_gemma4_globals())
    torch.manual_seed(6)
    return AutoModelForCausalLM.from_config(c).eval()


def _qwen35(moe=False):
    _family("qwen3_5_moe" if moe else "qwen3_5")
    kw = dict(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, layer_types=["linear_attention"] * 3 + ["full_attention"],
              linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
              linear_conv_kernel_dim=4, max_position_embeddings=512)
    if moe:
        kw.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32)
    c = AutoConfig.for_model("qwen3_5_moe_text" if moe else "qwen3_5_text", **kw)
    torch.manual_seed(7)
    return AutoModelForCausalLM.from_config(c).eval()


def _qwen35_moe():
    return _qwen35(moe=True)


def _greedy(model, prompt, n):
    """reference: full recomputation every step (HF's own masks, no cache), no stop tokens"""
    full, out = list(prompt), []
    with torch.inference_mode():
        for _ in range(n):
            t = int(model(torch.tensor([full])).logits[0, -1].argmax())
            out.append(t)
            full.append(t)
    return out


def _prompts(k, lo, hi, seed):
    g = torch.Generator().manual_seed(seed)
    return [tuple(torch.randint(3, 127, (int(torch.randint(lo, hi, (1,), generator=g)),), generator=g).tolist())
            for _ in range(k)]


FAMILIES = pytest.mark.parametrize("make", [_gemma3, _gemma4], ids=["gemma3", "gemma4"])


@FAMILIES
def test_geometry_groups(make):
    g = describe_model(make().config)
    kinds = {(x.kind, x.window) for x in g.groups}
    assert ("sliding", W) in kinds and ("full", None) in kinds


@FAMILIES
def test_batched_beyond_window_matches_hf(make):
    model = make()
    b = ContinuousBatcher(model, kv="paged", slots=3, max_context=128, prefill_chunk=8, page_size=8)
    prompts = _prompts(5, 10, 40, seed=1)                  # prompts and outputs longer than the window (16)
    fs = []
    for i, p in enumerate(prompts):
        fs.append(b.submit(PDRequest(f"r{i}", p, 24)))
        b.step()
    b.run_until_idle()
    for p, f in zip(prompts, fs):
        assert f.result() == _greedy(model, p, 24)


@FAMILIES
def test_prefix_cache_and_pd(make):
    model = make()
    engine = TorchPrefillEngine(copy.deepcopy(model), max_context=128, chunk=16, page_size=8)
    b = PDBatcher(model, PrefillWorker(engine), slots=2, max_context=128, page_size=8, prefix_cache=True)
    system = _prompts(1, 32, 33, seed=2)[0]
    prompts = [system + (i, i + 1, i + 2) for i in range(3, 7)]
    fs = [b.submit(PDRequest(f"p{i}", p, 12)) for i, p in enumerate(prompts)]
    b.run_until_idle()
    for p, f in zip(prompts, fs):
        assert f.result() == _greedy(model, p, 12)
    assert b.stats["decode_cached_tokens"] > 0


def test_slot_kv_rejects_hybrid():
    with pytest.raises(ValueError, match="paged"):
        ContinuousBatcher(_gemma3(), kv="slot", slots=2, max_context=64)


RECURRENT = pytest.mark.parametrize("make", [_qwen35, _qwen35_moe], ids=["qwen3_5", "qwen3_5_moe"])


@RECURRENT
def test_recurrent_batched_matches_hf(make):
    model = make()
    g = describe_model(model.config)
    assert len(g.recurrent) == 3 and [x.layers for x in g.groups] == [(3,)]
    b = ContinuousBatcher(model, kv="paged", slots=3, max_context=128, prefill_chunk=8, page_size=8)
    prompts = _prompts(5, 6, 30, seed=3)                   # several prefill chunks per prompt (state carried over)
    fs = []
    for i, p in enumerate(prompts):
        fs.append(b.submit(PDRequest(f"r{i}", p, 16)))
        b.step()
    b.run_until_idle()
    for p, f in zip(prompts, fs):
        assert f.result() == _greedy(model, p, 16)


@RECURRENT
def test_recurrent_pd_hands_over_state(make):
    model = make()
    engine = TorchPrefillEngine(copy.deepcopy(model), max_context=128, chunk=8, page_size=8, prefix_cache=False)
    b = PDBatcher(model, PrefillWorker(engine), slots=2, max_context=128, page_size=8)
    prompts = _prompts(4, 6, 30, seed=4)
    fs = [b.submit(PDRequest(f"p{i}", p, 12)) for i, p in enumerate(prompts)]
    b.run_until_idle()
    for p, f in zip(prompts, fs):
        assert f.result() == _greedy(model, p, 12)


@pytest.mark.parametrize("family", ["solar_open", "qwen3_moe", "llama", "qwen2"])
def test_full_attention_families_batched_and_pd(family):
    """Solar-Open / Qwen3-MoE / Llama (KONI) / Qwen2.5 tiny configs through the paged batcher and PD"""
    from opt_prime.accel.testing import build_tiny
    model = build_tiny(family).eval()
    vocab = model.config.get_text_config().vocab_size
    g = torch.Generator().manual_seed(9)
    prompts = [tuple(torch.randint(3, vocab - 1, (int(n),), generator=g).tolist()) for n in (7, 19, 30)]
    b = ContinuousBatcher(model, kv="paged", slots=2, max_context=128, prefill_chunk=8, page_size=8)
    fs = [b.submit(PDRequest(f"r{i}", p, 10)) for i, p in enumerate(prompts)]
    b.run_until_idle()
    for p, f in zip(prompts, fs):
        assert f.result() == _greedy(model, p, 10)
    engine = TorchPrefillEngine(copy.deepcopy(model), max_context=128, chunk=8, page_size=8)
    pd = PDBatcher(model, PrefillWorker(engine), slots=2, max_context=128, page_size=8)
    fs = [pd.submit(PDRequest(f"p{i}", p, 10)) for i, p in enumerate(prompts)]
    pd.run_until_idle()
    for p, f in zip(prompts, fs):
        assert f.result() == _greedy(model, p, 10)
