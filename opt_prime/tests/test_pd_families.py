"""PD (prefill engine -> KV copy -> decode engine) parity for the MoE and hybrid-attention families: tokens and
logits equal to the no-cache model, as in test_pd_torch for the dense ones."""
import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opt_prime.accel.pd import PDRequest, PDRuntime  # noqa: E402
from opt_prime.accel.pd.torch_engine import TorchCacheTransport, TorchCausalLMEngine  # noqa: E402
from opt_prime.accel.pd.torch_models import describe_model  # noqa: E402
_EXTRA = {"gemma4_text": dict(enable_moe_block=True, top_k_experts=2, global_head_dim=32)}


def _tiny(family):
    """small random model of the family (MoE variants with 8 experts, top-2)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        base = AutoConfig.for_model(family)
    except Exception as e:      # family unknown to this transformers version (gemma4 needs >= 5.5)
        pytest.skip(f"{family}: {type(e).__name__}")
    shrink = dict(hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=128,
                  max_position_embeddings=256, num_experts=8, n_routed_experts=8, num_local_experts=8,
                  num_experts_per_tok=2, n_shared_experts=1, n_group=1, topk_group=1, first_k_dense_replace=0,
                  pad_token_id=0, bos_token_id=1, eos_token_id=2)
    kw = {k: v for k, v in shrink.items() if hasattr(base, k)}
    kw.update(_EXTRA.get(family, {}))
    lt = getattr(base, "layer_types", None)
    if lt:
        kinds = list(dict.fromkeys(lt))
        kw["layer_types"] = [kinds[i % len(kinds)] for i in range(2)]
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(AutoConfig.for_model(family, **kw))
    m.config._experts_implementation = "eager"
    return m.eval()


@pytest.mark.parametrize("family", ["qwen3_moe", "solar_open", "gemma3_text", "gemma4_text"])
@pytest.mark.parametrize("prefix", [False, True])
def test_pd_parity(family, prefix):
    model = _tiny(family)
    geo = describe_model(model.config)
    if family == "gemma4_text":
        assert not geo.uniform, "Gemma 4 global layers keep larger K/V than sliding layers"
    kw = dict(model_id=family, revision="tiny-seed0", max_context=64)
    source = TorchCausalLMEngine(copy.deepcopy(model), endpoint="source", prefix_only=prefix, capture_logits=True, **kw)
    dest = TorchCausalLMEngine(copy.deepcopy(model), endpoint="destination", capture_logits=True, **kw)
    runtime = PDRuntime(source, dest, TorchCacheTransport(source="source", destination="destination", spec=source.spec))
    for length, count in ((2, 4), (9, 6), (23, 8)):
        prompt = tuple(3 + (7 * i) % 120 for i in range(length))
        full, logits = list(prompt), []
        with torch.inference_mode():
            for _ in range(count):
                row = model(torch.tensor([full]), use_cache=False).logits[0, -1]
                logits.append(row)
                full.append(int(row.argmax()))
        got = runtime.generate(PDRequest(f"{family}-{length}", prompt, count))
        assert got == full[length:]
        measured = source.last_logits + dest.last_logits
        torch.testing.assert_close(torch.stack(measured), torch.stack(logits), atol=2e-5, rtol=1e-4)
        assert runtime.last_receipt["bytes"] > 0 and runtime.state == "idle"


def test_sliding_window_context_is_bounded():
    model = _tiny("gemma3_text")
    model.config.sliding_window = 32
    window = describe_model(model.config).window
    assert window == 32
    with pytest.raises(ValueError, match="sliding window"):
        TorchCausalLMEngine(model, model_id="g", revision="r", max_context=window + 1, endpoint="e")
