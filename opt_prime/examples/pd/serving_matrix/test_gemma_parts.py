import pytest
import torch
from gemma_parts import GemmaLayer, attention_inputs
from transformers import Gemma3TextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3DecoderLayer,
    Gemma3RotaryEmbedding,
)


@pytest.mark.parametrize("index", [0, 5])
@pytest.mark.parametrize("width", [4, 12])
def test_gemma_layer_matches_hf_local_global_and_cache(index, width):
    torch.manual_seed(223)
    config = Gemma3TextConfig(
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        sliding_window=4,
        sliding_window_pattern=6,
        query_pre_attn_scalar=4,
        rope_theta=1000000,
        rope_local_base_freq=10000,
    )
    config._attn_implementation = "eager"
    ref = Gemma3DecoderLayer(config, index).eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if "norm" in name:
                p.uniform_(-0.2, 0.2)
    x = torch.randn(width, 12)
    rot, mask = attention_inputs(config.to_dict(), width, index)
    cos, sin = Gemma3RotaryEmbedding(config)(
        x[None], torch.arange(width)[None], config.layer_types[index]
    )
    distance = torch.arange(width)[:, None] - torch.arange(width)[None, :]
    allowed = distance >= 0
    if index == 0:
        allowed &= distance < 4
    expected_mask = torch.where(allowed, 0.0, -10000.0)
    torch.testing.assert_close(mask, expected_mask.repeat(4, 1)[None])
    norms = [
        getattr(ref, n).weight
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]
    ] + [ref.self_attn.q_norm.weight, ref.self_attn.k_norm.weight]
    matrices = [
        getattr(ref.self_attn, n).weight.T
        for n in ["q_proj", "k_proj", "v_proj", "o_proj"]
    ] + [getattr(ref.mlp, n).weight.T for n in ["gate_proj", "up_proj", "down_proj"]]
    cache = DynamicCache()
    with torch.no_grad():
        expected = ref(
            x[None],
            position_embeddings=(cos, sin),
            attention_mask=expected_mask[None, None],
            past_key_values=cache,
        )
        actual, k, v = GemmaLayer(config.to_dict())(x, norms, matrices, rot, mask)
    torch.testing.assert_close(actual, expected[0], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(
        k.permute(1, 0, 2), cache.layers[index].keys[0], atol=3e-6, rtol=3e-5
    )
    torch.testing.assert_close(
        v.permute(1, 0, 2), cache.layers[index].values[0], atol=3e-6, rtol=3e-5
    )
    # The layer's outputs before a changed suffix must remain unchanged.
    altered = x.clone()
    altered[width // 2 :] += 10
    with torch.no_grad():
        other = GemmaLayer(config.to_dict())(altered, norms, matrices, rot, mask)
    torch.testing.assert_close(actual[: width // 2], other[0][: width // 2])


@pytest.mark.parametrize(
    "change",
    [
        {"hidden_activation": "silu"},
        {"attention_bias": True},
        {"attn_logit_softcapping": 50.0},
        {"use_bidirectional_attention": True},
        {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
    ],
)
def test_rejects_semantics_not_implemented(change):
    cfg = Gemma3TextConfig().to_dict()
    cfg.update(change)
    with pytest.raises(ValueError):
        GemmaLayer(cfg)


@pytest.mark.parametrize("padding", [0, 4])
def test_full_model_embedding_norm_head_and_layer_packing(padding):
    from gemma_parts import final_logits, initial_hidden, layer_parameters
    from transformers import Gemma3ForCausalLM

    torch.manual_seed(223)
    cfg = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        sliding_window=4,
        sliding_window_pattern=6,
        query_pre_attn_scalar=4,
    )
    cfg._attn_implementation = "eager"
    model = Gemma3ForCausalLM(cfg).eval()
    weights = model.state_dict()
    ids = list(range(2, 14))
    with torch.no_grad():
        expected = model(input_ids=torch.tensor([ids]), logits_to_keep=1).logits[0, -1]
        x = initial_hidden(weights["model.embed_tokens.weight"], ids + [0] * padding)
        layer = GemmaLayer(cfg.to_dict())
        for i in range(6):
            norms, mats = layer_parameters(weights, i)
            rot, mask = attention_inputs(cfg.to_dict(), len(ids) + padding, i)
            x, _, _ = layer(x, norms, mats, rot, mask)
        actual = final_logits(
            x[len(ids) - 1],
            weights["model.norm.weight"],
            weights["lm_head.weight"],
            cfg.rms_norm_eps,
        )
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
