import math

import pytest
import torch
from koni_parts import AttentionPart, FFNPart, Parts
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)


class Add(torch.nn.Module):
    def forward(self, a, b):
        return (a + b,)


class Softmax(torch.nn.Module):
    def forward(self, x):
        return (x.softmax(-1),)


class SwiGLU(torch.nn.Module):
    def forward(self, gate, up):
        return (torch.nn.functional.silu(gate) * up,)


def fixture(width, kv_heads=2):
    torch.manual_seed(221)
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        head_dim=4,
        max_position_embeddings=131072,
        rope_theta=500000,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    )
    config._attn_implementation = "eager"
    layer = LlamaDecoderLayer(config, 0).eval()
    x = torch.randn(width, 16)
    positions = torch.arange(width)[None]
    cos, sin = LlamaRotaryEmbedding(config)(x[None], positions)
    rotation = torch.zeros(width, 4, 4)
    i = torch.arange(2)
    rotation[:, i, i] = rotation[:, i + 2, i + 2] = cos[0, :, :2]
    rotation[:, i, i + 2] = sin[0, :, :2]
    rotation[:, i + 2, i] = -sin[0, :, :2]
    mask = torch.triu(torch.full((width, width), -1e4), 1)
    return config, layer, x, positions, cos, sin, rotation, mask


def partitioned(layer, x, rotation, mask):
    width = len(x)
    kv_heads = layer.self_attn.config.num_key_value_heads
    attention = AttentionPart(width, 4, kv_heads, 4, Softmax(), Add())
    ffn = FFNPart(SwiGLU(), Add())
    a = layer.self_attn
    m = layer.mlp
    residual, keys, values = attention(
        layer.input_layernorm(x),
        x,
        a.q_proj.weight.T / math.sqrt(4),
        a.k_proj.weight.T,
        a.v_proj.weight.T,
        a.o_proj.weight.T,
        rotation,
        mask.repeat(4 // kv_heads, 1)[None],
    )
    out = ffn(
        layer.post_attention_layernorm(residual),
        residual,
        m.gate_proj.weight.T,
        m.up_proj.weight.T,
        m.down_proj.weight.T,
    )
    assert keys.shape == values.shape == (width, kv_heads, 4)
    return out, keys.permute(1, 2, 0), values.permute(1, 0, 2)


@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_parts_match_independent_hf_layer_and_cache(width, kv_heads):
    config, layer, x, pos, cos, sin, rotation, mask = fixture(width, kv_heads)
    cache = DynamicCache(config=config)
    with torch.no_grad():
        expected = layer(
            x[None],
            attention_mask=mask[None, None],
            position_ids=pos,
            position_embeddings=(cos, sin),
            past_key_values=cache,
            use_cache=True,
        )
        actual, keys, values = partitioned(layer, x, rotation, mask)
    torch.testing.assert_close(actual, expected[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        keys.transpose(-1, -2), cache.layers[0].keys[0], atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(values, cache.layers[0].values[0], atol=2e-6, rtol=2e-5)


def test_future_tokens_do_not_change_prefix_outputs():
    _, layer, x, _, _, _, rotation, mask = fixture(8)
    altered = x.clone()
    altered[4:] += 100
    with torch.no_grad():
        a = partitioned(layer, x, rotation, mask)
        b = partitioned(layer, altered, rotation, mask)
    torch.testing.assert_close(a[0][:4], b[0][:4])
    torch.testing.assert_close(a[1][..., :4], b[1][..., :4])
    torch.testing.assert_close(a[2][:, :4], b[2][:, :4])


class Norm(torch.nn.Module):
    def forward(self, x, w, eps, exponent):
        return x * (x.square().sum(-1, keepdim=True) + eps).pow(exponent) * w


def test_runtime_partition_glue_matches_hf_including_tiled_norm():
    config, layer, x, pos, cos, sin, rotation, mask = fixture(8, 1)
    parts = Parts(
        8,
        4,
        Norm(),
        AttentionPart(8, 4, 1, 4, Softmax(), Add()),
        FFNPart(SwiGLU(), Add()),
    )
    a, m = layer.self_attn, layer.mlp
    n1 = layer.input_layernorm.weight.repeat(4, 1)[None] * math.sqrt(16)
    n2 = layer.post_attention_layernorm.weight.repeat(4, 1)[None] * math.sqrt(16)
    cache = DynamicCache(config=config)
    with torch.no_grad():
        expected = layer(
            x[None],
            attention_mask=mask[None, None],
            position_ids=pos,
            position_embeddings=(cos, sin),
            past_key_values=cache,
            use_cache=True,
        )
        actual, keys, values = parts(
            x,
            torch.full((4, 16), config.rms_norm_eps * 16),
            torch.full((4, 16), -0.5),
            n1,
            a.q_proj.weight.T[None] / 2,
            a.k_proj.weight.T[None],
            a.v_proj.weight.T[None],
            a.o_proj.weight.T[None],
            n2,
            m.gate_proj.weight.T[None],
            m.up_proj.weight.T[None],
            m.down_proj.weight.T[None],
            rotation,
            mask.repeat(4, 1)[None],
        )
    torch.testing.assert_close(actual, expected[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        keys[0].transpose(-1, -2), cache.layers[0].keys[0], atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        values[0], cache.layers[0].values[0], atol=2e-6, rtol=2e-5
    )
