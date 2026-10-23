from types import SimpleNamespace

import pytest
import torch
from koni_export import export_groups, validate


def config():
    return {
        "model_type": "llama",
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "rope_scaling": {"rope_type": "llama3"},
    }


def test_rejects_geometry_or_semantics_the_fixed_graph_cannot_represent():
    validate(config())
    for change in [
        {"hidden_size": 2048},
        {"attention_bias": True},
        {"mlp_bias": True},
        {"rope_scaling": {"rope_type": "linear"}},
    ]:
        with pytest.raises(ValueError):
            validate(config() | change)


def stage():
    # Two groups, each with two layers; padding and axes have distinct values.
    keys = torch.arange(4 * 2 * 3 * 5).reshape(4, 2, 3, 5).to(torch.bfloat16)
    values = (keys.transpose(-1, -2) + 1000).contiguous()
    return SimpleNamespace(
        g=2,
        e=SimpleNamespace(ngroups=2),
        P=3,
        Ks=list(keys.split(2)),
        Vs=list(values.split(2)),
        x=torch.arange(20).reshape(5, 4).to(torch.bfloat16),
    )


def test_export_preserves_layer_head_position_and_crops_padding():
    s = stage()
    norm = torch.tensor([1.0, 2.0, 3.0, 4.0])
    head = torch.eye(4)
    k, v, logits = export_groups(s, norm, head, 1e-5)
    assert k.shape == v.shape == (4, 2, 3, 3)
    for layer in range(4):
        for pos in range(3):
            assert torch.equal(k[layer, :, pos], s.Ks[layer // 2][layer % 2, :, :, pos])
            assert torch.equal(v[layer, :, pos], s.Vs[layer // 2][layer % 2, :, pos])
    last = s.x[2].float()
    expected = last / torch.sqrt(last.square().mean() + 1e-5) * norm
    torch.testing.assert_close(logits, expected)


def test_incomplete_stage_is_not_exported():
    s = stage()
    s.g = 1
    with pytest.raises(ValueError, match="incomplete"):
        export_groups(s, torch.ones(4), torch.eye(4), 1e-5)
