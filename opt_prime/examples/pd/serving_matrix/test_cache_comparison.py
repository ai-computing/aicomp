import pytest
import torch
from cache_comparison import retained_export
from transformers import Gemma3TextConfig
from transformers.cache_utils import DynamicCache


def test_preserves_absolute_position_with_different_retained_lengths():
    cfg = Gemma3TextConfig(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
    )
    cache = DynamicCache(config=cfg)
    x = torch.arange(48).reshape(1, 1, 12, 4).float()
    for i in range(2):
        cache.update(x, x + 100, i)
    local = retained_export(x[0], cache.layers[0])
    full = retained_export(x[0], cache.layers[1])
    assert local.shape == (1, 3, 4) and full.shape == (1, 12, 4)
    torch.testing.assert_close(local, cache.layers[0].keys[0])
    torch.testing.assert_close(full, cache.layers[1].keys[0])
    with pytest.raises(ValueError):
        retained_export(x[0, :, :11], cache.layers[0])
