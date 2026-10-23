import pytest
from koni_prefill import Prefill


def test_rejects_wrong_geometry_before_loading_sdk():
    with pytest.raises(ValueError, match="Llama"):
        Prefill({}, {}, width=512)


@pytest.mark.parametrize("width", [0, 128, 513])
def test_rejects_unsupported_width_before_loading_sdk(width):
    config = {
        "model_type": "llama",
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "rope_scaling": {"rope_type": "llama3"},
    }
    with pytest.raises(ValueError, match="512"):
        Prefill(config, {}, width=width)
