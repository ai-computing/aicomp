"""Validated dense attention model families and physical cache geometry.

Adding a family here requires cached-vs-no-cache parity tests. A similar class
name or a shared Transformers base class alone does not establish compatibility.
"""
from dataclasses import dataclass

DENSE_MODEL_TYPES = frozenset({'gpt2', 'llama', 'qwen2', 'qwen3', 'mistral'})


@dataclass(frozen=True)
class ModelGeometry:
    model_type: str
    layers: int
    kv_heads: int
    head_dim: int


def describe_model(config) -> ModelGeometry:
    model_type = config.model_type
    if model_type not in DENSE_MODEL_TYPES:
        raise ValueError(f'unsupported model family: {model_type}')
    window = getattr(config, 'sliding_window', None)
    if window and getattr(config, 'use_sliding_window', True):
        raise ValueError('sliding-window cache requires a separate adapter')
    layer_types = getattr(config, 'layer_types', None)
    if layer_types and any(kind != 'full_attention' for kind in layer_types):
        raise ValueError('hybrid/sliding attention cache is unsupported')
    if getattr(config, 'is_encoder_decoder', False) or getattr(config, 'add_cross_attention', False):
        raise ValueError('cross attention requires additional cache state')
    layers = config.num_hidden_layers
    heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    if any(type(value) is not int or value < 1 for value in (layers, heads, dim)):
        raise ValueError('positive cache geometry required')
    return ModelGeometry(model_type, layers, heads, dim)
