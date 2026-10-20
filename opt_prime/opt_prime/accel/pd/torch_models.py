"""Validated attention model families and physical cache geometry.

Adding a family here requires cached-vs-no-cache parity tests. A similar class
name or a shared Transformers base class alone does not establish compatibility.

Families whose layers keep differently shaped K/V (Gemma 4: global layers use a
larger head dim) report per-layer geometry; layers with a sliding window are
accepted when the engine's context fits in the window (they then keep every
token, like full-attention layers).
"""
from dataclasses import dataclass
from typing import Optional, Tuple

DENSE_MODEL_TYPES = frozenset({'gpt2', 'llama', 'qwen2', 'qwen3', 'mistral'})
#: full-attention MoE families (attention cache identical to the dense ones)
MOE_MODEL_TYPES = frozenset({'qwen3_moe', 'solar_open', 'mixtral'})
#: hybrid sliding/full attention, per-layer geometry
HYBRID_MODEL_TYPES = frozenset({'gemma3_text', 'gemma4_text'})
SUPPORTED_MODEL_TYPES = DENSE_MODEL_TYPES | MOE_MODEL_TYPES | HYBRID_MODEL_TYPES


@dataclass(frozen=True)
class ModelGeometry:
    model_type: str
    layers: int
    kv_heads: int                  # largest over layers
    head_dim: int                  # largest over layers
    layer_dims: Tuple[Tuple[int, int], ...] = ()     # (kv_heads, head_dim) per layer
    window: Optional[int] = None   # smallest sliding window of the model's sliding layers

    @property
    def uniform(self) -> bool:
        return len(set(self.layer_dims)) <= 1


def _text_config(config):
    get = getattr(config, 'get_text_config', None)
    return get() if callable(get) else config


def describe_model(config) -> ModelGeometry:
    config = _text_config(config)
    model_type = config.model_type
    if model_type in HYBRID_MODEL_TYPES:
        return _describe_hybrid(config)
    if model_type not in DENSE_MODEL_TYPES | MOE_MODEL_TYPES:
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
    return ModelGeometry(model_type, layers, heads, dim, ((heads, dim),) * layers)


def _describe_hybrid(config) -> ModelGeometry:
    from opt_prime.accel.spec.config import config_value, per_layer_config
    if config_value(config, 'num_kv_shared_layers', 0):
        raise ValueError('layers sharing another layer\'s KV cache require a separate adapter')
    layers = config.num_hidden_layers
    kinds = list(config_value(config, 'layer_types') or ['full_attention'] * layers)
    if len(kinds) != layers or any(k not in ('full_attention', 'sliding_attention') for k in kinds):
        raise ValueError('unsupported layer types')
    plc = per_layer_config(config)
    if plc is not None:                   # transformers >= 5.15: per-layer configs carry each layer's geometry
        dims = tuple((getattr(c, 'num_key_value_heads', None) or config.num_attention_heads,
                      getattr(c, 'head_dim', None) or config.hidden_size // config.num_attention_heads) for c in plc)
        wins = [getattr(c, 'sliding_window', None) for c, k in zip(plc, kinds) if k == 'sliding_attention']
        window = min((w for w in wins if w), default=None)
    else:
        heads = config_value(config, 'num_key_value_heads') or config.num_attention_heads
        dim = config_value(config, 'head_dim') or config.hidden_size // config.num_attention_heads
        full_dim = config_value(config, 'global_head_dim') or dim          # Gemma 4 global layers
        full_heads = config_value(config, 'num_global_key_value_heads') or heads
        dims = tuple((full_heads, full_dim) if k == 'full_attention' else (heads, dim) for k in kinds)
        window = config_value(config, 'sliding_window') if 'sliding_attention' in kinds else None
    return ModelGeometry(config.model_type, layers, max(h for h, _ in dims), max(d for _, d in dims), dims, window)
