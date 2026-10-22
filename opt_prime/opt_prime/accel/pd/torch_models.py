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


def describe_model(config, model=None) -> ModelGeometry:
    """Cache geometry of a validated family. The family policy is checked here; the per-layer KV shapes and sliding
    windows come from opt_prime.accel.spec (`model_spec` of `model`, or of `config` built on the meta device), the
    one parser of model configs (#105)."""
    from opt_prime.accel.spec.config import config_value
    config = _text_config(config)
    model_type = config.model_type
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f'unsupported model family: {model_type}')
    if config_value(config, 'is_encoder_decoder', False) or config_value(config, 'add_cross_attention', False):
        raise ValueError('cross attention requires additional cache state')
    if config_value(config, 'num_kv_shared_layers', 0):
        raise ValueError('layers sharing another layer\'s KV cache require a separate adapter')
    kinds = config_value(config, 'layer_types')
    if model_type in HYBRID_MODEL_TYPES:
        if kinds and any(k not in ('full_attention', 'sliding_attention') for k in kinds):
            raise ValueError('unsupported layer types')
    else:
        window = config_value(config, 'sliding_window', None)
        if window and config_value(config, 'use_sliding_window', True):
            raise ValueError('sliding-window cache requires a separate adapter')
        if kinds and any(kind != 'full_attention' for kind in kinds):
            raise ValueError('hybrid/sliding attention cache is unsupported')
    spec = _spec(config, model)
    dims, windows = [], []
    for layer in spec.layers:
        a = layer.attention
        if not a.keeps_kv or a.kind == 'mla' or min(a.kv_heads, a.head_dim) < 1:
            raise ValueError('positive cache geometry required')
        dims.append((a.kv_heads, a.head_dim))
        if a.sliding_window:
            windows.append(a.sliding_window)
    if not dims:
        raise ValueError('positive cache geometry required')
    window = min(windows) if windows and model_type in HYBRID_MODEL_TYPES else None
    return ModelGeometry(model_type, len(dims), max(h for h, _ in dims), max(d for _, d in dims), tuple(dims), window)


def _spec(config, model):
    from opt_prime.accel.spec.hf import model_spec, model_spec_from_config
    if model is not None:
        return model_spec(model)
    return model_spec_from_config(config)
