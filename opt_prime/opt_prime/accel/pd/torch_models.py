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
#: linear-attention (recurrent state) layers mixed with full attention (#127)
RECURRENT_MODEL_TYPES = frozenset({'qwen3_5_text', 'qwen3_5_moe_text'})
SUPPORTED_MODEL_TYPES = DENSE_MODEL_TYPES | MOE_MODEL_TYPES | HYBRID_MODEL_TYPES | RECURRENT_MODEL_TYPES


@dataclass(frozen=True)
class KVGroup:
    """layers that share one KV geometry and attention kind (one page pool / block mask in the paged batcher)"""
    kind: str                      # "full" | "sliding"
    window: Optional[int]          # sliding window (tokens), None for full attention
    kv_heads: int
    head_dim: int
    layers: Tuple[int, ...]


@dataclass(frozen=True)
class ModelGeometry:
    model_type: str
    layers: int
    kv_heads: int                  # largest over layers
    head_dim: int                  # largest over layers
    layer_dims: Tuple[Tuple[int, int], ...] = ()     # (kv_heads, head_dim) per layer
    window: Optional[int] = None   # smallest sliding window of the model's sliding layers
    layer_windows: Tuple[Optional[int], ...] = ()    # sliding window per KV layer (None: full attention)
    kv_layers: Tuple[int, ...] = ()                    # model layer index of each KV layer (default 0..layers-1)
    recurrent: Tuple[int, ...] = ()                    # layers with a per-sequence recurrent state instead of KV

    @property
    def uniform(self) -> bool:
        return len(set(self.layer_dims)) <= 1

    @property
    def groups(self) -> Tuple[KVGroup, ...]:
        """layers grouped by (kind, window, kv_heads, head_dim), in order of first appearance"""
        windows = self.layer_windows or (None,) * self.layers
        index = self.kv_layers or tuple(range(self.layers))
        order, members = [], {}
        for i, dims, w in zip(index, self.layer_dims or ((self.kv_heads, self.head_dim),) * self.layers, windows):
            key = (w, dims)
            if key not in members:
                order.append(key)
                members[key] = []
            members[key].append(i)
        return tuple(KVGroup("sliding" if w else "full", w, d[0], d[1], tuple(members[(w, d)])) for w, d in order)


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
    elif model_type in RECURRENT_MODEL_TYPES:
        if kinds and any(k not in ('full_attention', 'linear_attention') for k in kinds):
            raise ValueError('unsupported layer types')
    else:
        window = config_value(config, 'sliding_window', None)
        if window and config_value(config, 'use_sliding_window', True):
            raise ValueError('sliding-window cache requires a separate adapter')
        if kinds and any(kind != 'full_attention' for kind in kinds):
            raise ValueError('hybrid/sliding attention cache is unsupported')
    spec = _spec(config, model)
    dims, windows, per_layer, kv_layers, recurrent = [], [], [], [], []
    for index, layer in enumerate(spec.layers):
        a = layer.attention
        if a.kind == 'linear' and model_type in RECURRENT_MODEL_TYPES:
            recurrent.append(index)
            continue
        if not a.keeps_kv or a.kind == 'mla' or min(a.kv_heads, a.head_dim) < 1:
            raise ValueError('positive cache geometry required')
        kv_layers.append(index)
        dims.append((a.kv_heads, a.head_dim))
        w = a.sliding_window if model_type in HYBRID_MODEL_TYPES else None
        per_layer.append(w or None)
        if w:
            windows.append(w)
    if not dims:
        raise ValueError('positive cache geometry required')
    window = min(windows) if windows else None
    return ModelGeometry(model_type, len(dims), max(h for h, _ in dims), max(d for _, d in dims), tuple(dims), window,
                         tuple(per_layer), tuple(kv_layers), tuple(recurrent))


def _spec(config, model):
    from opt_prime.accel.spec.hf import model_spec, model_spec_from_config
    if model is not None:
        return model_spec(model)
    return model_spec_from_config(config)
