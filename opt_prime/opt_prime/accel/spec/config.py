#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Reading transformers configs across versions.

transformers >= 5.15 has heterogeneous configs: attributes that may differ per decoder layer (Gemma 4's head_dim, ...)
live in `config.per_layer_config[i]`, and reading them globally raises AmbiguousGlobalPerLayerAttributeError, also from
`hasattr` and `getattr(cfg, name, default)`. Older versions keep one global value (Gemma 4: `head_dim` for sliding
layers, `global_head_dim` for full-attention layers).
"""


def config_value(cfg, name, default=None):
    """the global value of `name`, or `default` when it is absent or only defined per layer"""
    try:
        return getattr(cfg, name, default)
    except AttributeError:
        return default
    except Exception:                       # a per-layer attribute read globally (transformers >= 5.15)
        return default


def per_layer_config(cfg):
    """the per-layer configs of a heterogeneous config, or None"""
    plc = config_value(cfg, "per_layer_config")
    return plc if plc else None


def layer_value(cfg, name, layer, default=None):
    """`name` for decoder layer `layer`: from its per-layer config when the config has one, else the global value"""
    plc = per_layer_config(cfg)
    if plc is not None and layer is not None and layer < len(plc):
        try:
            v = getattr(plc[layer], name)
            if v is not None:
                return v
        except AttributeError:
            pass
    return config_value(cfg, name, default)


def has_value(cfg, name):
    """hasattr that also holds for per-layer attributes of heterogeneous configs"""
    try:
        getattr(cfg, name)
        return True
    except AttributeError:
        return False
    except Exception:
        return True
