#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""ModelSpec from a transformers (>= 5) causal LM: the module tree gives the structure (projection shapes, q/k norms,
MoE blocks, shared experts), the config gives scalars (eps, RoPE, routing parameters).

`model_spec(model)` reads an instantiated model; `model_spec_from_pretrained(path_or_id)` builds the model on the
meta device first, so no weights are read and nothing is allocated. Anything not recognized is described as kind
"custom" with its module class, never guessed: providers then fall back to running the original torch module.

Router scoring is code, not config, in transformers; ROUTER_SCORING lists the families whose router semantics were
checked against their modeling code. Unlisted families get scoring "custom".
"""
import json
from typing import Optional

import torch
import torch.nn as nn

from .blocks import (AttentionSpec, DenseMLPSpec, LayerSpec, MoESpec, ModelSpec, NormSpec, RopeSpec,
                     RouterSpec)

#: model_type -> (scoring, uses e_score_correction_bias for selection)
ROUTER_SCORING = {
    "qwen2_moe": ("softmax", False), "qwen3_moe": ("softmax", False), "qwen3_next": ("softmax", False),
    "mixtral": ("softmax", False), "olmoe": ("softmax", False), "flex_olmo": ("softmax", False),
    "deepseek_v3": ("sigmoid", True), "glm4_moe": ("sigmoid", True), "solar_open": ("sigmoid", True),
    "dots1": ("sigmoid", True), "glm4_moe_lite": ("sigmoid", True),
}
_ATTN_NAMES = ("self_attn", "attention", "self_attention", "attn")
_STATE_NAMES = ("linear_attn", "mamba", "conv", "mixer")
_IN_NORM = ("input_layernorm", "attention_norm", "ln_1")
_POST_NORM = ("post_attention_layernorm", "pre_ff_layernorm", "ffn_norm", "pre_feedforward_layernorm", "ln_2")
_MLP_NAMES = ("mlp", "block_sparse_moe", "feed_forward", "moe")


def _cfg(model):
    c = model.config
    return getattr(c, "text_config", None) or c


def _child(m, names):
    for n in names:
        if hasattr(m, n) and isinstance(getattr(m, n), nn.Module):
            return n, getattr(m, n)
    return None, None


def _out(lin) -> int:
    return lin.weight.shape[0] if hasattr(lin, "weight") else 0


def _norm(m: Optional[nn.Module]) -> Optional[NormSpec]:
    if m is None:
        return None
    cls = type(m).__name__
    kind = "rms" if "RMSNorm" in cls else ("layer" if isinstance(m, nn.LayerNorm) or "LayerNorm" in cls else "custom")
    eps = getattr(m, "variance_epsilon", None) or getattr(m, "eps", None)
    w = getattr(m, "weight", None)
    return NormSpec(kind, int(w.shape[0]) if w is not None else 0, float(eps) if eps is not None else None, cls)


def _rope(cfg, layer_type: Optional[str]) -> RopeSpec:
    rp = getattr(cfg, "rope_parameters", None) or {}
    if layer_type and isinstance(rp, dict) and isinstance(rp.get(layer_type), dict):
        rp = rp[layer_type]                                  # per layer-type RoPE (e.g. sliding vs full)
    rp = dict(rp) if isinstance(rp, dict) else {}
    theta = rp.pop("rope_theta", None) or getattr(cfg, "rope_theta", None)
    prf = float(rp.pop("partial_rotary_factor", getattr(cfg, "partial_rotary_factor", 1.0)) or 1.0)
    if rp.get("rope_type") == "default":
        rp.pop("rope_type")
    return RopeSpec(float(theta) if theta is not None else None, json.dumps(rp, sort_keys=True) if rp else None, prf)


def _attention(cfg, layer, layer_type) -> AttentionSpec:
    hidden = cfg.hidden_size
    name, a = _child(layer, _ATTN_NAMES)
    if a is None:
        sname, s = _child(layer, _STATE_NAMES)
        return AttentionSpec("linear" if sname in ("linear_attn",) else "custom", hidden,
                             module_class=type(s).__name__ if s is not None else "")
    cls = type(a).__name__
    heads = getattr(cfg, "num_attention_heads", 0)
    window = getattr(cfg, "sliding_window", None) if layer_type == "sliding_attention" else None
    if window is None and layer_type is None:
        window = getattr(a, "sliding_window", None)
    rope = _rope(cfg, layer_type)
    if hasattr(a, "kv_a_proj_with_mqa"):                     # multi-head latent attention (DeepSeek-V2/V3 family)
        nope, rd, vd = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        return AttentionSpec("mla", hidden, heads, heads, nope + rd, vd, False,
                             getattr(a.o_proj, "bias", None) is not None, False, window,
                             RopeSpec(rope.theta, rope.scaling, rope.partial_rotary_factor, rd),
                             cfg.kv_lora_rank, getattr(cfg, "q_lora_rank", None) or 0, cls)
    if all(hasattr(a, p) for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
        kvh = getattr(cfg, "num_key_value_heads", None) or heads
        hd = getattr(a, "head_dim", None) or getattr(cfg, "head_dim", None) or hidden // heads
        geometry_ok = _out(a.q_proj) == heads * hd and _out(a.k_proj) == kvh * hd and _out(a.v_proj) == kvh * hd
        return AttentionSpec("gqa" if geometry_ok else "custom", hidden, heads, kvh, hd, hd,
                             a.q_proj.bias is not None, a.o_proj.bias is not None,
                             hasattr(a, "q_norm") and hasattr(a, "k_norm"), window, rope, module_class=cls)
    return AttentionSpec("custom", hidden, heads, module_class=cls)


def _router(cfg, moe, experts_spec) -> RouterSpec:
    model_type = cfg.model_type
    E = experts_spec.num_experts
    rname, r = _child(moe, ("gate", "router"))
    cls = type(r).__name__ if r is not None else ""
    scoring, bias = ROUTER_SCORING.get(model_type, ("custom", False))
    names = [n for n, _ in moe.named_buffers()] + [n for n, _ in moe.named_parameters()]
    has_bias = any("e_score_correction_bias" in n for n in names)
    if has_bias != bias:
        scoring = "custom"                                   # the module does not match the checked semantics
    top_k = (getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "moe_k", None)
             or getattr(cfg, "top_k_experts", None) or 0)
    return RouterSpec(E, int(top_k), scoring, bool(getattr(cfg, "norm_topk_prob", True)), bias,
                      int(getattr(cfg, "n_group", None) or 1), int(getattr(cfg, "topk_group", None) or 1),
                      float(getattr(cfg, "routed_scaling_factor", None) or 1.0), cls)


def _shared(moe):
    _, s = _child(moe, ("shared_experts", "shared_expert", "shared_mlp"))
    if s is None:
        return 0, False
    ffn = _out(s.gate_proj) if hasattr(s, "gate_proj") else (_out(s.up_proj) if hasattr(s, "up_proj") else -1)
    return ffn, hasattr(moe, "shared_expert_gate")


def _mlp(cfg, layer):
    from ..moe_offload.adapters import adapter_for
    from ..moe_offload.discover import is_experts_module
    _, m = _child(layer, _MLP_NAMES)
    side = next((x for _, x in layer.named_children() if is_experts_module(x)), None)
    if side is not None:                                     # experts beside a dense MLP on the layer (Gemma 4)
        es = adapter_for(side).spec(side)
        dense = _mlp(cfg, nn.ModuleDict({"mlp": m}))[0] if m is not None else None
        return dense, MoESpec(_router(cfg, layer, es), es, 0, False, type(layer).__name__)
    if m is None:
        return None, None
    experts = next((x for x in m.modules() if is_experts_module(x)), None)
    if experts is not None:
        es = adapter_for(experts).spec(experts)
        ffn, gate = _shared(m)
        return None, MoESpec(_router(cfg, m, es), es, ffn, gate, type(m).__name__)
    cls, act = type(m).__name__, getattr(cfg, "hidden_act", "silu")
    if all(hasattr(m, p) for p in ("gate_proj", "up_proj", "down_proj")):
        return DenseMLPSpec("gated", cfg.hidden_size, _out(m.gate_proj), act, m.down_proj.bias is not None, cls), None
    if hasattr(m, "gate_up_proj") and hasattr(m, "down_proj"):
        return DenseMLPSpec("gated", cfg.hidden_size, _out(m.gate_up_proj) // 2, act, False, cls), None
    if hasattr(m, "up_proj") and hasattr(m, "down_proj"):
        return DenseMLPSpec("plain", cfg.hidden_size, _out(m.up_proj), act, m.down_proj.bias is not None, cls), None
    return DenseMLPSpec("custom", cfg.hidden_size, 0, act, module_class=cls), None


def _decoder_layers(model, n):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == n and name.endswith("layers"):
            return name, mod
    raise ValueError(f"no decoder layer list of length {n} found")


def model_spec(model: nn.Module, revision: str = "") -> ModelSpec:
    cfg = _cfg(model)
    n = cfg.num_hidden_layers
    prefix, layers = _decoder_layers(model, n)
    types = getattr(cfg, "layer_types", None)
    out = []
    for i, layer in enumerate(layers):
        lt = types[i] if isinstance(types, (list, tuple)) and i < len(types) else None
        mlp, moe = _mlp(cfg, layer)
        out.append(LayerSpec(i, _attention(cfg, layer, lt), _norm(_child(layer, _IN_NORM)[1]),
                             _norm(_child(layer, _POST_NORM)[1]), mlp, moe, f"{prefix}.{i}"))
    emb = model.get_input_embeddings()
    base = getattr(model, "model", model)
    dtype = getattr(cfg, "dtype", None) or next(model.parameters()).dtype
    return ModelSpec(cfg.model_type, cfg.hidden_size, int(emb.weight.shape[0]), tuple(out),
                     _norm(getattr(base, "norm", None)), bool(getattr(cfg, "tie_word_embeddings", False)),
                     str(dtype).replace("torch.", ""), getattr(cfg, "max_position_embeddings", None), revision)


def model_spec_from_pretrained(path_or_id: str, revision: str = "", dtype=None) -> ModelSpec:
    """ModelSpec without reading weights: config + the model built on the meta device."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(path_or_id)
    dt = dtype or getattr(cfg, "dtype", None) or torch.bfloat16
    if isinstance(dt, str):
        dt = getattr(torch, dt)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dt)
    return model_spec(model, revision or str(path_or_id).rstrip("/").split("/")[-1])
