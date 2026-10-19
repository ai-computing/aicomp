#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Model side: find a model's MoE layers and describe their experts as ExpertSpec / ExpertWeights.

GenericAdapter covers every transformers >= 5 experts module (stacked `gate_up_proj` / `down_proj`, optional biases,
`is_transposed`, `_apply_gate`). A family that stores experts differently registers its own adapter:

    @register_adapter
    class MyAdapter(ModelAdapter):
        def matches(self, module): ...
        def spec(self, module): ...
        def weights(self, module, ids): ...
"""
from typing import List

import copy

import torch
import torch.nn as nn

from .discover import ExpertsInfo, find_moe_experts, is_experts_module
from .spec import ExpertSpec, ExpertWeights

_ADAPTERS: List["ModelAdapter"] = []


def register_adapter(cls):
    _ADAPTERS.insert(0, cls())         # later registrations take precedence over the generic one
    return cls


class ModelAdapter:
    def matches(self, module: nn.Module) -> bool:
        raise NotImplementedError

    def spec(self, module: nn.Module) -> ExpertSpec:
        raise NotImplementedError

    def weights(self, module: nn.Module, ids: List[int]) -> ExpertWeights:
        raise NotImplementedError


def _activation_name(m):
    n = type(getattr(m, "act_fn", None)).__name__.lower()
    for k in ("silu", "gelu", "relu"):
        if k in n: return k
    return n or "unknown"


@register_adapter
class GenericAdapter(ModelAdapter):
    def matches(self, module):
        return is_experts_module(module)

    def spec(self, m):
        try:
            from transformers.integrations.moe import _default_apply_gate
            default_gate = getattr(type(m), "_apply_gate", _default_apply_gate) is _default_apply_gate
        except Exception:
            default_gate = True
        transposed = bool(getattr(m, "is_transposed", False))
        E = m.num_experts
        if transposed:     # [E, H, 2F], [E, F, H]
            H, F = m.gate_up_proj.shape[1], m.gate_up_proj.shape[2] // 2
        else:              # [E, 2F, H], [E, H, F]
            H, F = m.gate_up_proj.shape[2], m.gate_up_proj.shape[1] // 2
        return ExpertSpec(E, H, F, _activation_name(m), "swiglu" if default_gate else "custom",
                          bool(getattr(m, "has_bias", False)), transposed, m.gate_up_proj.dtype)

    @staticmethod
    def _shell(m):
        """the experts module without its stacked weights (structure, gate, activation, flags only), so holding it
        does not keep the full expert tensors alive."""
        c = copy.copy(m)
        E = m.num_experts
        c._parameters = {n: (nn.Parameter(torch.empty((0,) + tuple(p.shape[1:]), dtype=p.dtype), requires_grad=False)
                             if p is not None and p.dim() >= 1 and p.shape[0] == E else p)
                         for n, p in m._parameters.items()}
        c._buffers, c._modules = dict(m._buffers), dict(m._modules)
        return c

    def weights(self, m, ids):
        t = torch.tensor(list(ids), dtype=torch.long)
        sub = lambda name: getattr(m, name).data[t] if getattr(m, name, None) is not None else None
        has_bias = bool(getattr(m, "has_bias", False))
        return ExpertWeights(self.spec(m), list(ids), sub("gate_up_proj"), sub("down_proj"),
                             sub("gate_up_proj_bias") if has_bias else None, sub("down_proj_bias") if has_bias else None,
                             module=self._shell(m))


def adapter_for(module: nn.Module) -> ModelAdapter:
    for a in _ADAPTERS:
        if a.matches(module):
            return a
    raise ValueError(f"no ModelAdapter for {type(module).__name__}")


def moe_layers(model: nn.Module) -> List[ExpertsInfo]:
    return find_moe_experts(model)
