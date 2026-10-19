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

    def wrap_parent(self, parent: nn.Module) -> None:
        """optional: adapt the module that calls the experts (its MoE block) to the offloaded experts' call
        signature (hidden [T,H], top_k_index [T,k], top_k_weights [T,k]). Default: nothing to do."""


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


class _StdExperts(nn.Module):
    """transformers >= 5 experts interface over another layout (used as the weightless shell backends copy)."""

    def __init__(self, num_experts, act_fn, transposed):
        super().__init__()
        self.num_experts, self.act_fn, self.is_transposed = num_experts, act_fn, transposed
        self.has_bias, self.has_gate = False, True
        self.gate_up_proj = nn.Parameter(torch.empty(0), requires_grad=False)
        self.down_proj = nn.Parameter(torch.empty(0), requires_grad=False)

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def forward(self, hidden, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden)
        for j in range(top_k_index.shape[1]):
            for e in top_k_index[:, j].unique().tolist():
                rows = (top_k_index[:, j] == e).nonzero().squeeze(-1)
                x = hidden[rows]
                gu, dn = self.gate_up_proj[e], self.down_proj[e]
                h = self._apply_gate(x @ gu if self.is_transposed else x @ gu.t())
                y = h @ dn if self.is_transposed else h @ dn.t()
                out.index_add_(0, rows, (y * top_k_weights[rows, j, None]).to(out.dtype))
        return out


@register_adapter
class Llama4Adapter(ModelAdapter):
    """Llama 4 text MoE: experts stored [E, H, 2F] / [E, F, H] and called with every expert's input already scaled
    by its routing score (routed_in = x * sigmoid(score), zero for experts outside the top-k). Since expert(0) = 0
    (no biases), that equals sum over the top-k of expert_e(w_e * x). The MoE block is rewritten to that form -
    each (token, expert) pair becomes a row holding w * x, offloaded with weight 1 - so every backend runs it."""

    def matches(self, module):
        return type(module).__name__ == "Llama4TextExperts"

    def spec(self, m):
        E, H, F2 = m.gate_up_proj.shape
        return ExpertSpec(E, H, F2 // 2, _activation_name(m), "swiglu", False, True, m.gate_up_proj.dtype)

    def weights(self, m, ids):
        t = torch.tensor(list(ids), dtype=torch.long)
        shell = _StdExperts(m.num_experts, m.act_fn, True)
        return ExpertWeights(self.spec(m), list(ids), m.gate_up_proj.data[t], m.down_proj.data[t], module=shell)

    def wrap_parent(self, block):
        import types
        import torch.nn.functional as F

        def forward(self, hidden_states):
            h = hidden_states.reshape(-1, self.hidden_dim)
            logits = F.linear(h, self.router.weight)
            top_v, idx = torch.topk(logits, self.router.top_k, dim=1)
            w = torch.sigmoid(top_v.float()).to(h.dtype)
            k = idx.shape[1]
            rows = (h.unsqueeze(1) * w.unsqueeze(-1)).reshape(-1, h.shape[-1])          # one row per pair
            y = self.experts(rows, idx.reshape(-1, 1), torch.ones(rows.shape[0], 1, dtype=h.dtype, device=h.device))
            out = self.shared_expert(h) + y.view(-1, k, h.shape[-1]).sum(dim=1)
            return out, logits

        block.forward = types.MethodType(forward, block)


def adapter_for(module: nn.Module) -> ModelAdapter:
    for a in _ADAPTERS:
        if a.matches(module):
            return a
    raise ValueError(f"no ModelAdapter for {type(module).__name__}")


def moe_layers(model: nn.Module) -> List[ExpertsInfo]:
    return find_moe_experts(model)
