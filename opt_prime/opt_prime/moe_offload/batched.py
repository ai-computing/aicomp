#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Batched experts math over any transformers >= 5 experts module (or a shell with the same attributes): gather each
routed pair's expert weights, batched matmul, weighted sum. No in-place ops, no data-dependent branches, so it is
exportable and differentiable in any dtype. Used by the offloaded experts' backward and, registered as an experts
implementation, for training (opt_prime.moe_train)."""
import torch


def _linear(x, w, b, transposed):
    """x [S, in], w [S, out, in] (or [S, in, out] if transposed) -> [S, out]"""
    y = torch.bmm(x.unsqueeze(1), w).squeeze(1) if transposed else torch.bmm(w, x.unsqueeze(-1)).squeeze(-1)
    return y + b if b is not None else y


def opt_batched_mm(self, hidden_states, top_k_index, top_k_weights):
    k = top_k_index.shape[-1]
    T, H = hidden_states.shape
    x = hidden_states.repeat_interleave(k, dim=0)                         # [S, H], S = T * k
    ids = top_k_index.reshape(-1).clamp(0, self.num_experts - 1)          # out-of-place
    tr = bool(getattr(self, "is_transposed", False))
    bias = bool(getattr(self, "has_bias", False))
    if getattr(self, "has_gate", hasattr(self, "gate_up_proj")):
        h = _linear(x, self.gate_up_proj[ids], self.gate_up_proj_bias[ids] if bias else None, tr)
        if hasattr(self, "_apply_gate"):
            h = self._apply_gate(h)
        else:
            g, u = h.chunk(2, dim=-1)
            h = self.act_fn(g) * u
    else:
        h = self.act_fn(_linear(x, self.up_proj[ids], self.up_proj_bias[ids] if bias else None, tr))
    y = _linear(h, self.down_proj[ids], self.down_proj_bias[ids] if bias else None, tr)
    y = y * top_k_weights.reshape(-1, 1).to(y.dtype)
    return y.view(T, k, H).sum(dim=1).to(hidden_states.dtype)
