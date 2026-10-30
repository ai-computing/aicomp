#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Batched experts math over any transformers >= 5 experts module (or a shell with the same attributes): gather each
routed pair's expert weights, batched matmul, weighted sum. No in-place ops, no data-dependent branches, so it is
exportable and differentiable in any dtype. Used by the offloaded experts' backward and, registered as an experts
implementation, for training (opt_prime.accel.moe.train)."""
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
    valid = (top_k_index >= 0) & (top_k_index < self.num_experts)
    y = y * (top_k_weights * valid).reshape(-1, 1).to(y.dtype)
    return y.view(T, k, H).sum(dim=1).to(hidden_states.dtype)


def opt_grouped_mm(self, hidden_states, top_k_index, top_k_weights, device=None):
    """The same math as opt_batched_mm, one expert at a time: memory is one expert's weights (moved to `device` if
    they live elsewhere, e.g. on the host) instead of a weight copy per routed pair, which for large experts
    (Solar: 21 MB per pair, 1024 pairs per layer at S=128, top-8) does not fit. Differentiable w.r.t. the hidden
    states and the routing weights."""
    dev = device or hidden_states.device
    T, H = hidden_states.shape
    k = top_k_index.shape[-1]
    flat = top_k_index.reshape(-1)
    rows = torch.arange(T, device=flat.device).repeat_interleave(k)
    wf = top_k_weights.reshape(-1)
    tr = bool(getattr(self, "is_transposed", False))
    bias = bool(getattr(self, "has_bias", False))
    gated = getattr(self, "has_gate", hasattr(self, "gate_up_proj"))
    lin = lambda x, w, b: (x @ w if tr else x @ w.t()) + (b if b is not None else 0)
    get = lambda name, e: getattr(self, name)[e].to(dev) if getattr(self, name, None) is not None else None
    # zeros that stay connected to both inputs (all-padding calls still get zero gradients); empty slices, so an
    # inf in hidden_states does not turn into NaN
    out = (torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)
           + hidden_states[:0].float().sum() + top_k_weights[:0].float().sum())
    for e in torch.unique(flat).tolist():
        if not 0 <= e < self.num_experts:
            continue
        sel = (flat == e).nonzero().squeeze(1)
        r = rows[sel]
        x = hidden_states[r]
        if gated:
            h = lin(x, get("gate_up_proj", e), get("gate_up_proj_bias", e) if bias else None)
            if hasattr(self, "_apply_gate"):
                h = self._apply_gate(h)
            else:
                g, u = h.chunk(2, dim=-1)
                h = self.act_fn(g) * u
        else:
            h = self.act_fn(lin(x, get("up_proj", e), get("up_proj_bias", e) if bias else None))
        y = lin(h, get("down_proj", e), get("down_proj_bias", e) if bias else None)
        out = out.index_add(0, r, (y * wf[sel, None].to(y.dtype)).float())
    return out.to(hidden_states.dtype)
