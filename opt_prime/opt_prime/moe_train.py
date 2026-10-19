#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Training MoE models through the export-based IR (transformers >= 5 experts interface).

transformers' experts implementations do not all survive torch.export + autograd:
  eager       data-dependent loop over hit experts (nonzero) - not exportable
  grouped_mm  bf16 only (fp32 fails at export); some models' kernels are traced into by export (Gemma 4, bf16)
  batched_mm  clamps the routing indices in place (expert_ids.clamp_) - a view of top_k_index that the router saved
              for backward, so backward fails ("modified by an inplace operation")

  (5.1)       batched_mm returns early when the routing weights sum to 0 - data-dependent, not exportable

"opt_batched_mm" is our own batched form (gather each routed pair's expert weights, batched matmul, weighted sum,
no in-place ops, no data-dependent branches): exportable, differentiable, any dtype, any transformers 5.x.
Optimus_p selects it for models with experts modules when training (inference keeps the model's own choice).
"""
import torch

NAME = "opt_batched_mm"
_REGISTERED = False


def _register():
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        from transformers.integrations import moe
    except ImportError:
        return False
    if not hasattr(moe, "ExpertsInterface"):
        return False
    moe.ExpertsInterface.register(NAME, opt_batched_mm)
    _REGISTERED = True
    return True


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


def prepare_moe_for_training(model: torch.nn.Module) -> bool:
    """switch the model's experts to the export- and autograd-safe implementation; True if the model has experts."""
    from .moe_offload.discover import is_experts_module
    if not any(is_experts_module(m) for m in model.modules()):
        return False
    if not _register():
        return False
    for cfg in {id(c): c for c in (getattr(model, "config", None),
                                   getattr(getattr(model, "config", None), "get_text_config", lambda: None)())
                if c is not None}.values():
        cfg._experts_implementation = NAME
    return True
