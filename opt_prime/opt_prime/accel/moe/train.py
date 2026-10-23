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

from opt_prime.accel.moe.batched import opt_batched_mm

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


def prepare_moe_for_training(model: torch.nn.Module) -> bool:
    """switch the model's experts to the export- and autograd-safe implementation; True if the model has experts."""
    from opt_prime.accel.spec.discover import is_experts_module
    if not any(is_experts_module(m) for m in model.modules()):
        return False
    if not _register():
        return False
    for cfg in {id(c): c for c in (getattr(model, "config", None),
                                   getattr(getattr(model, "config", None), "get_text_config", lambda: None)())
                if c is not None}.values():
        cfg._experts_implementation = NAME
    return True
