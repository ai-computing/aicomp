#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Count which experts each MoE layer routes to while running calibration inputs."""
import torch
import torch.nn as nn

from .discover import find_moe_experts


class ExpertProfiler:
    """Context manager: `with ExpertProfiler(model) as p: model(...)`; then `p.counts` is [layers][experts].

    Hooks the experts modules (plain or already offloaded), so it can refine a placement in place.
    """
    def __init__(self, model: nn.Module):
        self.mods = [i.module for i in find_moe_experts(model)]
        if not self.mods:
            from .ops import OffloadedExperts
            self.mods = [m for m in model.modules() if isinstance(m, OffloadedExperts)]
        sizes = [m.num_experts for m in self.mods]
        self.counts = [torch.zeros(e, dtype=torch.long) for e in sizes]
        self._h = []

    def __enter__(self):
        for i, m in enumerate(self.mods):
            def hook(mod, args, i=i):
                idx = args[1].detach().reshape(-1).to("cpu", torch.long)
                self.counts[i] += torch.bincount(idx, minlength=self.counts[i].numel())[: self.counts[i].numel()]
            self._h.append(m.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._h: h.remove()
        self._h = []
        return False

    def as_lists(self):
        return [c.tolist() for c in self.counts]
