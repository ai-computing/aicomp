#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Count which experts each MoE layer routes to while running calibration inputs."""
import weakref

import torch
import torch.nn as nn

from opt_prime.accel.spec.discover import find_moe_experts


class ExpertProfiler:
    """Context manager: `with ExpertProfiler(model) as p: model(...)`; then `p.counts` is [layers][experts].

    Hooks the experts modules (plain or already offloaded), so it can refine a placement in place. Holds them weakly:
    a profiler kept alive for its counts must not keep the original expert weights alive once they are offloaded.
    """
    def __init__(self, model: nn.Module):
        mods = [i.module for i in find_moe_experts(model)]
        if not mods:
            from opt_prime.accel.moe.ops import OffloadedExperts
            mods = [m for m in model.modules() if isinstance(m, OffloadedExperts)]
        self.counts = [torch.zeros(m.num_experts, dtype=torch.long) for m in mods]
        self._mods = [weakref.ref(m) for m in mods]
        self._h = []

    @property
    def mods(self):
        return [r() for r in self._mods]

    def __enter__(self):
        for i, m in enumerate(self.mods):
            if m is None:
                continue
            def hook(mod, args, i=i):
                E = self.counts[i].numel()
                if len(args) >= 2:                  # experts(hidden, top_k_index, top_k_weights)
                    idx = args[1].detach().reshape(-1).to("cpu", torch.long)
                    self.counts[i] += torch.bincount(idx, minlength=E)[:E]
                else:                               # dense dispatch (Llama 4): [E * T, H], rows of unrouted pairs are 0
                    x = args[0].detach()
                    self.counts[i] += (x.reshape(E, -1, x.shape[-1]).abs().sum(-1) > 0).sum(-1).to("cpu", torch.long)
            self._h.append(m.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._h: h.remove()
        self._h = []
        return False

    def as_lists(self):
        return [c.tolist() for c in self.counts]
