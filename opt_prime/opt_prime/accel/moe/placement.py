#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Choose hot experts under a device memory budget.

counts[l][e]: routed tokens per (layer, expert) from a profile (ExpertProfiler or a saved list).
Global greedy: the hottest (layer, expert) pairs model-wide fill the budget, optionally with a per-layer cap and
floor, which maximizes the expected fraction of routed tokens served on the device.
"""
from typing import List, Optional, Sequence


def plan_hot_experts(counts: Sequence[Sequence[float]], budget_experts: int, per_layer_max: Optional[int] = None,
                     per_layer_min: int = 0) -> List[List[int]]:
    L = len(counts)
    hot = [sorted(range(len(c)), key=lambda e: (-c[e], e))[:per_layer_min] for c in counts]
    left = budget_experts - sum(len(h) for h in hot)
    cand = sorted(((counts[l][e], l, e) for l in range(L) for e in range(len(counts[l])) if e not in hot[l]),
                  key=lambda t: (-t[0], t[1], t[2]))
    for _, l, e in cand:
        if left <= 0: break
        if per_layer_max is not None and len(hot[l]) >= per_layer_max: continue
        hot[l].append(e); left -= 1
    return [sorted(h) for h in hot]


def uniform_hot(num_layers: int, num_experts: int, per_layer: int) -> List[List[int]]:
    return [list(range(min(per_layer, num_experts))) for _ in range(num_layers)]


def device_fraction(counts, hot) -> float:
    """fraction of routed tokens whose expert is on the device under `hot`."""
    tot = sum(sum(c) for c in counts)
    on = sum(sum(c[e] for e in h) for c, h in zip(counts, hot))
    return on / tot if tot else 0.0
