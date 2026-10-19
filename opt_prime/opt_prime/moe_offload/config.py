#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""User-facing configuration and run statistics."""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class TierSpec:
    """One device holding experts: a backend name, its device, a memory budget for expert weights and backend
    options. budget_gb=None means the backend's own capacity (unbounded for the host CPU)."""
    backend: str
    device: Any = None
    budget_gb: Optional[float] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OffloadStats:
    """Routed (token, expert) pairs per tier and host-side timings, accumulated over calls. Layers that never wait
    for the host keep their counts on the device; they are folded in when the stats are read (flush())."""
    tier_pairs: Dict[str, int] = field(default_factory=dict)
    host_pairs: int = 0         # pairs on blocking tiers (host CPU, NPU servers)
    device_pairs: int = 0       # pairs on asynchronous device tiers (GPUs)
    calls: int = 0
    _sources: list = field(default_factory=list, repr=False)

    def flush(self):
        for src in self._sources:
            src.flush_stats(self)
        return self
    swaps: int = 0
    host_s: float = 0.0         # time blocking tiers spent computing
    wait_s: float = 0.0         # time the caller waited for them after launching the device tiers
    #: per-phase host seconds of the layer calls when `MoEOffloadConfig.profile_phases` (or MOE_OFFLOAD_PROFILE=1):
    #: outside (between layer calls: rest of the model), sync (routing copy), split, launch (device tiers),
    #: inline / wait (blocking tiers), tier:<name> (time inside each blocking tier's run)
    phases: Dict[str, float] = field(default_factory=dict)

    @property
    def device_fraction(self) -> float:
        self.flush()
        n = self.device_pairs + self.host_pairs
        return self.device_pairs / n if n else 0.0

    def reset(self):
        self.flush()
        self.tier_pairs.clear()
        self.host_pairs = self.device_pairs = self.calls = self.swaps = 0
        self.host_s = self.wait_s = 0.0
        self.phases.clear()


@dataclass
class MoEOffloadConfig:
    """Where experts live and how tiers run.

    tiers           list of TierSpec, fastest first; the last tier must be able to hold every remaining expert
                    (normally the host CPU). Default: [cuda(gpu_budget_gb), cpu].
    profile         routing counts [layers][experts] (ExpertProfiler.as_lists()); the planner fills the
                    accelerator tiers with the hottest experts and balances each layer across them by cost.
                    Without a profile every layer gets the same number of experts 0..n-1.
    placement       explicit per-layer, per-tier expert ids (overrides the planner)
    hot             legacy two-tier form: hot expert ids per layer for tiers[0]; the rest go to tiers[-1]
    per_layer_max   cap on accelerator-resident experts per layer

    Shorthands for the default tiers: gpu_budget_gb, device_small_pairs (cuda: up to this many routed pairs use
    batched_mm), cpu_threads (12 was fastest on a 24-core host; start with OMP_WAIT_POLICY=PASSIVE),
    host_backend ("auto" | "native" | "torch"), numa_place.

    Adaptive re-placement (adapt_every > 0): each layer keeps an EMA of how often every expert is routed and
    periodically swaps the hottest host experts into accelerator tiers in place of their coldest ones (tiers whose
    backend can_update). Corrects profiles that do not match the served traffic.
    """
    tiers: Optional[List[TierSpec]] = None
    profile: Optional[Sequence[Sequence[float]]] = None
    placement: Optional[List[List[List[int]]]] = None
    hot: Optional[List[List[int]]] = None
    per_layer_max: Optional[int] = None
    gpu_budget_gb: float = 0.0
    profile_phases: bool = bool(int(os.environ.get("MOE_OFFLOAD_PROFILE", "0")))
    native_dispatch: bool = bool(int(os.environ.get("MOE_OFFLOAD_NATIVE", "1")))     # decode fast path in C (csrc/dispatch.c) when every blocking tier supports it
    fast_rows: int = 8               # blocking tiers get Python-list pairs (decode fast path) up to this many rows
    device_small_pairs: int = 256
    cpu_threads: int = 12
    host_backend: str = "auto"
    numa_place: bool = True
    adapt_every: int = 0          # adaptive re-placement every N calls of a layer (0: static placement)
    adapt_swaps: int = 2          # max experts promoted per accelerator tier and layer per adaptation
    adapt_decay: float = 0.995    # EMA decay of the per-expert routing counts (long memory: stable placements)
    adapt_margin: float = 2.0     # promote only if the host expert is this much hotter than the one it replaces
    stats: OffloadStats = field(default_factory=OffloadStats)

    def tier_specs(self) -> List[TierSpec]:
        if self.tiers is not None:
            return self.tiers
        import torch
        host = TierSpec("cpu", None, None, {"threads": self.cpu_threads, "kernel": self.host_backend,
                                            "numa_place": self.numa_place})
        if not torch.cuda.is_available():
            # no GPU (e.g. a CPU-only NPU host): the "hot" tier is a second host tier, so placements, hot lists
            # and adaptive re-placement keep their two-tier shape and results
            return [TierSpec("cpu", None, self.gpu_budget_gb, {"threads": self.cpu_threads, "kernel": "torch"}), host]
        return [TierSpec("cuda", None, self.gpu_budget_gb, {"small_pairs": self.device_small_pairs}), host]
