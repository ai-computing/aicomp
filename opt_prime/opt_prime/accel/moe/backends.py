#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""MoE expert tier contract (ExpertBackend, Handle) and registry. A backend instance is bound to one device ("tier");
the implementations are per vendor in opt_prime.accel.platforms.<vendor>.experts. See DESIGN.md."""
from dataclasses import dataclass
from typing import Any, Dict, Type

import torch

from opt_prime.accel.spec.experts import ExpertSpec, ExpertWeights

_REGISTRY: Dict[str, Type["ExpertBackend"]] = {}


def register_backend(name):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def _load():
    """the vendor tiers live in opt_prime.accel.platforms.<vendor>.experts and register on import"""
    from opt_prime.accel.platforms import load_expert_backends
    load_expert_backends()


def registered():
    _load()
    return sorted(_REGISTRY)


def get_backend(name: str, **kw) -> "ExpertBackend":
    _load()
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; registered: {registered()}")
    return _REGISTRY[name](**kw)


@dataclass
class Handle:
    """experts loaded on a backend: local ids 0..n-1 correspond to `global_ids`."""
    spec: ExpertSpec
    global_ids: list
    state: Any


class ExpertBackend:
    name = "base"
    #: True when run() blocks the calling thread for the whole computation (host CPU, remote/NPU servers);
    #: the layer runs such tiers in worker threads. False for asynchronous devices (CUDA streams).
    blocking = True
    #: True when run() costs per routed pair rather than per expert, so pairs it does not own should be removed
    #: (compacted) instead of passed with weight 0.
    compact_pairs = False

    def supports(self, spec: ExpertSpec) -> bool:
        return True

    def device_name(self):
        """'<kind>:<index>' of the device this tier drives, for opt_prime.accel.platforms leases; None = nothing to
        lease (host memory, emulation)."""
        return None

    def capacity_bytes(self) -> int:
        raise NotImplementedError

    def cost(self, spec: ExpertSpec, n_experts: int, n_pairs: int) -> float:
        """expected seconds for one call touching n_experts distinct experts over n_pairs routed pairs."""
        return self.overhead_s + n_experts * spec.expert_bytes / self.bandwidth_Bps

    overhead_s = 50e-6
    bandwidth_Bps = 50e9
    #: fixed cost per layer call as soon as the tier holds any expert of the layer: for blocking tiers the layer
    #: must wait for the routing on the host and hand work to a thread (~0.1 ms), even when no pair goes there
    presence_s = 0.0

    def device(self) -> torch.device:
        raise NotImplementedError

    def load(self, w: ExpertWeights) -> Handle:
        raise NotImplementedError

    #: whether update() is implemented (adaptive re-placement swaps experts in and out of this tier)
    can_update = False

    def update(self, h: Handle, slot: int, w: ExpertWeights) -> None:
        """replace the expert in local `slot` with the single expert in `w` (adaptive re-placement)."""
        raise NotImplementedError

    def native_tier(self, h: Handle, bf16_weights: bool = False):
        """optional: (C function address, context address) running this tier for the native dispatcher
        (csrc/dispatch.c, `moe_tier_fn` signature), or None. bf16_weights: the routing weights are bf16, so each
        weighted expert output is rounded to bf16 (as torch multiplies in the weights' dtype). Layers whose blocking tiers all provide one run decode
        steps without the Python GIL."""
        return None

    def run_pairs(self, h: Handle, x: torch.Tensor, pairs, bf16_weights: bool = False) -> torch.Tensor:
        """decode fast path for blocking tiers: x [R,H] on the host, pairs[r] = [(local_id, weight), ...] (non-empty,
        weights != 0, held as Python floats; bf16_weights: they came from bf16 routing weights) -> float32 [R,H]. The default packs the lists into tensors and calls run(); backends whose
        calls cost per Python op (servers, ctypes kernels) override it."""
        k = max(len(p) for p in pairs)
        idx = torch.tensor([[e for e, _ in p] + [p[0][0]] * (k - len(p)) for p in pairs], dtype=torch.long)
        w = torch.tensor([[a for _, a in p] + [0.0] * (k - len(p)) for p in pairs], dtype=torch.float32)
        return self.run(h, x, idx, w.to(torch.bfloat16) if bf16_weights else w).float()

    def run(self, h: Handle, hidden: torch.Tensor, local_index: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """partial output [T,H] in hidden's dtype on this backend's device. local_index/weights: [T,k]; weight-0
        pairs contribute nothing and may be skipped."""
        raise NotImplementedError
