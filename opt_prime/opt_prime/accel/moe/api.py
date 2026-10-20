#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""apply_moe_offload: plan the placement over the configured tiers and rewrite the model in place."""
import dataclasses
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch.nn as nn

from opt_prime.accel.moe import backends as B
from opt_prime.accel.spec.adapters import adapter_for, moe_layers
from opt_prime.accel.moe.config import MoEOffloadConfig
from opt_prime.accel.moe.layer import OffloadedLayer
from opt_prime.accel.moe.ops import OffloadedExperts, register_layer
from opt_prime.accel.moe.planner import device_fraction, plan

GiB = 1 << 30


@dataclass
class OffloadReport:
    moe_layers: int
    experts_per_layer: int
    spec: str
    tiers: List[dict] = field(default_factory=list)
    expected_accelerator_fraction: Optional[float] = None
    #: opt_prime.accel.platforms leases of the accelerator tiers (role "experts"); held until release()
    leases: list = field(default_factory=list, repr=False, compare=False)

    def release(self):
        """release the device leases (after the offloaded model is no longer used)."""
        for lease in self.leases:
            lease.release()

    @property
    def hot_experts(self) -> int:
        return sum(t["experts"] for t in self.tiers[:-1])

    def as_dict(self):
        d = asdict(dataclasses.replace(self, leases=[]))
        d["leases"] = [f"{l.device} gen {l.generation}" for l in self.leases]
        return d


def build_tiers(cfg: MoEOffloadConfig, leases: Optional[list] = None):
    """backend instances for cfg's tiers. With `leases` (a list), each tier's device is leased for role "experts"
    through opt_prime.accel.platforms first (exclusive for NPUs), and the leases are appended to it."""
    from opt_prime.accel.platforms import acquire
    tiers, names = [], []
    try:
        for ts in cfg.tier_specs():
            kw = dict(ts.options)
            if ts.budget_gb is not None: kw["budget_bytes"] = int(ts.budget_gb * GiB)
            if ts.device is not None: kw["device"] = ts.device
            tier = B.get_backend(ts.backend, **kw)
            name = f"{ts.backend}:{ts.device}" if ts.device is not None else ts.backend
            if name in names:                  # e.g. two cpu tiers: keep per-tier stats apart
                name = f"{name}#{len(names)}"
            dev = tier.device_name() if leases is not None else None
            if dev is not None:
                leases.append(acquire(dev, "experts", f"moe_offload tier {name}"))
            tiers.append(tier)
            names.append(name)
    except BaseException:
        for lease in leases or []:
            lease.release()
        raise
    return tiers, names


def apply_moe_offload(model: nn.Module, cfg: MoEOffloadConfig) -> OffloadReport:
    """Replace every MoE experts module of `model` (in place); returns what was placed where."""
    infos = moe_layers(model)
    if not infos:
        import transformers
        raise ValueError("no MoE experts module found (expects the transformers >= 5.0 experts interface: stacked "
                         f"gate_up_proj / down_proj and num_experts; installed transformers {transformers.__version__})")
    adapters = [adapter_for(i.module) for i in infos]
    specs = [a.spec(i.module) for a, i in zip(adapters, infos)]
    E = specs[0].num_experts
    leases = []
    tiers, names = build_tiers(cfg, leases)
    try:
        if cfg.placement is not None:
            placement = cfg.placement
        elif cfg.hot is not None:
            placement = [[sorted(h)] + [[] for _ in tiers[1:-1]] + [[e for e in range(E) if e not in set(h)]]
                         for h in cfg.hot]
        else:
            top_k = getattr(getattr(model, "config", None), "num_experts_per_tok", None) or 8
            placement = plan(cfg.profile, E, specs[0].expert_bytes, tiers, specs, cfg.per_layer_max, top_k)
        if len(placement) != len(infos):
            raise ValueError(f"placement for {len(placement)} layers, model has {len(infos)} MoE layers")
        for info, a, spec, pl in zip(infos, adapters, specs, placement):
            # accelerator tiers hold their subsets; the host tier holds every expert (local id == global id), so any
            # expert can be computed there and promoted from there by adaptive re-placement
            handles = [tier.load(a.weights(info.module, ids)) if ids else None for tier, ids in zip(tiers[:-1], pl[:-1])]
            handles.append(tiers[-1].load(a.weights(info.module, list(range(E)))))
            lid = register_layer(OffloadedLayer(E, tiers, handles, pl, names, cfg, adapter=a, source=info.module))
            parent, _, child = info.name.rpartition(".")
            parent_mod = model.get_submodule(parent) if parent else model
            setattr(parent_mod, child, OffloadedExperts(lid, E, [e for r in pl[:-1] for e in r]))
            a.wrap_parent(parent_mod)
        per_tier = [sum(len(pl[t]) for pl in placement) for t in range(len(tiers))]
        rep = OffloadReport(len(infos), E, specs[0].signature(),
                            [dict(tier=n, experts=c, gib=round(c * specs[0].expert_bytes / GiB, 3))
                             for n, c in zip(names, per_tier)])
        if cfg.profile is not None:
            rep.expected_accelerator_fraction = round(device_fraction(cfg.profile, placement, range(len(tiers) - 1)), 4)
        rep.leases = leases
    except BaseException:
        for lease in leases:
            lease.release()
        raise
    return rep
