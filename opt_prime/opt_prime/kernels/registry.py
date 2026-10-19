#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Provider contract and registry."""
from typing import Dict, List, Optional, Type

from ..model.blocks import AttentionSpec, DenseMLPSpec, MoESpec, NormSpec, RouterSpec
from ..moe_offload.spec import ExpertSpec

#: block kinds providers declare (the spec class each kind is described by)
BLOCK_KINDS = {
    "attention": AttentionSpec,
    "norm": NormSpec,
    "mlp": DenseMLPSpec,
    "router": RouterSpec,
    "experts": ExpertSpec,
    "moe": MoESpec,           # router + experts + shared experts as one block
}


def block_kind(block) -> str:
    for kind, cls in BLOCK_KINDS.items():
        if isinstance(block, cls):
            return kind
    raise TypeError(f"not a block spec: {type(block).__name__}")


class KernelProvider:
    """One implementation family of some block kinds on one device kind.

    supports(block)        -> bool   the provider can run this exact block (shape, dtype, variant)
    cost(block, tokens)    -> float  expected seconds per call (planner input); None if unknown
    check()                -> dict   optional numerical self-check against the torch reference
    """
    name = "base"
    device_kind = ""
    block_kinds: frozenset = frozenset()

    def supports(self, block) -> bool:
        return block_kind(block) in self.block_kinds

    def cost(self, block, tokens: int = 1) -> Optional[float]:
        return None

    def __repr__(self):
        return f"<provider {self.name} on {self.device_kind}: {sorted(self.block_kinds)}>"


_PROVIDERS: Dict[str, KernelProvider] = {}


def register_provider(provider):
    """register a provider instance (or a class, instantiated without arguments)."""
    p = provider() if isinstance(provider, type) else provider
    _PROVIDERS[f"{p.name}@{p.device_kind}"] = p
    return provider


def registered_providers() -> List[KernelProvider]:
    return list(_PROVIDERS.values())


def providers_for(block, device_kind: Optional[str] = None, tokens: int = 1) -> List[KernelProvider]:
    """providers that support `block` (on `device_kind` if given), cheapest first (unknown cost last)."""
    out = [p for p in _PROVIDERS.values()
           if (device_kind is None or p.device_kind == device_kind) and p.supports(block)]
    return sorted(out, key=lambda p: (p.cost(block, tokens) is None, p.cost(block, tokens) or 0.0, p.name))
