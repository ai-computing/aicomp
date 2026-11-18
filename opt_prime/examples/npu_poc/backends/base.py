"""Backend protocol — the boundary that keeps the PoC core unaware of which NPU it runs on.

This file is engine integration item #1.  On integration this protocol moves
into the engine, while the adapters (`rbln.py` · `rngd.py`) stay outside it.

It promotes the protocol from `rngd_poc/npu_backend.py` to the canonical one and
adds what PP×TP needs.  The existing `compile(..., devices=, tp_size=)`
signature is unchanged.
Note on porting the RNGD adapter: the furiosa SDK 2026.4.0 changed its API
(device name `furiosa:k`, `CompileModule.from_exported`), so the SDK version
compatibility layer in `npu_backend.py` must come along.

The one addition is a capability declaration (`BackendCaps`).  Things the PoC
core used to hard-code — the TP limit, the forced dtype, whether a bool may sit
on a stage boundary — are now stated by the backend itself.  The moment the core
grows an `if backend == "rngd"`, integration gets harder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import torch
from torch.fx import GraphModule


# ─────────────────────────────────────────────────────────────────────
# Capability declaration — replaces branching in the core
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BackendCaps:
    """Constraints stated by the backend.  The core builds the grid from these alone."""

    name: str
    n_devices: int                      # number of devices available now
    tp_sizes: tuple[int, ...]           # allowed tp_size values
    supports_async: bool                # whether micro-batch pipelining is supported

    # dtype policy (integration item #6)
    #   If forced_dtype is not None, no other dtype works.
    #   RNGD is used with bf16.  ATOM+ works with both (the default is an
    #   inherited fp32).
    forced_dtype: torch.dtype | None
    default_dtype: torch.dtype

    # Boundary tensor policy (integration item #1)
    #   False means the backend does not take a bool tensor (e.g. a bool
    #   attn_mask) as a stage input.  The core then checks the boundaries up
    #   front, and the boundary mask is built as float additive instead (C6).
    boundary_bool_ok: bool = True

    # Inter-stage transfer path.  This PoC moves stage tensors via host on both
    # backends.
    device_to_device: bool = False

    def max_tp(self) -> int:
        return max(self.tp_sizes)

    def check_grid(self, pp: int, tp: int) -> str | None:
        """Whether the (PP,TP) grid is feasible on this backend.  Returns the reason if not."""
        if tp not in self.tp_sizes:
            return (f"tp_size={tp} is not allowed by {self.name} "
                    f"(allowed: {', '.join(map(str, self.tp_sizes))})")
        if pp * tp > self.n_devices:
            return (f"PP={pp} × TP={tp} needs {pp * tp} devices but "
                    f"only {self.n_devices} are available")
        return None


@dataclass
class CompileStats:
    """Cost breakdown of one compilation.

    RNGD separates export, decomposition and compile, while ATOM+ does it in one
    call, so on ATOM+ only compile_s is filled.  Compare only totals across
    backends.
    """

    export_s: float = 0.0
    decompose_s: float = 0.0
    compile_s: float = 0.0
    n_nodes_before: int = 0
    n_nodes_after: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def total_s(self) -> float:
        return self.export_s + self.decompose_s + self.compile_s

    def __str__(self) -> str:
        parts = [f"compile={self.compile_s:.2f}s"]
        if self.export_s:
            parts.insert(0, f"export={self.export_s:.2f}s")
        if self.decompose_s:
            parts.insert(-1, f"decompose={self.decompose_s:.2f}s")
        parts.append(f"total={self.total_s:.2f}s")
        if self.n_nodes_before:
            parts.append(f"nodes {self.n_nodes_before}->{self.n_nodes_after}")
        return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Protocol
# ─────────────────────────────────────────────────────────────────────

@runtime_checkable
class CompiledStage(Protocol):
    """One compiled PP stage.  Takes tensors, returns tensors."""

    name: str
    stats: CompileStats

    def run(self, *tensors: torch.Tensor) -> Any: ...

    def run_async(self, *tensors: torch.Tensor) -> Any | None:
        """Asynchronous submit.  Backends without support return None."""
        ...


@runtime_checkable
class NpuBackend(Protocol):
    name: str

    def caps(self) -> BackendCaps: ...

    def device_count(self) -> int: ...

    def devices(self) -> list[str]: ...

    def compile(self, stage_gm: GraphModule,
                example_inputs: Sequence[torch.Tensor], *,
                name: str, devices: Sequence[int],
                tp_size: int = 1,
                input_names: Sequence[str] | None = None) -> CompiledStage:
        """Compile one stage onto `devices`.

        `devices` is the list of devices this stage uses, and its length is the
        tp_size — the signature makes explicit which side owns which axis: the
        core handles PP, the backend handles TP.

        `input_names` is for backends whose input description is a list of
        (name, shape, dtype) triples (ATOM+).  Once the stage is wrapped by the
        scalar-freezing wrapper the backend cannot read the placeholders
        directly, so the core passes them in.  RNGD ignores it.
        """
        ...

    def unsupported_ops(self, stage_gm: GraphModule) -> list[str]:
        """Pre-compile diagnosis.  Backends that cannot check support in advance return []."""
        ...


# ─────────────────────────────────────────────────────────────────────

def get_backend(kind: str, **kw) -> NpuBackend:
    """Select an adapter by name.  This is the only backend knowledge the core has."""
    if kind == "rbln":
        from .rbln import RblnBackend
        return RblnBackend(**kw)
    if kind == "rngd":
        from .rngd import RngdBackend
        return RngdBackend(**kw)
    raise SystemExit(f"unknown backend: {kind!r} (available: rbln, rngd)")
