"""Rebellions ATOM+ adapter — fits the `rebel` SDK to the protocol.

The ATOM+ interface is a single line:

    rebel.compile_from_torch(stage_gm, input_info=[...], tensor_parallel_size=N)

That is, opt_prime's split GraphModule is accepted with no adapter code.
This file is long not because the interface is hard, but because it shapes
boundaries, placement and stats into the protocol.

Parallelism ownership — keep it this way
────────────────────────────────────────
    PP               owned by opt_prime (IR · split · stage wiring)
    NPU-internal TP  owned by the rebel binary (collectives compiled in)

`tensor_parallel_size=N` is a compile argument; the runtime then binds that
binary to N cards (`Runtime(compiled, device=[d0, d1])`).  opt_prime's
`prepare_tp_group` (ColwiseParallel/RowwiseParallel + DTensor) assumes GPU·NCCL
and is not used on this path — stacking both TPs would shard twice.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import torch
from torch.fx import GraphModule

from .base import BackendCaps, CompileStats


class RblnCompiledStage:
    """A stage wrapping one rebel.Runtime."""

    def __init__(self, name: str, compiled, devices: Sequence[int],
                 stats: CompileStats, make_async: bool = False):
        import rebel
        self.name = name
        self.devices = list(devices)
        self.stats = stats
        self.compiled = compiled
        dev = self.devices if len(self.devices) > 1 else self.devices[0]
        self.runtime = rebel.Runtime(compiled, device=dev, tensor_type="pt")
        self.async_runtime = (rebel.AsyncRuntime(compiled, device=dev,
                                                 tensor_type="pt")
                              if make_async else None)

    def run(self, *tensors: torch.Tensor) -> Any:
        return self.runtime.run(*tensors)

    def run_async(self, *tensors: torch.Tensor) -> Any | None:
        if self.async_runtime is None:
            return None
        return self.async_runtime.run(*tensors)

    def __repr__(self) -> str:
        return f"<RblnCompiledStage {self.name} on {self.devices}>"


class RblnBackend:
    name = "rbln"

    def __init__(self, verbose: bool = True, make_async: bool = False):
        import rebel
        self._rebel = rebel
        self.verbose = verbose
        self.make_async = make_async
        self._n = rebel.device_count()
        if verbose:
            print(f"[rbln] rebel-compiler {rebel.__version__} · "
                  f"{self._n} devices · npu(0)={rebel.get_npu_name(0)}",
                  flush=True)

    # ── Capability declaration ─────────────────────────────────────
    def caps(self) -> BackendCaps:
        tp = tuple(t for t in (1, 2, 4, 8) if t <= self._n)
        return BackendCaps(
            name="rbln",
            n_devices=self._n,
            tp_sizes=tp,
            supports_async=True,
            # Both fp32 and bf16 work.  fp32 is inherited, not forced — it was
            # just the PoC default.  bf16 halves memory and bandwidth (transfer
            # 265.1 → 132.5 KiB, argmax unchanged).
            forced_dtype=None,
            default_dtype=torch.float32,
            # A bool attn_mask is not taken as a stage input — keep bools off
            #   the boundary (the mask is built as float additive, C6).
            boundary_bool_ok=False,
            # Stage tensors move via host.
            device_to_device=False,
        )

    def device_count(self) -> int:
        return self._n

    def devices(self) -> list[str]:
        return [f"rbln{i}" for i in range(self._n)]

    # ── Compile ─────────────────────────────────────────────────────
    def compile(self, stage_gm: GraphModule,
                example_inputs: Sequence[torch.Tensor], *,
                name: str, devices: Sequence[int],
                tp_size: int = 1,
                input_names: Sequence[str] | None = None):
        from ..stage_io import dtype_name

        if input_names is None:
            input_names = [n.target for n in stage_gm.graph.nodes
                           if n.op == "placeholder"][:len(example_inputs)]
        if len(input_names) != len(example_inputs):
            raise RuntimeError(f"{name}: {len(input_names)} names != "
                               f"{len(example_inputs)} inputs")

        info = [(nm, list(t.shape), dtype_name(t.dtype))
                for nm, t in zip(input_names, example_inputs)]

        kwargs = {}
        if tp_size > 1:
            kwargs["tensor_parallel_size"] = tp_size

        if self.verbose:
            print(f"[rbln] compile {name} · inputs={len(info)} · "
                  f"devices={list(devices)} · tp={tp_size} …", flush=True)

        # ── Compile cache slot (integration item #8) ────────────────
        # ATOM+ takes about 1.25s per layer, so a cache is optional.  If
        # needed, add a compiled.save(path) / load branch here — the key must
        # include (graph fingerprint, tp_size, input shapes, dtype).
        t0 = time.time()
        compiled = self._rebel.compile_from_torch(stage_gm, input_info=info,
                                                  **kwargs)
        dt = time.time() - t0

        stats = CompileStats(compile_s=dt,
                             extra={"tp_size": tp_size,
                                    "devices": list(devices)})
        if self.verbose:
            print(f"[rbln]   {name} {stats}", flush=True)

        return RblnCompiledStage(name, compiled, devices, stats,
                                 make_async=self.make_async)

    # ── Diagnosis ───────────────────────────────────────────────────
    def unsupported_ops(self, stage_gm: GraphModule) -> list[str]:
        """For NPUs without an API to query op support in advance.

        Unsupported graphs then surface only as compile errors, so an empty
        list is returned.
        """
        return []
