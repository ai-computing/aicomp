#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""FuriosaAI RNGD NPUs.

Default kernel: furiosa-torch compiles the generic SlotFFN graph (backends/npu_graph.py) per (shape, token bucket,
slot count); expert weights stay resident as device tensors and the host picks the slots per call. Measured on
Solar-Open shapes: K=8 experts in 0.67 ms (374 GB/s) plus ~0.5 ms round trip (npu-offload p0).

Shapes the per-shape furiosa-opt kernels accept (`npu_kernels.rngd.plan`) can instead use the `rngd-opt` backend
(~1.0 TB/s, one token per launch); the planner picks per layer by cost.
"""
import os
import torch

from opt_prime.accel.moe.backends import register_backend
from opt_prime.accel.platforms.npu_graph import GraphNPUBackend

@register_backend("rngd")
class RngdBackend(GraphNPUBackend):
    residency = "inputs"           # device tensors as graph inputs: no weight traffic per call
    # measured on RNGD (examples/moe_offload_rngd_check.py, Qwen3-30B-A3B shape, T=1): k=1 1.2-1.3 ms,
    # k=8 1.5-1.6 ms -> ~1.2 ms per graph call incl. host round trip, ~0.2 TB/s
    overhead_s = 1.2e-3
    bandwidth_Bps = 200e9
    _setup_done = False

    def __init__(self, device=0, budget_bytes=None, emulate=None, **kw):
        super().__init__(device, budget_bytes, emulate, **kw)
        self.dev = f"rngd:{device}"

    def default_capacity(self):
        return int(47.5 * (1 << 30) * 0.9)

    @classmethod
    def _setup(cls):
        if cls._setup_done:
            return
        import furiosa.torch  # noqa: F401  (registers the rngd device)
        from furiosa.torch import native_device as nd
        nd.set_fusion(8)                              # one program spans the chip's 8 PEs
        nd.set_mesh_kind(nd.MeshKind.Mesh(2))
        cls._setup_done = True

    def _compile(self, module, sample_inputs):
        self._setup()
        from furiosa.torch.custom_ops import CompileModule
        return CompileModule.from_module(module, tuple(sample_inputs))      # no device needed

    def _place(self, compiled):
        compiled.to(self.dev)
        return compiled

    def _to_device(self, t):
        self._setup()
        return t.to(self.dev)

    def version(self):
        try:
            from importlib.metadata import version
            return version("furiosa-torch")
        except Exception:
            return "unknown"

    def _save(self, compiled, path):
        """EDF blob (native serialize) + the exported weight spec (picklable): reload without recompiling."""
        with open(path + ".edf.tmp", "wb") as f:
            f.write(compiled.edf.serialize())
        os.replace(path + ".edf.tmp", path + ".edf")
        torch.save(compiled.module_weight, path + ".w.tmp")
        os.replace(path + ".w.tmp", path + ".w")

    def _load(self, path):
        if not (os.path.exists(path + ".edf") and os.path.exists(path + ".w")):
            raise FileNotFoundError(path)
        self._setup()
        from furiosa.native_torch import ir
        from furiosa.torch.custom_ops import CompileModule
        from furiosa.torch.custom_ops.edf import EdfModule
        with open(path + ".edf", "rb") as f:
            edf = ir.Edf.deserialize(f.read())
        return CompileModule(EdfModule(edf), torch.load(path + ".w", weights_only=False))
