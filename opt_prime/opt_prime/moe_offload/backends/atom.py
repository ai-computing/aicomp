#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Rebellions ATOM (RBLN SDK, rebel-compiler).

Experts run as `baked` graphs: one RBLN graph per expert with its weights compiled in as constants (the runtime
uploads graph inputs per call, so weights must not be inputs), only the experts a call uses are launched. Measured on
one RBLN-CA22 (rebel-compiler 0.11, Qwen3-30B-A3B expert shape, bf16): ~170 us per graph call + ~55 us per expert
of weights (~157 GB/s). A graph holding several experts with a dense gate streams all of them every call (8 experts:
0.62 ms), and selecting experts on the device by runtime ids (index_select) copies the weights (8 of 16: 2.4 ms),
so one graph per expert is the fastest layout for decode.

User-defined rebel.triton kernels do not pass the compiler's device graph conversion with rebel-compiler 0.11, so
this backend uses compiled torch graphs only.
"""
import os

import torch

from .base import register_backend
from .npu_graph import GraphNPUBackend


def visible_npu_id(physical: int) -> int:
    """the runtime's device index of physical card rbln<physical> (host group 0 in rsd_group), else the index itself."""
    path = "/sys/class/rebellions/rsd/rsd_group"
    try:
        import csv
        for r in csv.DictReader(open(path)):
            if r.get("DEVICE") == f"rbln{physical}" and r.get("GROUP") == "0":
                return int(r["NPU ID"])
    except (OSError, ValueError, KeyError):
        pass
    return physical


class _RblnGraph:
    def __init__(self, compiled, device):
        self.compiled, self.rt = compiled, compiled.create_runtime(device=device, tensor_type="pt")

    def __call__(self, *args):
        y = self.rt(*args)
        return y[0] if isinstance(y, (list, tuple)) else y


@register_backend("atom")
class AtomBackend(GraphNPUBackend):
    residency = "baked"
    overhead_s = 170e-6            # graph call
    bandwidth_Bps = 157e9          # expert weights streamed per call
    token_buckets = (1, 4, 16, 64)
    slot_buckets = (1,)

    def __init__(self, device=0, budget_bytes=None, emulate=None, **kw):
        super().__init__(device, budget_bytes, emulate, **kw)

    def device_name(self):
        return None if self.emulate else f"atom:{self.chip}"

    def default_capacity(self):
        return int(15.7 * (1 << 30) * 0.85)          # RBLN-CA22: 15.7 GiB, keep room for the runtime

    def version(self):
        try:
            import rebel
            return getattr(rebel, "__version__", "unknown")
        except ImportError:
            return "unknown"

    @staticmethod
    def _dtype_name(t):
        return str(t.dtype).replace("torch.", "")

    def _compile(self, module, sample_inputs):
        import rebel
        names = ["x", "g", "a", "b"][:len(sample_inputs)]
        info = [(n, list(t.shape), self._dtype_name(t)) for n, t in zip(names, sample_inputs)]
        return rebel.compile_from_torch(module, info)

    def _place(self, compiled):
        return _RblnGraph(compiled, visible_npu_id(self.chip))

    def _to_device(self, t):
        return t                                     # inputs are host tensors, uploaded by the runtime per call

    def _to_host(self, t):
        return t if isinstance(t, torch.Tensor) else torch.as_tensor(t)

    def _save(self, compiled, path):
        compiled.save(path + ".rbln.tmp")
        os.replace(path + ".rbln.tmp", path + ".rbln")

    def _load(self, path):
        if not os.path.exists(path + ".rbln"):
            raise FileNotFoundError(path)
        import rebel
        return rebel.RBLNCompiledModel(path + ".rbln")
