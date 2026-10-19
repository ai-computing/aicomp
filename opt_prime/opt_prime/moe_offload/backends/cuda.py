#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""GPUs through PyTorch's device API: NVIDIA CUDA, AMD ROCm (HIP builds expose the same `torch.cuda` API) and
Intel XPU. Runs the model's own experts class on the loaded subset, so any gate/bias/layout works. Few routed pairs
(decode) use transformers' batched_mm (gather + bmm), more use the model's configured implementation (grouped_mm by
default). Registered as "gpu" with the aliases "cuda" and "xpu"; a new GPU type that PyTorch supports needs only
an entry in _MEM_INFO."""
import copy

import torch
import torch.nn as nn

from ..spec import ExpertWeights
from .base import ExpertBackend, Handle, register_backend

try:
    from transformers.integrations.moe import batched_mm_experts_forward as _batched_mm
except Exception:  # pragma: no cover
    _batched_mm = None


def skeleton_with(w: ExpertWeights, device) -> nn.Module:
    """shallow clone of the model's experts module holding w's subset on `device`."""
    m = w.module
    c = copy.copy(m)
    c._parameters, c._buffers, c._modules = dict(m._parameters), dict(m._buffers), dict(m._modules)
    sub = {"gate_up_proj": w.gate_up, "down_proj": w.down, "gate_up_proj_bias": w.gate_up_bias,
           "down_proj_bias": w.down_bias}
    for n, p in m.named_parameters(recurse=False):
        data = sub[n] if sub.get(n) is not None else p.data
        c._parameters[n] = nn.Parameter(data.to(device), requires_grad=False)
    for n, b in m.named_buffers(recurse=False):
        c._buffers[n] = b.to(device)
    for n, s in m.named_children():
        c._modules[n] = copy.deepcopy(s).to(device)
    c.num_experts = len(w.global_ids)
    return c


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _mem_get_info(device):
    """(free, total) bytes on a torch device of any GPU type."""
    dev = torch.device(device) if device is not None else None
    kind = dev.type if dev is not None else ("cuda" if torch.cuda.is_available() else "xpu")
    mod = getattr(torch, kind)
    return mod.mem_get_info(dev if dev is not None and dev.index is not None else mod.current_device())


@register_backend("gpu")
class GpuBackend(ExpertBackend):
    blocking = False
    overhead_s = 60e-6
    bandwidth_Bps = 1.2e12            # A100 HBM, effective for gather+bmm

    def __init__(self, device=None, budget_bytes=None, small_pairs: int = 256):
        self._device = torch.device(device) if device is not None else None   # None: the hidden states' device
        self.budget = budget_bytes
        self.small_pairs = small_pairs

    def device_name(self):
        d = self._device if self._device is not None else torch.device("cuda", torch.cuda.current_device())
        return f"cuda:{d.index or 0}"

    def capacity_bytes(self):
        if self.budget is not None:
            return int(self.budget)
        free, _ = _mem_get_info(self._device)
        return int(free * 0.8)

    def device(self):
        return self._device

    def load(self, w):
        return Handle(w.spec, w.global_ids, {"weights": w, "module": None, "device": None})

    can_update = True

    def update(self, h, slot, w):
        """Overwrite one slot. Called from the adaptation worker: copies run on a side stream that first waits for
        the work already queued on the device, and the call returns when the copy has landed. The caller routes the
        old and the new expert elsewhere (host) while this runs, so no kernel reads a half-written slot."""
        st = h.state
        src = st["weights"]
        for name, new in (("gate_up", w.gate_up), ("down", w.down), ("gate_up_bias", w.gate_up_bias),
                          ("down_bias", w.down_bias)):
            if new is not None and getattr(src, name) is not None:
                getattr(src, name)[slot].copy_(new[0])             # host copy (re-materialization source)
        m = st["module"]
        if m is not None:                                        # resident copy on the GPU
            dev = st["device"]
            if "stream" not in st:
                st["stream"] = torch.cuda.Stream(dev) if dev.type == "cuda" else None
            ev = None
            if st["stream"] is not None:
                ev = torch.cuda.Event(); ev.record(torch.cuda.default_stream(dev))
            ctx = torch.cuda.stream(st["stream"]) if st["stream"] is not None else _null()
            with ctx:
                if ev is not None: st["stream"].wait_event(ev)
                for pname, new in (("gate_up_proj", w.gate_up), ("down_proj", w.down),
                                   ("gate_up_proj_bias", w.gate_up_bias), ("down_proj_bias", w.down_bias)):
                    if new is not None and getattr(m, pname, None) is not None:
                        getattr(m, pname).data[slot].copy_(new[0], non_blocking=True)
            if st["stream"] is not None: st["stream"].synchronize()
        h.global_ids[slot] = w.global_ids[0]

    @staticmethod
    def _bmm(m, hidden, idx, w):
        """gather + bmm over the routed pairs (transformers' batched_mm without its host-syncing zero check)."""
        T, k = idx.shape
        x = hidden.unsqueeze(1).expand(T, k, hidden.shape[-1]).reshape(T * k, -1)
        e = idx.reshape(-1)
        gu, dn = m.gate_up_proj[e], m.down_proj[e]
        tr = getattr(m, "is_transposed", False)
        y = torch.bmm(x.unsqueeze(1), gu).squeeze(1) if tr else torch.bmm(gu, x.unsqueeze(-1)).squeeze(-1)
        if getattr(m, "has_bias", False): y = y + m.gate_up_proj_bias[e]
        a = m._apply_gate(y)
        y = torch.bmm(a.unsqueeze(1), dn).squeeze(1) if tr else torch.bmm(dn, a.unsqueeze(-1)).squeeze(-1)
        if getattr(m, "has_bias", False): y = y + m.down_proj_bias[e]
        y = y * w.reshape(-1, 1).to(y.dtype)
        return y.view(T, k, -1).sum(dim=1).to(hidden.dtype)

    def run(self, h, hidden, local_index, weights):
        st = h.state
        if st["device"] != hidden.device:          # lazy: each pipeline stage materializes on its own GPU
            st["module"] = skeleton_with(st["weights"], hidden.device)
            st["device"] = hidden.device
        m = st["module"]
        if local_index.numel() <= self.small_pairs and hasattr(m, "_apply_gate"):
            return self._bmm(m, hidden, local_index, weights)
        return m(hidden, local_index, weights)


register_backend("cuda")(type("CudaBackend", (GpuBackend,), {}))
register_backend("xpu")(type("XpuBackend", (GpuBackend,), {"bandwidth_Bps": 0.5e12}))
