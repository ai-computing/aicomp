#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Host CPU. Standard SwiGLU bf16 experts use the native AVX-512 BF16 kernel (csrc/host_experts.c, NUMA
first-touch placement); other experts run a per-expert loop over the model's own layout, bias and gate."""
import torch

from ..spec import ExpertWeights
from .base import ExpertBackend, Handle, register_backend
from .cuda import skeleton_with


def _linear(x, w, b, transposed):
    y = x @ w if transposed else torch.nn.functional.linear(x, w)
    return y + b if b is not None else y


@register_backend("cpu")
class CpuBackend(ExpertBackend):
    blocking = True
    #: per-layer-call cost of holding any expert of the layer (host sync + hand-off). The real cost also includes
    #: kernel-launch time exposed when the CPU loses its lead over the GPU (~0.9 ms measured with OptimusPrime
    #: on A100), but planning with that value concentrated too many layers on the GPU and was slower end to end
    #: (Qwen3-30B-A3B, 15 GB: 9.6 vs 11.5 tok/s); 0.1 ms gave the best measured placements. Tunable per tier.
    presence_s = 100e-6
    overhead_s = 150e-6
    bandwidth_Bps = 60e9              # measured: native kernel, 12 threads, 2x Xeon 4410Y

    def __init__(self, threads: int = 12, kernel: str = "auto", numa_place: bool = True, budget_bytes=None,
                 presence_s=None):
        self.threads, self.kernel, self.numa_place, self.budget = threads, kernel, numa_place, budget_bytes
        if presence_s is not None: self.presence_s = presence_s

    def capacity_bytes(self):
        return int(self.budget) if self.budget is not None else 1 << 62

    def device(self):
        return torch.device("cpu")

    def load(self, w: ExpertWeights):
        native = None
        if self.kernel in ("auto", "native") and w.spec.standard and w.spec.dtype == torch.bfloat16:
            from .. import native as nat
            native = nat.load()
        if native is not None:
            gu, dn = w.gate_up.contiguous(), w.down.contiguous()
            if self.numa_place:
                gu, dn = self._place(native, gu), self._place(native, dn)
            return Handle(w.spec, w.global_ids, {"native": native, "gu": gu, "dn": dn, "shell": w.module})
        if self.kernel == "native":
            pass  # unsupported spec: fall back silently to the torch path below
        return Handle(w.spec, w.global_ids, {"native": None, "module": skeleton_with(w, "cpu"), "shell": w.module})

    def weights_of(self, h, ids):
        """ExpertWeights for local ids of a loaded handle (the source for promoting experts to other tiers)."""
        st = h.state
        t = torch.tensor(list(ids), dtype=torch.long)
        if st["native"] is not None:
            return ExpertWeights(h.spec, [h.global_ids[i] for i in ids], st["gu"][t], st["dn"][t], module=st["shell"])
        m = st["module"]
        bias = getattr(m, "has_bias", False)
        return ExpertWeights(h.spec, [h.global_ids[i] for i in ids], m.gate_up_proj.data[t], m.down_proj.data[t],
                             m.gate_up_proj_bias.data[t] if bias else None, m.down_proj_bias.data[t] if bias else None,
                             module=st["shell"])

    can_update = True

    def update(self, h, slot, w):
        """Overwrite one slot in place (a CPU tier holding a subset of experts, e.g. the hot tier of a GPU-less host).
        In place keeps the native dispatcher's weight pointers valid; the caller routes the old and new expert to
        the host tier while this runs."""
        st = h.state
        if st["native"] is not None:
            st["gu"][slot].copy_(w.gate_up[0])
            st["dn"][slot].copy_(w.down[0])
        else:
            m = st["module"]
            m.gate_up_proj.data[slot].copy_(w.gate_up[0])
            m.down_proj.data[slot].copy_(w.down[0])
            if getattr(m, "has_bias", False) and w.gate_up_bias is not None:
                m.gate_up_proj_bias.data[slot].copy_(w.gate_up_bias[0])
                m.down_proj_bias.data[slot].copy_(w.down_bias[0])
        h.global_ids[slot] = w.global_ids[0]

    def _place(self, lib, t):
        dst = torch.empty_like(t)
        lib.moe_host_place(dst.data_ptr(), t.data_ptr(), t.numel() // t.shape[-1], t.shape[-1], self.threads)
        return dst

    def native_tier(self, h):
        st = h.state
        if st["native"] is None:
            return None
        if "native_tier" not in st:
            from ..native import fn_addr, load_dispatch
            d = load_dispatch()
            if d is None:
                st["native_tier"] = None
            else:
                n, two_f, _ = st["gu"].shape
                ctx = d.moe_cpu_ctx_new(fn_addr(st["native"], "moe_host_forward"), st["gu"].data_ptr(),
                                        st["dn"].data_ptr(), n, two_f // 2, self.threads)
                st["native_tier"] = (fn_addr(d, "moe_tier_cpu"), ctx)
        return st["native_tier"]

    def run_pairs(self, h, x, pairs):
        st = h.state
        if st["native"] is None:
            return super().run_pairs(h, x, pairs)
        if torch.get_num_threads() != self.threads:
            torch.set_num_threads(self.threads)
        k = max(len(p) for p in pairs)
        idx = torch.tensor([[e for e, _ in p] + [p[0][0]] * (k - len(p)) for p in pairs], dtype=torch.int64)
        w = torch.tensor([[a for _, a in p] + [0.0] * (k - len(p)) for p in pairs], dtype=torch.float32)
        x = x.contiguous()
        out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32)
        n, two_f, H = st["gu"].shape
        st["native"].moe_host_forward(x.shape[0], k, H, two_f // 2, n, x.data_ptr(), idx.data_ptr(), w.data_ptr(),
                                      st["gu"].data_ptr(), st["dn"].data_ptr(), out.data_ptr(), self.threads)
        return out

    def run(self, h, hidden, local_index, weights):
        if torch.get_num_threads() != self.threads:
            torch.set_num_threads(self.threads)
        st = h.state
        if st["native"] is not None:
            x = hidden.contiguous()
            idx = local_index.to(torch.int64).contiguous()
            w = weights.float().contiguous()
            out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32)
            n, two_f, H = st["gu"].shape
            st["native"].moe_host_forward(x.shape[0], idx.shape[1], H, two_f // 2, n, x.data_ptr(), idx.data_ptr(),
                                          w.data_ptr(), st["gu"].data_ptr(), st["dn"].data_ptr(), out.data_ptr(),
                                          self.threads)
            return out.to(hidden.dtype)
        m = st["module"]
        out = torch.zeros(hidden.shape[0], hidden.shape[1], dtype=torch.float32)
        flat = local_index.reshape(-1)
        rows = torch.arange(local_index.shape[0]).repeat_interleave(local_index.shape[1])
        wf = weights.reshape(-1)
        nz = wf != 0
        flat, rows, wf = flat[nz], rows[nz], wf[nz]
        for e in torch.unique(flat).tolist():
            sel = flat == e
            r = rows[sel]
            gb = m.gate_up_proj_bias[e] if getattr(m, "has_bias", False) else None
            db = m.down_proj_bias[e] if getattr(m, "has_bias", False) else None
            tr = getattr(m, "is_transposed", False)
            y = _linear(m._apply_gate(_linear(hidden[r], m.gate_up_proj[e], gb, tr)), m.down_proj[e], db, tr)
            out.index_add_(0, r, (y * wf[sel, None]).float())
        return out.to(hidden.dtype)
