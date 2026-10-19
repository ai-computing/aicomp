#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""One offloaded MoE layer over N tiers: route each pair to the tier owning its expert, run the tiers concurrently
(asynchronous device tiers on the calling thread, blocking tiers - host CPU, NPU servers - in workers), sum."""
import threading
import time

_NATIVE = {}                 # process-wide native dispatcher (layer calls are sequential): {"lib", "disp"}
_LAST_END = [None]          # end of the previous layer call (phase profiling: time spent in the rest of the model)
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch

from .config import MoEOffloadConfig

def pin_for_cuda(t):
    """page-locked copy for async CUDA transfers. When another device extension (e.g. furiosa-torch) is the
    default accelerator, plain pin_memory() targets it, so name CUDA explicitly in that case."""
    try:
        p = t.pin_memory()
        if p.is_pinned():
            return p
    except Exception:
        pass
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return t.pin_memory("cuda")


_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="moe-offload-tier")
_ADAPT = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moe-offload-adapt")   # off the decode path


class OffloadedLayer:
    def __init__(self, num_experts: int, tiers: List, handles: List, placement: List[List[int]], names: List[str],
                 cfg: MoEOffloadConfig, adapter=None, source=None):
        self.E, self.tiers, self.handles, self.names, self.cfg = num_experts, tiers, handles, names, cfg
        self.adapter, self.host_weights = adapter, handles[-1]
        owner = torch.full((num_experts,), -1, dtype=torch.long)
        local = torch.zeros(num_experts, dtype=torch.long)
        host = len(tiers) - 1
        for t, ids in enumerate(placement):
            for i, e in enumerate(ids):
                owner[e], local[e] = t, (e if t == host else i)      # host tier holds every expert by global id
        self.counts = torch.zeros(num_experts, dtype=torch.float64)
        self.calls = 0
        self._lock = threading.Lock()      # guards owner/local tables against the adaptation worker
        self._adapting = None
        self._has, self._blocking_any, self._lists, self._native = None, None, None, None
        self._pin, self._dev_counts, self._counts_d = {}, None, None
        # keep the original module only if the host tier cannot hand out expert weights itself
        self._source = None if hasattr(tiers[-1], "weights_of") else source
        assert (owner >= 0).all(), "every expert must be placed on some tier"
        self.owner_c, self.local_c = owner, local
        self._dev_tables = {}

    def _tables(self, device):
        if device not in self._dev_tables:
            self._dev_tables[device] = (self.owner_c.to(device), self.local_c.to(device))
        return self._dev_tables[device]

    def _snapshot(self, device):
        """owner/local tables (host and device copies), which tiers own experts, and whether any expert is on a
        blocking tier - one consistent view per call."""
        with self._lock:
            if self._has is None:
                self._has = [bool((self.owner_c == t).any()) for t in range(len(self.tiers))]
                self._blocking_any = any(self._has[t] and self.tiers[t].blocking for t in range(len(self.tiers)))
                self._lists = (self.owner_c.tolist(), self.local_c.tolist(),      # for the decode fast path
                               self.owner_c.to(torch.int32).contiguous(), self.local_c.to(torch.int32).contiguous())
                self._native = None
            return (self.owner_c.clone(), self.local_c.clone(), self._tables(device), self._has, self._blocking_any,
                    self._lists)

    def _pinned(self, name, like):
        key = (name, tuple(like.shape), like.dtype)
        b = self._pin.get(key)
        if b is None:
            b = torch.empty(like.shape, dtype=like.dtype)
            if like.is_cuda:
                b = pin_for_cuda(b)
            self._pin[key] = b
        return b

    def _host_buf(self, name, shape, pinned):
        key = (name, shape)
        b = self._pin.get(key)
        if b is None:
            b = torch.zeros(shape, dtype=torch.float32)
            self._pin[key] = b = pin_for_cuda(b) if pinned else b
        return b

    def flush_stats(self, st):
        if self._dev_counts is not None:
            for t, n in enumerate(self._dev_counts.tolist()):
                if n:
                    st.tier_pairs[self.names[t]] = st.tier_pairs.get(self.names[t], 0) + int(n)
                    if self.tiers[t].blocking: st.host_pairs += int(n)
                    else: st.device_pairs += int(n)
            self._dev_counts.zero_()

    def forward(self, hidden, top_k_index, top_k_weights):
        cfg, st = self.cfg, self.cfg.stats
        prof = st.phases if cfg.profile_phases else None
        if prof is not None:
            tp = time.perf_counter()
            if _LAST_END[0] is not None: _add(prof, "outside", tp - _LAST_END[0])
        dev = hidden.device
        owner_c, local_c, (own_d, local_d), has, blocking_any, lists = self._snapshot(dev)   # consistent per call
        st.calls += 1
        own_idx = own_d[top_k_index]
        if cfg.adapt_every:                                 # routing EMA on the device (no host sync)
            if self._counts_d is None or self._counts_d.device != dev:
                self._counts_d = torch.zeros(self.E, dtype=torch.float32, device=dev)
            self._counts_d.mul_(cfg.adapt_decay).add_(torch.bincount(top_k_index.reshape(-1), minlength=self.E).float())
            self.calls += 1

        # 1) start copying the routing (and the hidden rows) to the host without waiting
        ev = None
        if blocking_any:
            if dev.type == "cuda":
                idx_p, w_p, h_p = (self._pinned(n, t) for n, t in (("idx", top_k_index), ("w", top_k_weights),
                                                                   ("h", hidden)))
                idx_p.copy_(top_k_index, non_blocking=True); w_p.copy_(top_k_weights, non_blocking=True)
                h_p.copy_(hidden, non_blocking=True)
                ev = torch.cuda.Event(); ev.record()
            else:
                idx_p, w_p, h_p = top_k_index.cpu(), top_k_weights.cpu(), hidden.cpu()

        # 2) host side first: wait for the (short) routing copy and hand the blocking tiers their pairs, so their
        #    work overlaps the device launches below
        futs, rows_c, inline, native = [], None, None, None
        if blocking_any:
            if ev is not None: ev.synchronize()
            if prof is not None: tp = _ph(prof, "sync", tp)
            if idx_p.shape[0] <= cfg.fast_rows:
                nat = self._native_plan(has) if cfg.native_dispatch and h_p.dtype == torch.bfloat16 else None
                if nat is not None:
                    native = self._native_start(nat, idx_p, w_p, h_p, lists)
                else:
                    futs, inline = self._split_small(idx_p, w_p, h_p, lists)
            else:
                futs, rows_c = self._split(idx_p, w_p, h_p, owner_c, local_c)
        else:
            if self._dev_counts is None or self._dev_counts.device != dev:
                self._dev_counts = torch.zeros(len(self.tiers), dtype=torch.long, device=dev)
                st._sources.append(self)
            self._dev_counts += torch.bincount(own_idx.reshape(-1), minlength=len(self.tiers))

        if prof is not None: tp = _ph(prof, "split", tp)
        # 3) launch every device tier that owns experts of this layer (needs nothing from the host)
        out = None
        for t, tier in enumerate(self.tiers):
            if tier.blocking or not has[t]:
                continue
            li, w = select_pairs(local_d[top_k_index], top_k_weights, own_idx == t)
            y = tier.run(self.handles[t], hidden, li, w)
            out = y if out is None else out + y
        if out is None:
            out = torch.zeros_like(hidden)

        # 4) the inline blocking tier (fast path) runs here, after the device launches; then collect the workers
        if prof is not None: tp = _ph(prof, "launch", tp)
        if native is not None:
            t0 = time.perf_counter()
            acc = self._host_buf("acc", tuple(out.shape), dev.type == "cuda")
            rc = _NATIVE["lib"].moe_disp_wait(_NATIVE["disp"], out.shape[0], out.shape[1], acc.data_ptr())
            if rc:
                raise RuntimeError(f"moe_offload: a blocking tier failed in the native dispatcher (rc={rc})")
            if prof is not None: tp = _ph(prof, "wait", tp)
            out = out + acc.to(dev, non_blocking=True).to(out.dtype)
            st.wait_s += time.perf_counter() - t0
            if prof is not None: tp = _ph(prof, "merge", tp)
        if futs or inline:
            t0 = time.perf_counter()
            done = [inline()] if inline else []
            if prof is not None: tp = _ph(prof, "inline", tp)
            done += [f.result() for f in futs]
            if prof is not None: tp = _ph(prof, "wait", tp)
            if rows_c is None:                     # fast path: few rows; sum the tiers on the host, copy once
                # reusing the pinned buffer is safe: the next call on this layer waits for its routing copy, which
                # the stream orders after this upload
                acc = self._host_buf("acc", tuple(out.shape), dev.type == "cuda")
                acc.zero_()
                for rows, y in done:
                    acc[rows] += y
                out = out + acc.to(dev, non_blocking=True).to(out.dtype)
            else:
                rows_d = rows_c.to(dev, non_blocking=True)
                for _, y in done:
                    out = out.index_add(0, rows_d, y.to(dev, out.dtype, non_blocking=True))
            st.wait_s += time.perf_counter() - t0
            if prof is not None: tp = _ph(prof, "merge", tp)
        if cfg.adapt_every and self.calls % cfg.adapt_every == 0 and (self._adapting is None or self._adapting.done()):
            self._adapting = _ADAPT.submit(self.adapt)         # in the background; tables change under the lock
        if prof is not None: _LAST_END[0] = time.perf_counter()
        return out

    def _split(self, idx_c, w_p, h_p, owner_c, local_c):
        """general path: tensors per blocking tier, all in workers."""
        own_c = owner_c[idx_c]
        per_tier = torch.bincount(own_c.reshape(-1), minlength=len(self.tiers)).tolist()
        self._count(per_tier)
        blocking = [t for t, n in enumerate(per_tier) if n and self.tiers[t].blocking]
        if not blocking:
            return [], None
        need = torch.zeros(idx_c.shape[0], dtype=torch.bool)
        for t in blocking: need |= (own_c == t).any(dim=-1)
        rows_c = need.nonzero().squeeze(-1)
        h_c, w_c, i_c, o_c = h_p[rows_c], w_p[rows_c], idx_c[rows_c], own_c[rows_c]
        return [_POOL.submit(lambda t=t: (rows_c, self._run_blocking(t, h_c, i_c, w_c, o_c, local_c)))
                for t in blocking], rows_c

    def _native_plan(self, has):
        """(fns, ctxs, blocking) ctypes arrays when every blocking tier holding experts of this layer has a native
        tier function; None otherwise (cached until the placement changes)."""
        if self._native is None:
            import ctypes
            from .native import load_dispatch
            lib = load_dispatch()
            T = len(self.tiers)
            plan = False
            if lib is not None and T <= 8:
                fns, ctxs = (ctypes.c_void_p * T)(), (ctypes.c_void_p * T)()
                blocking = (ctypes.c_int8 * T)()
                plan = (fns, ctxs, blocking)
                for t, tier in enumerate(self.tiers):
                    if not (tier.blocking and has[t]):
                        continue
                    nt = tier.native_tier(self.handles[t])
                    if nt is None:
                        plan = False
                        break
                    fns[t], ctxs[t], blocking[t] = nt[0], nt[1], 1
                if plan and "disp" not in _NATIVE:
                    _NATIVE.update(lib=lib, disp=ctypes.c_void_p(lib.moe_disp_new()))
            self._native = plan
        return self._native or None

    def _native_start(self, nat, idx_p, w_p, h_p, lists):
        fns, ctxs, blocking = nat
        T = len(self.tiers)
        idx = idx_p if idx_p.dtype == torch.int64 else idx_p.to(torch.int64)
        w = w_p if w_p.dtype == torch.float32 else w_p.float()
        counts = torch.zeros(T, dtype=torch.int64)
        R, k = idx.shape
        _NATIVE["lib"].moe_disp_start(_NATIVE["disp"], R, k, h_p.shape[1], idx.data_ptr(), w.data_ptr(), h_p.data_ptr(),
                                      lists[2].data_ptr(), lists[3].data_ptr(), T, blocking, fns, ctxs,
                                      counts.data_ptr())
        self._count(counts.tolist())
        return True

    def _split_small(self, idx_p, w_p, h_p, lists):
        """decode fast path (few rows): split pairs with Python lists, hand each blocking tier its rows and
        (local id, weight) lists; all but one tier go to workers, the last runs inline after the device launches."""
        owner_l, local_l = lists[0], lists[1]
        il, wl = idx_p.tolist(), w_p.float().tolist()
        T = len(self.tiers)
        per = [dict() for _ in range(T)]                  # tier -> {row: [(local, w)]}
        counts = [0] * T
        for r, (ir, wr) in enumerate(zip(il, wl)):
            for e, a in zip(ir, wr):
                t = owner_l[e]; counts[t] += 1
                if a != 0.0 and self.tiers[t].blocking:
                    per[t].setdefault(r, []).append((local_l[e], a))
        self._count(counts)
        jobs = []
        spec = self.handles[-1].spec
        for t in range(T):
            if per[t]:
                rows = list(per[t])
                n = sum(len(p) for p in per[t].values())
                cost = self.tiers[t].cost(spec, n, n)
                jobs.append((cost, lambda t=t, rows=rows: (rows, self._run_pairs(t, h_p[rows], [per[t][r] for r in rows]))))
        if not jobs:
            return [], None
        jobs.sort(key=lambda j: -j[0])
        futs = [_POOL.submit(j) for _, j in jobs[:-1]]
        return futs, jobs[-1][1]

    def _run_pairs(self, t, x, pairs):
        t0 = time.perf_counter()
        y = self.tiers[t].run_pairs(self.handles[t], x, pairs)
        dt = time.perf_counter() - t0
        self.cfg.stats.host_s += dt
        if self.cfg.profile_phases:
            _add(self.cfg.stats.phases, f"tier:{self.names[t]}", dt)
        return y

    def _count(self, per_tier):
        st = self.cfg.stats
        for t, n in enumerate(per_tier):
            if n:
                st.tier_pairs[self.names[t]] = st.tier_pairs.get(self.names[t], 0) + n
                if self.tiers[t].blocking: st.host_pairs += n
                else: st.device_pairs += n

    def adapt(self):
        """Swap the hottest host experts into accelerator tiers in place of their coldest experts, in two phases so
        every call sees a correct table: (1) route both experts to the host (it holds every expert), (2) overwrite
        the slot, (3) route the new expert to the slot."""
        host = len(self.tiers) - 1
        cfg = self.cfg
        counts = self._counts_d.cpu().double() if self._counts_d is not None else self.counts.clone()
        for t, tier in enumerate(self.tiers[:-1]):
            hdl = self.handles[t]
            if hdl is None or not tier.can_update:
                continue
            with self._lock:
                on_t = (self.owner_c == t).nonzero().squeeze(-1)
                on_h = (self.owner_c == host).nonzero().squeeze(-1)
            if not len(on_t) or not len(on_h):
                continue
            hot_h = on_h[torch.argsort(counts[on_h], descending=True)][:cfg.adapt_swaps].tolist()
            cold_t = on_t[torch.argsort(counts[on_t])][:cfg.adapt_swaps].tolist()
            for e_in, e_out in zip(hot_h, cold_t):
                if counts[e_in] <= cfg.adapt_margin * counts[e_out]:
                    break
                with self._lock:
                    slot = int(self.local_c[e_out])
                    self.owner_c[e_out], self.local_c[e_out] = host, e_out          # (1)
                    self._dev_tables.clear(); self._has = None; self._lists = None
                tier.update(hdl, slot, self._expert_weights(e_in))                 # (2) returns when landed
                with self._lock:
                    self.owner_c[e_in], self.local_c[e_in] = t, slot               # (3)
                    self._dev_tables.clear(); self._has = None; self._lists = None
                cfg.stats.swaps += 1

    def backward(self, hidden, top_k_index, top_k_weights, grad_out):
        """gradients of the layer's output w.r.t. hidden and the routing weights (experts are frozen): the experts
        the call used are taken from the host tier (it holds every expert), moved to the gradient's device and run
        again under autograd with opt_prime.moe_train's batched form."""
        from ..moe_train import opt_batched_mm
        from .backends.cuda import skeleton_with
        dev = grad_out.device
        used = torch.unique(top_k_index).tolist()
        host = self.tiers[-1]
        W = (host.weights_of(self.handles[-1], used) if hasattr(host, "weights_of")
             else self.adapter.weights(self._source, used))
        m = skeleton_with(W, dev)
        remap = torch.full((self.E,), 0, dtype=torch.long, device=top_k_index.device)
        remap[torch.tensor(used, device=top_k_index.device)] = torch.arange(len(used), device=top_k_index.device)
        x = hidden.detach().requires_grad_(True)
        w = top_k_weights.detach().requires_grad_(True)
        with torch.enable_grad():
            y = opt_batched_mm(m, x, remap[top_k_index], w)
            gx, gw = torch.autograd.grad(y, (x, w), grad_out.to(y.dtype))
        return gx.to(hidden.dtype), gw.to(top_k_weights.dtype)

    def _expert_weights(self, e):
        """one expert's weights from the host tier (which holds every expert)."""
        host = self.tiers[-1]
        if hasattr(host, "weights_of"):
            return host.weights_of(self.handles[-1], [e])
        return self.adapter.weights(self._source, [e])

    def _run_blocking(self, t, h, idx, w, own, local_c):
        t0 = time.perf_counter()
        keep = own == t
        li = local_c[idx]
        if self.tiers[t].compact_pairs:
            li, wt = compact_pairs(li, w, keep)
        else:
            li, wt = select_pairs(li, w, keep)
        y = self.tiers[t].run(self.handles[t], h, li, wt)
        self.cfg.stats.host_s += time.perf_counter() - t0
        return y


def _add(prof, name, dt):
    prof[name] = prof.get(name, 0.0) + dt


def _ph(prof, name, t0):
    """record the phase that started at t0; returns now (the next phase's start)."""
    now = time.perf_counter()
    _add(prof, name, now - t0)
    return now


def select_pairs(index, weights, keep):
    """Same [T,k] shape; pairs not in `keep` become (the row's first kept index, weight 0): they add exactly zero
    and touch no expert the row does not already use (rows with nothing kept fall back to index 0)."""
    first = torch.gather(index, 1, keep.int().argmax(dim=1, keepdim=True))
    first = torch.where(keep.any(dim=1, keepdim=True), first, torch.zeros_like(first))   # never an invalid id
    return torch.where(keep, index, first), weights * keep.to(weights.dtype)


def compact_pairs(index, weights, keep):
    """[R,k] -> [R,k'] with k' = max kept pairs per row: kept pairs first, the rest padded like select_pairs.
    For single-token decode k' is exactly the number of kept pairs, so nothing extra is computed."""
    order = torch.argsort((~keep).int(), dim=1, stable=True)
    k2 = max(1, int(keep.sum(dim=1).max()))
    index, weights, keep = (torch.gather(t, 1, order)[:, :k2] for t in (index, weights, keep))
    return select_pairs(index, weights, keep)
