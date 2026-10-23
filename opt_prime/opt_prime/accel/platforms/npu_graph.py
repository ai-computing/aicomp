#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Shared machinery for accelerators driven by an ahead-of-time graph compiler (FuriosaAI RNGD via furiosa-torch,
Rebellions ATOM via the RBLN SDK, ...). A new NPU of this kind only implements three hooks:

    _compile(module, sample_inputs) -> callable     # compile a torch module for the device
    _to_device(tensor) -> device tensor             # place a weight / input on the device
    _to_host(output) -> cpu tensor

The experts are run as `SlotFFN` graphs, compiled once per (spec, token bucket, slot count) and cached:

    y[T,H] = sum_k  G[:, k] * down_k( act(gate_k x) * (up_k x) )

Weight residency (choose per device):
  "inputs": every expert's weights are device tensors passed as graph inputs; the host picks K of them per call
            (pointer swap, no weight traffic). Needs a runtime that accepts device-resident inputs (RNGD).
  "baked":  one graph per expert with its weights compiled in as constants (K=1); for runtimes whose inputs are
            always uploaded from the host (weights would otherwise cross PCIe on every call).

Static shapes: token counts are padded up to `token_buckets`, expert groups to `slot_buckets`; the G matrix zeroes
padding, so padded rows/slots contribute nothing.
"""
import hashlib
import os
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from opt_prime.accel.spec.experts import ExpertSpec, ExpertWeights
from opt_prime.accel.moe.backends import ExpertBackend, Handle

_ACT = {"silu": nn.functional.silu, "gelu": nn.functional.gelu, "relu": nn.functional.relu}


class SlotFFN(nn.Module):
    """K standard experts on T tokens, masked/weighted by G [T, K]. Weights in [out, in] orientation."""

    def __init__(self, spec: ExpertSpec, K: int, with_bias: bool):
        super().__init__()
        self.F, self.K, self.bias, self.act = spec.ffn, K, with_bias, _ACT[spec.activation]

    def forward(self, x, G, *ws):
        per = 4 if self.bias else 2
        out = None
        for i in range(self.K):
            gu_w, dn_w = ws[per * i], ws[per * i + 1]
            gu = x @ gu_w.t()
            if self.bias: gu = gu + ws[per * i + 2]
            a = self.act(gu[:, :self.F]) * gu[:, self.F:]
            y = a @ dn_w.t()
            if self.bias: y = y + ws[per * i + 3]
            y = y * G[:, i:i + 1]
            out = y if out is None else out + y
        return out


class BakedFFN(nn.Module):
    """One expert with its weights as constants (for runtimes that upload every input)."""

    def __init__(self, spec: ExpertSpec, gu, dn, gb=None, db=None):
        super().__init__()
        self.F, self.act = spec.ffn, _ACT[spec.activation]
        self.register_buffer("gu", gu); self.register_buffer("dn", dn)
        self.gb = None if gb is None else nn.Parameter(gb, requires_grad=False)
        self.db = None if db is None else nn.Parameter(db, requires_grad=False)

    def forward(self, x, g):
        gu = x @ self.gu.t()
        if self.gb is not None: gu = gu + self.gb
        y = (self.act(gu[:, :self.F]) * gu[:, self.F:]) @ self.dn.t()
        if self.db is not None: y = y + self.db
        return y * g


def _bucket(n: int, buckets: Sequence[int]) -> int:
    for b in buckets:
        if n <= b: return b
    return n


def _standard(w: ExpertWeights):
    """per-expert weights in [out, in] orientation (transposed layouts are converted once, at load)."""
    gu, dn = w.gate_up, w.down
    if w.spec.transposed:
        gu, dn = gu.transpose(1, 2), dn.transpose(1, 2)
    return gu, dn, w.gate_up_bias, w.down_bias


class GraphNPUBackend(ExpertBackend):
    blocking = True
    presence_s = 100e-6            # the host thread drives the device call
    compact_pairs = True
    residency = "inputs"
    token_buckets = (1, 4, 16, 64, 256)
    slot_buckets = (1, 2, 4, 8)

    def __init__(self, device=0, budget_bytes=None, emulate=None, token_buckets=None, slot_buckets=None):
        self.chip, self.budget = device, budget_bytes
        self.emulate = bool(int(os.environ.get("MOE_OFFLOAD_EMULATE_NPU", "0"))) if emulate is None else emulate
        if token_buckets: self.token_buckets = tuple(token_buckets)
        if slot_buckets: self.slot_buckets = tuple(slot_buckets)
        self._graphs: Dict[tuple, object] = {}

    # ---- vendor hooks (emulation runs the same graphs on the host) ----
    def _compile(self, module, sample_inputs):
        raise NotImplementedError

    def _to_device(self, t):
        raise NotImplementedError

    def _to_host(self, t):
        return t.cpu()

    def _place(self, compiled):
        return compiled

    # optional: persist compiled graphs so each (shape, bucket) compiles once per machine
    def _save(self, compiled, path):
        raise NotImplementedError

    def _load(self, path):
        raise NotImplementedError

    def cache_dir(self):
        root = os.path.expanduser(os.environ.get("OPT_PRIME_CACHE", "~/.cache/opt_prime"))
        return os.path.join(root, "moe_offload", self.name)

    def _cached_compile(self, key, module, samples, place=True):
        if self.emulate:
            return module
        path = os.path.join(self.cache_dir(), hashlib.blake2b(repr(key).encode(), digest_size=12).hexdigest())
        try:
            compiled = self._load(path)
        except (NotImplementedError, FileNotFoundError, OSError):
            compiled = self._compile(module, samples)
            try:
                os.makedirs(self.cache_dir(), exist_ok=True)
                self._save(compiled, path)
            except NotImplementedError:
                pass
            except Exception as e:                     # caching must never fail a request
                print(f"[moe_offload] could not cache compiled graph {key}: {type(e).__name__}: {e}")
        return self._place(compiled) if place else compiled

    def _slot_samples(self, spec, T_b, K_b, with_bias):
        x = torch.zeros(T_b, spec.hidden, dtype=spec.dtype)
        G = torch.zeros(T_b, K_b, dtype=spec.dtype)
        per = [torch.zeros(2 * spec.ffn, spec.hidden, dtype=spec.dtype), torch.zeros(spec.hidden, spec.ffn, dtype=spec.dtype)]
        if with_bias:
            per += [torch.zeros(2 * spec.ffn, dtype=spec.dtype), torch.zeros(spec.hidden, dtype=spec.dtype)]
        return [x, G] + [t.clone() for _ in range(K_b) for t in per]       # distinct tensors: no input aliasing

    def precompile(self, spec: ExpertSpec, token_buckets=None, slot_buckets=None):
        """Compile and cache every (token bucket, slot bucket) graph for `spec` without using the device, e.g.
        while it is busy; later runs load them from the cache."""
        done = []
        for T_b in token_buckets or self.token_buckets:
            for K_b in slot_buckets or self.slot_buckets:
                key = (spec.signature(), T_b, K_b, spec.has_bias, self.version())
                self._cached_compile(key, SlotFFN(spec, K_b, spec.has_bias),
                                     self._slot_samples(spec, T_b, K_b, spec.has_bias), place=False)
                done.append((T_b, K_b))
        return done

    def version(self) -> str:
        """compiler/SDK version, part of the compiled-graph cache key."""
        return ""

    def _c(self, module, samples):
        return module if self.emulate else self._place(self._compile(module, samples))

    def _d(self, t):
        return t if self.emulate else self._to_device(t)

    def _h(self, t):
        return t if self.emulate else self._to_host(t)

    # ---- contract ----
    def device_name(self):
        return None if self.emulate else f"{self.name.split('-')[0]}:{self.chip}"

    def supports(self, spec):
        return spec.gate == "swiglu" and spec.activation in _ACT

    def capacity_bytes(self):
        return int(self.budget) if self.budget is not None else self.default_capacity()

    def default_capacity(self):
        raise NotImplementedError

    def device(self):
        return torch.device("cpu")    # inputs and outputs cross through host memory

    def load(self, w: ExpertWeights):
        gu, dn, gb, db = _standard(w)
        if self.residency == "baked":
            graphs = []
            for i in range(len(w.global_ids)):
                m = BakedFFN(w.spec, gu[i].contiguous(), dn[i].contiguous(),
                             None if gb is None else gb[i], None if db is None else db[i])
                # weights are constants of the graph: they are part of its cache key
                constants = [gu[i], dn[i]]
                if gb is not None:
                    constants.append(gb[i])
                if db is not None:
                    constants.append(db[i])
                digest = hashlib.blake2b(
                    b"".join(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                             for t in constants), digest_size=12).hexdigest()
                ent = {"module": m, "digest": digest}
                ent[self.token_buckets[0]] = self._baked_graph(w.spec, ent, self.token_buckets[0])   # decode bucket now
                graphs.append(ent)                 # other token buckets compile on first use: {T_b: callable}
            return Handle(w.spec, w.global_ids, {"graphs": graphs})
        # inputs: one device tensor per weight, distinct objects (aliased samples would be merged by compilers)
        per = []
        for i in range(len(w.global_ids)):
            ts = [gu[i].contiguous(), dn[i].contiguous()]
            if gb is not None: ts += [gb[i].contiguous(), db[i].contiguous()]
            per.append([self._d(t) for t in ts])
        return Handle(w.spec, w.global_ids, {"weights": per, "host_sample": [t.clone() for t in
                                                                                 ([gu[0], dn[0]] + ([gb[0], db[0]] if gb is not None else []))]})

    @property
    def can_update(self):
        return self.residency == "inputs"

    def update(self, h, slot, w):
        gu, dn, gb, db = _standard(w)
        ts = [gu[0].contiguous(), dn[0].contiguous()] + ([gb[0].contiguous(), db[0].contiguous()] if gb is not None else [])
        h.state["weights"][slot] = [self._d(t) for t in ts]
        h.global_ids[slot] = w.global_ids[0]

    def _baked_graph(self, spec, ent, T_b):
        samples = [torch.zeros(T_b, spec.hidden, dtype=spec.dtype), torch.zeros(T_b, 1, dtype=spec.dtype)]
        key = (spec.signature(), "baked", ent["digest"], T_b, self.version())
        return self._cached_compile(key, ent["module"], samples)

    def _slot_graph(self, spec, T_b, K_b, with_bias, host_sample=None):
        key = (spec.signature(), T_b, K_b, with_bias, self.version())
        if key not in self._graphs:
            self._graphs[key] = self._cached_compile(key, SlotFFN(spec, K_b, with_bias),
                                                     self._slot_samples(spec, T_b, K_b, with_bias))
        return self._graphs[key]

    def run(self, h: Handle, hidden, local_index, weights):
        spec = h.spec
        T, cap = hidden.shape[0], self.token_buckets[-1]
        if T > cap:                                # longer prefills run in chunks of the largest bucket (no graph per T)
            return torch.cat([self.run(h, hidden[i:i + cap], local_index[i:i + cap], weights[i:i + cap])
                              for i in range(0, T, cap)])
        T_b = _bucket(T, self.token_buckets)
        x = torch.zeros(T_b, spec.hidden, dtype=spec.dtype)
        x[:T] = hidden.to(spec.dtype)
        flat_i, flat_w = local_index.reshape(-1), weights.reshape(-1).float()
        rows = torch.arange(T).repeat_interleave(local_index.shape[1])
        nz = flat_w != 0
        flat_i, flat_w, rows = flat_i[nz], flat_w[nz], rows[nz]
        experts: List[int] = torch.unique(flat_i).tolist()
        out = torch.zeros(T, spec.hidden, dtype=torch.float32)
        if not experts:
            return out.to(hidden.dtype)
        xd = self._d(x)
        if self.residency == "baked":
            for e in experts:
                g = torch.zeros(T_b, 1, dtype=spec.dtype)
                sel = flat_i == e
                g.index_put_((rows[sel], torch.zeros_like(rows[sel])), flat_w[sel].to(spec.dtype), accumulate=True)
                ent = h.state["graphs"][e]
                if T_b not in ent:
                    ent[T_b] = self._baked_graph(spec, ent, T_b)
                out += self._h(ent[T_b](xd, self._d(g)))[:T].float()
            return out.to(hidden.dtype)
        with_bias = spec.has_bias
        K_b = _bucket(len(experts), self.slot_buckets)
        K_b = min(K_b, self.slot_buckets[-1])
        for c in range(0, len(experts), K_b):
            group = experts[c:c + K_b]
            G = torch.zeros(T_b, K_b, dtype=spec.dtype)
            for k, e in enumerate(group):
                sel = flat_i == e
                G.index_put_((rows[sel], torch.full_like(rows[sel], k)), flat_w[sel].to(spec.dtype), accumulate=True)
            ws = [t for e in group for t in h.state["weights"][e]]
            ws += [t for _ in range(K_b - len(group)) for t in h.state["weights"][group[0]]]   # padded slots, G=0
            graph = self._slot_graph(spec, T_b, K_b, with_bias, h.state["host_sample"])
            out += self._h(graph(xd, self._d(G), *ws))[:T].float()
        return out.to(hidden.dtype)
