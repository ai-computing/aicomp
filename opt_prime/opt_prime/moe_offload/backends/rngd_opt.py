#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""FuriosaAI RNGD with per-shape furiosa-opt expert kernels from the npu-kernels package (`npu_kernels.rngd`).

For shapes `npu_kernels.rngd.plan(H, F)` accepts, a crate is generated from the template, built once per machine,
and its `server` binary runs as a child process that owns one chip: every MoE layer's expert table stays resident
in HBM, and a call sends (layer, slots, x) and receives one output row per expert. The host weights the rows.

Compared with the furiosa-torch backend (`rngd`) this streams expert weights at ~1 TB/s instead of ~0.37 TB/s, but
handles one token per launch (decode). The planner chooses between the two through their cost models.

Build modes: on silicon the server is built with `cargo furiosa-opt build`; with `emulate=True` (or
MOE_OFFLOAD_EMULATE_NPU=1) a plain `cargo build` runs the same kernel on the furiosa-opt CPU emulator, so the
whole path (layouts, slot packing, protocol) is testable without a chip, only slowly.
"""
import os
import struct
import subprocess
import tempfile
import threading

import torch

from ..spec import ExpertSpec, ExpertWeights
from .base import ExpertBackend, Handle, register_backend

_OP_EXIT, _OP_LOAD, _OP_RUN, _OP_UPDATE, _OP_RUN_WSUM = 0, 1, 2, 3, 4


def perm_wg(w: torch.Tensor, p) -> torch.Tensor:
    """[F, H] -> WG layout m![F / R, H / 32, F / 8 % (R/8), H / 4 % 8, F % 8, H % 4] (flat)."""
    F, H = w.shape
    return w.reshape(F // p.R, p.R // 8, 8, H // 32, 8, 4).permute(0, 3, 1, 4, 2, 5).reshape(-1)


def perm_wd(w: torch.Tensor, p) -> torch.Tensor:
    """[H, F] -> WD layout m![H / HR, F / FC, H / 8 % (HR/8), F / 4 % (FC/4), H % 8, F % 4] (flat)."""
    H, F = w.shape
    return w.reshape(H // p.HR, p.HR // 8, 8, F // p.FC, p.FC // 4, 4).permute(0, 3, 1, 4, 2, 5).reshape(-1)


def build_server(plan, emulate: bool) -> str:
    """the expert server binary for `plan` (npu-kernels generates and builds it)."""
    from npu_kernels.rngd import build_server as _build
    return _build(plan, emulate)


class _NativeLock:
    def __init__(self, lib, pipe): self.lib, self.pipe = lib, pipe
    def __enter__(self): self.lib.moe_pipe_lock(self.pipe)
    def __exit__(self, *a): self.lib.moe_pipe_unlock(self.pipe)


class _Server:
    """one server process (one chip) speaking the stdin/stdout protocol of template/src/bin/server.rs.tpl."""

    def __init__(self, exe, k, chip=None):
        env = dict(os.environ)
        if chip is not None:
            env["FURIOSA_VISIBLE_CHIPS"] = str(chip)       # furiosa-opt-std opens a chip among these only
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0, env=env)
        hello = self._read(4)
        if hello != b"RDY1":
            raise RuntimeError(f"rngd-opt: server did not start (got {hello!r})")
        # one mutex for the pipe, shared with the native dispatcher (which talks to the server directly)
        from ..native import load_dispatch
        self.native = load_dispatch()
        if self.native is not None:
            self.pipe = self.native.moe_pipe_new(self.p.stdin.fileno(), self.p.stdout.fileno(), k)
            self.lock = _NativeLock(self.native, self.pipe)
        else:
            self.pipe, self.lock = None, threading.Lock()

    def _read(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("rngd-opt: server exited")
            buf += chunk
        return bytes(buf)

    def _status(self):
        st, = struct.unpack("<I", self._read(4))
        if st != 0:
            raise RuntimeError(f"rngd-opt: server returned status {st}")

    def load(self, layer, path):
        b = path.encode()
        with self.lock:
            self.p.stdin.write(struct.pack("<III", _OP_LOAD, layer, len(b)) + b); self._status()

    def update(self, layer, slot, path):
        b = path.encode()
        with self.lock:
            self.p.stdin.write(struct.pack("<IIII", _OP_UPDATE, layer, slot, len(b)) + b); self._status()

    def run(self, layer, slots, x_bf16: bytes, H):
        n = len(slots)
        with self.lock:
            self.p.stdin.write(struct.pack(f"<III{n}I", _OP_RUN, layer, n, *slots) + x_bf16)
            return self._read(2 * n * H)

    def run_wsum(self, layer, slots, weights, x_bf16: bytes, H):
        n = len(slots)
        with self.lock:
            self.p.stdin.write(struct.pack(f"<III{n}I{n}f", _OP_RUN_WSUM, layer, n, *slots, *weights) + x_bf16)
            return self._read(4 * H)

    def close(self):
        try:
            self.p.stdin.write(struct.pack("<I", _OP_EXIT)); self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


@register_backend("rngd-opt")
class RngdOptBackend(ExpertBackend):
    blocking = True
    compact_pairs = True
    presence_s = 100e-6
    # measured on RNGD (examples/moe_offload_rngd_opt_check.py, one token, k=1..8 experts): Qwen3-30B-A3B shape
    # 189-325 us, Solar-Open shape 263-640 us -> ~190 us per launch incl. pipe round trip, ~0.5-0.6 TB/s
    overhead_s = 190e-6
    bandwidth_Bps = 530e9
    can_update = True

    def __init__(self, device=0, budget_bytes=None, slots=32, k=8, emulate=None):
        self.chip, self.budget = device, budget_bytes
        self.slots_per_layer, self.k = slots, k
        self.emulate = bool(int(os.environ.get("MOE_OFFLOAD_EMULATE_NPU", "0"))) if emulate is None else emulate
        self._server = None
        self._layers = 0
        self._plan = None
        # expert tables go to the server through files (one layer at a time, deleted after loading): RAM-backed
        self._tmp = tempfile.mkdtemp(prefix="rngd_opt_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    def _plan_for(self, spec):
        from npu_kernels.rngd import plan          # optional dependency: pip install npu-kernels
        return plan(spec.hidden, spec.ffn, E=self.slots_per_layer, K=self.k)

    def device_name(self):
        return None if self.emulate else f"rngd:{self.chip}"

    def supports(self, spec: ExpertSpec) -> bool:
        if not spec.standard or spec.has_bias or spec.activation != "silu" or spec.dtype != torch.bfloat16:
            return False
        try:
            self._plan_for(spec)
            return True
        except (ValueError, ImportError):          # shape the kernels do not tile, or npu-kernels not installed
            return False

    def capacity_bytes(self):
        return int(self.budget) if self.budget is not None else int(40e9)   # 48 GB HBM minus runtime reserve

    def cost(self, spec, n_experts, n_pairs):
        # one launch per token, every routed pair streams its expert's weights
        return self.overhead_s + n_pairs * spec.expert_bytes / self.bandwidth_Bps

    def device(self):
        return torch.device("cpu")

    def precompile(self, spec):
        """generate and build the server for `spec` (no device needed); later runs reuse the binary."""
        return [build_server(self._plan_for(spec), self.emulate)]

    def cache_dir(self):
        from npu_kernels.rngd import cache_root
        return os.path.join(cache_root(), "target", "cpu" if self.emulate else "npu")

    def _ensure(self, spec):
        if self._server is None:
            self._plan = self._plan_for(spec)
            self._server = _Server(build_server(self._plan, self.emulate), self.k,
                                   None if self.emulate else self.chip)

    def _write(self, stem, gu, dn):
        """stacked [n,2F,H] / [n,H,F] (out,in orientation) -> <stem>.g/.u/.d in the kernel's HBM layouts."""
        p, F = self._plan, self._plan.F
        g = torch.cat([perm_wg(gu[i, :F].contiguous(), p) for i in range(gu.shape[0])])
        u = torch.cat([perm_wg(gu[i, F:].contiguous(), p) for i in range(gu.shape[0])])
        d = torch.cat([perm_wd(dn[i].contiguous(), p) for i in range(dn.shape[0])])
        for ext, t in ((".g", g), (".u", u), (".d", d)):
            t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tofile(stem + ext)

    @staticmethod
    def _standard(w: ExpertWeights):
        gu, dn = w.gate_up, w.down
        if w.spec.transposed:
            gu, dn = gu.transpose(1, 2), dn.transpose(1, 2)
        return gu, dn

    def load(self, w: ExpertWeights) -> Handle:
        spec = w.spec
        self._ensure(spec)
        n = len(w.global_ids)
        if n > self.slots_per_layer:
            raise ValueError(f"rngd-opt: {n} experts for a {self.slots_per_layer}-slot table")
        gu, dn = self._standard(w)
        pad = self.slots_per_layer - n                         # unused slots: zeros
        if pad:
            gu = torch.cat([gu, gu.new_zeros(pad, *gu.shape[1:])])
            dn = torch.cat([dn, dn.new_zeros(pad, *dn.shape[1:])])
        layer = self._layers; self._layers += 1
        stem = os.path.join(self._tmp, f"l{layer}")
        self._write(stem, gu, dn)
        self._server.load(layer, stem)
        for ext in (".g", ".u", ".d"): os.unlink(stem + ext)
        return Handle(spec, list(w.global_ids), {"layer": layer})

    def update(self, h: Handle, slot: int, w: ExpertWeights):
        gu, dn = self._standard(w)
        stem = os.path.join(self._tmp, f"u{h.state['layer']}_{slot}")
        self._write(stem, gu[:1], dn[:1])
        self._server.update(h.state["layer"], slot, stem)
        for ext in (".g", ".u", ".d"): os.unlink(stem + ext)
        h.global_ids[slot] = w.global_ids[0]

    def native_tier(self, h: Handle, bf16_weights: bool = False):
        srv = self._server
        if srv is None or srv.pipe is None:
            return None
        if "native_tier" not in h.state:
            from ..native import fn_addr
            h.state["native_tier"] = (fn_addr(srv.native, "moe_tier_pipe"),
                                      srv.native.moe_pipe_ctx_new(srv.pipe, h.state["layer"]))
        return h.state["native_tier"]

    def run_pairs(self, h: Handle, x, pairs, bf16_weights: bool = False):
        H = h.spec.hidden
        out = torch.empty(len(pairs), H, dtype=torch.float32)
        xb = x.to(torch.bfloat16).contiguous().view(torch.int16).numpy()
        for r, p in enumerate(pairs):
            acc = {}
            for e, a in p: acc[e] = acc.get(e, 0.0) + a
            items = list(acc.items())
            for c in range(0, len(items), self.k):
                grp = items[c:c + self.k]
                raw = self._server.run_wsum(h.state["layer"], [e for e, _ in grp], [a for _, a in grp], xb[r].tobytes(), H)
                y = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                if c: out[r] += y
                else: out[r] = y
        return out

    def run(self, h: Handle, hidden, local_index, weights):
        H = h.spec.hidden
        T = hidden.shape[0]
        out = torch.zeros(T, H, dtype=torch.float32)
        xb = hidden.to(torch.bfloat16).contiguous()
        for t in range(T):
            ws = {}
            for e, g in zip(local_index[t].tolist(), weights[t].float().tolist()):
                if g != 0.0: ws[e] = ws.get(e, 0.0) + g
            items = list(ws.items())
            for c in range(0, len(items), self.k):
                grp = items[c:c + self.k]
                raw = self._server.run(h.state["layer"], [e for e, _ in grp], xb[t].view(torch.int16).numpy().tobytes(), H)
                rows = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16).view(len(grp), H).float()
                out[t] += (torch.tensor([g for _, g in grp]).unsqueeze(1) * rows).sum(0)
        return out.to(hidden.dtype)

    def close(self):
        if self._server is not None:
            self._server.close(); self._server = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
