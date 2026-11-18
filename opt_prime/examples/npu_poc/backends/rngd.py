"""FuriosaAI RNGD adapter — fits the `furiosa.torch` SDK to the protocol.

Parallelism ownership
─────────────────────
    PP               owned by opt_prime (IR · split · stage wiring)
    NPU-internal TP  PE fusion: with fusion=4 one device is half a card and the
                     compiler spreads a stage over its 4 PEs

Compile path per stage:  torch.export → core-ATen decomposition → furiosa
compile → weights to the device.  Validated with furiosa-torch 2026.4.x; both
API generations below can be imported:

    furiosa-torch 2026.2.x   native_device · STD_DECOMPOSITIONS · run_by_rngd · "rngd:k"
    furiosa-torch 2026.3.x+  top-level API · CompileModule.from_exported · "furiosa:k"

Environment variables (optional):
    NPU_POC_RNGD_FUSION   cores per base device (default 4 = two devices per card);
                          --tp k fuses k base devices into one (fusion = 4·k)
    NPU_POC_RNGD_CACHE    compile-cache directory
    NPU_POC_RNGD_TACTIC_HINT  compiler tactic hint: auto (default — prefill for
                          multi-token stages, decode for single-token stages) ·
                          prefill · decode · default (the SDK's own setting)
    NPU_POC_RNGD_MAX_UNIQUE_SHAPES  compiler limit on the number of unique shapes it
                          generates tactics for: auto (default — 4 when one device spans
                          8 or more PEs, otherwise the SDK's own setting) · <integer> · default
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any, Sequence

import torch
import torch.utils._pytree as pytree
from torch.fx import GraphModule

from .base import BackendCaps, CompileStats

# furiosa-torch version this adapter has been validated with
VALIDATED_SDK = "2026.4"


def _sdk_new() -> bool:
    """True for the furiosa-torch 2026.4.x API (STD_DECOMPOSITIONS was removed)."""
    import furiosa.torch as ft
    return not hasattr(ft, "STD_DECOMPOSITIONS")


def _device_api():
    import furiosa.torch as ft
    if _sdk_new():
        return ft
    from furiosa.torch import native_device as nd
    return nd


def _split_with_sizes_as_slices(x, split_sizes, dim=0):
    """aten.split_with_sizes written as a chain of aten.slice (2026.2.x path)."""
    out, off = [], 0
    for s in split_sizes:
        out.append(torch.ops.aten.slice.Tensor(x, dim, off, off + s))
        off += s
    return out


def _strip_export_asserts(ep) -> int:
    """Remove unused export assertion nodes (`_assert_tensor_metadata` · `_assert_scalar`)."""
    g = ep.graph_module.graph
    n = 0
    for node in list(g.nodes):
        tgt = getattr(node.target, "_name", str(node.target))
        if ("_assert_tensor_metadata" in tgt or "_assert_scalar" in tgt) and not node.users:
            g.erase_node(node)
            n += 1
    if n:
        g.lint()
        ep.graph_module.recompile()
    return n


_HINTS = {  # setting → (furiosa-torch 2026.2.x name, 2026.4.x name)
    "prefill": ("ForLlmModelPrefill", "ForLlmModelComputeBound"),
    "decode": ("ForLlmModelDecode", "ForLlmModelIOBound"),
}


def _compiler_config(example_inputs, new_api: bool):
    """Compiler configuration for one stage, or None for the SDK default."""
    mode = os.environ.get("NPU_POC_RNGD_TACTIC_HINT", "auto")
    if mode == "default":
        return None
    if mode == "auto":
        # Sequence length from activations (B, S, D) or token ids (B, S).
        seq = [t.shape[1] for t in example_inputs if isinstance(t, torch.Tensor)
               and ((t.is_floating_point() and t.dim() == 3)
                    or (not t.is_floating_point() and t.dim() == 2))]
        mode = "decode" if seq and all(s == 1 for s in seq) else "prefill"
    if mode not in _HINTS:
        raise SystemExit(f"[rngd] NPU_POC_RNGD_TACTIC_HINT={mode!r} "
                         f"(expected default · prefill · decode · auto)")
    if new_api:
        from furiosa.torch._C.config import compiler as cc
        cfg = cc.CompilerConfig()
        cfg.tactic_hint = getattr(cc.TacticHintConfig, _HINTS[mode][1])
    else:
        from furiosa.native_torch import compiler as cc
        cfg = cc.Config()
        cfg.tactic_hint = getattr(cc.TacticHintConfig, _HINTS[mode][0])
    return cfg, mode


def _unique_shapes_limit(num_fusion: int) -> int | None:
    """Compiler limit on unique shapes for tactic generation, or None for the SDK's own setting.

    With 8 or more PEs per device (`num_fusion`) the compiler's tactic search on these stage
    graphs can exceed its memory budget; a limit of 4 keeps it bounded.  Measured on
    Llama-3.2-1B with 8 PEs per device: 4 is the only value tried (4–500) that compiles
    one-shot, prefill and decode stages, and results match the CPU.
    """
    mode = os.environ.get("NPU_POC_RNGD_MAX_UNIQUE_SHAPES", "auto")
    if mode == "default":
        return None
    if mode == "auto":
        return 4 if num_fusion >= 8 else None
    if mode.isdigit() and int(mode) > 0:
        return int(mode)
    raise SystemExit(f"[rngd] NPU_POC_RNGD_MAX_UNIQUE_SHAPES={mode!r} "
                     f"(expected auto · default · a positive integer)")


def _cross_chip(cfg_mode, num_fusion: int, new_api: bool):
    """Settings for a device that spans two or more chips (16+ PEs), unless the hint is set by hand.

    Measured on Llama-3.2-1B with 16 PEs per device: the stage graphs compile only with the
    SDK's default tactic hint together with estimate-ordered tactic search (plus the
    unique-shapes limit); the per-stage LLM hints do not.
    """
    if num_fusion < 16 or os.environ.get("NPU_POC_RNGD_TACTIC_HINT", "auto") != "auto":
        return cfg_mode
    if new_api:
        from furiosa.torch._C.config import compiler as cc
        cfg = cfg_mode[0] if cfg_mode else cc.CompilerConfig()
    else:
        from furiosa.native_torch import compiler as cc
        cfg = cfg_mode[0] if cfg_mode else cc.Config()
    cfg.tactic_hint = cc.TacticHintConfig.Default
    if hasattr(cc, "TacticSortingPolicy"):
        cfg.tactic_sorting_policy = cc.TacticSortingPolicy.ByEstimation
    return cfg, "default hint · estimate-ordered search"


def _with_unique_shapes(cfg_mode, limit: int | None, new_api: bool):
    """Return (cfg, label) with max_num_unique_shapes applied, creating a config if needed."""
    if limit is None:
        return cfg_mode
    if cfg_mode is None:
        if new_api:
            from furiosa.torch._C.config import compiler as cc
            cfg_mode = (cc.CompilerConfig(), "default")
        else:
            from furiosa.native_torch import compiler as cc
            cfg_mode = (cc.Config(), "default")
    cfg, mode = cfg_mode
    cfg.max_num_unique_shapes = limit
    return cfg, f"{mode} · unique shapes ≤ {limit}"


# Device memory kept for the life of the process (see _load_new).
_HELD: list = []


def _span(t) -> tuple[int, int]:
    return t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()


def _hold_range(lo: int, hi: int, dev, max_allocs: int = 1024) -> list:
    """Hold device memory [lo, hi) with placeholder blocks; returns the blocks that overlap it.

    Parts already held for an earlier program on the same device count as covered — programs
    on one device can share one scratch address.  Each request asks for the largest uncovered
    gap; after repeated misses the request size is halved.  Blocks that land elsewhere are kept
    until the end, so the allocator moves on to other free memory, then released.  A part the
    allocator never hands out is not free (for example, still in use by the program itself)
    and needs no hold; the weight check in `_load_new` remains the final guard.
    """
    gaps = [(lo, hi)]

    def cut(a, b):
        nonlocal gaps
        out = []
        for x, y in gaps:
            if b <= x or y <= a:
                out.append((x, y))
            else:
                if x < a:
                    out.append((x, a))
                if b < y:
                    out.append((b, y))
        gaps = out

    for d, blk in _HELD:
        if d == str(dev):
            cut(*_span(blk))
    hits, spare, misses = [], [], 0
    size = max((y - x for x, y in gaps), default=0)
    for _ in range(max_allocs):
        if not gaps:
            break
        blk = torch.empty(max(size, 512), dtype=torch.uint8, device=dev)
        a, b = _span(blk)
        if any(a < y and x < b for x, y in gaps):
            hits.append(blk)
            cut(a, b)
            misses = 0
            size = max((y - x for x, y in gaps), default=0)
        else:
            spare.append(blk)
            misses += 1
            if misses >= 16:
                size, misses = max(size // 2, 512), 0
    if os.environ.get("NPU_POC_RNGD_DEBUG"):
        print(f"[rngd-dbg] hold [{lo:#x},{hi:#x}) size {(hi-lo)/2**20:.1f} MiB · hits {len(hits)} "
              f"(holding {sum(_span(h)[1]-_span(h)[0] for h in hits)/2**20:.1f} MiB) · spare {len(spare)} · "
              f"uncovered {sum(y-x for x,y in gaps)/2**20:.1f} MiB in {len(gaps)} gaps {[(hex(x),hex(y)) for x,y in gaps[:4]]}", flush=True)
    del spare
    return hits


def _load_new(cm, dev, name: str):
    """Place a compiled program and its weights on `dev` (furiosa-torch 2026.3+ API).

    The program is moved first, which sets up its scratch area on the device.
    That address range is then held with placeholder blocks (`_hold_range`), so the
    weights and later allocations are placed elsewhere.  A weight that still lands on it
    is moved to a copy and its old storage kept as the hold.  Finally no weight may share
    memory with the scratch area; otherwise loading stops.  Programs without a
    separate scratch area are simply moved, program first.
    Weights of programs loaded earlier on the same device are checked the same way.
    """
    cm.runnable.to(dev)
    stacks = [s for s in getattr(cm.runnable._inner, "stacks", ()) if s.numel()]
    for st in stacks:
        _HELD.extend((str(dev), blk) for blk in _hold_range(*_span(st), dev))
    cm.module_weight.to(dev)
    weights = [(k, v) for k, v in list(cm.module_weight.named_parameters())
               + list(cm.module_weight.named_buffers()) if v.numel()]
    def clashing():
        return [(k, v) for k, v in weights for st in stacks
                if _span(v)[0] < _span(st)[1] and _span(st)[0] < _span(v)[1]]

    # A weight that still landed on the scratch area (the hold above cannot always reach it —
    # e.g. a range inside a free region the allocator uses only for large requests) is moved:
    # a copy is made while the old storage is still occupied, so the copy lands elsewhere, and
    # the old storage is kept as the hold.  The cost is that weight's size in device memory.
    for k, v in clashing():
        old = v.data
        new = torch.empty_like(old)
        new.copy_(old)
        v.data = new
        _HELD.append((str(dev), old))
        if os.environ.get("NPU_POC_RNGD_DEBUG"):
            print(f"[rngd-dbg] {name}: moved {k} off the scratch area "
                  f"({old.numel() * old.element_size() / 2**20:.1f} MiB kept as the hold)", flush=True)
    weights = [(k, v) for k, v in list(cm.module_weight.named_parameters())
               + list(cm.module_weight.named_buffers()) if v.numel()]
    clash = [k for k, _ in clashing()]
    if clash and os.environ.get("NPU_POC_RNGD_DEBUG"):
        for k, v_ in weights:
            if k in clash:
                print(f"[rngd-dbg] {k} span [{_span(v_)[0]:#x},{_span(v_)[1]:#x}) · stacks {[(hex(_span(t)[0]), hex(_span(t)[1])) for t in stacks]}", flush=True)
    if clash:
        raise SystemExit(f"[rngd] {name}: weights {clash[:3]} share device memory with the "
                         f"program scratch area — refusing to run")

    # Several programs on one device: a later program's scratch area can reach weights that an
    # earlier program loaded (seen when a small stage is followed by a larger one on the same
    # device).  Running the later program then overwrites them, and the earlier stage's output
    # changes between calls.  Move every earlier weight that overlaps the new scratch area, as
    # above, and stop if one still overlaps.
    for prev_dev, prev_cm in _LOADED:
        if prev_dev != str(dev):
            continue
        pw = prev_cm.module_weight
        for k, v in list(pw.named_parameters()) + list(pw.named_buffers()):
            if not v.numel() or not any(_span(v)[0] < _span(st)[1] and _span(st)[0] < _span(v)[1]
                                        for st in stacks):
                continue
            old = v.data
            new = torch.empty_like(old)
            new.copy_(old)
            v.data = new
            _HELD.append((str(dev), old))
            if os.environ.get("NPU_POC_RNGD_DEBUG"):
                print(f"[rngd-dbg] {name}: moved earlier weight {k} off this program's scratch area "
                      f"({old.numel() * old.element_size() / 2**20:.1f} MiB kept as the hold)", flush=True)
            if any(_span(v)[0] < _span(st)[1] and _span(st)[0] < _span(v)[1] for st in stacks):
                raise SystemExit(f"[rngd] {name}: earlier weight {k} still shares device memory with "
                                 f"this program's scratch area — refusing to run")
    _LOADED.append((str(dev), cm))
    return cm


_LOADED: list = []   # (device, compiled program) in load order, for the cross-program check above


class _Pending:
    """An in-flight call.  Holds its inputs until the result is read.

    Each device runs one in-order queue, and a plain copy is ordered behind
    everything already queued there.  So the result is read after an event
    recorded right behind this call, on a separate copy stream — otherwise
    reading call m would wait for calls m+1… as well.
    """

    def __init__(self, out, keep, event, copy_stream):
        self._out, self._keep, self._event, self._cs = out, keep, event, copy_stream

    def wait(self):
        import furiosa.torch as ft
        self._event.synchronize()
        self._cs.wait_event(self._event)
        with self._cs:
            outs = [o.cpu() if isinstance(o, torch.Tensor) else o
                    for o in pytree.tree_leaves(self._out)]
        self._out = self._keep = None
        return outs[0] if len(outs) == 1 else outs


class RngdCompiledStage:
    """One compiled stage bound to one RNGD device."""

    def __init__(self, name: str, module, device: torch.device,
                 stats: CompileStats, new_api: bool):
        self.name = name
        self.module = module
        self.device = device
        self.stats = stats
        self._new = new_api
        self._copy_stream = None
        if not new_api:
            from furiosa.torch.backend.exported_program import run_by_rngd
            self._run_by_rngd = run_by_rngd

    def run(self, *tensors: torch.Tensor) -> Any:
        # Inputs go to the device in contiguous layout: a stage boundary can
        # carry a strided view (e.g. a transposed attention tensor).
        args = tuple(t.contiguous().to(self.device) if isinstance(t, torch.Tensor) else t
                     for t in tensors)
        if self._new:
            out = self.module(*args)
        else:
            try:
                out = self._run_by_rngd(self.module, args, {},
                                        given_device=self.device)
            except ValueError as e:
                # When the compiled outputs outnumber the exported output spec
                # (e.g. caches updated in place), take the flat outputs.
                if "unflatten" not in str(e):
                    raise
                flat = self.module.module_weight.flatten_inputs(args, {})
                out = self.module.edf_module(flat, profiles=None, device=self.device)
        # Hand back host tensors: stage boundaries go via host, and a device
        # output must not be held across calls.
        outs = [o.cpu() if isinstance(o, torch.Tensor) else o
                for o in pytree.tree_leaves(out)]
        return outs[0] if len(outs) == 1 else outs

    def run_async(self, *tensors: torch.Tensor) -> Any | None:
        """Submit without waiting (furiosa-torch 2026.3+).  Returns a handle whose
        wait() gives host tensors.

        A call on a loaded program is queued on the device's in-order stream and
        returns device tensors at once; the host copy in wait() completes after
        the computation.  Calls queued on the same device keep their own outputs.
        """
        if not self._new:
            return None
        import furiosa.torch as ft
        # The input copy is queued, not waited for: a waiting copy would sit
        # behind every call already queued on this device.
        host = tuple(t.contiguous() if isinstance(t, torch.Tensor) else t for t in tensors)
        args = tuple(t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                     for t in host)
        out = self.module(*args)
        ev = ft.Event(self.device)
        ev.record()
        if self._copy_stream is None:
            self._copy_stream = ft.Stream(self.device)
        return _Pending(out, (host, args), ev, self._copy_stream)

    def __repr__(self) -> str:
        return f"<RngdCompiledStage {self.name} on {self.device}>"


class RngdBackend:
    name = "rngd"

    def __init__(self, verbose: bool = True, make_async: bool = False, tp_size: int = 1):
        import furiosa.torch  # noqa: F401 — registers the device backend
        self.verbose = verbose
        self._new = _sdk_new()
        if self._new and not os.environ.get("NPU_POC_RNGD_DEBUG"):
            logging.getLogger("furiosa_torch").setLevel(logging.WARNING)
        nd = _device_api()
        # PE fusion can be set once per process and is fixed by the first
        # device query, so it is set before anything else.
        # TP on RNGD is the number of cores fused into one device: --tp k fuses
        # k base devices (NPU_POC_RNGD_FUSION cores each) into one, so device ids
        # keep meaning base devices and a stage takes k consecutive ones.
        self.unit = int(os.environ.get("NPU_POC_RNGD_FUSION", "4"))
        self.tp = int(tp_size)
        fusion = self.unit * self.tp
        if fusion not in (1, 2, 4, 8, 16, 32):
            raise SystemExit(f"[rngd] {self.unit} cores per device × TP={self.tp} = {fusion} "
                             f"cores — must be 1, 2, 4, 8, 16 or 32")
        try:
            nd.set_fusion(fusion)
        except RuntimeError as e:
            cur = nd.get_fusion()
            if int(getattr(cur, "num_fusion", cur)) == fusion:
                pass                        # already set to the wanted value
            elif int(getattr(cur, "num_fusion", cur)) == 1:
                # Still at the default: the SDK did not accept this fusion size — e.g. the
                # cores it needs span more cards than this machine has available.
                raise SystemExit(f"[rngd] cannot form devices of {fusion} cores "
                                 f"({self.unit} cores per base device × TP={self.tp}) on this "
                                 f"machine: {e}.  Check the number of cards "
                                 f"(furiosa-smi info) or use a smaller TP")
            else:
                raise SystemExit(f"[rngd] PE fusion is already {cur} in this "
                                 f"process (wanted {fusion}) — create the backend first")
        self.fusion = nd.get_fusion()
        self._devs = list(nd.get_available_devices())
        self.cache_dir = os.environ.get("NPU_POC_RNGD_CACHE") or None
        if verbose:
            from importlib.metadata import version
            try:
                ver = version("furiosa-torch")
            except Exception:
                ver = "?"
            print(f"[rngd] furiosa-torch {ver} · fusion={self.fusion} · "
                  f"{len(self._devs)} devices {self._devs}"
                  + (f" · TP={self.tp}: each device = {self.tp} base devices "
                     f"({self.unit} cores each)" if self.tp > 1 else "")
                  + (f" · cache={self.cache_dir}" if self.cache_dir else ""),
                  flush=True)
            if not ver.startswith(VALIDATED_SDK):
                print(f"[rngd] Note: this adapter is validated with furiosa-torch "
                      f"{VALIDATED_SDK}.x; {ver} is not validated", flush=True)

    # ── Capability declaration ─────────────────────────────────────
    def caps(self) -> BackendCaps:
        return BackendCaps(
            name="rngd",
            n_devices=len(self._devs) * self.tp,   # in base devices (--device-ids)
            tp_sizes=(self.tp,),        # TP = PE fusion, fixed when the backend is created
            supports_async=self._new,   # stream submission (furiosa-torch 2026.3+)
            forced_dtype=torch.bfloat16,
            default_dtype=torch.bfloat16,
            boundary_bool_ok=False,
            device_to_device=False,
        )

    def device_count(self) -> int:
        return len(self._devs)

    def devices(self) -> list[str]:
        return list(self._devs)

    # ── Compile ─────────────────────────────────────────────────────
    def compile(self, stage_gm: GraphModule,
                example_inputs: Sequence[torch.Tensor], *,
                name: str, devices: Sequence[int],
                tp_size: int = 1,
                input_names: Sequence[str] | None = None):
        ids = sorted(int(d) for d in devices)
        k = ids[0] // self.tp if ids else -1
        if (tp_size != self.tp or ids != list(range(k * self.tp, (k + 1) * self.tp))
                or not 0 <= k < len(self._devs)):
            raise SystemExit(f"[rngd] {name}: TP={tp_size} (backend created with TP={self.tp}) "
                             f"needs {self.tp} consecutive base devices starting at a multiple "
                             f"of {self.tp} (got {list(devices)}; fused devices: {len(self._devs)})")
        dev = torch.device(self._devs[k])
        stats = CompileStats(extra={"device": str(dev)})
        args = tuple(example_inputs)
        if self.cache_dir:
            from furiosa.torch.config import cache_context
            with cache_context(root_dir=self.cache_dir):
                module = self._compile(stage_gm, args, dev, stats)
        else:
            module = self._compile(stage_gm, args, dev, stats)
        if self.verbose:
            print(f"[rngd]   {name} → {dev}  {stats}"
                  + (f" · {stats.extra['tactic_hint'].split(' · ', 1)[1]}"
                     if " · " in stats.extra.get("tactic_hint", "") else ""),
                  flush=True)
        return RngdCompiledStage(name, module, dev, stats, self._new)

    def _compile(self, stage_gm, args, dev, stats):
        import furiosa.torch as ft
        nf = int(getattr(self.fusion, "num_fusion", self.fusion))
        cfg = _with_unique_shapes(_cross_chip(_compiler_config(args, self._new), nf, self._new),
                                  _unique_shapes_limit(nf), self._new)
        if cfg is not None:
            cfg, stats.extra["tactic_hint"] = cfg
        t0 = time.time()
        ep = torch.export.export(stage_gm, args)
        stats.export_s = time.time() - t0
        stats.n_nodes_before = len(ep.graph.nodes)

        t0 = time.time()
        ep = ep.run_decompositions()                    # core ATen first
        if not self._new:
            table = {**ft.STD_DECOMPOSITIONS,
                     torch.ops.aten.split_with_sizes.default: _split_with_sizes_as_slices}
            ep = ep.run_decompositions(table)
        _strip_export_asserts(ep)
        stats.decompose_s = time.time() - t0
        stats.n_nodes_after = len(ep.graph.nodes)

        t0 = time.time()
        if self._new:
            # Compile from a private copy: moving the weights to `dev` must not
            # move tensors that other stages share with this one.
            module = _load_new(ft.CompileModule.from_exported(copy.deepcopy(ep),
                                                              compiler_config=cfg), dev,
                               str(dev))
        else:
            from furiosa.torch import compiler as fc
            from furiosa.torch.export.exported_program import ExportedProgramWeight
            from furiosa.torch.custom_ops.edf import EdfModule, CompileModule
            edf = fc.compile(ep, compiler_config=cfg)
            module = CompileModule(EdfModule(edf), ExportedProgramWeight(ep).to(dev))
        stats.compile_s = time.time() - t0
        return module

    # ── Diagnosis ───────────────────────────────────────────────────
    def unsupported_ops(self, stage_gm: GraphModule) -> list[str]:
        """Ops the compiler reports as not importable (checked by name)."""
        from furiosa.torch import compiler as fc
        bad = set()
        for n in stage_gm.graph.nodes:
            if n.op != "call_function":
                continue
            nm = getattr(n.target, "_name", None) or getattr(n.target, "__name__", str(n.target))
            try:
                if not fc.is_importable(nm):
                    bad.add(nm)
            except Exception:
                pass
        return sorted(bad)
