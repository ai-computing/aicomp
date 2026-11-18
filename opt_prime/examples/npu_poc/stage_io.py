"""Boundary contract — rules for values that cross *between* stages.

Cross-version validation (transformers x torch) surfaced three classes of
problems that occur at stage boundaries; they used to live in three different
places and are collected here.  This file covers integration items #1 and #6.

  (a) A non-tensor scalar crosses a boundary
       Dynamic capture leaves sym_size_int nodes, and after the PP split they
       cross the boundary as plain python ints.  For NPUs whose input
       description (input_info) cannot express a scalar, or whose tracer
       accepts only tensor outputs, they are baked in as compile-time
       constants (ConstSpecializedStage) — a binary is tied to one shape anyway.
       Note: with static capture (C2) there are zero boundary scalars and this
         mechanism is not needed at all — measured 0/0 for both PP=2 and PP=4.

  (b) A bool tensor crosses a boundary
       Some backends (BackendCaps.boundary_bool_ok=False) do not take a bool
       attn_mask as a stage *input*, so the boundaries are checked before
       compiling.  A bool inside a stage graph is fine; only stage inputs matter.
       Note: the fix belongs before capture (patches.float_causal_mask), not
         here.  This module only checks — changing the dtype of an already
         compiled graph would make it disagree with the graph.

  (c) Boundary dtype and compute dtype differ  (integration item #6)
       The dtype given here fixes the host↔NPU boundary only; it says nothing
       about the precision the NPU computes with.  Record the boundary dtype as
       such and do not read "fp32" as "exact".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


_TORCH_TO_RBLN_DTYPE = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.bool: "bool",
}


def dtype_name(t: torch.dtype) -> str:
    s = _TORCH_TO_RBLN_DTYPE.get(t)
    if s is None:
        raise NotImplementedError(f"add a dtype mapping for: {t}")
    return s


def is_freezable(v) -> bool:
    """Is this value a compile-time constant under a fixed shape?

    bool is rejected — it also reads as int, so we do not guess.
    """
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, torch.Size):
        return True
    if isinstance(v, tuple) and v and all(
            isinstance(x, int) and not isinstance(x, bool) for x in v):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────

@dataclass
class BoundarySpec:
    """Contract for what one stage receives and what it produces."""

    name: str
    tensors: list[tuple[str, list[int], str]] = field(default_factory=list)
    in_consts: dict[int, object] = field(default_factory=dict)
    out_kind: str = "single"                       # "single" | "tuple"
    out_consts: dict[int, object] = field(default_factory=dict)
    out_len: int = 1
    in_dtypes: list[torch.dtype] = field(default_factory=list)

    @property
    def n_args(self) -> int:
        return len(self.tensors) + len(self.in_consts)

    def bool_inputs(self) -> list[str]:
        return [n for (n, _, d) in self.tensors if d == "bool"]

    def summary(self) -> str:
        ph = ", ".join(f"{n}:{tuple(s)}/{d}" for n, s, d in self.tensors)
        s = f"inputs=[{ph}] out={self.out_kind}"
        if self.in_consts:
            s += " · frozen scalars " + ", ".join(
                f"arg{p}={v}" for p, v in sorted(self.in_consts.items()))
        return s


def collect(split_gm, stage_names, model, driver_inputs: dict):
    """Run once on CPU to collect each stage's boundary contract and reference values.

    The hooks fire during a normal forward pass, which may differ from the export
    graph the compiler sees — do not use the dtypes obtained here to claim what
    the compiler sees.  They are used for exactly two things: (1) rebel
    input_info (2) CPU reference values.
    """
    specs: dict[str, BoundarySpec] = {}
    handles = []

    def make_hook(tgt, submod):
        ph = [n.target for n in submod.graph.nodes if n.op == "placeholder"]

        def hook(_mod, inputs, output):
            if len(ph) != len(inputs):
                raise RuntimeError(f"{tgt}: placeholder {len(ph)} != "
                                   f"inputs {len(inputs)}")
            spec = BoundarySpec(name=tgt)
            for i, v in enumerate(inputs):
                if isinstance(v, torch.Tensor):
                    spec.tensors.append((ph[i], list(v.shape), dtype_name(v.dtype)))
                    spec.in_dtypes.append(v.dtype)
                elif is_freezable(v):
                    spec.in_consts[i] = v
                else:
                    raise RuntimeError(f"{tgt}: cannot handle input '{ph[i]}' of type "
                                       f"{type(v).__name__}")
            spec.out_kind = "tuple" if isinstance(output, (list, tuple)) else "single"
            if isinstance(output, (list, tuple)):
                spec.out_len = len(output)
                for i, v in enumerate(output):
                    if isinstance(v, torch.Tensor):
                        continue
                    if not is_freezable(v):
                        raise RuntimeError(f"{tgt}: cannot handle output[{i}] of type "
                                           f"{type(v).__name__}")
                    spec.out_consts[i] = v
            specs[tgt] = spec
        return hook

    for tgt in stage_names:
        sub = split_gm.get_submodule(tgt)
        handles.append(sub.register_forward_hook(make_hook(tgt, sub)))

    # Warning: order by placeholder order, not by dict order.
    #   Inputs are bound by name, so passing them by position can misalign
    #   them without any error.  This actually happened
    #   on the decode graph (input_ids · position_ids · attention_mask ·
    #   past_k_0 … past_v_15): a cache tensor landed in the embed slot and
    #   crashed (Expected ... Long, Int; but got CPUBFloat16Type).
    want = [n.target for n in split_gm.graph.nodes if n.op == "placeholder"]
    missing = [w for w in want if w not in driver_inputs]
    if missing:
        raise RuntimeError(f"inputs required by the graph are missing from driver: {missing}")
    ordered = [driver_inputs[w] for w in want]

    with torch.no_grad():
        full_out = model(**driver_inputs)
        split_out = split_gm(*ordered)
    for h in handles:
        h.remove()

    return specs, _logits_of(full_out), split_out


def _logits_of(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    return out[0] if isinstance(out, (list, tuple)) else out


# ─────────────────────────────────────────────────────────────────────
# (b) bool boundary guard
# ─────────────────────────────────────────────────────────────────────

def check_boundary_dtypes(specs: dict[str, BoundarySpec], caps, strict: bool = True):
    """Block bool tensors at stage boundaries on backends that do not take them.

    With strict=False only a warning is printed — for control experiments
    that want to measure anyway.
    """
    offenders = {n: s.bool_inputs() for n, s in specs.items() if s.bool_inputs()}
    if not offenders:
        return
    lines = "; ".join(f"{n}: {', '.join(b)}" for n, b in offenders.items())
    msg = (f"[stage_io] bool tensor at a stage boundary — {lines}\n"
           f"           {caps.name} does not take a bool attn_mask as a stage input.\n"
           f"           → use --float-mask to turn the mask into a float additive "
           f"mask before capture (measured 4/4 · logits cos 0.999996).")
    if caps.boundary_bool_ok:
        return
    if strict:
        raise SystemExit(msg)
    print(msg + "\n           (proceeding anyway due to --allow-bool-boundary)", flush=True)


# ─────────────────────────────────────────────────────────────────────
# (a) Boundary scalar freezing (C8)
# ─────────────────────────────────────────────────────────────────────

class ConstSpecializedStage(nn.Module):
    """Make a stage look like a module that takes tensors only.

    For NPUs whose input description (input_info) cannot express a scalar, or
    whose tracer accepts only tensor outputs.  A binary is tied to one input
    shape, so those scalars are constants.  Only tensors are exposed to the
    compiler; the constants are re-inserted at their original positions when
    calling the real stage.
    """

    def __init__(self, stage: nn.Module, spec: BoundarySpec):
        super().__init__()
        self.stage = stage
        self.const_items = sorted(spec.in_consts.items())
        self.n_args = spec.n_args
        self.drop_out_pos = set(spec.out_consts)

    def forward(self, *tensor_args):
        full = list(tensor_args)
        for pos, val in self.const_items:
            full.insert(pos, val)
        if len(full) != self.n_args:
            raise RuntimeError(f"argument reassembly produced {len(full)} "
                               f"(expected {self.n_args})")
        out = self.stage(*full)
        if self.drop_out_pos and isinstance(out, (list, tuple)):
            return tuple(v for i, v in enumerate(out)
                         if i not in self.drop_out_pos)
        return out


def specialize(stage: nn.Module, spec: BoundarySpec) -> nn.Module:
    """Wrap only when there is something to freeze.  Static capture returns the stage as-is."""
    if spec.in_consts or spec.out_consts:
        return ConstSpecializedStage(stage, spec)
    return stage


def traffic(specs: dict[str, BoundarySpec], stage_names) -> tuple[int, float]:
    """Transfer volume crossing between stages.  The first stage's inputs are not counted."""
    nbytes = {"float32": 4, "float16": 2, "bfloat16": 2,
              "int64": 8, "int32": 4, "bool": 1}
    elems = total = 0
    for tgt in stage_names[1:]:
        for _, shape, d in specs[tgt].tensors:
            n = 1
            for x in shape:
                n *= x
            elems += n
            total += n * nbytes.get(d, 4)
    return elems, total / 1024.0
