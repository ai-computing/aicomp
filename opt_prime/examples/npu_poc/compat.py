"""Version workarounds — one place that says which workaround is enabled on which torch/transformers.

Why this file exists
────────────────────
The three parties require different versions, and the requirements have no intersection:

    opt_prime  official torch 2.5.0 / tf 4.46.2 · additionally verified torch 2.13.0 / tf 5.15.0
    ATOM+      rebel-compiler lists torch only as an optional extra → version-agnostic
               (measured: 2.8 ↔ 2.13 give identical numerics, whole 34.7s → 28.6s)
    RNGD       furiosa-torch requires torch==2.10.0 (its native extension is built for that ABI;
               on 2.13: undefined symbol: at::native::diagonal(...))

    → RNGD's 2.10 pin is exclusive, so no single environment exists.  Two venvs remain.
      The code, however, can stay single — instead of scattering version-dependent
      logic, it is enabled declaratively here.

Previously these workarounds were scattered over three files, so the log alone could
not tell which workarounds had been active.  Every run now prints a [compat] line.

Workarounds (C1~C9)
───────────────────
  C1 prune dead call_module     specific to torch 2.8.x
  C2 static capture             required on torch 2.10.x · recommended elsewhere
  C3 replace example-input shape  always paired with C2
  C4 capture path               tf >= 5.0 has no HFTracer → dynamo forced
  C5 model patch set            backend decides whether needed, tf decides contents
  C6 boundary mask dtype        tf >= 5.0 × sdpa × backend that does not take bool at boundaries
  C7 attn default               model-dependent
  C8 boundary scalar freezing   needed when C2 is off
  C9 compute dtype              whatever the backend forces, if it does
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch


# ─────────────────────────────────────────────────────────────────────
# Version detection
# ─────────────────────────────────────────────────────────────────────

def _mm(ver: str) -> tuple[int, int]:
    """'2.8.0+cu128' → (2, 8).  Parse failure yields (0,0), which disables workarounds."""
    m = re.match(r"(\d+)\.(\d+)", ver or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


@dataclass(frozen=True)
class Versions:
    torch: str
    transformers: str

    @property
    def torch_mm(self) -> tuple[int, int]:
        return _mm(self.torch)

    @property
    def tf_mm(self) -> tuple[int, int]:
        return _mm(self.transformers)

    def __str__(self) -> str:
        return f"torch={self.torch} transformers={self.transformers}"


def detect() -> Versions:
    try:
        import transformers
        tf = transformers.__version__
    except Exception:                                    # works without transformers too
        tf = ""
    return Versions(torch=torch.__version__, transformers=tf)


def hf_tracer_available() -> bool:
    """C4 — `transformers.utils.fx` was removed in tf >= 5.0.

    The engine also detects this and exits telling the user to use `--dynamo-capture`
    (IR.py:1663, added in feature/compatibility1).  The PoC catches it earlier
    and gives a friendlier message.
    """
    try:
        import transformers.utils.fx          # noqa: F401
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────
# Workaround plan
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CompatPlan:
    versions: Versions

    prune_dead_call_modules: bool = False       # C1
    static_capture: bool = True                 # C2 (C3 always goes with it)
    use_dynamo: bool = True                     # C4
    patch_set: tuple[str, ...] = ()             # C5
    float_boundary_mask: bool = False           # C6
    attn: str = "sdpa"                          # C7
    freeze_boundary_scalars: bool = False       # C8
    compute_dtype: torch.dtype = torch.float32  # C9

    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def log(self) -> None:
        on = []
        if self.prune_dead_call_modules:
            on.append("C1:prune-dead-nodes")
        if self.static_capture:
            on.append("C2+C3:static-capture")
        on.append("C4:" + ("dynamo" if self.use_dynamo else "HFTracer"))
        on.append(f"C5:patches={len(self.patch_set)}")
        if self.float_boundary_mask:
            on.append("C6:float-boundary-mask")
        on.append(f"C7:attn={self.attn}")
        if self.freeze_boundary_scalars:
            on.append("C8:freeze-scalars")
        on.append(f"C9:dtype={str(self.compute_dtype).replace('torch.', '')}")
        print(f"[compat] {self.versions}", flush=True)
        print(f"[compat] enabled: {' · '.join(on)}", flush=True)
        for r in self.reasons:
            print(f"[compat]   - {r}", flush=True)
        for w in self.warnings:
            print(f"[compat]   ! {w}", flush=True)


def build_plan(caps, *, want_dynamo: bool = True, want_static: bool = True,
               attn: str | None = None, dtype: torch.dtype | None = None,
               patches: tuple[str, ...] = (), auto_patch: bool = True,
               versions: Versions | None = None) -> CompatPlan:
    """Build the workaround plan from installed versions and backend capabilities."""
    v = versions or detect()
    p = CompatPlan(versions=v)
    tmaj, tmin = v.torch_mm
    fmaj, _ = v.tf_mm

    # ── C1 : dead call_module ────────────────────────────────────────
    # torch 2.8 attributes `_assert_tensor_metadata` to nn_module_stack, so
    # reconstruction produces embed(embed(ids)).  Not present on 2.5·2.10·2.13.
    if (tmaj, tmin) == (2, 8):
        p.prune_dead_call_modules = True
        p.reasons.append("torch 2.8 — dead call_modules appear, so they are pruned "
                         "(unnecessary on 2.10·2.13)")

    # ── C2 / C3 : static capture ─────────────────────────────────────
    # torch 2.10 blocks Llama dynamic-seq export with a core guard
    #   Eq(Min(64*s, 256*s), 64*s)  @ _prims_common:280 check_contiguous_sizes_strides
    # It is recommended on other versions too — with static capture there are zero
    # non-tensor boundary values, so the scalar-freezing wrapper (~60 lines) disappears
    # entirely, and NPU binaries are bound to a fixed shape anyway.
    p.static_capture = want_static
    if (tmaj, tmin) == (2, 10) and not want_static:
        p.warnings.append("on torch 2.10, dynamic capture of Llama fails with "
                          "ConstraintViolationError (measured)")
    if want_static:
        p.reasons.append("static capture — 0 boundary scalars "
                         + ("(required on torch 2.10)" if (tmaj, tmin) == (2, 10)
                            else "(recommended: no scalar-freezing wrapper needed)"))

    # ── C8 : boundary scalar freezing ────────────────────────────────
    p.freeze_boundary_scalars = not want_static

    # ── C4 : capture path ───────────────────────────────────────────
    have_tracer = hf_tracer_available()
    p.use_dynamo = want_dynamo or not have_tracer
    if not have_tracer and not want_dynamo:
        p.reasons.append(f"transformers {v.transformers} has no `transformers.utils.fx`, "
                         f"so HFTracer is unavailable → dynamo forced")

    # ── C5 : model patch set ────────────────────────────────────────
    # The backend decides whether patches are needed; the transformers version decides which.
    #   ATOM+ × tf 4.46.2 : 0 (stock HF 1.24B compiles as a whole)
    #   ATOM+ × tf 5.x    : 1 (find_packed_sequence_indices → None)
    #   RNGD  × tf 5.x    : skip_causal_mask · float_mask_native · decode_mask_arith ·
    #                       rotary_expand · layernorm
    p.patch_set = patches
    if auto_patch and not patches and fmaj >= 5 and caps.name == "rbln":
        p.patch_set = ("packed_seq",)
        p.reasons.append("tf 5.x × ATOM+ — without the single patch "
                         "`find_packed_sequence_indices → None`, submod_0 does not compile")
    if auto_patch and not patches and caps.name == "rngd":
        p.patch_set = ("skip_causal_mask", "float_mask_native", "decode_mask_arith",
                       "rotary_expand", "layernorm")
        p.reasons.append("RNGD — skip causal-mask construction for unpadded prefill, "
                         "float masks in the model dtype, expand RoPE cos/sin to "
                         "the batch size, explicit LayerNorm")

    # ── C7 : attn ───────────────────────────────────────────────────
    # transformers defaults to sdpa, and on ATOM+ the attention implementation
    # matters and is model-dependent (e.g. GPT-2 is used with sdpa).
    # The general rule: keep sdpa and make the boundary mask float (C6).
    p.attn = attn or "sdpa"

    # ── C6 : boundary mask dtype ────────────────────────────────────
    if fmaj >= 5 and p.attn == "sdpa" and not caps.boundary_bool_ok:
        p.float_boundary_mask = True
        p.reasons.append("tf 5.x × sdpa × backend not taking bool at boundaries — boundary "
                         "mask converted to float additive")

    # ── C9 : compute dtype ──────────────────────────────────────────
    if caps.forced_dtype is not None:
        p.compute_dtype = caps.forced_dtype
        if dtype is not None and dtype != caps.forced_dtype:
            p.warnings.append(f"{caps.name} forces {caps.forced_dtype} "
                              f"— ignoring requested {dtype}")
    else:
        p.compute_dtype = dtype or caps.default_dtype

    return p


# ─────────────────────────────────────────────────────────────────────
# Unmeasured — placeholders only (do not delete)
# ─────────────────────────────────────────────────────────────────────
#
# The cells below have not been measured.  Each is listed with its cost — if
# cheap and expensive cells are not distinguished, even cheap ones go stale.
#
#   ATOM+ × torch 2.10          Must be opened for the two backends to share a venv.
#                               rebel does not pin torch, so it likely works, but
#                               there are 0 measurements.  Est. cost ~30 min (install
#                               rebel into venv_rngd).
#                               → If it works, the only change is that C2 becomes required.
#
#   ATOM+ × tf 5.15             Likely the same gate as 5.1, but the full PP path
#                               was not re-verified.  Est. cost ~1 hour.
#
#   opt_prime × torch 2.10      Outside the official verification list.  Only the RNGD
#                               PoC confirmed it works in practice (Phase 1·3 pass).
#
#   RNGD × torch 2.13           Not available — the native extension is built for torch 2.10.  No re-measurement needed.
#
# When a cell above opens up, do not add branches to build_plan(); adjust the
# conditions of C1~C9 above instead.  That is why the conditions are written as version ranges.
