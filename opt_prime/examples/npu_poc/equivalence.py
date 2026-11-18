"""Equivalence verdict — metric definitions, thresholds, and the rationale for the thresholds.

Numbers are not information unless you know how to read them.  Four metrics:

  |Δ|max   Absolute difference of the single worst-matching element.  Means
           nothing on its own — without knowing the logits' value scale you
           cannot tell whether 0.239 is large or small.
  rel      |Δ|max divided by the maximum absolute value of the reference,
           which makes it a relative magnitude.
  cos      Cosine similarity of the two tensors flattened to vectors.
           Measures direction.
           Note: computed in float64 — in fp32 the error over 4.1M elements
             is ~2e-4, producing values above 1 such as cos(x,x)=1.000188.
             The earlier PoC's cos 1.000075 was an artifact of this.
  argmax   Fraction of positions whose last-axis argmax matches — token-level
           prediction equivalence.
           Note: always report the denominator — with batch 1 × seq 16 one
             token is 6.25%.

Thresholds and rationale
────────────────────────
    rel ≤ 2e-2 · cos ≥ 0.999 · argmax ≥ 99%

NPU arithmetic is not bit-identical to fp32 on the CPU, and the per-op
difference accumulates over the layers (about 1e-2 relative over the 16 layers
of Llama-3.2-1B in our measurements).  The threshold is set at twice that.

→ Do not tighten this to allclose.  The verdict is three-tiered:
      (1) strict   allclose(atol, rtol)
      (2) relaxed  rel ≤ range_tol  ∧  cos ≥ cos_tol
      (3) argmax   argmax match ≥ argmax_tol
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


COS_TOL = 0.999
RANGE_TOL = 2e-2
ARGMAX_TOL = 0.99


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Computed in float64 — fp32 yields values above 1 on large tensors."""
    x = a.flatten().double()
    y = b.flatten().double()
    denom = x.norm() * y.norm()
    if denom == 0:
        return 1.0 if torch.equal(x, y) else 0.0
    return (x @ y / denom).item()


@dataclass
class Verdict:
    ok: bool
    level: str                   # "strict" | "relaxed" | "argmax" | "fail"
    d_max: float
    d_mean: float
    ref_max: float
    rel: float
    cos: float
    argmax: float | None
    n_argmax: int | None

    def __str__(self) -> str:
        s = (f"|Δ|max={self.d_max:.4e} rel={self.rel:.4e} "
             f"cos={self.cos:.6f}")
        if self.argmax is not None:
            s += f" argmax={self.argmax * 100:.2f}% ({self.n_argmax} positions)"
        return s + f" → {'PASS' if self.ok else 'FAIL'}({self.level})"


def judge(got: torch.Tensor, ref: torch.Tensor, *,
          atol: float = 1e-2, rtol: float = 1e-2,
          range_tol: float = RANGE_TOL, cos_tol: float = COS_TOL,
          argmax_tol: float = ARGMAX_TOL, want_argmax: bool = False) -> Verdict:
    got = got.detach().float().cpu()
    ref = ref.detach().float().cpu()
    if got.shape != ref.shape:
        return Verdict(False, "fail", float("inf"), float("inf"), 0.0,
                       float("inf"), 0.0, None, None)

    diff = (got - ref).abs()
    d_max = diff.max().item()
    d_mean = diff.mean().item()
    ref_max = ref.abs().max().item()
    rel = d_max / max(ref_max, 1e-9)
    cos = cosine(got, ref)

    am = n_am = None
    if want_argmax and got.dim() >= 2:
        g = got.argmax(-1)
        r = ref.argmax(-1)
        am = (g == r).float().mean().item()
        n_am = g.numel()

    if torch.allclose(got, ref, atol=atol, rtol=rtol):
        return Verdict(True, "strict", d_max, d_mean, ref_max, rel, cos, am, n_am)
    if rel <= range_tol and cos >= cos_tol:
        return Verdict(True, "relaxed", d_max, d_mean, ref_max, rel, cos, am, n_am)
    if am is not None and am >= argmax_tol:
        return Verdict(True, "argmax", d_max, d_mean, ref_max, rel, cos, am, n_am)
    return Verdict(False, "fail", d_max, d_mean, ref_max, rel, cos, am, n_am)


def thresholds_line() -> str:
    return (f"Verdict criteria: strict allclose(1e-2,1e-2) → "
            f"relaxed(rel ≤ {RANGE_TOL:g} ∧ cos ≥ {COS_TOL}) → "
            f"argmax ≥ {ARGMAX_TOL:.0%}  "
            f"[rationale: twice the ≈1e-2 relative difference measured over 16 layers]")
