#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# Usage: python3 moe_offload_check.py [--device cuda|cpu] [--dtype float32|bfloat16] [--families a,b,...]
#
# Tiny random-weight models of every transformers MoE family that uses the experts interface: offload part of the
# experts with opt_prime.accel.moe, compare logits with the unmodified model, and export the offloaded model.
# Families whose unmodified model cannot be built or run here are reported as SKIP. No downloads.
#
import argparse, copy, os, sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.accel.moe import MoEOffloadConfig, apply_moe_offload, ExpertProfiler  # noqa: E402
from opt_prime.accel.testing import FAMILIES, NUM_EXPERTS, build_tiny  # noqa: E402


class Logits(torch.nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, x): return self.inner(input_ids=x, use_cache=False).logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--families", default=",".join(FAMILIES))
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    tol = 1e-4 if dt == torch.float32 else 3e-2
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(1))
    summary = {}
    for fam in a.families.split(","):
        try:
            base = build_tiny(fam, dt).to(a.device)
            with torch.no_grad():
                ref = base(ids.to(a.device)).logits.float()
        except Exception as e:
            summary[fam] = "SKIP"; print(f"{fam:14s} SKIP (unmodified model: {type(e).__name__}: {str(e)[:90]})"); continue
        ok_f = True
        for frac in (0.0, 0.5, 1.0):
            m = copy.deepcopy(base).cpu()
            with ExpertProfiler(m) as prof, torch.no_grad():
                m(ids)
            hot = [sorted(range(NUM_EXPERTS), key=lambda e: -c[e])[:int(NUM_EXPERTS * frac)] for c in prof.as_lists()]
            cfg = MoEOffloadConfig(hot=hot)
            apply_moe_offload(m, cfg)
            m = m.to(a.device)
            with torch.no_grad():
                out = m(ids.to(a.device)).logits.float()
            err = (out - ref).abs().max().item() / max(1e-6, ref.abs().max().item())
            ok = err < tol; ok_f &= ok
            print(f"{fam:14s} hot {frac:4.0%}  rel.err {err:.1e}  pairs dev/host {cfg.stats.device_pairs}/{cfg.stats.host_pairs}"
                  f"  {'PASS' if ok else 'FAIL'}")
        try:
            ep = torch.export.export(Logits(m), (ids.to(a.device),), strict=False)
            n = sum(1 for nd in ep.graph.nodes if "offloaded_experts" in str(nd.target))
            exp = f"export OK ({n} nodes)"
        except Exception as e:
            exp = f"export FAIL ({str(e)[:70]})"
        print(f"{fam:14s} {exp}")
        summary[fam] = ("PASS" if ok_f else "FAIL") + (", export OK" if "OK" in exp else ", export FAIL")
    print("\nSUMMARY"); [print(f"  {k:14s} {v}") for k, v in summary.items()]
    sys.exit(0 if all(v.startswith(("PASS", "SKIP")) for v in summary.values()) else 1)


if __name__ == "__main__":
    main()
