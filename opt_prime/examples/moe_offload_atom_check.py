#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# Usage: python3 moe_offload_atom_check.py [--shape qwen3|solar|H,F] [--device 0] [--experts 4] [--tiny]
#
# Runs the atom backend (RBLN-compiled per-expert graphs) on random experts of a real shape against the cpu backend
# and reports per-call latency for 1..4 experts; --tiny also runs a tiny MoE model end to end with experts split
# between the atom tier and the host. Needs an idle ATOM card and the RBLN SDK (rebel-compiler).
import argparse, os, sys, time

import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.moe_offload.backends import get_backend  # noqa: E402
from opt_prime.moe_offload.spec import ExpertSpec, ExpertWeights  # noqa: E402

SHAPES = {"qwen3": (2048, 768), "solar": (4096, 1280)}


def tiny(device):
    from opt_prime.moe_offload import ExpertProfiler, MoEOffloadConfig, TierSpec, apply_moe_offload
    from opt_prime.moe_offload.testing import build_tiny
    m = build_tiny("qwen3_moe", torch.bfloat16)
    ids = torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        ref = m(ids).logits.float()
    with ExpertProfiler(m) as prof, torch.no_grad():
        m(ids)
    eb = 3 * 64 * 32 * 2
    cfg = MoEOffloadConfig(tiers=[TierSpec("atom", device, 4 * eb / (1 << 30)), TierSpec("cpu", options={"kernel": "torch"})],
                           profile=prof.as_lists())
    rep = apply_moe_offload(m, cfg)
    with torch.no_grad():
        out = m(ids).logits.float()
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    print(f"[atom] tiny qwen3_moe end to end: tiers {rep.tiers}, pairs {cfg.stats.flush().tier_pairs}, "
          f"rel max err {err:.4f} -> {'PASS' if err < 3e-2 else 'FAIL'}")
    rep.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="qwen3")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    H, F = SHAPES[a.shape] if a.shape in SHAPES else map(int, a.shape.split(","))
    E = a.experts
    spec = ExpertSpec(E, H, F)
    g = torch.Generator().manual_seed(0)
    gu = ((torch.rand(E, 2 * F, H, generator=g) * 2 - 1) * 0.03).bfloat16()
    dn = ((torch.rand(E, H, F, generator=g) * 2 - 1) * 0.03).bfloat16()
    w = ExpertWeights(spec, list(range(E)), gu, dn)
    npu, cpu = get_backend("atom", device=a.device, emulate=False), get_backend("cpu")
    t0 = time.time(); hn = npu.load(w); print(f"[atom] H={H} F={F}: compile+load {E} experts in {time.time() - t0:.1f}s")
    hc = cpu.load(w)
    x = torch.randn(1, H, generator=g).bfloat16()
    k = min(E, 4)
    idx = torch.arange(k).unsqueeze(0)
    wt = torch.softmax(torch.randn(1, k, generator=g), -1).bfloat16()
    ref, got = cpu.run(hc, x, idx, wt).float(), npu.run(hn, x, idx, wt).float()
    err = ((got - ref).abs().max() / ref.abs().max()).item()
    print(f"[atom] k={k} token vs cpu backend: rel max err {err:.4f} -> {'PASS' if err < 3e-2 else 'FAIL'}")
    for kk in range(1, k + 1):
        ts = []
        for _ in range(a.iters + 3):
            t = time.perf_counter(); npu.run(hn, x, idx[:, :kk], wt[:, :kk]); ts.append(time.perf_counter() - t)
        ts = sorted(ts[3:])
        print(f"[atom] k={kk}: p50 {ts[len(ts) // 2] * 1e6:.0f} us")
    if a.tiny:
        tiny(a.device)


if __name__ == "__main__":
    main()
