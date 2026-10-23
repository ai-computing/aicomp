#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# Usage: python3 moe_offload_rngd_check.py [--shape qwen3|solar] [--device 0] [--compile-only]
#
# Runs the rngd backend (furiosa-torch SlotFFN graphs) on real expert shapes against the cpu backend and reports
# per-call latency for each slot bucket. Needs a free RNGD card (and furiosa-torch).
#
import argparse, os, sys, time

import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.accel.moe.backends import get_backend  # noqa: E402
from opt_prime.accel.spec.experts import ExpertSpec, ExpertWeights  # noqa: E402

SHAPES = {"qwen3": (2048, 768), "solar": (4096, 1280)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="qwen3", choices=SHAPES)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--compile-only", action="store_true")
    a = ap.parse_args()
    H, F = SHAPES[a.shape]
    E = a.experts
    spec = ExpertSpec(E, H, F)
    torch.manual_seed(0)
    w = ExpertWeights(spec, list(range(E)), (torch.randn(E, 2 * F, H) * 0.02).to(torch.bfloat16),
                      (torch.randn(E, H, F) * 0.02).to(torch.bfloat16))
    npu = get_backend("rngd", device=a.device)
    if a.compile_only:
        npu.emulate = False
        from opt_prime.accel.platforms.npu_graph import SlotFFN
        RngdCls = type(npu)
        RngdCls._setup()
        from furiosa.torch.custom_ops import CompileModule
        x = torch.zeros(1, H, dtype=torch.bfloat16); G = torch.zeros(1, 8, dtype=torch.bfloat16)
        samples = [x, G] + [t.clone() for _ in range(8) for t in (w.gate_up[0], w.down[0])]
        t0 = time.time()
        CompileModule.from_module(SlotFFN(spec, 8, False), tuple(samples))
        print(f"compiled SlotFFN K=8 for H={H} F={F} in {time.time() - t0:.0f}s (no device used)")
        return
    cpu = get_backend("cpu")
    hn, hc = npu.load(w), cpu.load(w)
    for T in (1, 4):
        for k in (1, 2, 4, 8):
            idx = torch.stack([torch.randperm(E)[:k] for _ in range(T)])
            wt = torch.rand(T, k)
            x = torch.randn(T, H, dtype=torch.bfloat16)
            t0 = time.time(); y = npu.run(hn, x, idx, wt); first = time.time() - t0
            ref = cpu.run(hc, x, idx, wt)
            rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
            ts = []
            for _ in range(20):
                t = time.perf_counter(); npu.run(hn, x, idx, wt); ts.append(time.perf_counter() - t)
            ts.sort()
            print(f"T={T} k={k}: rel.err {rel:.3e}  first call {first:.1f}s  p50 {ts[10] * 1e3:.2f} ms  "
                  f"({k * spec.expert_bytes / ts[10] / 1e9:.0f} GB/s)", flush=True)


if __name__ == "__main__":
    main()
