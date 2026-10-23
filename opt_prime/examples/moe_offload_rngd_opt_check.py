#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# Usage: python3 moe_offload_rngd_opt_check.py [--shape qwen3|solar|H,F] [--device 0] [--emulate] [--iters 50]
#
# Runs the rngd-opt backend (generated per-shape furiosa-opt kernel + expert server) on random experts of a real
# shape: checks a decode token against the cpu backend and reports per-token latency for 1..8 experts, from which
# the backend's cost model (overhead_s, bandwidth_Bps) is calibrated. Runs on RNGD chip --device.
#
import argparse, os, sys, time

import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.accel.moe.backends import get_backend  # noqa: E402
from opt_prime.accel.spec.experts import ExpertSpec, ExpertWeights  # noqa: E402

SHAPES = {"qwen3": (2048, 768), "solar": (4096, 1280)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="qwen3")
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--emulate", action="store_true")
    a = ap.parse_args()
    H, F = SHAPES[a.shape] if a.shape in SHAPES else map(int, a.shape.split(","))
    E = a.experts
    spec = ExpertSpec(E, H, F)
    g = torch.Generator().manual_seed(0)
    gu = ((torch.rand(E, 2 * F, H, generator=g) * 2 - 1) * 0.03).bfloat16()
    dn = ((torch.rand(E, H, F, generator=g) * 2 - 1) * 0.03).bfloat16()
    w = ExpertWeights(spec, list(range(E)), gu, dn)

    npu = get_backend("rngd-opt", device=a.device, slots=E, emulate=a.emulate)
    cpu = get_backend("cpu")
    t0 = time.time(); hn = npu.load(w); print(f"[rngd-opt] H={H} F={F}: build+load {E} experts in {time.time() - t0:.1f}s")
    hc = cpu.load(w)

    x = torch.randn(1, H, generator=g).bfloat16()
    idx = torch.tensor([[(5 * k + 3) % E for k in range(8)]])
    wt = torch.softmax(torch.randn(1, 8, generator=g), -1)
    ref = cpu.run(hc, x, idx, wt).float()
    got = npu.run(hn, x, idx, wt).float()
    err = (got - ref).abs().max().item() / ref.abs().max().item()
    print(f"[rngd-opt] k=8 token vs cpu backend: rel max err {err:.4f} -> {'PASS' if err < 2e-2 else 'FAIL'}")

    eb = spec.expert_bytes
    lat = {}
    for k in (1, 2, 4, 8):
        ts = []
        for i in range(a.iters + 3):
            t = time.perf_counter(); npu.run(hn, x, idx[:, :k], wt[:, :k]); ts.append(time.perf_counter() - t)
        ts = sorted(ts[3:]); p50 = ts[len(ts) // 2]; lat[k] = p50
        print(f"[rngd-opt] k={k}: p50 {p50 * 1e6:.0f} us  ({k * eb / p50 / 1e9:.0f} GB/s incl. round trip)")
    bw = (8 - 1) * eb / (lat[8] - lat[1]) if lat[8] > lat[1] else float("inf")
    print(f"[rngd-opt] fit: overhead_s ~ {(lat[1] - eb / bw) * 1e6:.0f} us, bandwidth ~ {bw / 1e9:.0f} GB/s")
    npu.close()


if __name__ == "__main__":
    main()
