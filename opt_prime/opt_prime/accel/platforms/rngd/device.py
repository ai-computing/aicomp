#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""FuriosaAI RNGD: discovery (/dev/rngd, sysfs) and the busy probe (PE occupancy)."""
import os

from opt_prime.accel.platforms.lease import register_probe  # noqa: F401
from opt_prime.accel.platforms.registry import DeviceInfo, _numa_of_pci, _p, register_discoverer  # noqa: F401


@register_discoverer("rngd")
def _rngd():
    d = _p("/dev/rngd")
    if not os.path.isdir(d):
        return []
    idx = sorted({int(n[3:-4]) for n in os.listdir(d) if n.startswith("npu") and n.endswith("mgmt")
                  and n[3:-4].isdigit()})
    out = []
    for i in idx:
        bdf = None
        try:      # /sys/class/rngd_mgmt/rngd!npu<N>mgmt/busname = 0000:50:00.0
            bdf = open(_p(f"/sys/class/rngd_mgmt/rngd!npu{i}mgmt/busname")).read().strip() or None
        except OSError:
            pass
        out.append(DeviceInfo("rngd", i, "furiosa", 48 << 30, _numa_of_pci(bdf),
                              frozenset({"prefill", "decode", "experts", "graph_compile", "custom_kernels",
                                         "resident_weights", "p2p_dma"}),
                              {"model": "RNGD", "pci": bdf, "dev": f"/dev/rngd/npu{i}mgmt"}))
    return out


@register_probe("rngd")
def _rngd_busy(name):
    """PEs held by any process (Optimus or not): /sys/class/rngd_mgmt/rngd!npu<N>mgmt/pe_occupancy (1 = held)."""
    from opt_prime.accel.platforms.registry import _p
    idx = name.split(":")[1]
    try:
        occ = open(_p(f"/sys/class/rngd_mgmt/rngd!npu{idx}mgmt/pe_occupancy")).read().split()
    except OSError:
        return None                      # no sysfs (older driver / container): cannot tell, do not refuse
    held = [i for i, v in enumerate(occ) if v.strip() == "1"]
    return f"PEs {held} are held by another process (pe_occupancy)" if held else None
