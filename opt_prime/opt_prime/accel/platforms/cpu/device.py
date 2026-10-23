#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""The host CPU: discovery (one device, NUMA nodes in extra)."""
import os

from opt_prime.accel.platforms.registry import DeviceInfo, _p, register_discoverer  # noqa: F401


@register_discoverer("cpu")
def _cpu():
    nodes = []
    base = _p("/sys/devices/system/node")
    if os.path.isdir(base):
        nodes = sorted(int(n[4:]) for n in os.listdir(base) if n.startswith("node") and n[4:].isdigit())
    mem = None
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        pass
    return [DeviceInfo("cpu", 0, "host", mem, None, frozenset({"experts", "decode"}),
                       {"numa_nodes": nodes or [0], "cores": os.cpu_count()})]
