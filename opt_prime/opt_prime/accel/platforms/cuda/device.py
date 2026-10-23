#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""NVIDIA/AMD GPUs through torch.cuda: discovery."""
import os
import sys

from opt_prime.accel.platforms.registry import DeviceInfo, _numa_of_pci, register_discoverer  # noqa: F401


@register_discoverer("cuda")
def _cuda():
    if "torch" not in sys.modules and "CUDA_VISIBLE_DEVICES" not in os.environ:
        return []                                    # do not import torch / init CUDA just to look
    import torch
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        bdf = None
        if hasattr(p, "pci_bus_id"):
            bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        out.append(DeviceInfo("cuda", i, "nvidia" if torch.version.cuda else "amd", p.total_memory,
                              _numa_of_pci(bdf), frozenset({"prefill", "decode", "experts", "resident_weights"}),
                              {"model": p.name, "pci": bdf}))
    return out
